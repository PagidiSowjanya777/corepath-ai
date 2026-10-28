import os
import re
import time
import requests
from collections import defaultdict, deque
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

app = Flask(__name__,template_folder=".")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
USE_SEARCH = os.getenv("USE_SEARCH", "false").strip().lower() == "true"
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "15"))

ENGINEERING_BRANCHES = [
    "Computer Science and Engineering (CSE)",
    "Information Technology (IT)",
    "Electronics and Communication Engineering (ECE)",
    "Electrical and Electronics Engineering (EEE)",
    "Mechanical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Aerospace Engineering",
    "Biomedical Engineering",
    "Instrumentation and Control Engineering",
    "Mechatronics Engineering",
    "Industrial Engineering",
    "Environmental Engineering",
    "Materials / Metallurgical Engineering",
    "Engineering Mathematics",
    "Physics and Chemistry",
    "Other engineering subject",
]

# ---------- simple per-person request limit ----------
_hits = defaultdict(deque)


def client_ip():
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def rate_limited(ip):
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_HOUR:
        return True
    q.append(now)
    return False


# ---------- split AI text into answer + follow-up questions ----------
def split_followups(text):
    marker = "FOLLOWUPS:"
    if marker not in text:
        return text.strip(), []
    main, _, tail = text.rpartition(marker)
    questions = []
    for line in tail.splitlines():
        line = re.sub(r"^[-*\u2022\d.)\s]+", "", line.strip()).strip()
        if line:
            questions.append(line)
    return main.strip(), questions[:3]


@app.get("/")
def home():
    return render_template("index.html", branches=ENGINEERING_BRANCHES)


def extract_reference(upload):
    """Extract text from a user-provided TXT, MD, or text-based PDF file."""
    if not upload or not upload.filename:
        return ""
    filename = upload.filename.lower()
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(upload.stream)
            pages = []
            for page in reader.pages[:30]:
                pages.append(page.extract_text() or "")
            return "\n\n".join(pages)[:18000]
        if filename.endswith((".txt", ".md")):
            raw = upload.read()
            return raw.decode("utf-8", errors="replace")[:18000]
    except Exception as exc:
        raise ValueError("Could not read that file. Try a text-based PDF, TXT, or MD file.") from exc
    raise ValueError("Unsupported file type. Upload a PDF, TXT, or MD file.")


def clean_text(value, limit):
    value = (value or "").strip()
    return value[:limit]


@app.post("/ask")
def ask():
    if not GEMINI_API_KEY:
        return jsonify({
            "error": "Gemini API key is missing. Add GEMINI_API_KEY to your .env file, save it, and restart the app."
        }), 400

    question = clean_text(request.form.get("question"), 5000)
    branch = clean_text(request.form.get("branch"), 160)
    subject = clean_text(request.form.get("subject"), 160)
    level = clean_text(request.form.get("level", "Undergraduate"), 80)

    if not question:
        return jsonify({"error": "Please enter a question first."}), 400

    if rate_limited(client_ip()):
        return jsonify({
            "error": f"You have reached the limit of {RATE_LIMIT_PER_HOUR} questions per hour. Please try again later."
        }), 429

    try:
        reference_text = extract_reference(request.files.get("reference"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    reference_section = ""
    if reference_text:
        reference_section = (
            "\n\nUSER-PROVIDED REFERENCE MATERIAL (may contain errors; assess it critically):\n"
            + reference_text
            + "\n\nIf the reference conflicts with established facts, explain the conflict instead of silently repeating it."
        )

    prompt = f"""
You are CorePath AI, a careful engineering education tutor.

Student context:
- Engineering branch: {branch or "Not specified"}
- Subject: {subject or "Not specified"}
- Level: {level}

Answer the student's question:
{question}
{reference_section}

Answering rules:
1. Be educational, clear, and technically careful. Use headings and numbered steps when useful.
2. For numerical problems, state assumptions, list known values, show the formula, substitute values, calculate with units, and check whether the result is reasonable.
3. Do not invent facts, citations, standards, equations, or numerical values. If uncertain or if information depends on a current standard/code, say so and tell the student what authoritative source to check.
4. Distinguish assumptions from verified information. Do not claim you verified something unless you actually have a source for it.
5. For safety-critical engineering, real structures, electrical installations, medical devices, or industrial operations, provide educational context and recommend review by a qualified professional and applicable local codes.
6. If the question is ambiguous, state the ambiguity and ask a short clarifying question or answer with explicit assumptions.
7. Prefer SI units unless the student requests otherwise.
8. Finish the answer with a short "Quick check" or "Key takeaway" when useful.
9. After everything else, write a new line containing exactly FOLLOWUPS: and then 3 short, related follow-up questions the student might ask next, one per line, each starting with "- ". Write nothing after them.
"""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    headers = {"x-goog-api-key": GEMINI_API_KEY}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2200
        }
    }
    if USE_SEARCH:
        payload["tools"] = [{"google_search": {}}]

    try:
        response = None
        for attempt in range(3):
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            if response.status_code != 429:
                break
            time.sleep(3 * (attempt + 1))

        if response.status_code == 429:
            return jsonify({
                "error": (
                    f"Gemini quota exceeded for model '{GEMINI_MODEL}'. "
                    "Wait a minute and try again, or set a different GEMINI_MODEL in .env. "
                    "You can check available quota at https://ai.dev/rate-limit"
                )
            }), 429

        if not response.ok:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except ValueError:
                detail = response.text
            return jsonify({
                "error": f"Gemini API returned an error ({response.status_code}): {detail[:700]}"
            }), 502

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return jsonify({"error": "The AI did not return an answer. Please rephrase your question and try again."}), 502

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        full_text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
        if not full_text:
            full_text = "The AI response was empty. Please try again."

        answer, followups = split_followups(full_text)

        grounding = candidate.get("groundingMetadata", {})
        sources = []
        for chunk in grounding.get("groundingChunks", []):
            web = chunk.get("web", {})
            title = web.get("title") or web.get("uri") or "Web source"
            uri = web.get("uri")
            if uri and all(item["url"] != uri for item in sources):
                sources.append({"title": title, "url": uri})
        queries = grounding.get("webSearchQueries", [])

        return jsonify({
            "answer": answer,
            "followups": followups,
            "sources": sources[:8],
            "search_queries": queries[:5],
            "note": "AI-generated explanations can contain errors. Verify important calculations, standards, and safety-related decisions with textbooks, official documentation, and qualified instructors."
        })
    except requests.Timeout:
        return jsonify({"error": "The request timed out. Check your internet connection and try again."}), 504
    except requests.RequestException:
        return jsonify({"error": "Could not connect to Gemini. Check your internet connection and try again."}), 502
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "The AI response could not be read. Please try again."}), 502


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "The file is too large. Please upload a file smaller than 10 MB."}), 413


if __name__ == "__main__":
    # Local development only. For public deployment, use a production WSGI server.
    app.run(host="127.0.0.1", port=5000, debug=False)