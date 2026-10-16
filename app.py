"""የጥናት አጋር — Study Assistant backend (Fast & Optimized Version)."""
import io
import json
import os
import re
import time
import uuid
from pathlib import Path

from google import genai
from google.genai import types
import psycopg
from psycopg.types.json import Jsonb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pypdf import PdfReader

# Fast model and optimized token limit to eliminate latency
MODEL = os.getenv("MODEL", "gemini-2.0-flash")
MAX_CHARS = 15_000
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

# Initialize Gemini Client
client = genai.Client()
app = FastAPI(title="የጥናት አጋር")

# ---------- storage ----------
DATABASE_URL = os.getenv("DATABASE_URL")
_MEM: dict[str, dict] = {}


@app.on_event("startup")
def init_db():
    if not DATABASE_URL:
        return
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS modules (
                   id TEXT PRIMARY KEY,
                   name TEXT,
                   text TEXT NOT NULL,
                   notes TEXT NOT NULL,
                   quiz JSONB,
                   created_at TIMESTAMPTZ DEFAULT now()
               )"""
        )


def save_module(mid: str, name: str, text: str, notes: str) -> None:
    if not DATABASE_URL:
        _MEM[mid] = {"id": mid, "name": name, "text": text, "notes": notes, "quiz": None}
        return
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            "INSERT INTO modules (id, name, text, notes) VALUES (%s, %s, %s, %s)",
            (mid, name, text, notes),
        )


def load_module(mid: str) -> dict | None:
    if not DATABASE_URL:
        return _MEM.get(mid)
    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT id, name, text, notes, quiz FROM modules WHERE id = %s", (mid,)
        ).fetchone()
    if not row:
        return None
    return {"id": row[0], "name": row[1], "text": row[2], "notes": row[3], "quiz": row[4]}


def save_quiz(mid: str, quiz: list) -> None:
    if not DATABASE_URL:
        _MEM[mid]["quiz"] = quiz
        return
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("UPDATE modules SET quiz = %s WHERE id = %s", (Jsonb(quiz), mid))


TEACHER = """You are a patient, clear teacher helping a student study the module given below.
Language: write in Amharic by default. If the student writes in English, answer in English.
Base everything on the module. If something is not covered in the module, say so
briefly, then help with general knowledge while making clear it is outside the module.
Be accurate; never invent facts the module does not support."""


# ---------- Fast Auto-Retry Helper ----------
def ask(system: str, prompt: str, history: list[dict] = None) -> str:
    full_prompt = f"{prompt}"
    if history:
        formatted_hist = ""
        for h in history:
            role = "Student" if h["role"] == "user" else "Teacher"
            formatted_hist += f"{role}: {h['content']}\n"
        full_prompt = f"Previous conversation:\n{formatted_hist}\nStudent: {prompt}"

    max_retries = 3
    backoff_delay = 1.5  # Reduced delay for minimal waiting time

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.3,
                ),
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "UNAVAILABLE" in error_str or "high demand" in error_str:
                if attempt < max_retries - 1:
                    time.sleep(backoff_delay * (attempt + 1))
                    continue
            raise HTTPException(502, f"AI አገልግሎቱ አልመለሰም: {e}")


def teacher_system(text: str) -> str:
    return f"{TEACHER}\n\n<module>\n{text}\n</module>"


def parse_json(raw: str):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise HTTPException(502, "ከAI የመጣው መልስ ሊነበብ አልቻለም። እንደገና ሞክር።")


def extract_text(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if ext == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    if ext in (".txt", ".md"):
        return data.decode("utf-8", errors="ignore")
    raise HTTPException(400, "PDF፣ DOCX ወይም TXT ፋይል ብቻ ነው የሚደገፈው።")


def get_module(mid: str) -> dict:
    m = load_module(mid)
    if not m:
        raise HTTPException(404, "ሞጁሉ አልተገኘም። እባክህ እንደገና ጫነው።")
    return m


# ---------- routes ----------
@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "ፋይሉ በጣም ትልቅ ነው (ከ15MB መብለጥ የለበትም)።")
    try:
        text = extract_text(file.filename or "", data).strip()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "ፋይሉን ማንበብ አልተቻለም።")
    if len(text) < 50:
        raise HTTPException(
            400, "በፋይሉ ውስጥ ጽሑፍ አልተገኘም። ስካን የተደረገ (ምስል) PDF ከሆነ መጀመሪያ ወደ ጽሑፍ ቀይረው።"
        )
    text = text.replace("\x00", "")[:MAX_CHARS]

    prompt = (
        "Analyze this module and create structured study notes organized clearly by Chapters or Sections "
        "(e.g., 'ምዕራፍ 1: ...', 'ምዕራፍ 2: ...').\n"
        "Format (Markdown): Use '## ' heading for each Chapter/Main Section, followed by concise "
        "bullet points ('- '). Put key terms and definitions in **bold**. "
        "Keep each bullet point to one or two lines. At the very end, include a '## ' section "
        "summarizing the 3–5 most important key takeaways from the entire module."
    )

    notes = ask(teacher_system(text), prompt)
    mid = uuid.uuid4().hex[:12]
    save_module(mid, file.filename or "", text, notes.replace("\x00", ""))
    return {"id": mid, "name": file.filename, "notes": notes}


@app.get("/api/module/{mid}")
def open_module(mid: str):
    m = get_module(mid)
    return {"id": m["id"], "name": m["name"], "notes": m["notes"]}


class ChatIn(BaseModel):
    id: str
    message: str
    history: list[dict] = []


@app.post("/api/chat")
def chat(body: ChatIn):
    m = get_module(body.id)
    history = [
        {"role": h["role"], "content": h["content"]}
        for h in body.history[-10:]
        if h.get("role") in ("user", "assistant") and h.get("content")
    ]
    reply = ask(teacher_system(m["text"]), body.message, history=history)
    return {"reply": reply}


class QuizIn(BaseModel):
    id: str
    count: int = 5


@app.post("/api/quiz")
def quiz(body: QuizIn):
    m = get_module(body.id)
    n = max(3, min(body.count, 10))
    prompt = (
        f"Write {n} practice questions covering different chapters of this module, in Amharic: about 60% "
        "multiple choice (4 options, exactly one correct) and the rest short-answer. "
        "Vary the difficulty across chapters.\n"
        "Return ONLY JSON, no other text, in this shape:\n"
        '{"questions":[{"type":"mcq","question":"...","options":["..","..","..",".."],'
        '"answer":0,"explanation":"..."},'
        '{"type":"short","question":"...","answer":"model answer","explanation":"..."}]}\n'
        'For "mcq", "answer" is the 0-based index of the correct option.'
    )
    raw = ask(teacher_system(m["text"]), prompt)
    qs = parse_json(raw).get("questions", [])
    clean = []
    for q in qs:
        if q.get("type") == "mcq" and isinstance(q.get("options"), list) and isinstance(q.get("answer"), int):
            if 0 <= q["answer"] < len(q["options"]):
                clean.append(q)
        elif q.get("type") == "short" and q.get("answer"):
            clean.append(q)
    if not clean:
        raise HTTPException(502, "ጥያቄዎችን ማዘጋጀት አልተቻለም። እንደገና ሞክር።")
    save_quiz(body.id, clean)
    return {
        "questions": [
            {"type": q["type"], "question": q["question"], "options": q.get("options")}
            for q in clean
        ]
    }


class GradeIn(BaseModel):
    id: str
    answers: list[str]


GRADER = """You are a fair, encouraging teacher marking a student's answers.
For each item compare the student's answer with the reference answer. Accept correct answers
worded differently. Score 1 (correct), 0.5 (partly correct), or 0 (wrong or blank).
Write feedback in Amharic, 1-3 sentences: say what was right, correct any mistake, and add a
short extra explanation that helps the student understand.
Return ONLY JSON: {"results":[{"id":0,"score":1,"feedback":"..."}]}"""


@app.post("/api/grade")
def grade(body: GradeIn):
    m = get_module(body.id)
    qs = m.get("quiz")
    if not qs:
        raise HTTPException(400, "መጀመሪያ ጥያቄዎችን አዘጋጅ።")

    results: list[dict | None] = [None] * len(qs)
    shorts = []
    for i, q in enumerate(qs):
        a = body.answers[i].strip() if i < len(body.answers) else ""
        if q["type"] == "mcq":
            ok = a == str(q["answer"])
            results[i] = {
                "score": 1 if ok else 0,
                "feedback": q.get("explanation", ""),
                "correct_answer": q["options"][q["answer"]],
            }
        else:
            shorts.append(
                {"id": i, "question": q["question"], "reference": q["answer"], "student": a}
            )

    if shorts:
        sys = f"{GRADER}\n\n<module>\n{m['text']}\n</module>"
        raw = ask(sys, json.dumps(shorts, ensure_ascii=False))
        by_id = {r.get("id"): r for r in parse_json(raw).get("results", [])}
        for s in shorts:
            r = by_id.get(s["id"], {})
            results[s["id"]] = {
                "score": float(r.get("score", 0)),
                "feedback": r.get("feedback", ""),
                "correct_answer": s["reference"],
            }

    return {
        "results": results,
        "total": sum(r["score"] for r in results),
        "count": len(qs),
    }
