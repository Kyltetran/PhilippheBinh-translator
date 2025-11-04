from flask import Flask, jsonify, request, render_template, session
import json
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from get_embedding_function import get_embedding_function

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = "123456"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

# -------------------------------
# Load exact dictionaries
# -------------------------------

with open(DATA_DIR / "exact_dicts.json", "r", encoding="utf8") as f:
    EXACT = json.load(f)

# -------------------------------
# Load vector stores (1 per type)
# -------------------------------

embedding = get_embedding_function()

VECTORS = {
    t: Chroma(
        persist_directory=str(Path(CHROMA_PATH) / t),
        embedding_function=embedding
    )
    for t in ["sound_change", "grammar_pattern", "fixed_phrase", "vocab", "phonology_rule"]
}

print("✅ Loaded Chroma collections:", list(VECTORS.keys()))

# -------------------------------
# Translator Prompt
# -------------------------------

PROMPT_TEMPLATE = """
Bạn là chuyên gia ngôn ngữ học lịch sử, chuyên dịch tiếng Việt hiện đại sang **cổ ngữ Philíphe Bỉnh thế kỷ XVII**.

Dữ liệu ngôn ngữ và quy tắc đi kèm dưới đây là CỐ ĐỊNH, TUYỆT ĐỐI ĐÚNG:

----------------------
EVIDENCE:
{evidence}
----------------------

YÊU CẦU DỊCH:
Câu gốc (hiện đại):
"{input_text}"

HƯỚNG DẪN BẮT BUỘC:
1. Nếu tồn tại **cụm cổ cố định** → dùng ngay.
2. Nếu từ hiện đại có trong bảng biến âm → chuyển sang dạng cổ tương ứng.
3. Nếu từ không có trong bảng → tham khảo quy tắc ngữ âm & văn cảnh trong evidence.
4. Tôn trọng văn phong và chính tả Philíphe Bỉnh.
5. KHÔNG được bịa từ mới.
6. Giữ ý nghĩa gốc, chỉ chuyển hình thức ngôn ngữ.

Hãy xuất ra JSON:
{{
  "translation": "...",
  "explanation": "..."
}}
"""

# -------------------------------
# Helper functions
# -------------------------------


def tokenize(sentence):
    """Whitespace tokens + 2-word + 3-word phrases."""
    words = sentence.strip().split()
    phrases = []
    for i in range(len(words)):
        phrases.append(words[i])
        if i+1 < len(words):
            phrases.append(" ".join(words[i:i+2]))
        if i+2 < len(words):
            phrases.append(" ".join(words[i:i+3]))
    return list(dict.fromkeys(phrases))


def exact_lookup(phrases):
    found = {
        "fixed_phrases": {},
        "sound_change": {}
    }
    for p in phrases:
        if p in EXACT["fixed_phrases"]:
            found["fixed_phrases"][p] = EXACT["fixed_phrases"][p]
        if p in EXACT["sound_change_modern_to_ancient"]:
            found["sound_change"][p] = EXACT["sound_change_modern_to_ancient"][p]
    return found


def rag_retrieve(sentence, phrases):
    """Hybrid targeted retrieval strategy."""
    evidence = {}

    # Sentence-level retrieval
    for t in ["grammar_pattern", "phonology_rule"]:
        docs = VECTORS[t].similarity_search(sentence, k=3)
        evidence[t] = [d.page_content for d in docs]

    # Token-level retrieval
    for t in ["vocab", "sound_change", "fixed_phrase"]:
        evidence[t] = []
        for p in phrases:
            docs = VECTORS[t].similarity_search(p, k=2)
            for d in docs:
                evidence[t].append({
                    "token": p,
                    "content": d.page_content,
                    "meta": d.metadata
                })

    return evidence


# -------------------------------
# Hybrid Translator
# -------------------------------

def translate_text(input_text):

    phrases = tokenize(input_text)
    exact = exact_lookup(phrases)
    rag = rag_retrieve(input_text, phrases)

    combined_evidence = {
        "exact_matches": exact,
        "retrieved": rag
    }

    prompt = PROMPT_TEMPLATE.format(
        input_text=input_text,
        evidence=json.dumps(combined_evidence, ensure_ascii=False, indent=2)
    )

    model = ChatOpenAI(model="gpt-4o", temperature=0)
    result = model.invoke(prompt).content

    # Try JSON parse
    try:
        return json.loads(result)
    except:
        return {
            "translation": result,
            "explanation": "⚠ Model did not output strict JSON."
        }


# -------------------------------
# Flask Routes
# -------------------------------

@app.route("/")
def index():
    session.clear()
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def api_translate():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    out = translate_text(text)
    return jsonify(out)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


# deploy
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)

# run local
if __name__ == "__main__":
    app.run()
