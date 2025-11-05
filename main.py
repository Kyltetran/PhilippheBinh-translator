from flask import Flask, jsonify, request, render_template, session
import json
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from get_embedding_function import get_embedding_function

from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------------------------
# App Configuration
# ------------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "123456"  # change for production

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

# ------------------------------------------------------------------------------
# Load Exact Dictionaries
# ------------------------------------------------------------------------------
with open(DATA_DIR / "exact_dicts.json", "r", encoding="utf8") as f:
    EXACT = json.load(f)

# ------------------------------------------------------------------------------
# Load Chroma Collections
# ------------------------------------------------------------------------------
embedding = get_embedding_function()

VECTORS = {
    t: Chroma(
        persist_directory=str(Path(CHROMA_PATH) / t),
        embedding_function=embedding
    )
    for t in ["sound_change", "grammar_pattern", "fixed_phrase", "vocab", "phonology_rule"]
}

print("✅ Loaded Chroma collections:", list(VECTORS.keys()))

# ------------------------------------------------------------------------------
# Translator Prompt
# ------------------------------------------------------------------------------
PROMPT_TEMPLATE = """
Bạn là chuyên gia ngôn ngữ học lịch sử, tái dựng tiếng Việt thế kỷ XVII theo văn phong Philiphê Bỉnh.

========================================
EVIDENCE (quy tắc, từ vựng, cụm cố định, ngữ âm):
{evidence}
========================================

Câu cần dịch:
"{input_text}"

========================================
QUY TẮC DỊCH – THỨ TỰ BẮT BUỘC
========================================

1. **Biến âm / ngữ âm**
   - Nếu từ hiện đại có dạng cổ tương ứng → dùng.
   - KHÔNG được thay đổi nghĩa.

2. **Từ vựng cổ / đồng nghĩa cùng trường nghĩa**
   - Nếu có từ cổ mang nghĩa TRÙNG KHỚP → dùng.
   - Nếu chỉ có đồng nghĩa hiện đại → dùng đồng nghĩa rồi áp dụng biến âm.

3. **Cú pháp cổ**
   - Áp dụng NẾU phù hợp & KHÔNG làm đổi nghĩa.

4. **Cụm cổ cố định**
   ✅ Chỉ dùng nếu cụm hiện đại TRÙNG KHỚP 100% với mục “modern” trong dữ liệu.  
   ❌ Không được dùng cụm cổ dài là mô tả hoặc câu giải nghĩa.  
   ❌ Không được dùng cụm cổ thuộc TRƯỜNG NGHĨA KHÁC.

========================================
QUY TẮC QUAN TRỌNG
========================================

❗Nếu KHÔNG tìm thấy bất kỳ dạng cổ phù hợp (âm, từ, ngữ pháp, đồng nghĩa):
→ **GIỮ NGUYÊN từ hiện đại**.  
→ Không được thay bằng cụm cổ khác nghĩa.  
→ Không được “chuyển phong cách” hoặc tự sáng tác câu cổ.

Ví dụ:
- “tôi thật xinh đẹp” → nếu không có từ cổ cho “xinh đẹp”, giữ nguyên “xinh đẹp”.

========================================
ĐẦU RA (KHÔNG dùng code block)
========================================

Dịch cổ ngữ: <kết quả dịch – giữ nguyên các từ không có dạng cổ>
Giải thích: <tóm tắt quy tắc đã áp dụng, và ghi rõ từ nào được giữ nguyên vì không có dạng cổ phù hợp>
"""

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------


def tokenize(sentence):
    """Whitespace tokens + bigrams + trigrams."""
    words = sentence.strip().split()
    phrases = []

    for i in range(len(words)):
        phrases.append(words[i])
        if i + 1 < len(words):
            phrases.append(" ".join(words[i:i+2]))
        if i + 2 < len(words):
            phrases.append(" ".join(words[i:i+3]))

    return list(dict.fromkeys(phrases))


def exact_lookup(phrases):
    found = {
        "fixed_phrases": {},
        "sound_change": {},
        "vocabulary": {}
    }

    fixed_dict = EXACT.get("fixed_phrases_modern_to_ancient", {})
    sound_dict = EXACT.get("sound_change_modern_to_ancient", {})
    vocab_dict = EXACT.get("vocabulary_modern_to_ancient", {})

    for p in phrases:
        if p in fixed_dict:
            found["fixed_phrases"][p] = fixed_dict[p]
        if p in sound_dict:
            found["sound_change"][p] = sound_dict[p]
        if p in vocab_dict:
            found["vocabulary"][p] = vocab_dict[p]

    return found


def rag_retrieve(sentence, phrases):
    evidence = {}

    # Sentence level rules (grammar + phonology)
    for t in ["grammar_pattern", "phonology_rule"]:
        docs = VECTORS[t].similarity_search(sentence, k=3)
        evidence[t] = [d.page_content for d in docs]

    # Token-level retrieval for vocab + sound-change + fixed-phrase
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


# ------------------------------------------------------------------------------
# Translator Core
# ------------------------------------------------------------------------------
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

    try:
        return json.loads(result)
    except:
        return {
            "translation": result,
            "explanation": "⚠️ Model did not output valid JSON."
        }


# ------------------------------------------------------------------------------
# Flask Routes
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    session.clear()
    return render_template("index.html")


@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})


# ✅ Main Translator Endpoint
@app.route("/translate", methods=["POST"])
def api_translate():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    out = translate_text(text)

    return jsonify({
        "success": True,
        "translation": out["translation"],
        "explanation": out.get("explanation", ""),
        "confidence": 0.92,  # placeholder
        "method": data.get("method", "llm_refined"),
        "word_breakdown": []  # optional
    })


# ✅ System Stats (required by index.html)
@app.route("/stats", methods=["GET"])
def stats():
    try:
        vocab_count = len(EXACT.get("vocabulary_modern_to_ancient", {}))
        fixed_phrase_count = len(
            EXACT.get("fixed_phrases_modern_to_ancient", {}))
        sound_change_count = len(
            EXACT.get("sound_change_modern_to_ancient", {}))

        collections_count = {}
        for name, store in VECTORS.items():
            try:
                collections_count[name] = store._collection.count()
            except:
                collections_count[name] = 0

        return jsonify({
            "vocabulary_entries": vocab_count,
            "fixed_phrases": fixed_phrase_count,
            "sound_changes": sound_change_count,
            "collections": collections_count
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Alternative translations (required by index.html)
@app.route("/alternatives", methods=["POST"])
def alternatives():
    data = request.json
    text = data.get("text", "")
    n = int(data.get("n", 5))

    if not text:
        return jsonify({"success": False, "error": "Missing text"})

    model = ChatOpenAI(model="gpt-4o", temperature=0.8)

    prompt = f"""
Sinh ra {n} bản dịch cổ ngữ thế kỷ XVII cho câu sau:

"{text}"

Xuất ra danh sách JSON:
[
  {{"ancient": "...", "source": "model"}},
  ...
]
"""

    raw = model.invoke(prompt).content

    try:
        alts = json.loads(raw)
        return jsonify({"success": True, "alternatives": alts})
    except:
        return jsonify({"success": False, "error": "Model output invalid JSON"})


# deploy
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)
# run local
if __name__ == "__main__":
    app.run()
