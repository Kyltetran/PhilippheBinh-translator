# convert_json.py
import os
import json
import pandas as pd
from docx import Document
from pathlib import Path

# ✅ ALWAYS point ROOT to the folder containing THIS script
ROOT = Path(__file__).resolve().parent

# ✅ Data folder always relative to project root
DATA_DIR = ROOT / "data"
CSV_DIR = DATA_DIR   # scan all CSV here
DOCS_DIR = DATA_DIR  # scan all DOCX here

# ✅ Output folder
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_phonology_rule(csv_path):
    df = pd.read_csv(csv_path)
    out = []
    for _, r in df.iterrows():
        out.append({
            "type": "sound_change",
            "ancient": str(r.get('Dạng cổ (Philiphê Bỉnh)', '')).strip(),
            "modern": str(r.get('Dạng hiện đại', '')).strip(),
            "rule": str(r.get('Loại biến âm', '')).strip(),
            "note": str(r.get('Nhận xét', '')).strip(),
            "source": csv_path.name
        })
    return out


def parse_generic_table(csv_path, kind, mapping):
    df = pd.read_csv(csv_path)
    out = []
    for _, r in df.iterrows():
        chunk = {"type": kind, "source": csv_path.name}
        for k, col in mapping.items():
            chunk[k] = str(r.get(col, '')).strip()
        out.append(chunk)
    return out


def parse_docx_rules(docx_path):
    doc = Document(docx_path)
    out = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            out.append({
                "type": "phonology_rule",
                "text": text,
                "source": docx_path.name
            })
    return out


def main():
    all_chunks = []

    print("🔍 Searching for CSV & DOCX in:", DATA_DIR)

    # ✅ Automatically detect files
    csv_files = list(DATA_DIR.glob("*.csv"))
    docx_files = list(DATA_DIR.glob("*.docx"))

    print("✅ Found CSV files:", [f.name for f in csv_files])
    print("✅ Found DOCX files:", [f.name for f in docx_files])

    # ✅ Identify files by their expected names
    for csv in csv_files:
        name = csv.name

        if "phonology" in name or "bien" in name:
            all_chunks += parse_phonology_rule(csv)

        elif "grammar" in name:
            mapping = {
                "old_pattern": 'Cấu trúc / Từ ngữ',
                "meaning": 'Ngữ nghĩa',
                "context": 'Ngữ cảnh (trích nguyên văn)'
            }
            all_chunks += parse_generic_table(csv, 'grammar_pattern', mapping)

        elif "fixed" in name:
            mapping = {
                "ancient": 'Cụm từ cố định',
                "context": 'Ngữ cảnh (trích nguyên văn)',
                "gloss": 'Giải nghĩa'
            }
            all_chunks += parse_generic_table(csv, 'fixed_phrase', mapping)

        elif "vocab" in name or "ancient" in name:
            mapping = {
                "ancient": 'Từ ngữ',
                "gloss": 'Ngữ nghĩa',
                "context": 'Ngữ cảnh (trích nguyên văn)'
            }
            all_chunks += parse_generic_table(csv, 'vocab', mapping)

        else:
            print(f"⚠️ WARNING: CSV '{name}' not recognized, skipping.")

    # ✅ Parse docx files
    for docx in docx_files:
        all_chunks += parse_docx_rules(docx)

    # ✅ Save chunks.jsonl inside the correct folder
    out_file = OUT_DIR / 'chunks.jsonl'
    with open(out_file, 'w', encoding='utf8') as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(all_chunks)} chunks to {out_file}")

    # ✅ Build exact-match dictionaries
    exact = {
        "fixed_phrases": {},
        "vocab_ancient_to_modern": {},
        "vocab_modern_to_ancient": {},
        "sound_change_modern_to_ancient": {}
    }

    for c in all_chunks:
        t = c.get('type')
        if t == 'fixed_phrase':
            exact['fixed_phrases'][c.get('ancient')] = c.get('gloss')
        if t == 'vocab':
            exact['vocab_ancient_to_modern'][c.get('ancient')] = c.get('gloss')
        if t == 'sound_change':
            modern = c.get('modern')
            if modern:
                exact['sound_change_modern_to_ancient'][modern] = {
                    "ancient": c.get('ancient'),
                    "rule": c.get('rule'),
                    "note": c.get('note')
                }

    exact_path = OUT_DIR / 'exact_dicts.json'
    with open(exact_path, 'w', encoding='utf8') as f:
        json.dump(exact, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved exact dictionaries to {exact_path}")


if __name__ == '__main__':
    main()
