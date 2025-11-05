"""
Convert all CSV files to structured JSON format
Run this first to prepare your data
"""

import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. PHONOLOGY RULES
# pb_phonology_rule.csv
# Columns:
#   Dạng âm cổ, Dạng âm hiện đại, Luật biến đổi, Ngữ cảnh
# => JSON: { ancient, modern, rule, quote }
# ============================================================

def convert_phonology_rules():
    df = pd.read_csv(RAW_DIR / "pb_phonology_rule.csv")

    sound_changes = []
    for idx, row in df.iterrows():
        sound_changes.append({
            "id": f"sc_{idx+1:03d}",
            "ancient": str(row["Dạng âm cổ"]).strip(),
            "modern": str(row["Dạng âm hiện đại"]).strip(),
            "rule": str(row["Luật biến đổi"]).strip(),
            "quote": str(row.get("Ngữ cảnh", "")).strip(),
        })

    with open(PROCESSED_DIR / "sound_changes.json", "w", encoding="utf-8") as f:
        json.dump(sound_changes, f, ensure_ascii=False, indent=2)

    print(f"✓ Converted {len(sound_changes)} sound changes")
    return sound_changes


# ============================================================
# 2. VOCABULARY
# pb_ancient_vocab.csv
# Columns:
#   Từ vựng cổ, Từ vựng hiện đại, Ngữ cảnh
# => JSON: { ancient, modern, quote }
# ============================================================

def convert_vocabulary():
    df = pd.read_csv(RAW_DIR / "pb_ancient_vocab.csv")

    vocabulary = []
    for idx, row in df.iterrows():
        ancient = str(row["Từ vựng cổ"]).strip()
        modern_raw = str(row["Từ vựng hiện đại"]).strip()

        # Split modern equivalents by comma
        modern_list = [x.strip() for x in modern_raw.split(",")]

        # Parse contexts
        contexts = []
        if pd.notna(row["Ngữ cảnh"]):
            for ctx in str(row["Ngữ cảnh"]).split("\n"):
                if ctx.strip():
                    contexts.append({"context": ctx.strip()})

        vocabulary.append({
            "id": f"vocab_{idx+1:03d}",
            "ancient": ancient,
            "modern": modern_list,
            "quote": contexts
        })

    with open(PROCESSED_DIR / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)

    print(f"✓ Converted {len(vocabulary)} vocabulary entries")
    return vocabulary


# ============================================================
# 3. GRAMMAR
# pb_grammar.csv
# Columns:
#   Cấu trúc cổ, Cấu trúc hiện đại, Ngữ cảnh
# => JSON: { ancient, modern, quote }
# ============================================================

def convert_grammar_patterns():
    df = pd.read_csv(RAW_DIR / "pb_grammar.csv")

    grammar_patterns = []
    for idx, row in df.iterrows():
        ancient = str(row["Cấu trúc cổ"]).strip()
        modern = str(row["Cấu trúc hiện đại"]).strip()

        # Parse quote
        quotes = []
        if pd.notna(row["Ngữ cảnh"]):
            for q in str(row["Ngữ cảnh"]).split("\n"):
                if q.strip():
                    quotes.append({"context": q.strip()})

        grammar_patterns.append({
            "id": f"gram_{idx+1:03d}",
            "ancient": ancient,
            "modern": modern,
            "quote": quotes
        })

    with open(PROCESSED_DIR / "grammar_patterns.json", "w", encoding="utf-8") as f:
        json.dump(grammar_patterns, f, ensure_ascii=False, indent=2)

    print(f"✓ Converted {len(grammar_patterns)} grammar patterns")
    return grammar_patterns


# ============================================================
# 4. FIXED PHRASES / IDIOMS
# pb_fixed_phrase.csv
# Columns:
#   Cụm từ cổ, Ngữ cảnh, Giải nghĩa hiện đại
# => JSON: { ancient, modern, quote }
# ============================================================

def convert_fixed_phrases():
    df = pd.read_csv(RAW_DIR / "pb_fixed_phrase.csv")

    fixed_phrases = []
    for idx, row in df.iterrows():
        quotes = []
        if pd.notna(row["Ngữ cảnh"]):
            for q in str(row["Ngữ cảnh"]).split("\n"):
                if q.strip():
                    quotes.append({"context": q.strip()})

        fixed_phrases.append({
            "id": f"phrase_{idx+1:03d}",
            "ancient": str(row["Cụm từ cổ"]).strip(),
            "modern": str(row["Giải nghĩa hiện đại"]).strip(),
            "quote": quotes
        })

    with open(PROCESSED_DIR / "fixed_phrases.json", "w", encoding="utf-8") as f:
        json.dump(fixed_phrases, f, ensure_ascii=False, indent=2)

    print(f"✓ Converted {len(fixed_phrases)} fixed phrases")
    return fixed_phrases


# ============================================================
# 5. EXACT DICTIONARY (ALL LOOKUPS)
# Keep all original structure, only adjust to new JSON format
# ============================================================

# def build_exact_dictionaries(sound_changes, vocabulary, fixed_phrases):
#     exact = {
#         "sound_change_modern_to_ancient": {},
#         "sound_change_ancient_to_modern": {},
#         "vocabulary_modern_to_ancient": {},
#         "vocabulary_ancient_to_modern": {},
#         "fixed_phrases_modern_to_ancient": {},
#         "fixed_phrases_ancient_to_modern": {},
#     }

#     # Sound changes
#     for sc in sound_changes:
#         exact["sound_change_modern_to_ancient"][sc["modern"]] = sc["ancient"]
#         exact["sound_change_ancient_to_modern"][sc["ancient"]] = sc["modern"]

#     # Vocabulary
#     for v in vocabulary:
#         for modern_word in v["modern"]:
#             exact["vocabulary_modern_to_ancient"][modern_word] = v["ancient"]
#         exact["vocabulary_ancient_to_modern"][v["ancient"]] = v["modern"]

#     # Fixed phrases
#     for fp in fixed_phrases:
#         exact["fixed_phrases_modern_to_ancient"][fp["modern"]] = fp["ancient"]
#         exact["fixed_phrases_ancient_to_modern"][fp["ancient"]] = fp["modern"]

#     with open(PROCESSED_DIR / "exact_dicts.json", "w", encoding="utf-8") as f:
#         json.dump(exact, f, ensure_ascii=False, indent=2)

#     print(f"✓ Built exact lookup dictionaries")


# ============================================================
# 6. CHROMA CHUNKS
# ============================================================

def create_chunks_jsonl(sound_changes, vocabulary, grammar_patterns, fixed_phrases):
    with open(PROCESSED_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:

        # Sound changes
        for sc in sound_changes:
            f.write(json.dumps({
                "type": "sound_change",
                "ancient": sc["ancient"],
                "modern": sc["modern"],
                "rule": sc["rule"],
                "quote": sc["quote"]
            }, ensure_ascii=False) + "\n")

        # Vocabulary
        for v in vocabulary:
            f.write(json.dumps({
                "type": "vocab",
                "ancient": v["ancient"],
                "modern": ", ".join(v["modern"]),
                "quote": " | ".join([x["context"] for x in v["quote"]])
            }, ensure_ascii=False) + "\n")

        # Grammar
        for g in grammar_patterns:
            f.write(json.dumps({
                "type": "grammar_pattern",
                "ancient": g["ancient"],
                "modern": g["modern"],
                "quote": " | ".join([x["context"] for x in g["quote"]])
            }, ensure_ascii=False) + "\n")

        # Fixed phrases
        for fp in fixed_phrases:
            f.write(json.dumps({
                "type": "fixed_phrase",
                "ancient": fp["ancient"],
                "modern": fp["modern"],
                "quote": " | ".join([x["context"] for x in fp["quote"]])
            }, ensure_ascii=False) + "\n")

    print("✓ Created chunks.jsonl")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Converting CSV files to structured JSON...")
    print("=" * 60)

    sound_changes = convert_phonology_rules()
    vocabulary = convert_vocabulary()
    grammar_patterns = convert_grammar_patterns()
    fixed_phrases = convert_fixed_phrases()

    # build_exact_dictionaries(sound_changes, vocabulary, fixed_phrases)
    create_chunks_jsonl(sound_changes, vocabulary,
                        grammar_patterns, fixed_phrases)

    print("\n" + "=" * 60)
    print("✓ ALL CONVERSIONS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
