import json
import os
from pathlib import Path
from dotenv import load_dotenv
from get_embedding_function import get_embedding_function
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma


load_dotenv()

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))


def load_chunks(path):
    """Load chunks.jsonl and return dict {type: [documents]}"""
    grouped = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            t = obj.get("type", "misc")

            # Build page content
            if t == "sound_change":
                content = f"sound_change | ancient:{obj.get('ancient')} | modern:{obj.get('modern')} | rule:{obj.get('rule')} | note:{obj.get('note')}"
            elif t == "grammar_pattern":
                content = f"grammar_pattern | old:{obj.get('old_pattern')} | meaning:{obj.get('meaning')} | context:{obj.get('context')}"
            elif t == "fixed_phrase":
                content = f"fixed_phrase | ancient:{obj.get('ancient')} | gloss:{obj.get('gloss')} | context:{obj.get('context')}"
            elif t == "vocab":
                content = f"vocab | ancient:{obj.get('ancient')} | gloss:{obj.get('gloss')} | context:{obj.get('context')}"
            elif t == "phonology_rule":
                content = f"phonology_rule | {obj.get('text')}"
            else:
                content = json.dumps(obj, ensure_ascii=False)

            grouped.setdefault(t, []).append({
                "content": content,
                "metadata": obj
            })
    return grouped


def main():
    print("📥 Loading data from:", PROCESSED)
    grouped = load_chunks(PROCESSED)

    embedding = get_embedding_function()

    # Create separate vectorstores
    for t, docs in grouped.items():
        save_dir = CHROMA_DIR / t
        save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"🚀 Building Chroma collection for '{t}' with {len(docs)} docs...")

        Chroma.from_texts(
            texts=[d["content"] for d in docs],
            embedding=embedding,
            metadatas=[d["metadata"] for d in docs],
            persist_directory=str(save_dir)
        )

        print(f"✅ Saved collection '{t}' into:", save_dir)

    print("\n✅ ALL DONE: Vector DB created!")


if __name__ == "__main__":
    main()
