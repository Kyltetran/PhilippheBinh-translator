"""
Populate ChromaDB with linguistic data
Run this after convert_csv_to_json.py
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from get_embedding_function import get_embedding_function
from tqdm import tqdm
from langchain.schema import Document
from langchain_chroma import Chroma

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

            # Build page content for embeddings
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
    print("=" * 60)
    print("Populating ChromaDB with linguistic data...")
    print("=" * 60)

    print(f"\n📥 Loading data from: {PROCESSED}")
    grouped = load_chunks(PROCESSED)

    embedding = get_embedding_function()

    # Create separate vectorstores for each type
    for t, docs in grouped.items():
        save_dir = CHROMA_DIR / t
        save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n🚀 Building Chroma collection for '{t}' with {len(docs)} docs...")

        # Use batch processing for efficiency
        batch_size = 100
        for i in tqdm(range(0, len(docs), batch_size), desc=f"Processing {t}"):
            batch = docs[i:i+batch_size]

            if i == 0:
                # Create new collection
                db = Chroma.from_texts(
                    texts=[d["content"] for d in batch],
                    embedding=embedding,
                    metadatas=[d["metadata"] for d in batch],
                    persist_directory=str(save_dir)
                )
            else:
                # Add to existing collection
                db = Chroma(
                    persist_directory=str(save_dir),
                    embedding_function=embedding
                )
                db.add_texts(
                    texts=[d["content"] for d in batch],
                    metadatas=[d["metadata"] for d in batch]
                )

        print(f"✅ Saved collection '{t}' into: {save_dir}")
        print(f"   Total documents: {len(docs)}")

    print("\n" + "=" * 60)
    print("✅ ALL DONE: Vector DB created successfully!")
    print("=" * 60)

    # Print summary
    print("\n📊 Collection Summary:")
    for t, docs in grouped.items():
        print(f"  • {t}: {len(docs)} documents")


if __name__ == "__main__":
    main()
