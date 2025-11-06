"""
Populate ChromaDB with linguistic data
Run this after processing JSON files
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from get_embedding_function import get_embedding_function
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/processed"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))


def load_json_data():
    """Load all JSON files and prepare for embedding"""
    data = {
        "vocabulary": [],
        "grammar_patterns": [],
        "fixed_phrases": [],
        "sound_changes": []
    }

    # Load vocabulary.json
    vocab_file = DATA_DIR / "vocabulary.json"
    if vocab_file.exists():
        with open(vocab_file, "r", encoding="utf8") as f:
            vocab_items = json.load(f)
            for item in vocab_items:
                # Create searchable content for each modern meaning
                for modern in item.get("modern", []):
                    content = f"modern: {modern} | ancient: {item['ancient']}"
                    if item.get("quote"):
                        content += f" | context: {item['quote'][0].get('context', '')[:100]}"

                    data["vocabulary"].append({
                        "id": f"vocab_{item.get('id', '')}_{modern}",
                        "content": content,
                        "metadata": {
                            "ancient": item["ancient"],
                            "modern": modern,
                            "quote": item.get("quote", [{}])[0].get("context", ""),
                            "type": "vocabulary"
                        }
                    })

    # Load grammar_patterns.json
    grammar_file = DATA_DIR / "grammar_patterns.json"
    if grammar_file.exists():
        with open(grammar_file, "r", encoding="utf8") as f:
            grammar_items = json.load(f)
            for item in grammar_items:
                content = f"modern: {item['modern']} | ancient: {item['ancient']}"
                if item.get("quote"):
                    content += f" | example: {item['quote'][0].get('context', '')[:150]}"

                data["grammar_patterns"].append({
                    "id": item.get("id", ""),
                    "content": content,
                    "metadata": {
                        "ancient": item["ancient"],
                        "modern": item["modern"],
                        "quote": item.get("quote", [{}])[0].get("context", ""),
                        "type": "grammar_pattern"
                    }
                })

    # Load fixed_phrases.json
    phrases_file = DATA_DIR / "fixed_phrases.json"
    if phrases_file.exists():
        with open(phrases_file, "r", encoding="utf8") as f:
            phrase_items = json.load(f)
            for item in phrase_items:
                content = f"modern meaning: {item['modern']} | ancient phrase: {item['ancient']}"
                if item.get("quote"):
                    content += f" | example: {item['quote'][0].get('context', '')[:150]}"

                data["fixed_phrases"].append({
                    "id": item.get("id", ""),
                    "content": content,
                    "metadata": {
                        "ancient": item["ancient"],
                        "modern": item["modern"],
                        "quote": item.get("quote", [{}])[0].get("context", ""),
                        "type": "fixed_phrase"
                    }
                })

    # Load sound_changes.json
    sound_file = DATA_DIR / "sound_changes.json"
    if sound_file.exists():
        with open(sound_file, "r", encoding="utf8") as f:
            sound_items = json.load(f)
            for item in sound_items:
                content = f"modern sound: {item['modern']} | ancient sound: {item['ancient']} | rule: {item['rule']}"

                data["sound_changes"].append({
                    "id": item.get("id", ""),
                    "content": content,
                    "metadata": {
                        "ancient": item["ancient"],
                        "modern": item["modern"],
                        "rule": item["rule"],
                        "quote": item.get("quote", ""),
                        "type": "sound_change"
                    }
                })

    return data


def populate_collection(client, collection_name, documents, embedding_function):
    """Populate a single ChromaDB collection"""

    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
        print(f"  ⚠️  Deleted existing collection '{collection_name}'")
    except:
        pass

    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Add documents in batches
    batch_size = 100
    total = len(documents)

    print(f"  📦 Adding {total} documents to '{collection_name}'...")

    for i in tqdm(range(0, total, batch_size), desc=f"  Processing"):
        batch = documents[i:i+batch_size]

        # Extract data
        ids = [doc["id"] for doc in batch]
        texts = [doc["content"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]

        # Get embeddings using the embedding function
        embeddings = embedding_function.embed_documents(texts)

        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    print(
        f"  ✅ Collection '{collection_name}' created with {total} documents\n")


def main():
    print("=" * 70)
    print("🚀 POPULATING CHROMADB WITH LINGUISTIC DATA")
    print("=" * 70)

    # Load data
    print(f"\n📥 Loading JSON files from: {DATA_DIR}")
    data = load_json_data()

    # Print summary
    print("\n📊 Data Summary:")
    for collection_name, docs in data.items():
        print(f"  • {collection_name}: {len(docs)} documents")

    if sum(len(docs) for docs in data.values()) == 0:
        print("\n❌ No data found! Please check your JSON files.")
        return

    # Initialize ChromaDB client
    print(f"\n🔧 Initializing ChromaDB at: {CHROMA_DIR}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Get embedding function
    print("🔑 Loading OpenAI embeddings...")
    embedding_function = get_embedding_function()

    # Populate each collection
    print("\n" + "=" * 70)
    print("📚 CREATING COLLECTIONS")
    print("=" * 70 + "\n")

    for collection_name, documents in data.items():
        if documents:
            populate_collection(client, collection_name,
                                documents, embedding_function)

    print("=" * 70)
    print("✅ ALL DONE! ChromaDB populated successfully!")
    print("=" * 70)

    # Verification
    print("\n🔍 Verification:")
    for collection_name in data.keys():
        try:
            collection = client.get_collection(collection_name)
            count = collection.count()
            print(f"  • {collection_name}: {count} documents stored")
        except Exception as e:
            print(f"  ❌ {collection_name}: Error - {e}")


if __name__ == "__main__":
    main()
