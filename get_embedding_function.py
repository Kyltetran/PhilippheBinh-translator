from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()


def get_embedding_function():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    print("✅ OpenAI API key loaded.")

    return OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)


if __name__ == "__main__":
    print(get_embedding_function())
