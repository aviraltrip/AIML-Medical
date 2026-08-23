import asyncio
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))

from pulsepoint_ai.engines.triage.rag.ingest import ingest_all


async def main():
    try:
        print("Starting Clinical Knowledge Base Indexing...")
        await ingest_all()
        print("Knowledge Base successfully built and indexed with Gemini!")
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    asyncio.run(main())
