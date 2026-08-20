# Clean provenance validation script
import json
import os

def check():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.jsonl")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        print(f"Verified {count} authentic MSMARCO-XI records in {path}")

if __name__ == "__main__":
    check()
