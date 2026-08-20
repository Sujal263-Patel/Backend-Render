import fsspec
import pyarrow.parquet as pq

url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/train/hintrain.parquet"
fs = fsspec.filesystem("http", block_size=4 * 1024 * 1024)

with fs.open(url, "rb") as f:
    parquet_file = pq.ParquetFile(f)
    table = parquet_file.read_row_group(0, columns=["query", "Eng_Query", "Answer", "Eng_Answer", "passages"])
    rows = table.to_pylist()

print(f"Total rows in first row group: {len(rows)}")
for i, r in enumerate(rows[:15]):
    print(f"\n--- Question {i+1} ---")
    print(f"Hindi Question: {r.get('query')}")
    print(f"English Question: {r.get('Eng_Query')}")
    print(f"Hindi Answer/Grounding: {r.get('Answer')}")
    print(f"English Answer: {r.get('Eng_Answer')}")
