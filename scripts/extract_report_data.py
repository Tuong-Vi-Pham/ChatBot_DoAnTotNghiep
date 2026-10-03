import json
import os
from pathlib import Path
import pandas as pd

def load_benchmark_queries(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry["query"].strip(): entry for entry in data}

def parse_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

def main():
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "chatbot_60_test_results.csv"
    json_path = repo_root / "evaluation" / "benchmark_queries.json"
    out_path = repo_root / "evaluation" / "report_data.json"

    benchmarks = load_benchmark_queries(json_path)
    rows = parse_csv(csv_path)
    entries = []
    for row in rows:
        query = row.get("Query", "").strip()
        bench = benchmarks.get(query)
        if not bench:
            for q, b in benchmarks.items():
                if query in q:
                    bench = b
                    break
        entry = {
            "Test_ID": row.get("Test_ID"),
            "Query": query,
            "Category": bench.get("category") if bench else None,
            "Expected_Answer": bench.get("ground_truth_answer") if bench else None,
            "System_Response": row.get("Actual_Output"),
            "Judge_1": row.get("Judge_1"),
            "Judge_2": row.get("Judge_2"),
            "Judge_3": row.get("Judge_3"),
            "Final_Label": row.get("Final_Label"),
            "Agreement": row.get("Agreement"),
            "Comments": row.get("Comments"),
        }
        entries.append(entry)
    entries = [e for e in entries if e["Test_ID"]]
    if len(entries) != 60:
        print(f"Warning: expected 60 entries, got {len(entries)}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(entries)} entries to {out_path}")

if __name__ == "__main__":
    main()
