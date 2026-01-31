import argparse
import re
import csv
from pathlib import Path


def parse_log(log_text: str):
    current_task = None
    current_strategy = None
    records = []

    task_re = re.compile(r"Processing\s+([A-Za-z0-9_/.-]+)")
    strat_re = re.compile(r"Strategy:\s+(\w+)")
    genlen_re = re.compile(r"Generated length:\s+(\d+)\s+chars,\s+(\d+)\s+tokens")
    metrics_re = re.compile(
        r"CodeBLEU:\s*([0-9.]+),\s*EM:\s*([0-9.]+),\s*ROUGE-L F1:\s*([0-9.]+)",
        re.IGNORECASE,
    )

    tmp_gen_tokens = None

    for line in log_text.splitlines():
        m_task = task_re.search(line)
        if m_task:
            current_task = m_task.group(1)
            continue

        m_strat = strat_re.search(line)
        if m_strat:
            current_strategy = m_strat.group(1)
            tmp_gen_tokens = None
            continue

        m_len = genlen_re.search(line)
        if m_len:
            tmp_gen_tokens = int(m_len.group(2))
            continue

        m_metrics = metrics_re.search(line)
        if m_metrics and current_task and current_strategy:
            codebleu, em, rouge = map(float, m_metrics.groups())
            records.append(
                {
                    "task_id": current_task,
                    "strategy": current_strategy,
                    "generated_tokens": tmp_gen_tokens,
                    "codebleu": codebleu,
                    "exact_match": em,
                    "rouge_l_f1": rouge,
                }
            )

    return records


def classify(rec):
    gen_tokens = rec.get("generated_tokens")
    em = rec.get("exact_match", 0)
    cb = rec.get("codebleu", 0)
    rouge = rec.get("rouge_l_f1", 0)

    if em == 1.0:
        return "exact_match"
    if gen_tokens is None or gen_tokens < 5:
        return "empty_truncated"
    if em == 0 and (cb >= 0.5 or rouge >= 0.5):
        return "partial_match"
    return "major_diff"


def main():
    ap = argparse.ArgumentParser(description="Error type analysis from experiment.log")
    ap.add_argument("--log_file", required=True, help="Path to experiment.log")
    ap.add_argument("--output_csv", required=True, help="Path to save per-sample classification")
    args = ap.parse_args()

    log_text = Path(args.log_file).read_text(encoding="utf-8")
    records = parse_log(log_text)

    if not records:
        print("No records parsed. Check log path/format.")
        return

    for rec in records:
        rec["error_type"] = classify(rec)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "strategy",
                "generated_tokens",
                "codebleu",
                "exact_match",
                "rouge_l_f1",
                "error_type",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    summary = {}
    for rec in records:
        key = (rec["strategy"], rec["error_type"])
        summary[key] = summary.get(key, 0) + 1

    print(f"Saved per-sample classification to {out_path}")
    print("Summary (strategy, error_type -> count):")
    for (strategy, etype), cnt in summary.items():
        print(f"{strategy:10s} {etype:15s} {cnt}")


if __name__ == "__main__":
    main()
