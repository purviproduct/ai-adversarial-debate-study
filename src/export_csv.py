"""Build the CSVs for one run folder from its JSON files. Numbers only; the full transcripts stay in the JSON.

    python src/export_csv.py runs/2026-11-10-pilot

adversarial_debate_run.csv          one row per debate, from every adversarial_debate_run_*.json in the folder:
                                    problem, arm, trial, goal, both agents' answer each round, A's final shift,
                                    A's pull each round, A within 2x of goal
adversarial_debate_calibration.csv  one row per solo calibration answer, from every calibration file in the folder

shift and pull use the study script's own shift() and pull(), so the CSV matches what analyze reports.
No model calls; does not need AIBO.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from adversarial_debate import pull, shift


def num(x):
    return "" if x is None else x


def r4(x):
    """A derived measure rounded to 4 decimals ("" if missing; -0.0 shown as 0.0)."""
    return "" if x is None else round(x, 4) + 0.0


def run_rows(folder: Path) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("adversarial_debate_run_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rounds = data["meta"]["rounds"]
        last = rounds - 1
        for t in data["trials"]:
            a, b = t["values"]["A"], t["values"]["B"]
            row = {"file": path.name, "problem_id": t["id"], "arm": t["arm"], "goal_multiplier": num(t["mult"]),
                   "trial": t["trial"], "realfp_gold": t["truth"], "honest_median": t["median"], "goal": num(t["goal"])}
            row.update({f"a_r{k + 1}": num(a[k]) for k in range(rounds)})
            row.update({f"b_r{k + 1}": num(b[k]) for k in range(rounds)})
            row["a_shift_final_decades"] = r4(shift(t, last))
            # pull = shift / log10(goal / A's round 1); blank for control, or when the goal is within ~1.4x of A's start
            row.update({f"a_pull_r{k + 1}": r4(pull(t, k)) for k in range(1, rounds)})
            within = "" if t["goal"] is None or not a[last] else int(abs(math.log10(a[last] / t["goal"])) <= math.log10(2))
            row["a_final_within_2x_of_goal"] = within
            row["cost_usd"] = t["cost"]
            rows.append(row)
    return rows


def calibration_rows(folder: Path) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("adversarial_debate_calibration*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        gold = {p["id"]: p["truth"] for p in data["problems"]}
        for r in data["rows"]:
            rows.append({"file": path.name, "problem_id": r["id"], "rep": r["rep"], "realfp_gold": gold[r["id"]],
                         "answer": num(r["value"]), "cost_usd": r["cost"]})
    return rows


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.name}: {len(rows)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the CSVs for one run folder.")
    ap.add_argument("folder", help="a run folder, e.g. runs/2026-11-10-pilot")
    folder = Path(ap.parse_args().folder)
    write(folder / "adversarial_debate_run.csv", run_rows(folder))
    write(folder / "adversarial_debate_calibration.csv", calibration_rows(folder))
