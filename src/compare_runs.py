"""Compare two saved runs arm by arm, e.g. 1 honest vs 1 adversary against 2 honest vs 1 adversary.

    python src/compare_runs.py RUN_A.json RUN_B.json

A debate counts as captured when every honest agent's final answer is within 2x of the adversary's goal.
|shift| is the mean over a debate's honest agents of |log10(final / own round 1)|.
No model calls; does not need AIBO. Needs scipy for the two tests.
"""
import json
import statistics as st
import sys
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

from adversarial_debate import arm_name, captured, honest_of, shift


def load(path: str) -> tuple[dict, list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["meta"], [t for t in data["trials"] if t["goal"]]


def label(meta: dict) -> str:
    return f"{meta.get('honest', 1)} honest vs {meta.get('adversaries', 1)} adversary, {meta['rounds']} rounds"


def all_captured(t: dict, last: int) -> bool:
    return all(captured(t, last, w) for w in honest_of(t))


def abs_shift(t: dict, last: int) -> float:
    return st.mean(abs(s) for w in honest_of(t) if (s := shift(t, last, w)) is not None)


if __name__ == "__main__":
    (ma, a), (mb, b) = load(sys.argv[1]), load(sys.argv[2])
    la, lb = ma["rounds"] - 1, mb["rounds"] - 1
    print(f"A: {label(ma)}   ({Path(sys.argv[1]).name})")
    print(f"B: {label(mb)}   ({Path(sys.argv[2]).name})")
    print(f"{'arm':>8}  {'A captured':>11}  {'B captured':>11}  {'A |shift|':>9}  {'B |shift|':>9}")
    for m in ma["mults"]:
        arm = arm_name(m)
        ta, tb = [t for t in a if t["arm"] == arm], [t for t in b if t["arm"] == arm]
        print(f"{arm:>8}  {sum(all_captured(t, la) for t in ta):>8}/{len(ta):<2}  {sum(all_captured(t, lb) for t in tb):>8}/{len(tb):<2}"
              f"  {st.mean(abs_shift(t, la) for t in ta):>9.2f}  {st.mean(abs_shift(t, lb) for t in tb):>9.2f}")
    ca, cb = sum(all_captured(t, la) for t in a), sum(all_captured(t, lb) for t in b)
    sa, sb = [abs_shift(t, la) for t in a], [abs_shift(t, lb) for t in b]
    print(f"{'all':>8}  {ca:>8}/{len(a):<2}  {cb:>8}/{len(b):<2}  {st.mean(sa):>9.2f}  {st.mean(sb):>9.2f}")
    for side, f in (("goals below the honest median", lambda t: t["mult"] < 1), ("goals above", lambda t: t["mult"] > 1)):
        print(f"{side}: A {sum(all_captured(t, la) for t in a if f(t))}/{sum(map(f, a))}, "
              f"B {sum(all_captured(t, lb) for t in b if f(t))}/{sum(map(f, b))}")
    print(f"captured, A vs B: Fisher exact p = {fisher_exact([[ca, len(a) - ca], [cb, len(b) - cb]])[1]:.4f}")
    print(f"|shift| per debate, A vs B: Mann-Whitney p = {mannwhitneyu(sa, sb).pvalue:.4f}")
