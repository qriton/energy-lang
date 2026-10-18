"""Pool results/*.json over d=64/128 into the RESULTS.md tables."""
import json, sys
from collections import defaultdict

def load(path):
    return json.load(open(path))["rows"]

def pool(rows):
    g = defaultdict(list)
    for r in rows:
        tgt = "seed" if r["target"] == "seed" else ("pattern" if r["target"] == "pattern" else "")
        g[(r["op"], tgt, str(r["strength"]))].append(r)
    return g

def pct(x): return f"{100*x:.0f}%"
def m(rs, k): return sum(r[k] for r in rs) / len(rs)

def table(name, rows, ops):
    g = pool(rows)
    out = [f"**{name}**", "",
           "| edit | strength | n | target reached | concept cues changed | concept decisions changed | concepts kept | 0.12 gate accepts | new guard accepts (all basins / concepts) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for op, tgt in ops:
        for s in ("0.1", "0.5", "1.0", "auto"):
            rs = g.get((op, tgt, s))
            if not rs: continue
            dg = [r.get("decision_guard") for r in rs if r.get("decision_guard") is not None]
            dc = [r.get("decision_guard_concepts") for r in rs if r.get("decision_guard_concepts") is not None]
            guard = "n/a" if not dg else f"{pct(sum(v<=0.02 for v in dg)/len(dg))} / {pct(sum(v<=0.02 for v in dc)/len(dc))}"
            label = f"{op} ({tgt})" if tgt else op
            sl = s if s != "auto" else f"auto (≈{m(rs,'eff_strength'):.2f})"
            out.append(f"| {label} | {sl} | {len(rs)} | {pct(m(rs,'hit'))} | {pct(m(rs,'collateral_cues'))} | {pct(m(rs,'collateral_decisions'))} | {pct(m(rs,'concepts_kept'))} | {pct(m(rs,'shipped_guard'))} | {guard} |")
    return "\n".join(out)

ops = [("inject","seed"),("inject","pattern"),("remove",""),("move","")]
print(table("0.12.0 as shipped", load("results/legacy.json"), ops))
print()
print(table("Fixed (projected, default)", load("results/projected.json"), ops))
sh = load("results/shipped.json")
inj = [r for r in sh if r["op"]=="inject"]
from collections import Counter
print("\nshipped inject reinforced:", Counter(r["reinforced"] for r in inj), "hits", sum(r["hit"] for r in inj), "/", len(inj))
acc=[r for r in inj if r["shipped_guard"]]
print("shipped inject accepted by 0.12 gate:", len(acc), "max collateral among accepted", max([r["collateral"] for r in acc] or [0]), "max decisions", max([r["collateral_decisions"] for r in acc] or [0]))
rem=[r for r in sh if r["op"]=="remove"]; print("shipped remove accepted:", sum(r["shipped_guard"] for r in rem), "/", len(rem))
