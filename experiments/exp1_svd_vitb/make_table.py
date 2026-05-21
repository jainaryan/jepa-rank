"""
Build a comparison table:
  layer | stable_rank | rank@95 | rank@99 | shannon | probe_top1 | best_C
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svd_json", required=True)
    ap.add_argument("--probe_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    svd = json.loads(Path(args.svd_json).read_text())
    probe = json.loads(Path(args.probe_json).read_text())

    layers = sorted(int(l) for l in svd["metrics"].keys())
    D = svd["embed_dim"]

    rows = []
    for l in layers:
        m = svd["metrics"][str(l)]
        p = probe["results"][str(l)]
        rows.append({
            "layer": l,
            "stable_rank": round(m["stable_rank"], 2),
            "rank_at_95": m["rank_at_95"],
            "rank_at_99": m["rank_at_99"],
            "shannon": round(m["shannon_rank"], 1),
            "probe_top1_pct": round(p["best_top1"] * 100, 2),
            "best_C": p["best_C"],
        })

    # Markdown
    md = ["| layer | stable_rank | rank@95 | rank@99 | shannon | probe_top1 (%) | best_C |",
          "|------:|------------:|--------:|--------:|--------:|---------------:|-------:|"]
    for r in rows:
        md.append(f"| {r['layer']} | {r['stable_rank']} | {r['rank_at_95']} | {r['rank_at_99']} "
                  f"| {r['shannon']} | {r['probe_top1_pct']} | {r['best_C']:g} |")
    md.append("")
    md.append(f"_d = {D}, d/2 = {D//2}_")

    # Transition points
    rank95 = [r["rank_at_95"] for r in rows]
    cross_half = next((rows[i]["layer"] for i, v in enumerate(rank95) if v >= D // 2), None)
    cross_full = next((rows[i]["layer"] for i, v in enumerate(rank95) if v >= int(0.9 * D)), None)
    md.append("")
    md.append(f"- First layer where rank@95 ≥ d/2 ({D//2}): **layer {cross_half}**")
    md.append(f"- First layer where rank@95 ≥ 0.9·d ({int(0.9*D)}): **layer {cross_full}**")

    Path(args.out_md).write_text("\n".join(md) + "\n")

    # CSV
    keys = list(rows[0].keys())
    lines = [",".join(keys)] + [",".join(str(r[k]) for k in keys) for r in rows]
    Path(args.out_csv).write_text("\n".join(lines) + "\n")

    print("\n".join(md))


if __name__ == "__main__":
    main()
