#!/usr/bin/env python3
"""Performance profile of gpuecm2's stage 2 over modulus sizes and B2 values.

For every modulus size in --bits and every B1 in --b1 (with GMP-ECM's default
B2, or B2 = --mult times it) this runs bench_stage2 on a random modulus of that
size: the time per curve with the automatic batch size ("batched", what gpuecm2
does for consecutive residues of one number), one curve at a time ("single"),
and the per-phase times of a synchronised run. The results go to a JSON file
(runs are cached there) and a report that lists, per B1, time against size, and
flags points whose time per unit of work jumps against their neighbours.

Usage: profile_grid.py --bench BENCH_STAGE2 [--out FILE.json] [--md FILE] [--bits ...] [--b1 ...] [--mult ...]
Needs an idle GPU.
"""
import argparse, json, math, os, re, subprocess, sys

BITS = [200, 256, 384, 512, 513, 640, 768, 769, 1024, 1025, 1280, 1536, 1792, 1920, 1921, 2048, 2049, 2560, 3072,
        3584, 4096, 4097, 4608]
B1S = [1e6, 3e6, 11e6, 43e6, 11e7, 26e7]


def run(bench, bits, b1, b2, batch, timed, reps):
    cmd = [bench, str(bits), "%.0f" % b1] + ([str(b2)] if b2 else []) + \
          ["-batch", str(batch), "-reps", str(reps), "-untimed", "0" if timed else "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout
    m = re.search(r"total ([0-9.]+) ms", out)
    res = {"total_ms": float(m.group(1)) if m else None}
    m = re.search(r"dF=(\d+), k=(\d+), S=(-?\d+), GPU estimate (\d+) MB", out)
    if m:
        res.update(dF=int(m.group(1)), k=int(m.group(2)), S=int(m.group(3)), mem_mb=int(m.group(4)))
    m = re.search(r"B2=(\d+)", out)
    if m: res["B2"] = int(m.group(1))
    m = re.search(r"batch of (\d+) curves", out)
    res["batch"] = int(m.group(1)) if m else 1
    m = re.search(r"rootsF ([0-9.]+)\s+treeF ([0-9.]+)\s+invF ([0-9.]+)\s+rootsG ([0-9.]+)\s+treesG ([0-9.]+)\s+"
                  r"mulmod ([0-9.]+)\s+eval ([0-9.]+)\s+prod ([0-9.]+)", out)
    if m and timed:
        res["phases"] = dict(zip(["rootsF", "treeF", "invF", "rootsG", "treesG", "mulmod", "eval", "prod"],
                                 [float(x) for x in m.groups()]))
    if res["total_ms"] is None:
        res["error"] = (out + r.stderr)[-300:]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", default="profile_grid.json")
    ap.add_argument("--md")
    ap.add_argument("--bits", type=lambda s: [int(x) for x in s.split(",")], default=BITS)
    ap.add_argument("--b1", type=lambda s: [float(x) for x in s.split(",")], default=B1S)
    ap.add_argument("--mult", type=lambda s: [float(x) for x in s.split(",")], default=[1.0])
    ap.add_argument("--max-ms", type=float, default=60000, help="skip larger configurations once a smaller size took this long")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for r in res.values():  # files of the first version divided the per-curve batched time by the batch size
        if r.get("v") != 2 and r.get("batched_ms") and r.get("batch", 1) > 1:
            r["batched_ms"] *= r["batch"]
        r["v"] = 2
    for mult in a.mult:
        for b1 in a.b1:
            for bits in a.bits:
                key = f"{bits}_{b1:.0e}_{mult:g}"
                if key in res:
                    continue
                b2 = None if mult == 1.0 else int(round((11.0 / 6.0 * b1) ** 1.43 * mult))
                r = {"bits": bits, "B1": b1, "mult": mult}
                single = run(a.bench, bits, b1, b2, 1, False, 3)
                r["single_ms"] = single["total_ms"]
                r.update({k: v for k, v in single.items() if k not in ("total_ms", "batch", "phases")})
                batched = run(a.bench, bits, b1, b2, 0, False, 3)
                r["batched_ms"], r["batch"] = batched["total_ms"], batched["batch"]  # already per curve
                r["v"] = 2
                timed = run(a.bench, bits, b1, b2, 1, True, 2)
                r["phases"] = timed.get("phases")
                res[key] = r
                json.dump(res, open(a.out, "w"), indent=1)
                print(key, {k: v for k, v in r.items() if k != "phases"}, flush=True)
    report(res, a)


def report(res, a):
    lines = []
    for mult in a.mult:
        for b1 in a.b1:
            rows = sorted((r for r in res.values() if r["B1"] == b1 and r["mult"] == mult and r.get("single_ms")),
                          key=lambda r: r["bits"])
            if not rows:
                continue
            lines += [f"\n### B1 = {b1:.0e}" + (f", B2 = {mult:g} x default" if mult != 1 else ", default B2") + "\n",
                      "| bits | dF, k | per curve (batched) | one curve | batch | roots F+G | trees, 1/F, mulmod | eval, product | ns per (coefficient x limb) | |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            prev = None
            for r in rows:
                ph = r.get("phases") or {}
                roots = ph.get("rootsF", 0) + ph.get("rootsG", 0)
                poly = ph.get("treeF", 0) + ph.get("invF", 0) + ph.get("treesG", 0) + ph.get("mulmod", 0)
                ev = ph.get("eval", 0) + ph.get("prod", 0)
                # work unit: k*dF coefficients times limbs (32-bit) of the modulus
                unit = r["k"] * r["dF"] * math.ceil(r["bits"] / 32) if r.get("k") else None
                norm = r["single_ms"] * 1e6 / unit if unit else None
                flag = ""
                if prev and norm and prev[1] and norm > 1.35 * prev[1] and r["bits"] - prev[0] <= 300:
                    flag = "jump"
                lines.append("| %d | %s, %s | %.3g ms | %.3g ms | %d | %.3g | %.3g | %.3g | %.2f | %s |" % (
                    r["bits"], r.get("dF"), r.get("k"), r["batched_ms"] or float("nan"), r["single_ms"], r["batch"],
                    roots, poly, ev, norm or float("nan"), flag))
                prev = (r["bits"], norm)
    text = "\n".join(lines) + "\n"
    print(text)
    if a.md:
        open(a.md, "w").write(text)


if __name__ == "__main__":
    sys.exit(main())
