#!/usr/bin/env python3
"""How large should B2 be when stage 2 runs on the GPU?

For each number size and B1 of GMP-ECM's table (target factor size in
digits), and B2 = m * (GMP-ECM's default B2) for several m, this combines
  - GMP-ECM's expected number of curves to find a factor of the target size
    (`ecm -v -param 3`, the parametrization of GPU stage 1),
  - gpuecm2's GPU stage 2 time per curve (bench_stage2, batched as gpuecm2
    runs consecutive residues of one number),
  - a GPU stage 1 time per curve (GMP-ECM -gpu measured with
    defaults_bench.py gpus1: ns per bit and curve, times 1.4427 B1 bits),
into the expected time to find such a factor, curves * (stage 1 + stage 2),
and reports the best m.

Usage: b2_tradeoff.py --ecm ECM --bench BENCH_STAGE2 --stage1 FILE [--out FILE.json] [--md FILE]
  --stage1 FILE: GPU stage 1 cost: defaults_bench.py's gpus1.json (measured ecm -gpu),
                 or "bits ns_per_bit" lines
"""
import argparse, json, os, re, resource, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RSA_FILE = os.path.join(HERE, "..", "tests", "data", "rsa_numbers.txt")
NUMBERS = [(512, "RSA-155"), (1024, "RSA-1024"), (2048, "RSA-2048"), (4095, "RSA-2048*RSA-1536*RSA-155")]
TABLE = [(35, 1e6), (40, 3e6), (45, 11e6), (50, 43e6), (55, 11e7), (60, 26e7)]
MULT = [0.25, 0.5, 1, 2, 3, 5, 10, 20, 50, 100]

def number(spec):
    vals = dict(l.split()[:2] for l in open(RSA_FILE) if l.strip() and not l.startswith("#"))
    n = 1
    for part in spec.split("*"): n *= int(vals[part])
    return n

def default_b2(b1):
    return int(round((11.0 / 6.0 * b1) ** 1.43))

def expected_curves(ecm, n, b1, b2, digits):
    def lim(): resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    try:
        r = subprocess.run(["timeout", "20", ecm, "-v", "-param", "3", "-c", "1", "%.0f" % b1, str(b2)],
                           input=f"{n}\n", capture_output=True, text=True, preexec_fn=lim)
    except Exception:
        return None
    lines = r.stdout.splitlines()
    for i, l in enumerate(lines):
        if l.startswith("Expected number of curves"):
            head, vals = lines[i + 1].split(), lines[i + 2].split()
            if str(digits) in head:
                return float(vals[head.index(str(digits))])
    return None

def stage2_ms(bench, spec, b1, b2):
    r = subprocess.run([bench, spec, "%.0f" % b1, str(b2), "-batch", "0", "-reps", "3", "-untimed", "1"],
                       capture_output=True, text=True)
    m = re.search(r"total ([0-9.]+) ms", r.stdout)
    k = re.search(r"dF=(\d+), k=(\d+)", r.stdout)
    return (float(m.group(1)) if m else None), (k.groups() if k else None)

def stage1_model(path):
    """ns per bit and curve by modulus size: the gpus1.json of defaults_bench.py (measured
    GMP-ECM -gpu, largest B1 of each number), or "bits ns" lines"""
    ns = {}
    if path.endswith(".json"):
        best = {}
        for v in json.load(open(path)).values():
            if "ms_per_curve" in v and (v["bits"] not in best or v["B1"] > best[v["bits"]]["B1"]):
                best[v["bits"]] = v
        return {b: v["ms_per_curve"] * 1e6 / (1.4427 * v["B1"]) for b, v in best.items()}
    for l in open(path):
        m = re.match(r"\s*(\d+) bits:.*?([0-9.]+) ns per bit", l) or re.match(r"\s*(\d+)\s+([0-9.]+)\s*$", l)
        if m: ns[int(m.group(1))] = float(m.group(2))
    return ns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecm", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--out", default="b2_tradeoff.json")
    ap.add_argument("--md")
    ap.add_argument("--bits", type=lambda x: [int(v) for v in x.split(",")], default=None,
                    help="only these sizes (512,1024,2048,4095)")
    ap.add_argument("--remeasure", action="store_true", help="time stage 2 again for the points already measured")
    ap.add_argument("--stop-factor", type=float, default=1.0,
                    help="go on until past the optimum for a stage 1 this many times slower than the model")
    ap.add_argument("--max-s2-ms", type=float, default=20000, help="skip B2 values whose stage 2 would take longer")
    a = ap.parse_args()
    s1 = stage1_model(a.stage1)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    if a.remeasure:  # time stage 2 again for every point (e.g. after a disturbed run), keep the curve counts
        for key, r in sorted(res.items()):
            spec = dict(NUMBERS)[r["bits"]]
            r["s2_ms"], r["dfk"] = stage2_ms(a.bench, spec, r["B1"], r["B2"])
            json.dump(res, open(a.out, "w"), indent=1)
            print("remeasured", key, r["s2_ms"], flush=True)
    for bits, spec in NUMBERS:
        if a.bits and bits not in a.bits:
            continue
        n = number(spec)
        size = min((b for b in s1 if b >= bits), default=None)
        for digits, b1 in TABLE:
            t1 = s1.get(size, 0) * 1.4427 * b1 * 1e-6 * a.stop_factor  # ms (for the stopping rule)
            worse = 0
            prev = None
            for m in MULT:
                key = f"{bits}_{b1:.0e}_{m}"
                if key not in res:
                    if prev and prev["s2_ms"] and prev["s2_ms"] * m / prev["mult"] > a.max_s2_ms:
                        break  # too slow to measure here
                    b2 = int(default_b2(b1) * m)
                    c = expected_curves(a.ecm, n, b1, b2, digits)
                    t2, dfk = stage2_ms(a.bench, spec, b1, b2)
                    res[key] = dict(bits=bits, digits=digits, B1=b1, mult=m, B2=b2, curves=c, s2_ms=t2, dfk=dfk,
                                    s1_ns_per_bit=s1.get(size), s1_bits=size)
                    json.dump(res, open(a.out, "w"), indent=1)
                    print(key, res[key], flush=True)
                r = res[key]
                # stop two steps past the optimum
                if prev and r["curves"] and prev["curves"] and r["s2_ms"] and prev["s2_ms"]:
                    if r["curves"] * (t1 + r["s2_ms"]) > prev["curves"] * (t1 + prev["s2_ms"]):
                        worse += 1
                if worse >= 2:
                    break
                prev = r
    # report
    lines = ["| N | digits | B1 | t1 (GPU stage 1) | best B2 | curves | t2 | expected time | at default B2 | gain |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for bits, _ in NUMBERS:
        for digits, b1 in TABLE:
            rows = [res.get(f"{bits}_{b1:.0e}_{m}") for m in MULT]
            rows = [r for r in rows if r]
            rows = [r for r in rows if r and r["curves"] and r["s2_ms"]]
            if not rows:
                continue
            size = min((b for b in s1 if b >= bits), default=None)
            if size is None:
                continue
            t1 = s1[size] * 1.4427 * b1 * 1e-6  # ms, from the --stage1 data of this run
            def total(r): return r["curves"] * (t1 + r["s2_ms"]) / 1000.0
            base = next((r for r in rows if r["mult"] == 1), None)
            best = min(rows, key=total)
            lines.append("| %d | %d | %.0e | %.0f ms | %g × default (%.2g) | %.0f | %.1f ms | %.3g s | %.3g s | %.2f |" % (
                bits, digits, b1, t1, best["mult"], best["B2"], best["curves"], best["s2_ms"], total(best),
                total(base) if base else float("nan"), total(base) / total(best) if base else float("nan")))
    text = "\n".join(lines) + "\n"
    print(text)
    if a.md:
        open(a.md, "w").write(text)

if __name__ == "__main__":
    sys.exit(main())
