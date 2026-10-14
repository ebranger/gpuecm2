#!/usr/bin/env python3
"""Check an `ecm` built with gmp-ecm/build_ecm.sh: for every curve of a GPU run,
stage 2 with -gpu2 (gpuecm2) must give the same result as GMP-ECM's own stage 2
on the CPU (-gpu): the same factors, and for the other curves the same
"Product of G(f_i)" value (printed at -v -v -v).

Usage: test_gpu2.py [path/to/ecm] [--quick]
Runs GPU stage 1 twice and CPU stage 2 for all curves of each case (a few
minutes; --quick: the first case only).
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ECM = args[0] if args else os.path.join(HERE, "..", "build-gmp-ecm", "ecm-7.0.6", "ecm")
rsa = {}
for line in open(os.path.join(HERE, "..", "tests", "data", "rsa_numbers.txt")):
    if line.strip() and not line.startswith("#"):
        name, val = line.split()[:2]
        rsa[name] = int(val)
# (number, B1, curves, extra options)
cases = [("RSA-1024", "5e4", "256", []), ("RSA-155", "1e5", "256", ["-dickson", "12"]),
         ("RSA-2048", "2e4", "128", [])]
if "--quick" in sys.argv:
    cases = cases[:1]
fails = 0
for name, b1, curves, opts in cases:
    n = rsa[name]
    res = {}
    for mode in ("-gpu2", "-gpu"):
        r = subprocess.run([ECM, mode, "-v", "-v", "-v", "-gpucurves", curves, "-sigma", "3:1000"] + opts + [b1],
                           input=f"{n}\n", capture_output=True, text=True)
        res[mode] = (re.findall(r"Product of G\(f_i\) = (\d+)", r.stdout),
                     re.findall(r"GPU: factor (\d+) found in Step (\d) with curve (\d+)", r.stdout),
                     re.search(r"Step 2 on (GPU|CPU) took (\d+)ms", r.stdout), r.stdout + r.stderr)
    g, c = res["-gpu2"], res["-gpu"]
    ok = g[0] == c[0] and g[1] == c[1] and len(g[0]) + len(g[1]) > 0 and g[2] and g[2].group(1) == "GPU"
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name} ({n.bit_length()} bits), B1={b1} {' '.join(opts)}, {curves} curves: "
          f"{len(g[0])} values and {len(g[1])} factors identical" if ok else
          f"FAIL {name}: -gpu2 gave {len(g[0])} values / {len(g[1])} factors, -gpu {len(c[0])} / {len(c[1])}",
          f"(stage 2: GPU {g[2].group(2) if g[2] else '?'} ms, CPU {c[2].group(2) if c[2] else '?'} ms)", flush=True)
    if not ok:
        print(g[3][-800:])
print("all cases agree" if fails == 0 else f"{fails} case(s) differ")
sys.exit(1 if fails else 0)
