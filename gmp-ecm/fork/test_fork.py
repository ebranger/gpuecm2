#!/usr/bin/env python3
"""Check an `ecm` built with gmp-ecm/fork/build_fork.sh (the GMP-ECM fork with
gpuecm2 as its stage 2): for every curve of a GPU run, stage 2 with gpuecm2
(-gpu) must give the same result as GMP-ECM's stage 2 on the CPU
(-gpu -cpustage2): the same factors, and for the other curves the same
"Product of G(f_i)" value (printed at -v -v -v).

Usage: test_fork.py [path/to/ecm] [--quick]
Runs GPU stage 1 twice and CPU stage 2 for all curves of each case (a few
minutes; --quick: the first case only). The last case is a number with a
44-bit factor that three of the curves find in stage 2 (none in stage 1).
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ECM = args[0] if args else os.path.join(HERE, "..", "..", "build-gmp-ecm-fork", "gmp-ecm", "ecm")
rsa = {}
for line in open(os.path.join(HERE, "..", "..", "tests", "data", "rsa_numbers.txt")):
    if line.strip() and not line.startswith("#"):
        name, val = line.split()[:2]
        rsa[name] = int(val)
P44 = 17592186044399  # the largest prime below 2^44
# (number, B1, curves, extra options)
cases = [("RSA-1024", "5e4", "256", []), ("RSA-155", "1e5", "256", ["-dickson", "12"]),
         ("RSA-2048", "2e4", "128", []), ("P44*RSA-155", "2e3", "64", [])]
if "--quick" in sys.argv:
    cases = cases[:1]
def number(spec):
    n = 1
    for f in spec.split("*"):
        n *= P44 if f == "P44" else rsa[f]
    return n
fails = 0
for name, b1, curves, opts in cases:
    n = number(name)
    res = {}
    for mode in (["-gpu"], ["-gpu", "-cpustage2"]):
        r = subprocess.run([ECM] + mode + ["-v", "-v", "-v", "-gpucurves", curves, "-sigma", "3:1000"] + opts + [b1],
                           input=f"{n}\n", capture_output=True, text=True)
        res[mode[-1]] = (re.findall(r"Product of G\(f_i\) = (\d+)", r.stdout),
                         re.findall(r"GPU: factor (\d+) found in Step (\d) with curve (\d+)", r.stdout),
                         "GPU stage 2: gpuecm2" in r.stdout, r.stdout + r.stderr)
    g, c = res["-gpu"], res["-cpustage2"]
    s2 = sum(1 for f in g[1] if f[1] == "2")
    ok = g[0] == c[0] and g[1] == c[1] and len(g[0]) + len(g[1]) > 0 and g[2] and (s2 > 0 or "P44" not in name)
    fails += not ok
    print(f"ok   {name} ({n.bit_length()} bits), B1={b1} {' '.join(opts)}, {curves} curves: {len(g[0])} values and "
          f"{len(g[1])} factors ({s2} in stage 2) identical" if ok else
          f"FAIL {name}: gpuecm2 gave {len(g[0])} values / {len(g[1])} factors, the CPU {len(c[0])} / {len(c[1])}"
          f"{'' if g[2] else ' (gpuecm2 not used)'}", flush=True)
    if not ok:
        print(g[3][-800:])
print("all cases agree" if fails == 0 else f"{fails} case(s) differ")
sys.exit(1 if fails else 0)
