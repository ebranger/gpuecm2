#!/usr/bin/env python3
"""Checks gpuecm2's curve setup (A from sigma, parametrizations 0-3) against
GMP-ECM without needing GMP-ECM to print A:
  1. GMP-ECM runs stage 1 for many curves and saves the residues (-save).
  2. Each save line is duplicated with SIGMA= replaced by A=<gpuecm2's A>.
  3. GMP-ECM resumes both files with the same stage 2; the two runs must find
     factors in step 2 on exactly the same curves.
Usage: check_curves.py <gpuecm2> <ecm> [curves_per_param]
Small numbers only: little RAM, no GPU.
"""
import os, re, resource, subprocess, sys, tempfile

GPUECM2, ECM = sys.argv[1], sys.argv[2]
CURVES = int(sys.argv[3]) if len(sys.argv) > 3 else 60  # 250 per param takes ~20 min
CAP = 1024 * 1024 * 1024

def cap():
    resource.setrlimit(resource.RLIMIT_AS, (CAP, CAP))

def run(args, inp=None):
    # only GMP-ECM is capped: a CUDA program reserves far more address space than it uses
    return subprocess.run(args, input=inp, capture_output=True, text=True,
                          preexec_fn=cap if args[0] == ECM else None).stdout

# N = p * q with a 16-digit prime p: B1=5000, B2=5e9 finds p in step 2 on a few % of curves
p, q = 1000000000000037, 2**127 - 1
N = p * q
B1 = 5000
B2 = "5e9"
total_found = mism = 0
with tempfile.TemporaryDirectory() as td:
    for param in (0, 1, 2, 3):
        save = os.path.join(td, f"s{param}.txt")
        for _ in range(CURVES):  # one curve per run: a stage-1 factor must not stop the series
            run([ECM, "-q", "-param", str(param), "-savea", save, str(B1), "1"], f"{N}\n")
        lines = [l for l in open(save) if l.startswith("METHOD=ECM") and f"N={N};" in l]
        with open(save, "w") as fh:
            fh.writelines(lines)
        # B2 < B1: no stage 2, so that no factor ends the series (and every curve is printed)
        curves = run([GPUECM2, "-q", "-printcurve", "-resume", save, str(B1), "1"])
        A = dict(re.findall(r"CURVE line=(\d+) A=(0x[0-9a-f]+)", curves))
        alt = os.path.join(td, f"a{param}.txt")
        with open(alt, "w") as fh:
            for i, l in enumerate(lines, 1):
                l = re.sub(r"SIGMA=\d+", f"A={A[str(i)]}", l)
                l = re.sub(r" CHECKSUM=\d+;", "", l)
                fh.write(l)
        def found(fn):
            """resume each line separately (GMP-ECM stops after a factor whose cofactor is prime)"""
            res = []
            for l in open(fn):
                one = os.path.join(td, "one.txt")
                with open(one, "w") as fh:
                    fh.write(l)
                out = run([ECM, "-resume", one, str(B1), B2])
                m = re.search(r"Factor found in step (\d): (\d+)", out)
                res.append(f"{m.group(2)} (step {m.group(1)})" if m else "")
            return res
        a, b = found(save), found(alt)
        n_found = sum(1 for x in a if x)
        diff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        total_found += n_found; mism += diff
        print(f"param {param}: {len(lines)} curves, factor found on {n_found} with SIGMA; "
              f"{diff} curves differ when using gpuecm2's A")
print("OK" if mism == 0 and total_found > 0 else "FAIL")
sys.exit(0 if mism == 0 and total_found > 0 else 1)
