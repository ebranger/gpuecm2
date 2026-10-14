#!/usr/bin/env python3
"""M5 differential test: gpuecm2 vs GMP-ECM stage 2 on the same saved residues.

For N = p*q with a planted p, GMP-ECM runs stage 1 for many curves (one per
run, -savea). Each residue is then resumed by both programs with the same
B1/B2. For every curve they must agree exactly: the same factor found in
step 2, or the same "Product of G(f_i) = ..." value (printed at -v -v when no
factor is found; it is H(-f_0) and depends on every root and polynomial).

Usage: compare_stage2.py <gpuecm2> <ecm> [curves] [maxbits] [case-filter]
GMP-ECM runs are memory-capped; nothing is run in parallel.
"""
import os, random, re, resource, subprocess, sys, tempfile

GPU, ECM = sys.argv[1], sys.argv[2]
CURVES = int(sys.argv[3]) if len(sys.argv) > 3 else 40
MAXBITS = int(sys.argv[4]) if len(sys.argv) > 4 else 4600
FILTER = sys.argv[5] if len(sys.argv) > 5 else ""  # only cases whose cofactor name contains this
CAP = 3 << 30

def cap():
    resource.setrlimit(resource.RLIMIT_AS, (CAP, CAP))

def is_prime(n):
    if n < 2: return False
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % a == 0: return n == a
    d, s = n - 1, 0
    while d % 2 == 0: d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1: break
        else: return False
    return True

def prime(bits, rnd):
    while True:
        x = rnd.getrandbits(bits) | (1 << (bits - 1)) | 1
        if is_prime(x): return x

def outcome(text):
    """per resumed curve: ('F', factor) or ('P', product value) or ('?', '')"""
    m = re.search(r"Factor found in step 2: (\d+)", text)
    if m: return ("F", m.group(1))
    m = re.search(r"Product of G\(f_i\) = (\d+)", text)
    if m: return ("P", m.group(1))
    return ("?", text[-300:])

# Cofactors are RSA challenge numbers (tests/data/rsa_numbers.txt): fixed
# composites without small factors, so every run is repeatable. N = p * cofactor
# with a planted prime p that stage 2 finds on some curves.
RSA = {}
for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rsa_numbers.txt")):
    if l.strip() and not l.startswith("#"):
        name, val = l.split()
        RSA[name] = int(val)
# (cofactor, p bits, B1, extra B2 argument, options, param)
cases = [
    ("RSA-100", 66, 2e4, None, [], 3),
    ("RSA-155", 66, 5e4, None, [], 1),
    ("RSA-232", 70, 1e5, None, ["-dickson", "12"], 0),
    ("RSA-300", 70, 3e4, "300000000", [], 2),
    ("RSA-290", 72, 1e5, None, ["-power", "1"], 3),
    ("RSA-450", 70, 5e4, None, [], 3),
    ("RSA-2048", 70, 5e4, None, ["-dickson", "30"], 3),
    ("RSA-2048*RSA-1024", 70, 2e4, None, [], 3),
    ("RSA-2048*RSA-2048*RSA-100", 70, 1e4, "20000000", [], 3),
]
rnd = random.Random(2026)
total = agree = found = 0
with tempfile.TemporaryDirectory() as td:
    for cof, pbits, B1, B2, opts, param in cases:
        if FILTER not in cof: continue
        q = 1
        for part in cof.split("*"): q *= RSA[part]
        p = prime(pbits, rnd)
        N = p * q
        nbits = N.bit_length()
        if nbits > MAXBITS: continue
        save = os.path.join(td, "s.txt")
        if os.path.exists(save): os.remove(save)
        for _ in range(CURVES):
            subprocess.run([ECM, "-q", "-param", str(param), "-savea", save, "%.0f" % B1, "1"],
                           input=f"{N}\n", capture_output=True, text=True, preexec_fn=cap)
        lines = [l for l in open(save) if f"N={N};" in l] if os.path.exists(save) else []
        hits = mism = 0
        for l in lines:
            one = os.path.join(td, "one.txt")
            with open(one, "w") as fh:
                fh.write(l)
            args = opts + ["-v", "-v", "-resume", one, "%.0f" % B1] + ([B2] if B2 else [])
            e = subprocess.run([ECM] + args, capture_output=True, text=True, preexec_fn=cap)
            g = subprocess.run([GPU] + args, capture_output=True, text=True)
            oe, og = outcome(e.stdout), outcome(g.stdout)
            total += 1
            if oe == og and oe[0] != "?":
                agree += 1
                hits += oe[0] == "F"
            else:
                mism += 1
                print(f"  MISMATCH {nbits}-bit: ecm {oe[0]} {oe[1][:60]} | gpuecm2 {og[0]} {og[1][:60]}")
                if og[0] == "?": print("  gpuecm2 stderr:", g.stderr[-300:])
        found += hits
        print(f"{'ok  ' if mism == 0 else 'FAIL'} {nbits}-bit N, B1={B1:.0f} {' '.join(opts)}: "
              f"{len(lines)} curves, {hits} found p in step 2, {mism} mismatches", flush=True)
        if len(lines) > 1:
            # the whole file in one run: gpuecm2 batches the residues; after a
            # factor both continue on the cofactor
            args = opts + ["-v", "-v", "-resume", save, "%.0f" % B1] + ([B2] if B2 else [])
            e = subprocess.run([ECM] + args, capture_output=True, text=True, preexec_fn=cap)
            g = subprocess.run([GPU] + args, capture_output=True, text=True)
            be = [outcome(t) for t in e.stdout.split("Resuming ECM residue")[1:]]
            bg = [outcome(t) for t in g.stdout.split("Resuming ECM residue")[1:]]
            ok = be == bg and all(o[0] != "?" for o in be)
            total += 1
            agree += ok
            sizes = ", ".join(re.findall(r"Stage 2 of (\d+) residues", g.stdout)) or "none"
            print(f"{'ok  ' if ok else 'FAIL'}   whole file: {len(bg)} residues (gpuecm2 batches: {sizes})",
                  flush=True)
            if not ok:
                for i, (a, b) in enumerate(zip(be, bg)):
                    if a != b: print(f"    residue {i}: ecm {a[0]} {a[1][:50]} | gpuecm2 {b[0]} {b[1][:50]}")
                if len(be) != len(bg): print(f"    ecm {len(be)} residues, gpuecm2 {len(bg)}")
print(f"{agree}/{total} curves and whole-file runs agree exactly ({found} with a step 2 factor)")
sys.exit(0 if agree == total and total > 0 else 1)
