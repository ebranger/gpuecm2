#!/usr/bin/env python3
"""Tests for large numbers (4609 to 32768 bits): gpuecm2 against GMP-ECM on
cached stage 1 residues, over the CGBN sizes above 4608 bits and a range of
B1 (with GMP-ECM's default B2, and a few other stage 2 settings).

GMP-ECM's CPU runs at these sizes take long (up to about half an hour per
curve at 32768 bits and B1 = 1e6), so they are done once by `prepare`, which
stores the residues and GMP-ECM's results ("Product of G(f_i)" or the factor,
the stage 2 parameters and the CPU time per curve) in a cache directory;
`check` then runs gpuecm2 on the same residues and compares, and `bench`
reports gpuecm2's stage 2 time against GMP-ECM's.

The numbers are N = p * c: c a product of primes of about 1024 bits with N
close to 2^bits (as in boundary_tests.py), p a planted prime that some curves
find in stage 2 (or 1).

Usage:
  large_tests.py prepare <ecm> [--cache DIR] [--jobs N] [--only REGEX]
  large_tests.py check <gpuecm2> [--cache DIR] [--only REGEX] [--one-by-one]
  large_tests.py bench <gpuecm2> [--cache DIR] [--only REGEX] [--reps N]
  large_tests.py list
GMP-ECM runs are memory-capped (--memcap GB, default 3) and run in parallel
(--jobs, default 4; mind the RAM: each may use up to the cap).
"""
import argparse, concurrent.futures, json, os, random, re, resource, subprocess, sys, tempfile, time
if hasattr(sys, "set_int_max_str_digits"):  # numbers of up to 32768 bits (9865 digits)
    sys.set_int_max_str_digits(0)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from boundary_tests import number, is_prime, program_max_bits  # noqa: E402

# the CGBN sizes above 4608 bits (exact sizes and one bit more at some), and sizes in between
SIZES = [4609, 5120, 6144, 7000, 8192, 8193, 10240, 12288, 14336, 16384, 16385, 20480, 24576, 28672, 32768]

def cases():
    """(name, bits, B1, [(param, sigma)], planted p bits, [(variant, B2 or None, options)])"""
    C = []
    rnd = random.Random(4609)
    def curves(params):
        return [(p, rnd.randrange(10 ** 6, 2 ** 31)) for p in params]
    for bits in SIZES:
        C.append((f"{bits}-1e4", bits, 1e4, curves([3, 3, 0, 1]), 56,
                  [("default", None, []), ("dickson12", None, ["-dickson", "12"]),
                   ("B2x100", "200000000", []), ("k5", None, ["-k", "5"])]))
        C.append((f"{bits}-1e5", bits, 1e5, curves([3, 2]), 64,
                  [("default", None, []), ("power1", None, ["-power", "1"])]))
        C.append((f"{bits}-1e6", bits, 1e6, curves([3, 3]), 72, [("default", None, [])]))
        if bits <= 16385:
            C.append((f"{bits}-3e6", bits, 3e6, curves([3]), 76, [("default", None, [])]))
        if bits <= 10240:
            C.append((f"{bits}-11e6", bits, 11e6, curves([3]), 80, [("default", None, [])]))
    # high-degree Brent-Suyama polynomials (long scalar multiplications) at the largest size
    C.append(("32768-1e5-dickson30", 32768, 1e5, curves([3]), 1, [("dickson30", None, ["-dickson", "30"])]))
    return C

def case_number(bits, pbits, name):
    """N = p * c with exactly `bits` bits; deterministic"""
    rnd = random.Random(name)
    if pbits <= 1:
        return number(bits, 7)
    while True:  # near 2^pbits, like the factors of c, so that N has `bits` bits
        p = rnd.getrandbits(pbits) | (0xff << (pbits - 8)) | 1
        if is_prime(p): break
    return p * number(bits - pbits, 7)

def outcome(text):
    """('F', factor) or ('P', product value) or ('?', tail of the output)"""
    m = re.search(r"Factor found in step 2: (\d+)", text)
    if m: return ["F", m.group(1)]
    m = re.search(r"Product of G\(f_i\) = (\d+)", text)
    if m: return ["P", m.group(1)]
    return ["?", text[-300:]]

def found_via(text):
    """how GMP-ECM found a step 2 factor: in the product (the gcd that gpuecm2 computes too) or
    through a value its affine Weierstrass arithmetic could not invert while computing the
    roots ("Found factor while computing ..."), which gpuecm2's Edwards arithmetic need not meet"""
    return "inversion" if "Found factor while" in text else "product"

def params_of(text):
    m = re.search(r"dF=(\d+), k=(\d+), d=(\d+), d2=(\d+), i0=(-?\d+)", text)
    return m.group(0) if m else None

def step2_ms(text):
    m = re.findall(r"Step 2 took (\d+)ms", text)
    return int(m[-1]) if m else None

class Cache:
    def __init__(self, d):
        self.d = d
        os.makedirs(d, exist_ok=True)
    def path(self, name, ext):
        return os.path.join(self.d, f"{name}.{ext}")
    def load(self, name):
        p = self.path(name, "json")
        return json.load(open(p)) if os.path.exists(p) else None
    def store(self, name, obj):
        p = self.path(name, "json")
        with open(p + ".tmp", "w") as fh:
            json.dump(obj, fh)
        os.replace(p + ".tmp", p)

# ---------------------------------------------------------------- prepare (GMP-ECM)
def prepare_case(ecm, cache, case, memcap):
    name, bits, B1, cv, pbits, variants = case
    rec = cache.load(name) or {}
    def cap():
        resource.setrlimit(resource.RLIMIT_AS, (memcap, memcap))
    if "N" not in rec:
        rec = {"name": name, "bits": bits, "B1": B1, "N": str(case_number(bits, pbits, name)), "residues": [],
               "stage1": [], "s2": {}}
    N = rec["N"]
    assert int(N).bit_length() == bits
    log = []
    with tempfile.TemporaryDirectory() as td:
        # stage 1 of each curve (once)
        for i, (param, sigma) in enumerate(cv):
            if i < len(rec["stage1"]): continue
            save = os.path.join(td, "s1.txt")
            t0 = time.time()
            r = subprocess.run([ecm, "-param", str(param), "-sigma", f"{param}:{sigma}", "-save", save,
                                "%.0f" % B1, "1"], input=f"{N}\n", capture_output=True, text=True, preexec_fn=cap)
            t1 = time.time() - t0
            line = open(save).read().strip() if os.path.exists(save) else ""
            if os.path.exists(save): os.remove(save)
            found = re.search(r"Factor found in step 1: (\d+)", r.stdout)
            if found: line = ""  # no stage 2 for this curve
            rec["stage1"].append({"param": param, "sigma": sigma, "s": round(t1, 1),
                                  "factor": found.group(1) if found else None, "rc": r.returncode})
            rec["residues"].append(line)
            log.append(f"stage 1 curve {i} ({param}:{sigma}) {t1:.0f} s" + (f", factor {found.group(1)}" if found else ""))
            cache.store(name, rec)
        # stage 2 of each residue and variant
        for vname, B2, opts in variants:
            res = rec["s2"].setdefault(vname, {"B2": B2, "opts": opts, "curves": []})
            for i, line in enumerate(rec["residues"]):
                if i < len(res["curves"]): continue
                if not line:
                    res["curves"].append(None)
                    continue
                one = os.path.join(td, "one.txt")
                with open(one, "w") as fh:
                    fh.write(line + "\n")
                t0 = time.time()
                r = subprocess.run([ecm, "-v", "-v"] + opts + ["-resume", one, "%.0f" % B1] + ([B2] if B2 else []),
                                   capture_output=True, text=True, preexec_fn=cap)
                t1 = time.time() - t0
                o = outcome(r.stdout)
                mem = re.search(r"Estimated memory usage: ([\d.]+)([KMG]B)", r.stdout)
                res["curves"].append({"out": o, "params": params_of(r.stdout), "ms": step2_ms(r.stdout),
                                      "s": round(t1, 1), "mem": mem.group(1) + mem.group(2) if mem else None,
                                      "rc": r.returncode, "via": found_via(r.stdout) if o[0] == "F" else None})
                log.append(f"{vname} curve {i}: {o[0]} {t1:.0f} s" + ("" if o[0] != "?" else " " + o[1][-120:]))
                cache.store(name, rec)
    return name, log

def cmd_prepare(a):
    cache = Cache(a.cache)
    todo = [c for c in cases() if re.search(a.only, c[0])]
    # the largest first: they take longest
    todo.sort(key=lambda c: -(c[1] ** 1.6 * c[2]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(prepare_case, a.ecm, cache, c, int(a.memcap * (1 << 30))) for c in todo]
        for f in concurrent.futures.as_completed(futs):
            name, log = f.result()
            print(f"{name}: " + ("; ".join(log) if log else "cached"), flush=True)

# ---------------------------------------------------------------- check / bench (gpuecm2)
def gpu_run(gpu, lines, B1, B2, opts, env=None):
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "r.txt")
        with open(f, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        t0 = time.time()
        r = subprocess.run([gpu, "-v", "-v"] + opts + ["-resume", f, "%.0f" % B1] + ([B2] if B2 else []),
                           capture_output=True, text=True, env={**os.environ, **(env or {})})
        return r, time.time() - t0

def cmd_check(a):
    cache = Cache(a.cache)
    total = bad = skipped = 0
    maxbits = program_max_bits(a.gpuecm2)
    for name, bits, B1, cv, pbits, variants in cases():
        if not re.search(a.only, name): continue
        if bits > maxbits:  # a build for smaller moduli (GPUECM2_MAX_BITS)
            skipped += 1
            continue
        rec = cache.load(name)
        if not rec:
            print(f"skip {name}: not prepared")
            continue
        for vname, B2, opts in variants:
            ref = rec["s2"].get(vname)
            if not ref or len(ref["curves"]) < len(rec["residues"]):
                print(f"skip {name} {vname}: not prepared")
                continue
            idx = [i for i, l in enumerate(rec["residues"]) if l and ref["curves"][i] and ref["curves"][i]["out"][0] != "?"]
            if a.one_by_one:
                groups = [[i] for i in idx]
            else:  # after a factor the following residues continue with the cofactor: start a new run
                groups, g = [], []
                for i in idx:
                    g.append(i)
                    if ref["curves"][i]["out"][0] == "F":
                        groups.append(g)
                        g = []
                if g: groups.append(g)
            outs = {}
            for g in groups:
                r, t = gpu_run(a.gpuecm2, [rec["residues"][i] for i in g], B1, B2, opts)
                blocks = r.stdout.split("Resuming ECM residue")[1:]
                for j, i in enumerate(g):
                    outs[i] = outcome(blocks[j]) if j < len(blocks) else ["?", r.stdout[-200:] + r.stderr[-200:]]
                    outs[i].append(params_of(blocks[j]) if j < len(blocks) else None)
            mism = [i for i in idx if outs[i][:2] != ref["curves"][i]["out"] or outs[i][2] != ref["curves"][i]["params"]]
            # a factor GMP-ECM found through a non-invertible value in its root computation, which
            # gpuecm2 need not meet (docs/TESTING.md): a known difference if gpuecm2 gives a value
            known = [i for i in mism if ref["curves"][i].get("via") == "inversion" and outs[i][0] == "P" and
                     outs[i][2] == ref["curves"][i]["params"]]
            mism = [i for i in mism if i not in known]
            total += 1
            bad += bool(mism)
            hits = sum(ref["curves"][i]["out"][0] == "F" for i in idx)
            print(f"{'ok  ' if not mism else 'FAIL'} {name:22s} {vname:10s} {len(idx)} curves"
                  f"{f', {hits} found p in step 2' if hits else ''}"
                  + (f", known difference at {known} (GMP-ECM's factor from a failed inversion)" if known else "")
                  + (f"  mismatches: {mism}" if mism else ""), flush=True)
            for i in mism:
                print(f"    curve {i}: ecm {ref['curves'][i]['out'][0]} {ref['curves'][i]['params']} | "
                      f"gpuecm2 {outs[i][0]} {outs[i][2]} {outs[i][1][:100]}")
    print(f"{total - bad}/{total} case variants agree exactly"
          + (f" ({skipped} cases above the build's {maxbits} bits skipped)" if skipped else ""))
    return 1 if bad or not total else 0

def cmd_bench(a):
    cache = Cache(a.cache)
    print(f"{'case':22s} {'variant':10s} {'params':48s} {'ecm s2/curve':>12s} {'gpuecm2 (1)':>11s} "
          f"{'(batch)':>9s} {'speedup':>8s}")
    maxbits = program_max_bits(a.gpuecm2)
    for name, bits, B1, cv, pbits, variants in cases():
        if not re.search(a.only, name) or bits > maxbits: continue
        rec = cache.load(name)
        if not rec: continue
        for vname, B2, opts in variants:
            ref = rec["s2"].get(vname)
            if not ref: continue
            idx = [i for i, l in enumerate(rec["residues"]) if l and ref["curves"][i]]
            if not idx: continue
            ecm_ms = [ref["curves"][i]["ms"] for i in idx if ref["curves"][i]["ms"]]
            cpu = sum(ecm_ms) / len(ecm_ms) / 1e3 if ecm_ms else float("nan")
            def best(lines):
                t = []
                for _ in range(a.reps):
                    r, _ = gpu_run(a.gpuecm2, lines, B1, B2, opts)
                    ms = [int(x) for x in re.findall(r"Step 2 took (\d+)ms", r.stdout)]
                    if ms: t.append(sum(ms) / 1e3 / len(lines))
                return min(t) if t else float("nan")
            one = best([rec["residues"][idx[0]]])
            # a batch: the residues repeated to 8 curves (the same work as 8 different curves)
            many = [rec["residues"][i] for i in idx] * (8 // len(idx) + 1)
            batch = best(many[:8])
            print(f"{name:22s} {vname:10s} {ref['curves'][idx[0]]['params'] or '':48s} {cpu:10.1f} s "
                  f"{one:9.2f} s {batch:7.2f} s {cpu / one:7.0f}x", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "check", "bench", "list"])
    ap.add_argument("prog", nargs="?")
    ap.add_argument("--cache", default=os.path.join(os.getcwd(), "large-cache"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--memcap", type=float, default=3.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--one-by-one", action="store_true")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "list":
        for c in cases():
            print(c[0], f"{len(c[3])} curves", " ".join(v[0] for v in c[5]))
        return 0
    if not a.prog:
        ap.error("the program is missing")
    if a.cmd == "prepare":
        a.ecm = a.prog
        return cmd_prepare(a) or 0
    a.gpuecm2 = a.prog
    return cmd_check(a) if a.cmd == "check" else cmd_bench(a) or 0

if __name__ == "__main__":
    sys.exit(main())
