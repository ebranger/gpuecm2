#!/usr/bin/env python3
"""Boundary tests: gpuecm2 against GMP-ECM on both sides of every input size or
parameter value where gpuecm2 switches between code paths, and gpuecm2 against
itself where only a hidden choice changes (batch size, NTT length, memory).

Thresholds covered (see docs/TESTING.md):
  - modulus sizes: the CGBN size ladder (512, 768, ..., 2048, 2560, ..., 4608
    bits) and one bit more, 32-bit limb counts, GMP-ECM's NTT threshold
    (1920 bits: power-of-two dF or not), the kernels of the RNS->limbs
    conversion (tiled from 993 to 3552 bits, otherwise one dot product per
    thread with CB 32/16/8, switching at 2254-2258 and 4577 bits), sizes where the RNS
    primes leave no spare bit over the CRT bound, tiny moduli and 4608 bits
    (the largest supported; 4609 bits must be refused);
  - the NTT length: products longer than the longest NTT use the split and
    exact-product paths; at small sizes forced with GPUECM2_MAX_LOGT, at their
    natural size switches with --heavy (dF = 2^21 at 483/484 bits, 2^20 at
    1186/1187 bits, a non-power-of-two dF > 2^18 at 2520/2521 bits);
  - stage 2 parameters: Brent-Suyama polynomials x^S and Dickson(S) of several
    degrees, k = 1 .. many blocks, tiny B2 (d2 = 1), B2 <= B1, B2min-B2
    ranges, huge B2 (i0 beyond 64 bits), -no-ntt / -ntt, parametrizations 0-3;
  - curve batches: sizes 1, 2, 3, 7, automatic, the maximum (512) and a batch
    cut by the GPU memory budget (GPUECM2_GPU_MEM); a budget too small for
    GMP-ECM's dF (gpuecm2 then uses more blocks: compared with GMP-ECM -k).

Every comparison uses GMP-ECM's "Product of G(f_i)" value (-v -v), which
depends on every root and every polynomial operation, or the factor found.
Residues come from GMP-ECM stage 1 with B1 = 10000 (stage 2 parameters are
set through B2), cached in the cache directory.

Usage: boundary_tests.py <gpuecm2> <ecm> [--heavy] [--cache DIR] [--only REGEX] [--list]
GMP-ECM runs one at a time with a memory cap (4 GB; 10 GB for --heavy).
"""
import argparse, os, random, re, resource, subprocess, sys, time

B1 = "10000"

# ---------------------------------------------------------------- numbers
SMALL_PRIMES = [p for p in range(3, 2000) if all(p % q for q in range(2, int(p ** 0.5) + 1))]

def is_prime(n):
    if n < 2: return False
    for p in SMALL_PRIMES:
        if n % p == 0: return n == p
    d, s = n - 1, 0
    while d % 2 == 0: d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19):
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1: break
        else: return False
    return True

def number(bits, tag=0):
    """A product of two primes with exactly `bits` bits, close to 2^bits (so
    that the CRT bounds are as tight as they get); deterministic."""
    rnd = random.Random(bits * 1009 + tag)
    if bits < 40:
        raise ValueError("too small")
    a = bits // 2
    b = bits - a
    while True:
        def prime_near_top(k):
            while True:
                x = rnd.getrandbits(k) | (0xff << (k - 8)) | 1
                if is_prime(x): return x
        n = prime_near_top(a) * prime_near_top(b)
        if n.bit_length() == bits:
            return n


# ---------------------------------------------------------------- models
# GMP-ECM's choice of dF and k (bestd.c, without -maxmem), to pick B2 values
# that give a wanted dF
L_ALL = [12, 18, 30, 42, 60, 90, 120, 150, 210, 240, 270, 330, 420, 510, 630, 840, 1050, 1260, 1470, 1680, 1890,
         2310, 2730, 3150, 3570, 3990, 4620, 5460, 6090, 6930, 8190, 9240, 10920, 12180, 13860, 16170, 18480, 20790,
         23100, 30030, 34650, 39270, 43890, 48510, 60060, 66990, 78540, 90090, 99330, 120120, 133980, 150150,
         180180, 210210, 240240, 270270, 300300, 334950, 371280, 420420, 510510, 570570, 600600, 630630, 746130,
         870870, 1021020, 1141140, 1291290, 1531530, 1711710, 1891890, 2081310, 2312310, 2552550, 2852850, 3183180,
         3573570, 3993990, 4594590, 5105100, 5705700, 6322470, 7147140, 7987980, 8978970, 10210200, 11741730,
         13123110, 14804790, 16546530, 19399380, 21411390, 23993970, 26816790, 29609580, 33093060, 36606570,
         40330290, 44414370, 49639590, 54624570, 60090030, 67897830, 77597520, 87297210, 96996900, 107056950,
         118107990]
L_PO2 = [12, 30, 60, 120, 240, 510, 1020, 2310, 4620, 9240, 19110, 39270, 79170, 158340, 324870, 690690, 1345890,
         2852850, 5705700, 11741730, 23130030, 48498450, 96996900]

def phi(n):
    r, p = 1, 2
    while p * p <= n:
        if n % p == 0:
            r *= p - 1; n //= p
            while n % p == 0: r *= p; n //= p
        p += 1
    return r * (n - 1) if n > 1 else r

def clog2(n):
    k = 0
    while (1 << k) < n: k += 1
    return k

def best_dF(B2, k, po2, B2min=10000):
    for d1 in (L_PO2 if po2 else L_ALL):
        dF = phi(d1) // 2
        if po2: dF = 1 << clog2(dF)
        d2 = next((x for x in range(5, 25, 2) if x % 3 and d1 % x), 25)
        if d2 >= 25 or d2 - 1 > dF: d2 = 1
        i0 = -((-(B2min - (d1 - 1) * d2)) // d1)
        i1 = (B2 + (d1 - 1) * d2) // d1
        j = i1 - i0 + 1
        if d2 > 1: j -= i1 // d2 - i0 // d2
        if -(-j // dF) <= k: return dF
    return None

def b2_for_dF(dF, k, po2):
    """the smallest B2 (to 2 significant digits) for which GMP-ECM picks dF with k blocks"""
    for e in range(4, 20):
        for m in range(10, 100):
            B2 = m * 10 ** (e - 1)
            if best_dF(B2, k, po2) == dF: return str(B2)
            if (best_dF(B2, k, po2) or 0) > dF: break
    return None

# gpuecm2's GPU memory estimate (stage2_gpu_bytes in src/api/stage2.cu)
def rns_primes_count(bits, dF):
    need = 2 * bits + clog2(2 * dF + 2) + 2
    for k in range(max(12, clog2(2 * dF)), 11, -1):
        bitsM, P = 0.0, 0
        c = ((1 << 31) - 1) >> k
        while c >= 1 and ((c << k) + 1) >= (1 << 28):
            if is_prime((c << k) + 1):
                import math
                bitsM += math.log2((c << k) + 1); P += 1
                if bitsM >= need - 1: return P
            c -= 1
    return None

def gpu_bytes(bits, dF, B=1):
    nl4 = 4 * ((bits + 31) // 32)
    P = (2 * bits + clog2(2 * dF + 2) + 2) // 30 + 2
    cg = 512
    while cg < bits: cg += 256 if cg < 2048 else 512
    L, T2 = clog2(dF), 1 << clog2(2 * dF)
    b = nl4 * dF * (L + 2 + 4) + 7 * (cg // 8) * dF + 4 * P * 2 * T2
    b *= B
    b += max(768 << 20, 4 * P * 3 * T2 * B)
    b += 8 * (cg // 8) * 9 * 16384
    return b + (256 << 20)

# ---------------------------------------------------------------- running
class Runner:
    def __init__(self, gpu, ecm, cache, memcap):
        self.gpu, self.ecm, self.cache, self.memcap = gpu, ecm, cache, memcap
        os.makedirs(cache, exist_ok=True)

    def cap(self):
        resource.setrlimit(resource.RLIMIT_AS, (self.memcap, self.memcap))

    def residues(self, bits, curves, tag=0, param=3):
        """save file with `curves` stage 1 residues (B1 = 10000) of number(bits, tag)"""
        path = os.path.join(self.cache, f"n{bits}_t{tag}_p{param}_c{curves}.save")
        if os.path.exists(path):
            return path
        n = number(bits, tag)
        tmp = path + ".tmp"
        if os.path.exists(tmp): os.remove(tmp)
        for c in range(curves):
            sigma = f"{param}:{1000 + 17 * c + tag}" if param != 0 else f"0:{1000 + 17 * c + tag}"
            r = subprocess.run([self.ecm, "-q", "-sigma", sigma, "-savea", tmp, B1, "1"], input=f"{n}\n",
                               capture_output=True, text=True, preexec_fn=self.cap)
            if r.returncode not in (0, 8):  # 8: composite cofactor is not an error here
                pass
        os.rename(tmp, path)
        return path

    @staticmethod
    def outcomes(text):
        out = []
        for blk in re.split(r"(?=^Resuming ECM residue)", text, flags=re.M)[1:]:
            m = re.search(r"Factor found in step (\d): (\d+)", blk)
            if m: out.append(("F" + m.group(1), m.group(2))); continue
            m = re.search(r"Product of G\(f_i\) = (\d+)", blk)
            if m: out.append(("P", m.group(1))); continue
            out.append(("-", ""))
        return out

    def run_ecm(self, save, b2, opts=()):
        r = subprocess.run([self.ecm, "-v", "-v"] + list(opts) + ["-resume", save, B1, b2],
                           capture_output=True, text=True, preexec_fn=self.cap)
        return self.outcomes(r.stdout), r

    def run_gpu(self, save, b2, opts=(), env=None):
        e = dict(os.environ)
        e.update(env or {})
        r = subprocess.run([self.gpu, "-v", "-v"] + list(opts) + ["-resume", save, B1, b2],
                           capture_output=True, text=True, env=e)
        return self.outcomes(r.stdout), r

def params_line(text):
    m = re.search(r"dF=(\d+), k=(\d+), d=(\d+), d2=(\d+), i0=(-?\d+)", text)
    return m.groups() if m else None

# ---------------------------------------------------------------- cases
def cases(heavy):
    C = []
    def add(name, fn): C.append((name, fn))

    # modulus sizes (dF = 256 at the default B2 for B1 = 1e4)
    sizes = [64, 100, 256, 487, 490, 511, 512, 513, 767, 768, 769, 992, 993, 1014, 1017, 1023, 1024, 1025, 1280, 1281,
             1536, 1537, 1792, 1793, 1912, 1916, 1920, 1921, 2048, 2049, 2253, 2254, 2257, 2258, 2560, 2561,
             3072, 3073, 3552, 3553, 3584, 3585, 4008, 4032, 4096, 4097, 4504, 4527, 4576, 4577, 4607, 4608]
    for bits in sizes:
        add(f"size/{bits}", lambda R, bits=bits: compare(R, R.residues(bits, 2), "1873422"))
    # larger polynomials (dF = 16384) at the sizes where the conversions switch tiles
    for bits in (512, 992, 993, 1920, 1921, 2254, 2258, 3552, 3553, 4576, 4577, 4608):
        add(f"size-dF16384/{bits}", lambda R, bits=bits: compare(R, R.residues(bits, 1), "5706890290",
                                                                ["-dickson", "6"]))
    add("size/4609-refused", lambda R: refused(R, R.residues(4609, 1), "1873422"))

    # Brent-Suyama polynomials
    for opt in (["-power", "1"], ["-power", "2"], ["-power", "3"], ["-power", "6"], ["-power", "7"],
                ["-dickson", "3"], ["-dickson", "5"], ["-dickson", "6"], ["-dickson", "12"], ["-dickson", "30"],
                ["-dickson", "60"]):
        add(f"poly/{opt[0][1:]}{opt[1]}", lambda R, opt=opt: compare(R, R.residues(1024, 2), "1000000000", opt))
    # blocks, tiny and degenerate B2, ranges, huge B2
    for k in ("1", "2", "7", "50"):
        add(f"k/{k}", lambda R, k=k: compare(R, R.residues(1024, 2), "100000000", ["-k", k]))
    for b2 in ("20000", "100000", "1000000"):  # tiny dF (d2 = 1 for the smallest)
        add(f"b2/{b2}", lambda R, b2=b2: compare(R, R.residues(1024, 2), b2))
    add("b2/equal-B1", lambda R: compare(R, R.residues(1024, 2), B1, allow_none=True))
    add("b2/below-B1", lambda R: compare(R, R.residues(1024, 2), "5000", allow_none=True))
    add("b2/range", lambda R: compare(R, R.residues(1024, 2), "500000000-700000000"))
    add("b2/range-single", lambda R: compare(R, R.residues(1024, 2), "123456789-123456789", allow_none=True))
    for b2 in ("18446744073709551616-18446744173709551616",
               "1000000000000000000000000000000000000000-1000000000000000000000000000100000000000"):
        add(f"b2/huge-{len(b2) // 2}digits", lambda R, b2=b2: compare(R, R.residues(512, 2), b2))
    add("b2/huge-2048bits", lambda R: compare(R, R.residues(2048, 1),
                                              "18446744073709551616-18446744173709551616"))
    # large B2 (not a range): 100 times GMP-ECM's default for B1 = 3e6 and beyond
    add("b2/large-512-4.4e11", lambda R: compare(R, R.residues(512, 1), "435229822400"))
    add("b2/large-1024-1e11-k20", lambda R: compare(R, R.residues(1024, 1), "100000000000", ["-k", "20"]))
    add("ntt/no-ntt-1024", lambda R: compare(R, R.residues(1024, 2), "1000000000", ["-no-ntt"]))
    add("ntt/ntt-2048", lambda R: compare(R, R.residues(2048, 2), "1000000000", ["-ntt"]))
    for param in (0, 1, 2):
        add(f"param/{param}", lambda R, param=param: compare(R, R.residues(1024, 2, param=param), "100000000"))

    # forced short NTTs: the split products and exact-product fallbacks
    for bits in (512, 2049, 4608):
        for lt in ("12", "13", "14"):
            add(f"maxlogt/{bits}/{lt}", lambda R, bits=bits, lt=lt: compare(
                R, R.residues(bits, 1), "5706890290", ["-dickson", "6"], env={"GPUECM2_MAX_LOGT": lt}))

    # batches (consecutive residues of one number)
    add("batch/1024-dF256", lambda R: batches(R, R.residues(1024, 520), "1873422", [1, 2, 3, 7, 64, 0, 512]))
    add("batch/2049-nonpo2", lambda R: batches(R, R.residues(2049, 40), "1873422", [1, 3, 7, 0]))
    add("batch/1024-dF16384", lambda R: batches(R, R.residues(1024, 12), "5706890290", [1, 5, 0, 12],
                                                ["-dickson", "6"]))
    # a budget that holds about 40 curves of dF = 256 (the automatic batch is 512)
    budget = (gpu_bytes(1024, 256, 40) >> 20) + 1
    add("batch/budget-cut", lambda R: batches(R, R.residues(1024, 520), "1873422", [0, 512],
                                              env={"GPUECM2_GPU_MEM": str(budget)}, expect_batches_max=40))
    # a budget below what dF = 16384 needs, above dF = 8192
    budget2 = ((gpu_bytes(1024, 16384) + gpu_bytes(1024, 8192)) >> 21)
    add("memory/more-blocks", lambda R: more_blocks(R, R.residues(1024, 1), "5706890290", str(budget2)))

    if heavy:
        add("heavy/b2-512-1e13", lambda R: compare(R, R.residues(512, 1), "10000000000000", selfcheck=True))
        add("heavy/b2-2048-1e12", lambda R: compare(R, R.residues(2048, 1), "1000000000000", selfcheck=True))
        # natural NTT length switches: (bits, B2 giving the wanted dF with k = 2)
        for bits, dF in ((483, 1 << 21), (484, 1 << 21), (1186, 1 << 20), (1187, 1 << 20),
                         (2520, 362880), (2521, 362880)):
            add(f"heavy/natural-ntt/{bits}", lambda R, bits=bits, dF=dF: compare(
                R, R.residues(bits, 1), b2_for_dF(dF, 2, bits <= 1920), ["-k", "2"], want_dF=dF, selfcheck=True))
    return C

def compare(R, save, b2, opts=(), env=None, allow_none=False, want_dF=None, selfcheck=False):
    a, ra = R.run_ecm(save, b2, opts)
    gopts = list(opts) + (["-selfcheck"] if selfcheck else [])
    b, rb = R.run_gpu(save, b2, gopts, env)
    pa, pb = params_line(ra.stdout), params_line(rb.stdout)
    if allow_none and pa and pb and pa[:2] == pb[:2] == ("0", "0"):
        pa = pb  # no stage 2: GMP-ECM then prints an uninitialised d
    if pa != pb:
        return False, f"parameters differ: GMP-ECM {pa}, gpuecm2 {pb}"
    if want_dF and pb and int(pb[0]) != want_dF:
        return False, f"dF {pb[0]}, wanted {want_dF} (adjust B2)"
    if not a or len(a) != len(b):
        if allow_none and len(a) == len(b):
            return True, "no stage 2 (both)"
        return False, f"{len(a)} GMP-ECM results, {len(b)} gpuecm2 results; {rb.stderr[-300:]}{ra.stderr[-200:]}"
    if allow_none and all(x[0] == "-" for x in a) and a == b:
        return True, "no stage 2 (both)"
    bad = [i for i in range(len(a)) if a[i] != b[i] or a[i][0] == "-"]
    if bad:
        i = bad[0]
        return False, f"residue {i}: GMP-ECM {a[i][0]} {a[i][1][:30]}, gpuecm2 {b[i][0]} {b[i][1][:30]}"
    lt = re.search(r"NTT length up to 2\^(\d+)", rb.stdout)
    return True, f"{len(a)} residues same{', dF=' + pb[0] + ' k=' + pb[1] if pb else ''}" + \
                 (f", NTT 2^{lt.group(1)}" if lt else "")

def refused(R, save, b2):
    """a residue of a too large number is skipped with a message (and the others still run)"""
    with open(save) as fh:
        big = fh.readline()
    small = open(R.residues(1024, 2)).read()
    path = save + ".mixed"
    with open(path, "w") as fh:
        fh.write(small + big + small)
    b, rb = R.run_gpu(path, b2)
    a, ra = R.run_ecm(R.residues(1024, 2), b2)
    # (the exit code is that of the last residue, as with GMP-ECM)
    ok = "at most" in rb.stderr and b[:2] == a and b[3:] == a and b[2][0] == "-"
    msg = (rb.stderr.strip().splitlines() or [""])[0]
    return ok, f"skipped, other residues same as GMP-ECM; {msg[:90]}"

def batches(R, save, b2, sizes, opts=(), env=None, expect_batches_max=None):
    ref, rr = R.run_ecm(save, b2, opts)
    if not ref or any(x[0] == "-" for x in ref):
        return False, "GMP-ECM gave no results"
    notes = []
    for bsz in sizes:
        b, rb = R.run_gpu(save, b2, list(opts) + ["-batch", str(bsz)], env)
        if b != ref:
            bad = next((i for i in range(min(len(b), len(ref))) if b[i] != ref[i]), min(len(b), len(ref)))
            return False, f"-batch {bsz}: residue {bad} differs ({len(b)} results)"
        sizes_seen = [int(x) for x in re.findall(r"Stage 2 of (\d+) residues together", rb.stdout)]
        biggest = max(sizes_seen) if sizes_seen else 1
        if expect_batches_max is not None and biggest > expect_batches_max:
            return False, f"-batch {bsz}: batches of {biggest} residues, budget allows {expect_batches_max}"
        notes.append(f"{bsz or 'auto'}:{biggest}")
    return True, f"{len(ref)} residues same for batch sizes (requested:largest) " + " ".join(notes)

def more_blocks(R, save, b2, mem_mb):
    b, rb = R.run_gpu(save, b2, ["-dickson", "6"], {"GPUECM2_GPU_MEM": mem_mb})
    m = re.search(r"using dF=(\d+), k=(\d+)", rb.stdout)
    if not m:
        return False, "no 'using more blocks' note: " + rb.stdout[-200:] + rb.stderr[-200:]
    a, ra = R.run_ecm(save, b2, ["-dickson", "6", "-k", m.group(2)])
    if params_line(ra.stdout)[:2] != m.groups():
        return False, f"GMP-ECM -k {m.group(2)} gives {params_line(ra.stdout)}"
    return (a == b and a and a[0][0] != "-"), f"dF={m.group(1)}, k={m.group(2)}: same as GMP-ECM -k {m.group(2)}" \
        if a == b else "values differ from GMP-ECM -k"

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="gpuecm2 boundary tests")
    ap.add_argument("gpuecm2"); ap.add_argument("ecm")
    ap.add_argument("--heavy", action="store_true", help="also the natural NTT-length switches (GMP-ECM needs ~6 GB, ~5 min each)")
    ap.add_argument("--cache", default=os.path.join(os.getcwd(), "boundary_cache"))
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    R = Runner(a.gpuecm2, a.ecm, a.cache, (10 if a.heavy else 4) << 30)
    todo = [(n, f) for n, f in cases(a.heavy) if re.search(a.only, n)]
    if a.list:
        for n, _ in todo: print(n)
        return 0
    fails = 0
    t0 = time.time()
    for name, fn in todo:
        t = time.time()
        try:
            ok, msg = fn(R)
        except Exception as e:  # report and go on
            ok, msg = False, f"exception {e!r}"
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:28s} {msg} ({time.time() - t:.0f} s)", flush=True)
    print(f"{len(todo) - fails}/{len(todo)} boundary cases passed ({time.time() - t0:.0f} s)")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
