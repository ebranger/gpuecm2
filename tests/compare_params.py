#!/usr/bin/env python3
"""M1 check: gpuecm2 must print the same 'Input number', 'Using B1=...' and
'dF=..., k=..., d=..., d2=..., i0=...' lines as GMP-ECM for resumed residues.

Usage: compare_params.py <gpuecm2> <ecm> [--quick]
Each ecm run is memory-capped (ulimit -v) and killed right after it has printed
its stage 2 parameters, so this needs little RAM and no GPU.
"""
import os, random, resource, subprocess, sys, tempfile

GPUECM2, ECM = sys.argv[1], sys.argv[2]
QUICK = "--quick" in sys.argv
MEM_CAP = 1536 * 1024 * 1024
KEYS = ("Input number is", "Using B1=", "dF=", "Error")

def cap():
    resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP, MEM_CAP))

def run(cmd):
    """Collect the interesting lines; stop once the dF= line (or an error) is seen."""
    # only GMP-ECM is capped: a CUDA program reserves far more address space than it uses
    # gpuecm2 would pick a larger k when dF does not fit in the free GPU memory; here
    # GMP-ECM's own rule is compared, so the memory check is switched off and
    # gpuecm2 only prints its parameters
    env = dict(os.environ, GPUECM2_GPU_MEM="1e12", GPUECM2_PARAMS_ONLY="1")
    p = subprocess.Popen(["stdbuf", "-oL"] + cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, preexec_fn=cap if cmd[0] == ECM else None, env=env)
    out = []
    try:
        for line in p.stdout:
            line = line.rstrip("\n")
            if line.startswith(KEYS):
                out.append(line)
            if line.startswith(("dF=", "Error")):
                break
    finally:
        p.kill()
        p.wait()
    return out

def save_line(n_text, n, param, sigma, b1=3e9):
    x = random.randrange(2, n)
    return f"METHOD=ECM; PARAM={param}; SIGMA={sigma}; B1={b1:.0f}; N={n_text}; X=0x{x:x}; PROGRAM=test;\n"

random.seed(1)
numbers = []
for bits in ([200, 1024, 1921, 3000] if QUICK else
             [60, 100, 200, 330, 512, 700, 1024, 1500, 1919, 1920, 1921, 1983, 2048, 3000, 4400]):
    n = random.getrandbits(bits) | (1 << (bits - 1)) | 1
    numbers.append((str(n), n))
# expressions: X only needs to be some integer (both programs reduce it mod N)
for expr in ("2^1277-1", "(10^200+5)/7", "Phi(105,2)", "53!+1", "17#5+2"):
    numbers.append((expr, 2**64))

cases = []
B1s = [1e3, 5e4, 1e6, 11e6, 43e6, 11e7, 26e7, 85e7, 29e8]
for b1 in (B1s[2:6] if QUICK else B1s):
    cases.append(([], f"{b1:.0f}", None))
for opts, b2 in (([], "1e10"), ([], "123456789012"), ([], "5e14"), ([], "1000000000-5000000000000"),
                 (["-k", "3"], None), (["-k", "20"], None), (["-power", "1"], None), (["-dickson", "12"], None),
                 (["-maxmem", "100"], None), (["-maxmem", "1000"], None), (["-no-ntt"], None), (["-ntt"], None),
                 (["-k", "1"], "1e15")):
    cases.append((opts, "11e7", b2))

fails = total = 0
with tempfile.TemporaryDirectory() as td:
    for idx, (n_text, n) in enumerate(numbers):
        f = os.path.join(td, f"r{idx}.txt")
        with open(f, "w") as fh:
            fh.write(save_line(n_text, n, param=idx % 4, sigma=random.randrange(10**6, 2**31)))
        for opts, b1, b2 in (cases if idx < len(numbers) - 5 else cases[:3]):
            args = opts + ["-v", "-resume", f, b1] + ([b2] if b2 else [])
            a, b = run([GPUECM2] + args), run([ECM] + args)
            total += 1
            if a != b:
                fails += 1
                print(f"MISMATCH N={n_text[:40]}... bits={n.bit_length()} args={' '.join(opts)} {b1} {b2 or ''}")
                for x, y in zip(a + [''] * len(b), b + [''] * len(a)):
                    if x != y:
                        print(f"   gpuecm2: {x}\n   ecm:     {y}")
print(f"{total - fails}/{total} cases identical")
sys.exit(1 if fails else 0)
