#!/usr/bin/env python3
"""CPU (GMP-ECM) versus GPU (gpuecm2) stage 2 at GMP-ECM's default parameters.

For a few composites (512, 1024, 2048 and 4096 bits, made from RSA challenge
numbers) and the B1 values of GMP-ECM's table of optimal parameters, this
  1. runs GMP-ECM stage 1 where that is cheap (residues saved with -save);
     for larger B1 the residue is "raised": the point of the largest real
     stage 1 is kept and the B1 field set to the target (checksum
     recomputed). Stage 2 does exactly the same work for any point, so this
     does not change the stage 2 timings;
  2. times GMP-ECM stage 2 (ecm -v -v -resume) one job at a time, waiting
     while compilers, GPU programs or other ecm processes run, and repeating
     a job that was disturbed;
  3. times gpuecm2 on the same residues and checks that both print the same
     "Product of G(f_i)" value (or find the same factor).

Usage:
  defaults_bench.py setup  --data DIR --ecm ECM            (stage 1 + raised residues)
  defaults_bench.py cpu    --data DIR --ecm ECM [--max-est S] [--only NAME]
  defaults_bench.py gpu    --data DIR --gpu GPUECM2 [--selfcheck] [--batch]
  defaults_bench.py gpus1  --data DIR --ecm-gpu ECM_GPU [--b1 1e5,1e6,3e6] [--gpucurves N]
  defaults_bench.py report --data DIR [--md FILE]
A file DIR/pause stops the CPU queue from starting new jobs.
"""
import argparse
import json
import math
import os
import random
import re
import resource
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RSA_FILE = os.path.join(HERE, "..", "tests", "data", "rsa_numbers.txt")

# GMP-ECM's table: digits, optimal B1
B1_TABLE = [(20, 11000), (25, 50000), (30, 250000), (35, 1000000), (40, 3000000), (45, 11000000),
            (50, 43000000), (55, 110000000), (60, 260000000), (65, 850000000), (70, 2900000000),
            (75, 7600000000), (80, 25000000000)]
NUMBERS = [("c512", "RSA-155"), ("c1024", "RSA-1024"), ("c2048", "RSA-2048"),
           ("c4096", "RSA-2048*RSA-1536*RSA-155")]
CHKSUMMOD = 4294967291
MAXMEM_MB = 8000            # GMP-ECM's -maxmem for both programs (same parameters)
ECM_AS_LIMIT = 10 << 30     # address-space cap for ecm processes
S1_CAP = 600.0              # real stage 1 only if estimated below this (seconds)
BATCH_CURVES = 64           # curves with real small-B1 stage 1 for batch runs
BUSY = {"nvcc", "cicc", "ptxas", "cudafe++", "fatbinary", "nvlink", "cc1plus", "ld", "make", "cmake",
        "gpuecm2", "bench_stage2", "bench_poly", "bench_roots", "test_roots", "test_poly", "test_polyalg",
        "test_stage2", "test_smoke", "test_host", "ecm", "ctest"}


def rsa(name):
    for line in open(RSA_FILE):
        if line.strip() and not line.startswith("#"):
            nm, val = line.split()[:2]
            if nm == name:
                return int(val)
    raise KeyError(name)


def number(spec):
    n = 1
    for part in spec.split("*"):
        n *= rsa(part)
    return n


def est_stage1(bits, b1):  # rough planning estimates (i7-14700F, GMP-ECM 7.0.6)
    return 1.0e-6 * (bits / 512) ** 1.6 * b1


def est_stage2(bits, dF, k):  # GMP-ECM 7.0.6, from runs at B1=11e7
    a = {512: 0.6e-6, 1024: 1.26e-6, 2048: 3.3e-6, 4096: 1.0e-5}[
        min((512, 1024, 2048, 4096), key=lambda x: abs(x - bits))]
    return a * dF * math.log2(dF) ** 1.5 * (k + 2.5)


def load_params(data):
    """GMP-ECM's parameters per case, from DATA/params.txt (query_params.sh)"""
    out = {}
    p = os.path.join(data, "params.txt")
    if os.path.exists(p):
        for line in open(p):
            m = re.match(r"(\S+) .*dF=(\d+), k=(\d+)", line)
            if m:
                out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def b1s(b1):  # GMP-ECM style: 11e3, 3e6, 26e7
    m, e = int(b1), 0
    while m % 10 == 0:
        m //= 10
        e += 1
    return "%de%d" % (m, e)


# ------------------------------------------------------------------ save lines

def parse_save(line):
    kv = []
    for f in line.strip().split(";"):
        f = f.strip()
        if f:
            k, v = f.split("=", 1)
            kv.append((k.strip(), v.strip()))
    return kv


def checksum(kv):
    d = dict(kv)
    c = int(float(d["B1"]))
    if "SIGMA" in d:
        c *= int(d["SIGMA"], 0) % CHKSUMMOD
    if "A" in d:
        c *= int(d["A"], 0) % CHKSUMMOD
    c *= int(d["N"], 0) % CHKSUMMOD
    c *= int(d["X"], 0) % CHKSUMMOD
    if "Z" in d:
        c *= int(d["Z"], 0) % CHKSUMMOD
    c *= (int(d.get("PARAM", "0")) + 1) % CHKSUMMOD
    return c % CHKSUMMOD


def raise_b1(line, b1, from_b1):
    kv = [(k, v) for k, v in parse_save(line) if k != "COMMENT"]
    kv = [(k, str(b1) if k == "B1" else v) for k, v in kv]
    kv = [(k, str(checksum(kv)) if k == "CHECKSUM" else v) for k, v in kv]
    kv.append(("COMMENT", "point from stage 1 to B1=%d, B1 raised for stage 2 timing" % from_b1))
    return "; ".join("%s=%s" % p for p in kv) + ";\n"


# ------------------------------------------------------------------ job running

def busy_procs(exclude):
    found = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) in exclude:
            continue
        try:
            with open("/proc/%s/comm" % pid) as f:
                name = f.read().strip()
        except OSError:
            continue
        if name in BUSY:
            found.add(name)
    return found


def wait_quiet(data, exclude):
    said = False
    while True:
        b = busy_procs(exclude)
        paused = os.path.exists(os.path.join(data, "pause"))
        if not b and not paused and os.getloadavg()[0] < 2.0:
            return
        if not said:
            print("  waiting (%s)" % (", ".join(sorted(b)) if b else "paused" if paused else "load"), flush=True)
            said = True
        time.sleep(10)


def limit_as():
    resource.setrlimit(resource.RLIMIT_AS, (ECM_AS_LIMIT, ECM_AS_LIMIT))


def run_job(data, cmd, stdin_text, out_path, attempts=3, before=None):
    """Runs cmd with nothing else heavy going on; returns (output, seconds, clean).
    before() is called before every attempt."""
    for attempt in range(attempts):
        wait_quiet(data, {os.getpid()})
        if before:
            before()
        disturbed = []
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, preexec_fn=limit_as)
        stop = threading.Event()

        def watch():
            while not stop.wait(3):
                b = busy_procs({os.getpid(), p.pid})
                if b:
                    disturbed.append(sorted(b))
        th = threading.Thread(target=watch, daemon=True)
        th.start()
        t0 = time.time()
        out, _ = p.communicate(stdin_text)
        dt = time.time() - t0
        stop.set()
        th.join()
        with open(out_path, "w") as f:
            f.write(out)
        if not disturbed:
            return out, dt, True
        print("  disturbed by %s, repeating" % disturbed[0], flush=True)
    return out, dt, False


def parse_ecm(out):
    r = {}
    m = re.search(r"Step 1 took (\d+)ms", out)
    if m:
        r["s1_ms"] = int(m.group(1))
    m = re.search(r"Step 2 took (\d+)ms", out)
    if m:
        r["s2_ms"] = int(m.group(1))
    m = re.search(r"Using B1=[^,]*, B2=([0-9-]+), polynomial ([^,]+),", out)
    if m:
        r["B2"], r["poly"] = m.group(1), m.group(2)
    m = re.search(r"dF=(\d+), k=(\d+), d=(\d+), d2=(\d+), i0=(-?\d+)", out)
    if m:
        r["dF"], r["k"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"Estimated memory usage: ([0-9.]+[KMG]?B)", out)
    if m:
        r["mem"] = m.group(1)
    m = re.search(r"Product of G\(f_i\) = (\d+)", out)
    if m:
        r["prod"] = m.group(1)
    m = re.search(r"Factor found in step (\d): (\d+)", out)
    if m:
        r["factor"] = m.group(2)
    m = re.search(r"Note: GPU memory.*", out)
    if m:
        r["note"] = m.group(0)
    return r


# ------------------------------------------------------------------ commands

def save_path(data, name, b1, suffix=""):
    return os.path.join(data, "res", "%s_%s%s.save" % (name, b1s(b1), suffix))


def cmd_setup(a):
    os.makedirs(os.path.join(a.data, "res"), exist_ok=True)
    os.makedirs(os.path.join(a.data, "out"), exist_ok=True)
    with open(os.path.join(a.data, "numbers.txt"), "w") as f:
        for name, spec in NUMBERS:
            n = number(spec)
            f.write("%s %s %d %d\n" % (name, spec, n.bit_length(), n))
    meta_path = os.path.join(a.data, "stage1.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    jobs = []
    for name, spec in NUMBERS:
        n = number(spec)
        bits = n.bit_length()
        for _, b1 in B1_TABLE:
            if est_stage1(bits, b1) <= S1_CAP:
                jobs.append((est_stage1(bits, b1), name, n, b1, 1))
        jobs.append((est_stage1(bits, 11000) * BATCH_CURVES, name, n, 11000, BATCH_CURVES))
    for est, name, n, b1, curves in sorted(jobs):
        path = save_path(a.data, name, b1, "_x%d" % curves if curves > 1 else "")
        key = os.path.basename(path)
        if key in meta and meta[key]["s1_ms"] and os.path.exists(path):
            continue
        print("stage 1 %s B1=%d (%d curve%s, ~%.0f s)" % (name, b1, curves, "s" * (curves > 1), est), flush=True)
        # a residue without a recorded time (its run was disturbed) is kept, as
        # later results depend on it: only the time is measured again
        target = path if not os.path.exists(path) else os.path.join(a.data, "out", "timing_only.save")

        def remove():
            if os.path.exists(target):
                os.remove(target)
        cmd = [a.ecm, "-save", target, "-c", str(curves), str(b1), "1"]
        out, dt, clean = run_job(a.data, cmd, "%d\n" % n, os.path.join(a.data, "out", key + ".s1.txt"),
                                 before=remove)
        times = [int(x) for x in re.findall(r"Step 1 took (\d+)ms", out)]
        meta[key] = {"name": name, "B1": b1, "curves": curves, "s1_ms": times, "clean": clean}
        json.dump(meta, open(meta_path, "w"), indent=1)
    # raised residues for the B1 values without a real stage 1
    for name, spec in NUMBERS:
        bits = number(spec).bit_length()
        real = [b1 for _, b1 in B1_TABLE if est_stage1(bits, b1) <= S1_CAP]
        src_b1 = max(real)
        line = open(save_path(a.data, name, src_b1)).readline()
        many = open(save_path(a.data, name, 11000, "_x%d" % BATCH_CURVES)).readlines()
        for _, b1 in B1_TABLE:
            if b1 not in real:
                with open(save_path(a.data, name, b1), "w") as f:
                    f.write(raise_b1(line, b1, src_b1))
            with open(save_path(a.data, name, b1, "_x%d" % BATCH_CURVES), "w") as f:
                for l in many:
                    f.write(raise_b1(l, b1, 11000))
    print("setup done")


def cases(data):
    out = []
    for line in open(os.path.join(data, "numbers.txt")):
        name, spec, bits, n = line.split()
        for digits, b1 in B1_TABLE:
            out.append((name, int(bits), digits, b1))
    return out


def load_results(data, kind):
    p = os.path.join(data, kind + ".json")
    return json.load(open(p)) if os.path.exists(p) else {}


def store_results(data, kind, res):
    p = os.path.join(data, kind + ".json")
    json.dump(res, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)


def cmd_cpu(a):
    res = load_results(a.data, "cpu")
    par = load_params(a.data)
    todo = sorted(((est_stage2(bits, *par["%s_%s" % (name, b1s(b1))]), name, bits, b1)
                   for name, bits, _, b1 in cases(a.data) if not a.only or name == a.only), key=lambda t: t[0])
    for est, name, bits, b1 in todo:
        key = "%s_%s" % (name, b1s(b1))
        if key in res and res[key].get("clean") and "s2_ms" in res[key]:
            continue
        if est > a.max_est:
            continue
        print("cpu stage 2 %s (~%.0f s)" % (key, est), flush=True)
        cmd = [a.ecm, "-v", "-v", "-maxmem", str(MAXMEM_MB), "-resume", save_path(a.data, name, b1), str(b1)]
        out, dt, clean = run_job(a.data, cmd, "", os.path.join(a.data, "out", key + ".cpu.txt"))
        r = parse_ecm(out)
        r.update({"wall": dt, "clean": clean, "load": os.getloadavg()[0], "when": time.strftime("%Y-%m-%d %H:%M")})
        res[key] = r
        store_results(a.data, "cpu", res)
        print("  %s" % {k: v for k, v in r.items() if k != "prod"}, flush=True)


def cmd_gpu(a):
    kind = "gpu_batch" if a.batch else ("gpu_check" if a.selfcheck else "gpu")
    res = load_results(a.data, kind)
    for name, bits, _, b1 in cases(a.data):
        if a.only and name != a.only:
            continue
        if a.max_b1 and b1 > a.max_b1:
            continue
        key = "%s_%s" % (name, b1s(b1))
        if (key in res and not a.redo) or (a.keys and key not in a.keys.split(",")):
            continue
        path = save_path(a.data, name, b1, "_x%d" % BATCH_CURVES if a.batch else "")
        cmd = [a.gpu, "-v", "-v", "-maxmem", str(MAXMEM_MB)] + (["-selfcheck"] if a.selfcheck else []) + \
              a.extra.split() + ["-resume", path, str(b1)]
        t0 = time.time()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        dt = time.time() - t0
        with open(os.path.join(a.data, "out", key + "." + kind + ".txt"), "w") as f:
            f.write(p.stdout)
        r = parse_ecm(p.stdout)
        if a.batch:
            r["s2_ms_all"] = [int(x) for x in re.findall(r"Step 2 took (\d+)ms", p.stdout)]
            r["curves"] = len(r["s2_ms_all"])
        r.update({"wall": dt, "rc": p.returncode, "when": time.strftime("%Y-%m-%d %H:%M")})
        if "self-check failed" in p.stdout:
            r["check_failed"] = True
        res[key] = r
        store_results(a.data, kind, res)
        print("%-14s %s" % (key, {k: v for k, v in r.items() if k not in ("prod", "s2_ms_all")}), flush=True)


def cmd_gpubench(a):
    """steady-state GPU time per curve with bench_stage2 (random curves, engine reused):
    unbatched, and batched with the automatic batch size"""
    res = load_results(a.data, "gpubench")
    specs = dict((name, spec) for name, spec in NUMBERS)
    for name, bits, _, b1 in cases(a.data):
        if (a.only and name != a.only) or (a.max_b1 and b1 > a.max_b1):
            continue
        key = "%s_%s" % (name, b1s(b1))
        if key in res and not a.redo:
            continue
        r = {}
        for tag, batch in (("single", 1), ("batched", 0)):
            if batch != 1 and b1 > 11e7:
                continue
            reps = 5 if b1 <= 11e7 else 2
            cmd = [a.bench, specs[name], str(b1), "-maxmem", str(MAXMEM_MB), "-reps", str(reps), "-untimed", "1",
                   "-batch", str(batch)]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            m = re.search(r"total ([0-9.]+) ms", p.stdout)
            mb = re.search(r"batch of (\d+) curves", p.stdout)
            r[tag] = float(m.group(1)) if m else None
            r[tag + "_B"] = int(mb.group(1)) if mb else 1
            if not m:
                r[tag + "_err"] = p.stdout[-300:]
        r["when"] = time.strftime("%Y-%m-%d %H:%M")
        res[key] = r
        store_results(a.data, "gpubench", res)
        print("%-14s %s" % (key, r), flush=True)


def cmd_gpus1(a):
    """GMP-ECM's GPU stage 1 (ecm -gpu, CGBN): GPU time per curve for each
    number at the B1 values in --b1 (default 1e5,1e6,3e6); the report scales
    the largest measured B1 linearly to the other B1 values (stage 1 costs
    about 1.44 B1 doublings and additions)."""
    res = load_results(a.data, "gpus1")
    b1list = [float(x) for x in (a.b1 or "1e5,1e6,3e6").split(",")]
    for line in open(os.path.join(a.data, "numbers.txt")):
        name, spec, bits, n = line.split()
        if a.only and name != a.only:
            continue
        for b1 in b1list:
            key = "%s_%s" % (name, b1s(b1))
            if key in res and not a.redo:
                continue
            wait_quiet(a.data, set())
            cmd = [a.ecm_gpu, "-gpu", "-v"] + (["-gpucurves", str(a.gpucurves)] if a.gpucurves else []) + \
                  ["%.0f" % b1, "1"]
            t0 = time.time()
            p = subprocess.run(cmd, input=n + "\n", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            m = re.search(r"Computing (\d+) Step 1 took (\d+)ms of CPU time / (\d+)ms of GPU time", p.stdout)
            kb = re.search(r"CGBN<(\d+), *(\d+)>", p.stdout)
            r = {"name": name, "bits": int(bits), "B1": b1, "wall": time.time() - t0,
                 "when": time.strftime("%Y-%m-%d %H:%M")}
            if m:
                r.update({"curves": int(m.group(1)), "gpu_ms": int(m.group(3)),
                          "ms_per_curve": int(m.group(3)) / int(m.group(1))})
            else:
                r["error"] = p.stdout[-300:]
            if kb:
                r["kernel"] = "%s bits, TPI %s" % kb.groups()
            with open(os.path.join(a.data, "out", key + ".gpus1.txt"), "w") as f:
                f.write(p.stdout)
            res[key] = r
            store_results(a.data, "gpus1", res)
            print("%-14s %s" % (key, r), flush=True)


def gpu_stage1_ms(gs1, name, b1):
    """GPU stage 1 time per curve at b1: measured, or scaled linearly from the
    largest measured B1 of that number; (ms, measured?)"""
    m = gs1.get("%s_%s" % (name, b1s(b1)))
    if m and "ms_per_curve" in m:
        return m["ms_per_curve"], True
    best = max((v for v in gs1.values() if v.get("name") == name and "ms_per_curve" in v),
               key=lambda v: v["B1"], default=None)
    return (best["ms_per_curve"] * b1 / best["B1"], False) if best else (None, False)


def fmt_ms(ms):
    if ms is None:
        return "-"
    s = ms / 1000.0
    if ms < 10:
        return "%.2g ms" % ms
    if s < 1:
        return "%.0f ms" % ms
    if s < 100:
        return "%.1f s" % s
    if s < 6000:
        return "%.0f s" % s
    return "%.1f h" % (s / 3600)


def cmd_report(a):
    cpu, gpu, chk, gb = (load_results(a.data, k) for k in ("cpu", "gpu", "gpu_check", "gpubench"))
    s1 = load_results(a.data, "stage1")
    # stage 1 time per unit of B1, from the largest real stage 1 of each number
    rate = {}
    for v in s1.values():
        if v["curves"] == 1 and v["s1_ms"] and (v["name"] not in rate or v["B1"] > rate[v["name"]][0]):
            rate[v["name"]] = (v["B1"], v["s1_ms"][0] / v["B1"])
    lines = ["| N | digits | B1 | B2 (GMP-ECM default) | dF, k | CPU stage 1 | CPU stage 2 | GPU stage 2 | GPU, batched | "
             "speedup | speedup, batched | check |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, bits, digits, b1 in cases(a.data):
        key = "%s_%s" % (name, b1s(b1))
        c, g, k, b = cpu.get(key, {}), gpu.get(key, {}), chk.get(key, {}), gb.get(key, {})
        m = s1.get(os.path.basename(save_path(a.data, name, b1)))
        if m and m["s1_ms"]:
            st1 = fmt_ms(m["s1_ms"][0])
        elif name in rate:
            st1 = "≈" + fmt_ms(rate[name][1] * b1)
        else:
            st1 = ""
        info = g or c or k
        b2s = str(info.get("B2", ""))
        dfk = "%s, %s" % (info.get("dF", ""), info.get("k", "")) if info else ""
        gs = b.get("single") or g.get("s2_ms")
        gbt = b.get("batched")
        sp1 = "%.0f" % (c["s2_ms"] / gs) if c.get("s2_ms") and gs else ""
        sp2 = "%.0f" % (c["s2_ms"] / gbt) if c.get("s2_ms") and gbt else ""
        if c.get("prod") and g.get("prod"):
            check = "same" if c["prod"] == g["prod"] else "**DIFFERENT**"
        elif c.get("factor") or g.get("factor"):
            check = "factor same" if c.get("factor") == g.get("factor") else "**factor differs**"
        elif k:
            check = "self-check ok" if k.get("rc") == 0 and not k.get("check_failed") else "**self-check failed**"
        else:
            check = ""
        bt = "%s (%d)" % (fmt_ms(gbt), b.get("batched_B", 1)) if gbt else ""
        lines.append("| %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            bits, digits, b1s(b1), b2s, dfk, st1, fmt_ms(c.get("s2_ms")), fmt_ms(gs), bt, sp1, sp2, check))
    # stage 1 against stage 2 on each kind of hardware
    gs1 = load_results(a.data, "gpus1")
    if gs1:
        lines += ["", "| N | B1 | B2 | GPU stage 1 / curve | kernel | GPU stage 2 / curve | GPU stage 2 / stage 1 | "
                  "CPU stage 2 / stage 1 |", "|---|---|---|---|---|---|---|---|"]
        kern = {}
        for v in gs1.values():
            if v.get("kernel"):
                kern[v["name"]] = v["kernel"]
        for name, bits, digits, b1 in cases(a.data):
            key = "%s_%s" % (name, b1s(b1))
            c, g, b = cpu.get(key, {}), gpu.get(key, {}), gb.get(key, {})
            info = g or c or chk.get(key, {})
            t1, measured = gpu_stage1_ms(gs1, name, b1)
            t2 = b.get("batched") or b.get("single") or g.get("s2_ms")
            m = s1.get(os.path.basename(save_path(a.data, name, b1)))
            c1 = m["s1_ms"][0] if m and m["s1_ms"] else (rate[name][1] * b1 if name in rate else None)
            lines.append("| %d | %s | %s | %s | %s | %s | %s | %s |" % (
                bits, b1s(b1), info.get("B2", ""), ("" if measured else "≈") + fmt_ms(t1) if t1 else "-",
                kern.get(name, ""), fmt_ms(t2), "%.3g" % (t2 / t1) if t1 and t2 else "",
                "%.2g" % (c["s2_ms"] / c1) if c.get("s2_ms") and c1 else ""))
    text = "\n".join(lines) + "\n"
    print(text)
    if a.md:
        open(a.md, "w").write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "cpu", "gpu", "gpubench", "gpus1", "report"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--ecm")
    ap.add_argument("--gpu")
    ap.add_argument("--bench")
    ap.add_argument("--max-est", type=float, default=1800)
    ap.add_argument("--max-b1", type=float, default=0)
    ap.add_argument("--only")
    ap.add_argument("--keys")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--extra", default="")
    ap.add_argument("--md")
    ap.add_argument("--ecm-gpu", help="GMP-ECM built with --enable-gpu (gpus1)")
    ap.add_argument("--gpucurves", type=int, default=0)
    ap.add_argument("--b1", help="comma-separated B1 values for gpus1")
    a = ap.parse_args()
    a.data = os.path.abspath(a.data)
    {"setup": cmd_setup, "cpu": cmd_cpu, "gpu": cmd_gpu, "gpubench": cmd_gpubench, "gpus1": cmd_gpus1,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
