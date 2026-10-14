#!/usr/bin/env python3
"""Improper input: gpuecm2 must refuse it with a message, never crash, and never
run a stage 2 with bad parameters.

Each case is a command line and/or a resume file. A case passes when gpuecm2
exits normally (no signal, no CUDA or GMP abort) and the output shows the
expected reaction: an error for the option or bound, or the residue skipped
with a message (other residues in the same file still run). Valid residues
around bad ones check that a bad line does not disturb the others.

Usage: input_tests.py <gpuecm2> [-v]
Needs a GPU only for the few cases that run a (tiny) stage 2.
"""
import os, re, subprocess, sys, tempfile

GOOD_N = 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139  # RSA-100
# a residue of RSA-100 (any point will do for these tests; the CHECKSUM is computed below)
GOOD = dict(METHOD="ECM", PARAM="3", SIGMA="1000", B1="10000", N=str(GOOD_N), X=hex(12345678901234567890123))
CHKSUMMOD = 4294967291


def line(d, checksum=True):
    d = dict(d)
    if checksum and "CHECKSUM" not in d:
        try:
            c = int(float(d["B1"]))
            c = c * (int(d["SIGMA"]) % CHKSUMMOD) if "SIGMA" in d else c * (int(d["A"]) % CHKSUMMOD)
            n = int(d["N"]) if d["N"].lstrip("-").isdigit() else None
            if n is not None and n > 0:
                c = c * (n % CHKSUMMOD) * (int(d["X"], 0) % CHKSUMMOD) * ((int(d.get("PARAM", 0)) + 1) % CHKSUMMOD)
                d["CHECKSUM"] = str(c % CHKSUMMOD)
        except (ValueError, ZeroDivisionError, OverflowError, KeyError):
            pass
    return "; ".join(f"{k}={v}" for k, v in d.items()) + ";"


def with_(**kw):
    d = dict(GOOD)
    for k, v in kw.items():
        if v is None: d.pop(k, None)
        else: d[k] = v
    return d


# (name, gpuecm2 arguments before -resume, resume lines or None for the good one, B1, B2,
#  expectation: "error" (refused, nothing run), "skip" (bad line skipped, the good lines run),
#  "run" (accepted and run))
CASES = [
    ("B1 negative", [], None, "-1e4", None, "error"),
    ("B1 zero", [], None, "0", None, "error"),
    ("B1 nan", [], None, "nan", None, "error"),
    ("B1 inf", [], None, "inf", None, "error"),
    ("B1 1e400", [], None, "1e400", None, "error"),
    ("B1 above 2^53", [], None, "1e16", None, "error"),
    ("B1 text", [], None, "abc", None, "error"),
    ("B1 trailing text", [], None, "10000x", None, "error"),
    ("B2 inf", [], None, "10000", "inf", "error"),
    ("B2 1e400", [], None, "10000", "1e400", "error"),
    ("B2 negative", [], None, "10000", "-5", "error"),
    ("B2 text", [], None, "10000", "big", "error"),
    ("B2 range with text", [], None, "10000", "1e6-x", "error"),
    ("B2 range inf", [], None, "10000", "1e6-inf", "error"),
    ("-k 0", ["-k", "0"], None, "10000", None, "error"),
    ("-k negative", ["-k", "-1"], None, "10000", None, "error"),
    ("-k text", ["-k", "two"], None, "10000", None, "error"),
    ("-k huge", ["-k", "99999999999999999999"], None, "10000", None, "error"),
    ("-power 0", ["-power", "0"], None, "10000", None, "error"),
    ("-power negative", ["-power", "-3"], None, "10000", None, "error"),
    ("-dickson 0", ["-dickson", "0"], None, "10000", None, "error"),
    ("-dickson text", ["-dickson", "6x"], None, "10000", None, "error"),
    ("-dickson 100000", ["-dickson", "100000"], None, "10000", None, "error"),
    ("-maxmem negative", ["-maxmem", "-100"], None, "10000", None, "error"),
    ("-maxmem text", ["-maxmem", "lots"], None, "10000", None, "error"),
    ("-batch negative", ["-batch", "-2"], None, "10000", None, "error"),
    ("-batch text", ["-batch", "many"], None, "10000", None, "error"),
    ("unknown option", ["-frobnicate"], None, "10000", None, "error"),
    ("option without value", ["-k"], None, None, None, "error"),
    ("no B1", [], None, None, None, "error"),
    ("too many arguments", [], None, "10000", "1e6 1e7", "error"),
    ("missing file", ["MISSING"], None, "10000", None, "error"),
    ("B1 above saved B1", [], None, "20000", None, "error"),
    ("N = 0", [], [with_(N="0")], "10000", None, "skip"),
    ("N = 1", [], [with_(N="1")], "10000", None, "skip"),
    ("N negative", [], [with_(N="-15")], "10000", None, "skip"),
    ("N even", [], [with_(N=str(2 * GOOD_N))], "10000", None, "skip"),
    ("N bad expression", [], [with_(N="10^(")], "10000", None, "skip"),
    ("N over 4608 bits", [], [with_(N=str(2 ** 4700 + 1))], "10000", None, "skip"),
    ("B1 field nan", [], [with_(B1="nan")], "10000", None, "skip"),
    ("B1 field inf", [], [with_(B1="inf")], "10000", None, "skip"),
    ("B1 field negative", [], [with_(B1="-5")], "10000", None, "skip"),
    ("bad checksum", [], [dict(with_(), CHECKSUM="12345")], "10000", None, "skip"),
    ("PARAM 5", [], [with_(PARAM="5")], "10000", None, "skip"),
    ("PARAM -1", [], [with_(PARAM="-1")], "10000", None, "skip"),
    ("PARAM text", [], [with_(PARAM="x")], "10000", None, "skip"),
    ("param 0 sigma 5", [], [with_(PARAM="0", SIGMA="5")], "10000", None, "skip"),
    ("param 3 sigma 0", [], [with_(SIGMA="0")], "10000", None, "skip"),
    ("ETYPE 2", [], [dict(with_(), ETYPE="2")], "10000", None, "skip"),
    ("METHOD P-1", [], [with_(METHOD="P-1")], "10000", None, "skip"),
    ("METHOD unknown", [], [with_(METHOD="QS")], "10000", None, "skip"),
    ("no X", [], [with_(X=None)], "10000", None, "skip"),
    ("no sigma or A", [], [with_(SIGMA=None)], "10000", None, "skip"),
    ("A = 2 (singular)", [], [with_(SIGMA=None, PARAM=None, A="2")], "10000", None, "skip"),
    ("A = -2 (singular)", [], [with_(SIGMA=None, PARAM=None, A=str(GOOD_N - 2))], "10000", None, "skip"),
    ("X text", [], [with_(X="0xzz")], "10000", None, "skip"),
    ("unknown field", [], [dict(with_(), COLOR="blue")], "10000", None, "skip"),
    ("no semicolons", [], ["METHOD=ECM SIGMA=1000"], "10000", None, "skip"),
    ("garbage line", [], ["this is not a save file line"], "10000", None, "skip"),
    ("empty file", [], [], "10000", None, "run"),
    ("tiny N = 15", [], [with_(N="15")], "10000", None, "run"),
    ("tiny prime N = 101", [], [with_(N="101")], "10000", None, "run"),
    ("B2 below B1", [], None, "10000", "5000", "run"),
    ("B2min above B2", [], None, "10000", "1000000-500000", "run"),
]


def run_case(gpu, name, opts, lines, b1, b2, expect, td):
    path = os.path.join(td, "in.save")
    goodl = line(GOOD)
    if lines is None:
        body = [goodl]
    else:
        body = [goodl] + [l if isinstance(l, str) else line(l) for l in lines] + [goodl]
    with open(path, "w") as fh:
        fh.write("\n".join(body) + ("\n" if body else ""))
    if lines == []:
        open(path, "w").close()
    args = [gpu, "-v", "-v"] + [a for a in opts if a != "MISSING"]
    args += ["-resume", os.path.join(td, "nonexistent") if "MISSING" in opts else path]
    args += [b1] if b1 else []
    args += b2.split() if b2 else []
    r = subprocess.run(args, capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    crashed = r.returncode < 0 or "CUDA error" in out or "GNU MP" in out or "Assertion" in out
    ran = len(re.findall(r"Step 2 took", r.stdout))
    if crashed:
        return False, f"crashed (exit {r.returncode}): {out.strip()[-160:]}"
    if expect == "error":
        ok = r.returncode != 0 and ran == 0
        return ok, f"exit {r.returncode}: {(r.stderr.strip() or r.stdout.strip()).splitlines()[-1][:110] if out.strip() else ''}"
    if expect == "skip":
        # the bad line is skipped with a message, the two good lines around it run
        msgs = [l for l in r.stderr.splitlines() + r.stdout.splitlines()
                if re.search(r"(?i)error|skip|invalid|lacks|unknown|bad|not supported|only|must|cannot|warning", l)]
        ok = ran == 2 and bool(msgs)
        return ok, f"{ran} good residues ran; {msgs[0][:100] if msgs else 'NO MESSAGE'}"
    ok = r.returncode in (0, 2, 6, 8, 10, 14) and not crashed
    return ok, f"exit {r.returncode}, {ran} stage 2 runs: {out.strip().splitlines()[-1][:100] if out.strip() else ''}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    gpu = sys.argv[1]
    verbose = "-v" in sys.argv[2:]
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        for name, opts, lines, b1, b2, expect in CASES:
            ok, msg = run_case(gpu, name, opts, lines, b1, b2, expect, td)
            fails += not ok
            if verbose or not ok:
                print(f"{'ok  ' if ok else 'FAIL'} {name:24s} [{expect}] {msg}")
    print(f"{len(CASES) - fails}/{len(CASES)} improper-input cases handled")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
