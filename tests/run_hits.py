#!/usr/bin/env python3
"""Reproduce known ECM stage 2 hits with gpuecm2.

Each line of the hits file (default: tests/data/stage2_hits.txt) holds a
GMP-ECM stage 1 residue of a curve that is known to find a factor in stage 2,
with the stage 2 bound and Brent-Suyama polynomial of the logged run and the
factor that was found. gpuecm2 resumes every residue with those parameters
and must report exactly that factor in step 2. No GMP-ECM is needed.

Usage: run_hits.py <gpuecm2> [hits-file] [-only regex] [-verbose] [gpuecm2 options...]
(gpuecm2 may also be tests/capi_resume, which runs stage 2 through the C interface)

Line format (tab-separated; '#' starts a comment line):
  B2 [or B2min-B2, or "default": GMP-ECM's default B2]  options (as for gpuecm2 and ecm, e.g.
  "-dickson 12")  factor  source  resume line
The residues with the same parameters are run together (one gpuecm2 process
per group, at most one residue per number in a process, so that a factor
found on one curve cannot turn the next curve's number into a cofactor).
"""
import collections, os, re, subprocess, sys, tempfile, time

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    gpu = args.pop(0)
    here = os.path.dirname(os.path.abspath(__file__))
    hits_file = os.path.join(here, "data", "stage2_hits.txt")
    if args and not args[0].startswith("-"):
        hits_file = args.pop(0)
    only, extra, verbose = None, [], False
    while args:
        a = args.pop(0)
        if a == "-only": only = re.compile(args.pop(0))
        elif a == "-verbose": verbose = True
        else: extra.append(a)

    hits = []
    for ln, line in enumerate(open(hits_file), 1):
        if not line.strip() or line.startswith("#"):
            continue
        B2, opts, factor, source, resume = line.rstrip("\n").split("\t")
        if only and not only.search(source):
            continue
        f = dict(kv.strip().split("=", 1) for kv in resume.split(";") if "=" in kv)
        hits.append(dict(B1=f["B1"], B2=B2, popt=opts.split(), poly=opts, factor=factor, source=source, n=f["N"],
                         resume=resume, digits=len(f["N"]) if f["N"].isdigit() else 0))

    # groups of identical parameters, split so that a number occurs once per run
    groups = collections.OrderedDict()
    for h in hits:
        groups.setdefault((h["B1"], h["B2"], tuple(h["popt"])), []).append(h)
    runs = []
    for key, hs in groups.items():
        parts = []
        for h in hs:
            for p in parts:
                if h["n"] not in {x["n"] for x in p}:
                    p.append(h)
                    break
            else:
                parts.append([h])
        runs += [(key, p) for p in parts]

    ok = bad = skipped = 0
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        for (B1, B2, popt), hs in runs:
            path = os.path.join(td, "run.save")
            with open(path, "w") as fh:
                for h in hs:
                    fh.write(h["resume"] + "\n")
            bounds = [B1] if B2 == "default" else [B1, B2]
            r = subprocess.run([gpu, "-v"] + list(popt) + extra + ["-resume", path] + bounds,
                               capture_output=True, text=True)
            blocks = re.split(r"(?=^Resuming ECM residue)", r.stdout, flags=re.M)[1:]
            # a build for smaller moduli only (-DGPUECM2_ONLY_BITS) refuses larger numbers
            lim = re.search(r"at most (\d+)", r.stdout + r.stderr)
            for i, h in enumerate(hs):
                text = blocks[i] if i < len(blocks) else ""
                m = re.search(r"Factor found in step 2: (\d+)", text)
                found = m.group(1) if m else None
                if not found and lim and h["digits"] and int(h["n"]).bit_length() > int(lim.group(1)):
                    skipped += 1
                    continue
                good = found == h["factor"]
                ok += good
                bad += not good
                if not good:
                    print(f"FAIL {h['source']}: c{h['digits']}, B1={B1}, B2={B2}, {' '.join(popt)}: "
                          f"{'found ' + found if found else 'no factor'}, expected {h['factor']}")
                    if not text:
                        print("  (no output for this residue)", r.stdout[-500:], r.stderr[-500:])
            if verbose:
                print(f"  {len(hs):3d} residues, B1={B1}, B2={B2}, {' '.join(popt)} ({time.time() - t0:.0f} s)",
                      flush=True)
    print(f"{ok}/{ok + bad} known stage 2 hits reproduced" +
          (f" ({skipped} skipped: numbers too large for this build)" if skipped else ""))
    return 0 if bad == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
