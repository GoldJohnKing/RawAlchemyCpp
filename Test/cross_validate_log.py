"""
Cross-validate the C++ log-encoding curves in include/color_data.h against the
authoritative `colour-science` library for ALL 14 registered log spaces.

Strategy (header-only probe — does not require linking the project binary):
  1. Write a tiny /tmp/ra_log_probe.cpp that #includes color_data.h, looks up
     LOG_SPACES.at(<name>).curve, and calls logEncode(x, curve) for each ramp
     value passed on argv.
  2. Compile the probe with g++ -std=c++17 -O2 (header-only inline, no linking).
  3. For each of the 14 log spaces, run the probe over a cut-crossing ramp:
        x in {0, cut-1e-4, cut, cut+1e-4, 0.18, 0.5, 1.0, 2.0}
     (cut per curve, derived below).
  4. Compare to colour.cctf_encoding(x, function=NAME).
     PASS criterion: max-abs-diff < 1e-5 for every curve.

colour-science is authoritative per the Phase 7 spec.

NOTE: This file EXTENDS the earlier F-Log2-only cross-val. The original
F-Log2-against-NEF behaviour is subsumed by curve-level validation here.
"""
import os
import subprocess
import sys
import tempfile

import colour
import numpy as np

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INCLUDE_DIR = os.path.join(REPO_ROOT, "include")
PROBE_CPP = "/tmp/ra_log_probe.cpp"
PROBE_BIN = "/tmp/ra_log_probe"

# ------------------------------------------------------------------
# The 14 registered log spaces, mapped to colour.cctf_encoding function names.
# Aliases: F-Log2C == F-Log2 OETF (only the gamut matrix differs);
#          S-Log3.Cine == S-Log3 OETF.
# ------------------------------------------------------------------
LOG_SPACES = [
    ("F-Log",        "F-Log"),
    ("F-Log2",       "F-Log2"),
    ("F-Log2C",      "F-Log2"),
    ("V-Log",        "V-Log"),
    ("N-Log",        "N-Log"),
    ("L-Log",        "L-Log"),
    ("Canon Log 2",  "Canon Log 2"),
    ("Canon Log 3",  "Canon Log 3"),
    ("S-Log3",       "S-Log3"),
    ("S-Log3.Cine",  "S-Log3"),
    ("Arri LogC3",   "ARRI LogC3"),
    ("Arri LogC4",   "ARRI LogC4"),
    ("Log3G10",      "Log3G10"),
    ("D-Log",        "D-Log"),
]

# Per-curve cut (the toe/mid or toe/log boundary in *input* linear x).
# Curves without a meaningful cut (Canon-style reflection curves, ARRI LogC4,
# Log3G10) use 0.0 so the ramp exercises the toe↔mid neighbourhood at small x.
CUTS = {
    "F-Log":       0.00089,
    "F-Log2":      0.000889,
    "F-Log2C":     0.000889,
    "V-Log":       0.01,
    "N-Log":       0.328,
    "L-Log":       0.006,
    "Canon Log 2": 0.0,        # reflection curve; toe cut at x=0 after /0.9
    "Canon Log 3": 0.0,        # cuts at s = ±0.014 (s = x/0.9); probe ramp covers it
    "S-Log3":      0.01125000,
    "S-Log3.Cine": 0.01125000,
    "Arri LogC3":  0.010591,
    "Arri LogC4":  0.0,        # LogC4 has no simple cut; boundary computed internally
    "Log3G10":     0.0,        # toe at xs = x+0.01 = 0 → x = -0.01; ramp uses x>=0
    "D-Log":       0.0078,
}


def build_probe():
    """Write & compile the C++ probe. Returns path to the executable."""
    src = r"""
#include "color_data.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    // argv[1] = log space name (must exist in LOG_SPACES)
    // argv[2..] = float ramp values
    if (argc < 3) { std::fprintf(stderr, "usage: probe <name> <x...>\n"); return 2; }
    const std::string name = argv[1];
    auto it = rawalchemy::LOG_SPACES.find(name);
    if (it == rawalchemy::LOG_SPACES.end()) {
        std::fprintf(stderr, "unknown log space: %s\n", name.c_str()); return 3;
    }
    rawalchemy::LogCurve curve = it->second.curve;
    for (int i = 2; i < argc; ++i) {
        float x = static_cast<float>(std::strtod(argv[i], nullptr));
        float y = rawalchemy::logEncode(x, curve);
        std::printf("%.10g\n", static_cast<double>(y));
    }
    return 0;
}
"""
    with open(PROBE_CPP, "w") as f:
        f.write(src)
    cmd = [
        "g++", "-std=c++17", "-O2",
        "-I", INCLUDE_DIR,
        PROBE_CPP, "-o", PROBE_BIN,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("PROBE COMPILE FAILED:", res.stderr, file=sys.stderr)
        sys.exit(1)
    return PROBE_BIN


def run_probe(name, xs):
    """Run the compiled probe for one log space over a list of x values."""
    args = [PROBE_BIN, name] + [f"{float(x):.12g}" for x in xs]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"PROBE RUN FAILED for {name}:", res.stderr, file=sys.stderr)
        sys.exit(1)
    return [float(line) for line in res.stdout.splitlines() if line.strip()]


def ramp_for(cut):
    """Build a cut-crossing ramp as specified by the Phase 7 task."""
    return [0.0, cut - 1e-4, cut, cut + 1e-4, 0.18, 0.5, 1.0, 2.0]


def main():
    build_probe()

    print("=" * 72)
    print("Phase 7 — log-curve C++ vs colour-science cross-validation (14 curves)")
    print(f"C++ probe : {PROBE_BIN}")
    print(f"colour    : colour {colour.__version__} (venv)")
    print("Tolerance : max-abs-diff < 1e-5 per curve")
    print("=" * 72)
    print(f"{'Log Space':<14}{'colour fn':<14}{'max-abs-diff':>16}{'  verdict':>10}")
    print("-" * 72)

    overall_pass = True
    worst_overall = 0.0
    for log_name, colour_fn in LOG_SPACES:
        cut = CUTS[log_name]
        xs = ramp_for(cut)
        cpp_vals = run_probe(log_name, xs)
        # colour.cctf_encoding expects array input; clamp tiny negatives to 0
        # only where the formula is undefined — but feed the same ramp to be
        # faithful. Most colour functions are vectorised and handle 0 fine.
        # NB: colour uses np.where/np.select internally, so the unused branch
        # (e.g. log of a negative for the reflection toe at x<0) raises a
        # benign RuntimeWarning that is masked by np.where. Silence it.
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cs_vals = [float(v) for v in colour.cctf_encoding(np.array(xs), function=colour_fn)]

        diffs = [abs(c - s) for c, s in zip(cpp_vals, cs_vals)]
        max_diff = max(diffs) if diffs else 0.0
        worst_overall = max(worst_overall, max_diff)
        verdict = "PASS" if max_diff < 1e-5 else "FAIL"
        if verdict == "FAIL":
            overall_pass = False
        print(f"{log_name:<14}{colour_fn:<14}{max_diff:16.3e}{verdict:>10}")

        # Detail on failure
        if verdict == "FAIL":
            print(f"    detail ({log_name}, cut={cut}):")
            for x, c, s, d in zip(xs, cpp_vals, cs_vals, diffs):
                flag = " <-- DIFF" if d >= 1e-5 else ""
                print(f"      x={x:12.6g}  cpp={c:14.8f}  colour={s:14.8f}  diff={d:.3e}{flag}")

    print("-" * 72)
    print(f"WORST overall max-abs-diff: {worst_overall:.3e}")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'} (all 14 curves < 1e-5)")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
