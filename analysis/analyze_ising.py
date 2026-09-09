"""Analyse Ising temperature scans written by src/ising_demo.cpp.

Runs are grouped by L. Susceptibility, specific heat and the Binder cumulant
are blocked at 2 tau_int and jackknifed over the pooled blocks.

    python3 analyze_ising.py DIR [DIR ...] [--plot] [--csv FILE]
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))
U_STAR = 0.6106

# `m` changes sign only on a tunnelling event, so its tau and R-hat describe
# the +-m degeneracy rather than the sampling. Everything blocked below is
# sign-symmetric.
SYMMETRIC = ("e", "abs_m")

DEFAULT_MIN_BLOCKS = 32
REQUIRED = ("e", "m", "abs_m")
I_E, I_E2, I_ABSM, I_M2, I_M4 = range(5)


def load_run(json_path):
    """Load one run, checking the .json and .npy agree."""
    with open(json_path) as fh:
        meta = json.load(fh)

    npy = os.path.join(os.path.dirname(json_path),
                       os.path.basename(meta["data_file"]))
    data = np.load(npy)

    names = meta["observables"]
    expected = (meta["num_chains"], meta["num_draws"], len(names))
    if data.shape != expected:
        raise ValueError(
            f"{json_path}: metadata says {expected}, {npy} holds {data.shape}")

    missing = [n for n in REQUIRED if n not in names]
    if missing:
        raise ValueError(f"{json_path}: missing observable(s) {missing}")

    return meta, data, {name: i for i, name in enumerate(names)}


def block_means(series, block_len):
    """(chains, draws, k) -> (chains * nblock, k) of block means."""
    nchain, ndraw, k = series.shape
    nblock = ndraw // block_len
    if nblock < 2:
        raise ValueError(
            f"block_len {block_len} leaves {nblock} blocks per chain")
    trimmed = series[:, : nblock * block_len]
    return trimmed.reshape(nchain, nblock, block_len, k).mean(axis=2).reshape(-1, k)


def choose_block_len(meta, nchain, ndraw, min_blocks):
    """Return (block_len, starved, tau_sweeps); starved means the cap bound,
    not tau, so the resulting errors are underestimates."""
    taus = [t for name, t in zip(meta["observables"], meta["tau_int_sweeps"])
            if name in SYMMETRIC and t is not None]
    if not taus:
        raise ValueError(f"no usable tau_int among {SYMMETRIC}")

    tau_sweeps = max(taus)
    want = max(1, int(np.ceil(2.0 * tau_sweeps / meta["thin"])))
    cap = max(1, nchain * ndraw // min_blocks)
    return min(want, cap), want > cap, tau_sweeps


def jackknife(blocks, estimator):
    """Delete-one jackknife. Returns (bias-corrected value, error, bias)."""
    B = blocks.shape[0]
    if B < 2:
        raise ValueError("jackknife needs at least 2 blocks")

    total = blocks.sum(axis=0)
    full = estimator(total / B)
    loo = np.array([estimator((total - blocks[i]) / (B - 1)) for i in range(B)])

    err = np.sqrt((B - 1) / B * np.sum((loo - loo.mean()) ** 2))
    bias = (B - 1) * (loo.mean() - full)
    return full - bias, err, bias


def onsager_m(T, J=1.0):
    """Bulk spontaneous magnetisation, zero above Tc."""
    x = np.sinh(2.0 * J / T) ** -4
    return (1.0 - x) ** 0.125 if x < 1.0 else 0.0


def _ellipk(k):
    """Complete elliptic integral K of modulus k, by the AGM."""
    a, b = 1.0, np.sqrt(max(0.0, 1.0 - k * k))
    for _ in range(80):
        if abs(a - b) <= 1e-16 * abs(a):
            break
        a, b = 0.5 * (a + b), np.sqrt(a * b)
    return 0.5 * np.pi / a


def onsager_energy(T, J=1.0):
    """Bulk internal energy per site, exact at every T."""
    t = 2.0 * J / T
    k = 2.0 * np.sinh(t) / np.cosh(t) ** 2
    kp = 2.0 * np.tanh(t) ** 2 - 1.0
    return -J / np.tanh(t) * (1.0 + (2.0 / np.pi) * kp * _ellipk(k))


def analyze(json_path, min_blocks=DEFAULT_MIN_BLOCKS):
    """Derive the observables for one run."""
    meta, data, cols = load_run(json_path)

    beta = meta["beta"]
    N = meta["N"]

    e = data[:, :, cols["e"]]
    m = data[:, :, cols["m"]]
    am = data[:, :, cols["abs_m"]]
    nchain, ndraw = e.shape

    block_len, starved, tau_sweeps = choose_block_len(
        meta, nchain, ndraw, min_blocks)

    series = np.stack([e, e ** 2, am, m ** 2, m ** 4], axis=-1)
    blocks = block_means(series, block_len)

    rhats = [r for name, r in zip(meta["observables"], meta["r_hat"])
             if name in SYMMETRIC and r is not None]
    esses = [s for name, s in zip(meta["observables"], meta["ess"])
             if name in SYMMETRIC and s is not None]

    out = {
        "T": meta["T"],
        "beta": beta,
        "L": meta["L"],
        "N": N,
        "block_len": block_len,
        "n_blocks": blocks.shape[0],
        "dropped_per_chain": ndraw - (ndraw // block_len) * block_len,
        "tau_sweeps": tau_sweeps,
        "starved": starved,
        "r_hat": max(rhats) if rhats else float("nan"),
        "ess": min(esses) if esses else float("nan"),
        "sectors": sorted(set(int(np.sign(v)) for v in m.mean(axis=1))),
        "accept": float(np.mean([a for a in meta["accepted_fraction"]
                                 if a is not None])),
    }

    def jk(key, estimator):
        out[key], out[key + "_err"], out[key + "_bias"] = jackknife(
            blocks, estimator)

    jk("e", lambda a: a[I_E])
    jk("absm", lambda a: a[I_ABSM])
    jk("C", lambda a: beta ** 2 * N * (a[I_E2] - a[I_E] ** 2))
    # chi is the finite-size form using <|m|>; chi_m2 assumes <m> = 0 and is
    # the correct one above Tc.
    jk("chi", lambda a: beta * N * (a[I_M2] - a[I_ABSM] ** 2))
    jk("chi_m2", lambda a: beta * N * a[I_M2])
    jk("U", lambda a: 1.0 - a[I_M4] / (3.0 * a[I_M2] ** 2))

    return out


def peak_temperature(rows):
    """Susceptibility peak by parabolic interpolation through the maximum."""
    i = max(range(len(rows)), key=lambda j: rows[j]["chi"])
    if i == 0 or i == len(rows) - 1:
        return rows[i]["T"]

    (x0, y0), (x1, y1), (x2, y2) = [
        (rows[j]["T"], rows[j]["chi"]) for j in (i - 1, i, i + 1)]
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if d == 0.0:
        return x1
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
    return x1 if a == 0.0 else -b / (2.0 * a)


def report(by_L):
    """Print one table per lattice size, then extrapolate Tc if possible."""
    for L in sorted(by_L):
        rows = by_L[L]
        print(f"\nL = {L}   N = {rows[0]['N']}   exact T_c = {TC_EXACT:.6f}")
        print(f"{'T':>7} {'e':>19} {'|m|':>18} {'chi':>16} {'C':>15} "
              f"{'U':>15} {'tau':>8} {'Rhat':>6} {'blk':>6} {'nblk':>6}")
        for r in rows:
            flags = ""
            if r["starved"]:
                flags += " blocks<2tau"
            if r["r_hat"] > 1.01:
                flags += f" Rhat={r['r_hat']:.3f}"
            if abs(r["chi_bias"]) > r["chi_err"]:
                flags += " chi-bias>err"
            print(f"{r['T']:7.3f} "
                  f"{r['e']:10.5f}+-{r['e_err']:<7.5f} "
                  f"{r['absm']:9.5f}+-{r['absm_err']:<7.5f} "
                  f"{r['chi']:9.3f}+-{r['chi_err']:<6.3f} "
                  f"{r['C']:8.4f}+-{r['C_err']:<6.4f} "
                  f"{r['U']:8.4f}+-{r['U_err']:<6.4f} "
                  f"{r['tau_sweeps']:8.2f} {r['r_hat']:6.3f} "
                  f"{r['block_len']:6d} {r['n_blocks']:6d}"
                  f"{flags}")

        worst = max(rows, key=lambda r: abs(r["e"] - onsager_energy(r["T"])))
        print(f"  largest |e - Onsager| = "
              f"{abs(worst['e'] - onsager_energy(worst['T'])):.4f} at T = "
              f"{worst['T']:.3f}")

        starved = [r["T"] for r in rows if r["starved"]]
        if starved:
            print(f"  {len(starved)} temperature(s) blocked below 2 tau "
                  f"(T = {min(starved):.2f}..{max(starved):.2f}); "
                  f"those error bars are underestimates")

        print(f"  chi peaks at T = {peak_temperature(rows):.4f} "
              f"(exact bulk T_c = {TC_EXACT:.4f})")

    if len(by_L) >= 2:
        Ls = np.array(sorted(by_L), dtype=float)
        Tp = np.array([peak_temperature(by_L[int(L)]) for L in Ls])
        slope, intercept = np.polyfit(1.0 / Ls, Tp, 1)
        print(f"\nfinite-size extrapolation over L = "
              f"{', '.join(str(int(L)) for L in Ls)}:")
        print(f"  T_c(L) = {intercept:.4f} + {slope:.3f}/L   "
              f"(exact {TC_EXACT:.4f}, error {intercept - TC_EXACT:+.4f})")
    elif len(by_L) == 1:
        print("\nonly one L present; add more sizes to extrapolate T_c "
              "and to use the Binder crossing")


def write_csv(rows, path):
    """Write the derived table."""
    fields = [k for k in rows[0] if k != "sectors"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


def plot(by_L, path):
    """Write the five-panel summary figure."""
    import matplotlib.pyplot as plt

    panels = [
        ("absm", r"$\langle |m| \rangle$", None),
        ("chi", r"$\chi$", "log"),
        ("C", r"$C$", None),
        ("U", "Binder $U$", None),
        ("e", "Energy $e$", None),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    flat = axes.ravel()
    flat[-1].axis("off")

    for ax, (key, label, scale) in zip(flat, panels):
        for L in sorted(by_L):
            rows = by_L[L]
            T = np.array([r["T"] for r in rows])
            y = np.array([r[key] for r in rows])
            yerr = np.array([r[key + "_err"] for r in rows])
            ax.errorbar(T, y, yerr=yerr, fmt="o-", ms=3, lw=1, capsize=2,
                        label=f"L = {L}")
        ax.axvline(TC_EXACT, ls="--", c="k", lw=0.8)
        ax.set_xlabel("T")
        ax.set_ylabel(label)
        if scale:
            ax.set_yscale(scale)

    Tmin = min(r["T"] for rows in by_L.values() for r in rows)
    Tmax = max(r["T"] for rows in by_L.values() for r in rows)

    dense = np.linspace(Tmin, TC_EXACT, 400)
    flat[0].plot(dense, [onsager_m(t) for t in dense], "r-", lw=1,
                 zorder=0, label="Onsager")

    dense_all = np.linspace(Tmin, Tmax, 400)
    flat[4].plot(dense_all, [onsager_energy(t) for t in dense_all], "r-",
                 lw=1, zorder=0, label="Onsager")

    flat[3].axhline(U_STAR, ls=":", c="r", lw=1, label=f"$U^* = {U_STAR}$")

    for ax in flat[:5]:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directories", nargs="+",
                    help="directories of .json/.npy run pairs")
    ap.add_argument("--plot", action="store_true", help="write ising.png")
    ap.add_argument("--csv", metavar="FILE", help="write the derived table")
    ap.add_argument("--min-blocks", type=int, default=DEFAULT_MIN_BLOCKS,
                    help=f"jackknife block floor (default {DEFAULT_MIN_BLOCKS})")
    args = ap.parse_args(argv)

    files = []
    for d in args.directories:
        files.extend(sorted(glob.glob(os.path.join(d, "*.json"))))
    if not files:
        print(f"no runs found in {', '.join(args.directories)}", file=sys.stderr)
        return 1

    rows, failed = [], []
    for f in files:
        try:
            rows.append(analyze(f, args.min_blocks))
        except (ValueError, KeyError, OSError) as exc:
            failed.append((f, exc))

    for f, exc in failed:
        print(f"skipped {os.path.basename(f)}: {exc}", file=sys.stderr)
    if not rows:
        return 1

    by_L = {}
    for r in rows:
        by_L.setdefault(r["L"], []).append(r)
    for group in by_L.values():
        group.sort(key=lambda r: r["T"])

    report(by_L)

    if args.csv:
        write_csv(sorted(rows, key=lambda r: (r["L"], r["T"])), args.csv)
    if args.plot:
        plot(by_L, os.path.join(args.directories[0], "ising.png"))

    return 0


if __name__ == "__main__":
    sys.exit(main())