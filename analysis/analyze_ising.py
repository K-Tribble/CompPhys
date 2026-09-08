"""
Analyze an Ising temperature scan written by src/ising_demo.cpp
Sample writes e, m, |m| series. Then susceptability, specific heat, 
Binder cumulant are calculated here with blocking and jackknife.

Cut each chains into blocks of length >= 2 * tau_int so block means are 
effectively independant. Then jackknife over the blocks.
"""

import glob
import json
import os
import sys

import numpy as np


def load_run(json_path):
    meta = json.load(open(json_path))
    npy = os.path.join(os.path.dirname(json_path), os.path.basename(meta["data_file"]))
    data = np.load(npy)
    cols = {name: i for i, name in enumerate(meta["observables"])}
    return meta, data, cols


def block_means(series, block_len):
    """series: (chains, draws) -> (total_blocks,) of block means."""
    nchain, ndraw = series.shape
    nblock = ndraw // block_len
    if nblock < 8:
        block_len = max(1, ndraw // 8)
        nblock = ndraw // block_len
    trimmed = series[:, : nblock * block_len]
    return trimmed.reshape(nchain, nblock, block_len).mean(axis=2).ravel()


def jackknife(blocks, estimator):
    # blocks: (B, k) array of per-block means of k quantities.
    # estimator maps a (k,) vector of averages to a scalar.
    B = blocks.shape[0]
    total = blocks.sum(axis=0)
    full = estimator(total / B)
    loo = np.array([estimator((total - blocks[i]) / (B - 1)) for i in range(B)])
    err = np.sqrt((B - 1) / B * np.sum((loo - loo.mean()) ** 2))
    return full, err


def analyze(json_path):
    meta, data, cols = load_run(json_path)

    T = meta["T"]
    beta = meta["beta"]
    N = meta["N"]

    e = data[:, :, cols["e"]]
    m = data[:, :, cols["m"]]
    am = data[:, :, cols["abs_m"]]

    # Block length from the longest tau_int in the run, so one choice covers
    # every observable. tau is reported in sweeps; draws are thinned sweeps.
    tau_sweeps = max(t for t in meta["tau_int_sweeps"] if t is not None)
    tau_draws = tau_sweeps / meta["thin"]
    block_len = max(1, int(np.ceil(2 * tau_draws)))

    # Primary quantities, blocked together so the jackknife keeps them aligned.
    prim = np.stack(
        [
            block_means(e, block_len),
            block_means(e ** 2, block_len),
            block_means(am, block_len),
            block_means(m ** 2, block_len),
            block_means(m ** 4, block_len),
        ],
        axis=1,
    )

    out = {"T": T, "beta": beta, "L": meta["L"], "block_len": block_len,
           "n_blocks": prim.shape[0], "tau_sweeps": tau_sweeps,
           "r_hat": max(r for r in meta["r_hat"] if r is not None)}

    out["e"], out["e_err"] = jackknife(prim, lambda a: a[0])
    out["absm"], out["absm_err"] = jackknife(prim, lambda a: a[2])

    # C = beta^2 N (<e^2> - <e>^2)
    out["C"], out["C_err"] = jackknife(prim, lambda a: beta ** 2 * N * (a[1] - a[0] ** 2))

    # chi = beta N (<m^2> - <|m|>^2)   -- the finite-size form using |m|
    out["chi"], out["chi_err"] = jackknife(prim, lambda a: beta * N * (a[3] - a[2] ** 2))

    # Binder cumulant U = 1 - <m^4> / (3 <m^2>^2)
    out["U"], out["U_err"] = jackknife(prim, lambda a: 1.0 - a[4] / (3.0 * a[3] ** 2))

    return out


def onsager_m(T):
    x = np.sinh(2.0 / T) ** -4
    return (1.0 - x) ** 0.125 if x < 1.0 else 0.0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    directory = sys.argv[1]
    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    if not files:
        print(f"no runs found in {directory}")
        return 1

    rows = [analyze(f) for f in files]
    rows.sort(key=lambda r: r["T"])

    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    print(f"L = {rows[0]['L']}   exact T_c = {Tc:.6f}\n")
    print(f"{'T':>7} {'e':>10} {'|m|':>18} {'chi':>16} {'C':>14} "
          f"{'U':>14} {'tau':>7} {'Rhat':>7}")
    for r in rows:
        print(f"{r['T']:7.3f} {r['e']:10.5f} "
              f"{r['absm']:9.5f}+-{r['absm_err']:<7.5f} "
              f"{r['chi']:9.3f}+-{r['chi_err']:<6.3f} "
              f"{r['C']:7.4f}+-{r['C_err']:<6.4f} "
              f"{r['U']:7.4f}+-{r['U_err']:<6.4f} "
              f"{r['tau_sweeps']:7.2f} {r['r_hat']:7.4f}")

    peak = max(rows, key=lambda r: r["chi"])
    print(f"\nsusceptibility peaks at T = {peak['T']:.3f} "
          f"(exact bulk T_c = {Tc:.4f}; the shift is finite-size, "
          f"T_c(L) - T_c ~ L^-1)")

    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt

        T = np.array([r["T"] for r in rows])
        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        for a, key, label in [
            (ax[0, 0], "absm", r"$\langle |m| \rangle$"),
            (ax[0, 1], "chi", r"$\chi$"),
            (ax[1, 0], "C", r"$C$"),
            (ax[1, 1], "U", "Binder $U$"),
        ]:
            a.errorbar(T, [r[key] for r in rows], yerr=[r[key + "_err"] for r in rows],
                       fmt="o-", ms=3, lw=1, capsize=2)
            a.axvline(Tc, ls="--", c="k", lw=0.8)
            a.set_xlabel("T")
            a.set_ylabel(label)
        ax[0, 0].plot(T, [onsager_m(t) for t in T], "r-", lw=1, label="Onsager")
        ax[0, 0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(directory, "ising.png"), dpi=140)
        print(f"wrote {os.path.join(directory, 'ising.png')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())