"""Analyse anharmonic oscillator PIMC runs written by src/anharmonic_pimc.cpp.

Reads <stem>.npy (chains, draws, observables) and <stem>.json. The first
eight observables are scalars; the rest, when present, are the raw bead
positions x_0 .. x_{N-1}.

Everything derived from more than one observable (the heat capacity, the
kurtosis, the correlator, the gap) is blocked at a multiple of tau_int and
jackknifed over the pooled blocks, because the driver's per-observable
standard errors do not propagate through a ratio or a covariance.

    python3 analyze_anharmonic.py data/anharmonic_hmc [--plot] [--csv FILE]
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

SCALARS = ("E_vir", "E_prim", "E_vir_E_prim", "E_vir_sq",
           "m2", "m4", "xbar", "Rg2")


# --------------------------------------------------------------------- io

def load(stem):
    if stem.endswith(".json") or stem.endswith(".npy"):
        stem = os.path.splitext(stem)[0]
    with open(stem + ".json") as fh:
        meta = json.load(fh)
    data = np.load(stem + ".npy", mmap_mode="r")
    return meta, data


def column(meta, data, name):
    """(chains, draws) view of one named observable."""
    return np.asarray(data[:, :, meta["observables"].index(name)])


def path_block(meta, data):
    """(chains, draws, N) view of the bead positions, or None."""
    n_scalars = int(meta.get("num_scalars", len(SCALARS)))
    if data.shape[2] <= n_scalars:
        return None
    return data[:, :, n_scalars:]


# -------------------------------------------------------------- jackknife

def blocks_of(arrays, block_len):
    """Pool (chains, draws) arrays into per-block sums. Returns (B, k) sums
    and the block occupancy, with whole blocks only."""
    stacked = np.stack([np.asarray(a) for a in arrays], axis=-1)
    n_chains, n_draws, k = stacked.shape
    n_blocks = n_draws // block_len
    if n_blocks < 2:
        raise ValueError("not enough draws for two blocks at this tau")
    trimmed = stacked[:, : n_blocks * block_len, :]
    sums = trimmed.reshape(n_chains, n_blocks, block_len, k).sum(axis=2)
    return sums.reshape(n_chains * n_blocks, k), block_len


def jackknife(sums, count, func):
    """Leave-one-block-out jackknife of func(means)."""
    total = sums.sum(axis=0)
    n_blocks = sums.shape[0]
    n_total = count * n_blocks
    full = func(total / n_total)
    leave = np.array([func((total - sums[b]) / (n_total - count))
                      for b in range(n_blocks)])
    err = np.sqrt((n_blocks - 1) / n_blocks * np.sum((leave - leave.mean(axis=0)) ** 2,
                                                     axis=0))
    return full, err


def block_len_for(meta, names, factor=10.0):
    taus = [meta["tau_int_sweeps"][meta["observables"].index(n)] / meta["thin"]
            for n in names]
    return max(1, int(np.ceil(factor * max(taus))))


# ------------------------------------------------------------ derived bits

def energy_and_cv(meta, data):
    """E from the virial estimator, and Cv = beta^2 Cov(E_vir, E_prim).

    The fluctuation formula beta^2 Var(E) does NOT hold for a path integral
    estimator: the path weight carries its own beta dependence, so
        Cv = beta^2 [ Cov(E_vir, E_prim) - <dE_vir/dbeta> ]
    and E_vir has no explicit beta, killing the second term.
    """
    beta = meta["beta"]
    ev = column(meta, data, "E_vir")
    ep = column(meta, data, "E_prim")
    cross = column(meta, data, "E_vir_E_prim")

    bl = block_len_for(meta, ("E_vir", "E_prim"))
    sums, count = blocks_of((ev, ep, cross), bl)

    e, e_err = jackknife(sums, count, lambda m: m[0])
    cv, cv_err = jackknife(sums, count, lambda m: beta * beta * (m[2] - m[0] * m[1]))

    e_prim, e_prim_err = jackknife(sums, count, lambda m: m[1])
    return dict(E=e, E_err=e_err, E_prim=e_prim, E_prim_err=e_prim_err,
                Cv=cv, Cv_err=cv_err, block_len=bl, n_blocks=sums.shape[0])


def shape_moments(meta, data):
    """<x^2>, <x^4>, kurtosis and the radius of gyration."""
    m2 = column(meta, data, "m2")
    m4 = column(meta, data, "m4")
    rg = column(meta, data, "Rg2")

    bl = block_len_for(meta, ("m2", "m4", "Rg2"))
    sums, count = blocks_of((m2, m4, rg), bl)

    x2, x2_err = jackknife(sums, count, lambda m: m[0])
    x4, x4_err = jackknife(sums, count, lambda m: m[1])
    kurt, kurt_err = jackknife(sums, count, lambda m: m[1] / m[0] ** 2)
    rg2, rg2_err = jackknife(sums, count, lambda m: m[2])
    return dict(x2=x2, x2_err=x2_err, x4=x4, x4_err=x4_err,
                kurtosis=kurt, kurtosis_err=kurt_err,
                Rg2=rg2, Rg2_err=rg2_err)


# ------------------------------------------------------------- correlator

def spectra_by_block(paths, block_len, global_mean):
    """Per-block summed periodograms of the centred paths.

    Periodicity in imaginary time makes the path covariance circulant, so the
    circularly averaged autocovariance c(k) is both the best-conditioned
    covariance estimate available and the correlator <x(0)x(k dtau)> itself.
    Centre by the GLOBAL mean, never per draw: per-draw centring would delete
    the q=0 centroid mode, which is physical.
    """
    n_chains, n_draws, n_beads = paths.shape
    n_blocks = n_draws // block_len
    out = np.zeros((n_chains * n_blocks, n_beads // 2 + 1))
    idx = 0
    for c in range(n_chains):
        for b in range(n_blocks):
            chunk = np.asarray(paths[c, b * block_len:(b + 1) * block_len, :],
                               dtype=np.float64) - global_mean
            spec = np.abs(np.fft.rfft(chunk, axis=-1)) ** 2
            out[idx] = spec.sum(axis=0)
            idx += 1
    return out


def _logcosh(z):
    """log cosh z, stable for large |z| (cosh(500) overflows outright)."""
    a = np.abs(z)
    return a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)


def effective_mass(c, n_beads, dtau):
    """Periodic effective mass, vectorised over any leading axes.

    Solves  C(t)/C(t+1) = cosh(E(t-N/2)) / cosh(E(t+1-N/2))  for E by
    bisection. Bisection on the periodic form rather than a single global
    cosh fit, because the shape of the effective mass curve is what tells
    you where excited states have died out and where noise takes over.
    """
    half = n_beads / 2.0
    t = np.arange(n_beads // 2)
    c = np.asarray(c, dtype=np.float64)

    num = c[..., :n_beads // 2]
    den = c[..., 1:n_beads // 2 + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den
    valid = np.isfinite(ratio) & (ratio > 1.0) & (num > 0) & (den > 0)
    ratio = np.where(valid, ratio, 2.0)

    lo = np.full(ratio.shape, 1e-8)
    hi = np.full(ratio.shape, 5.0)

    def resid(e):
        return np.exp(_logcosh(e * (t - half)) - _logcosh(e * (t + 1 - half))) - ratio

    flo = resid(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = resid(mid)
        same = (flo * fmid) > 0
        lo = np.where(same, mid, lo)
        flo = np.where(same, fmid, flo)
        hi = np.where(same, hi, mid)

    out = 0.5 * (lo + hi) / dtau
    return np.where(valid, out, np.nan)


def plateau_fit(mass, mass_err, lo, hi):
    """Inverse-variance weighted average over a window, with chi^2/dof.

    chi^2/dof near 1 means the window really is a plateau. Much above 1 means
    residual excited state contamination at the near end, or the noise at the
    far end has been let in."""
    sel = np.arange(lo, hi)
    m = mass[sel]
    e = mass_err[sel]
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
    m, e = m[ok], e[ok]
    if m.size < 2:
        return float("nan"), float("nan"), float("nan"), 0
    w = 1.0 / e ** 2
    val = np.sum(w * m) / np.sum(w)
    err = 1.0 / np.sqrt(np.sum(w))
    chi2 = np.sum(w * (m - val) ** 2) / (m.size - 1)
    return float(val), float(err), float(chi2), int(m.size)


def correlator(meta, data, target_blocks=192, window=None):
    paths = path_block(meta, data)
    if paths is None:
        return None

    n_chains, n_draws, n_beads = paths.shape
    dtau = meta["dtau"]

    # global mean, streamed so the mmap is not pulled in whole
    total, count = 0.0, 0
    for c in range(n_chains):
        block = np.asarray(paths[c], dtype=np.float64)
        total += block.sum()
        count += block.size
    global_mean = total / count

    # blocks at least 10 tau long, but few enough that the jackknife over
    # replicate effective-mass curves stays cheap
    bl = max(block_len_for(meta, ("m2",)),
             (n_chains * n_draws) // max(1, target_blocks))
    spec_sums = spectra_by_block(paths, bl, global_mean)

    def c_of_k(mean_spec):
        return np.fft.irfft(mean_spec / n_beads, n=n_beads)

    c_full, c_err = jackknife(spec_sums, bl, c_of_k)

    # jackknife the effective mass itself, so each slice gets a real error bar
    total_spec = spec_sums.sum(axis=0)
    n_blocks = spec_sums.shape[0]
    n_total = bl * n_blocks
    reps = (total_spec - spec_sums) / (n_total - bl)
    c_reps = np.fft.irfft(reps / n_beads, n=n_beads, axis=-1)

    mass = effective_mass(c_full, n_beads, dtau)
    mass_reps = effective_mass(c_reps, n_beads, dtau)
    mass_err = np.sqrt((n_blocks - 1) / n_blocks
                       * np.nansum((mass_reps - np.nanmean(mass_reps, axis=0)) ** 2,
                                   axis=0))

    if window is None:
        # start once the near-end drift has settled, stop before the signal
        # is swamped: keep slices whose relative error is under 2 percent
        rel = np.where(np.isfinite(mass) & (mass > 0), mass_err / mass, np.inf)
        usable = np.where(rel < 0.02)[0]
        if usable.size >= 4:
            lo = max(int(0.15 * n_beads / 2), int(usable[0]))
            hi = int(usable[-1]) + 1
            if hi - lo < 4:
                lo, hi = int(usable[0]), int(usable[-1]) + 1
        else:
            lo, hi = max(1, n_beads // 16), max(3, n_beads // 6)
    else:
        lo, hi = window

    val, err, chi2, npts = plateau_fit(mass, mass_err, lo, hi)

    return dict(c=c_full, c_err=c_err, global_mean=global_mean,
                gap=mass, gap_err=mass_err,
                plateau=val, plateau_err=err, chi2=chi2, n_fit=npts,
                fit_range=(lo, hi), dtau=dtau, n_beads=n_beads,
                block_len=bl, n_blocks=n_blocks)


def position_histogram(meta, data, bins=201, span=None):
    paths = path_block(meta, data)
    if paths is None:
        return None
    if span is None:
        sample = np.asarray(paths[0, ::max(1, paths.shape[1] // 2000)], dtype=np.float64)
        span = 1.15 * np.abs(sample).max()
    edges = np.linspace(-span, span, bins + 1)
    hist = np.zeros(bins)
    for c in range(paths.shape[0]):
        block = np.asarray(paths[c], dtype=np.float64).ravel()
        hist += np.histogram(block, bins=edges)[0]
    centres = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]
    return centres, hist / (hist.sum() * width)


# ------------------------------------------------------------------ report

def report(meta, data, args):
    lam = meta["lambda"]
    beta = meta["beta"]
    print(f"N={meta['N']}  beta={beta}  dtau={meta['dtau']:.4g}  "
          f"omega={meta['omega']}  lambda={lam}  m={meta['mass_param']}")
    print(f"chains={meta['num_chains']}  draws={meta['num_draws']}  thin={meta['thin']}  "
          f"acceptance={meta['accepted_fraction_mean']:.3f}  "
          f"divergences={int(meta.get('divergences', 0))}")
    print(f"mass matrix: {meta.get('mass_source', '?')}   "
          f"leapfrog eps={meta.get('leapfrog_eps', float('nan')):.4g} "
          f"L={int(meta.get('leapfrog_steps', 0))}")

    worst = max(meta["r_hat"])
    print(f"worst R-hat over all observables: {worst:.4f}"
          f"{'   <-- CHECK' if worst > 1.01 else ''}\n")

    en = energy_and_cv(meta, data)
    sh = shape_moments(meta, data)

    print(f"blocking: {en['n_blocks']} blocks of {en['block_len']} draws\n")
    print(f"  E   (virial)     {en['E']:.6f} +- {en['E_err']:.6f}")
    print(f"  E   (primitive)  {en['E_prim']:.6f} +- {en['E_prim_err']:.6f}"
          f"   [consistency: {abs(en['E'] - en['E_prim']) / max(en['E_prim_err'], 1e-30):.2f} sigma]")
    print(f"  Cv               {en['Cv']:.6f} +- {en['Cv_err']:.6f}"
          f"   [{'resolved' if abs(en['Cv']) > 3 * en['Cv_err'] else 'NOT resolved'}]")
    print(f"  <x^2>            {sh['x2']:.6f} +- {sh['x2_err']:.6f}")
    print(f"  <x^4>            {sh['x4']:.6f} +- {sh['x4_err']:.6f}")
    print(f"  <x^4>/<x^2>^2    {sh['kurtosis']:.6f} +- {sh['kurtosis_err']:.6f}"
          f"   [excess {sh['kurtosis'] - 3.0:+.6f}; 0 = gaussian]")
    print(f"  <Rg^2>           {sh['Rg2']:.6f} +- {sh['Rg2_err']:.6f}")

    if lam == 0.0:
        ex = exact_lattice(meta)
        print(f"\n  lambda = 0 reference (exact at this lattice spacing):")
        print(f"    E     {ex['E']:.8f}   dev {(en['E'] - ex['E']) / en['E_err']:+.2f} sigma")
        print(f"    Cv    {ex['Cv']:.8f}   dev {(en['Cv'] - ex['Cv']) / en['Cv_err']:+.2f} sigma")
        print(f"    gap   {meta['omega']:.6f} (continuum)")

    corr = correlator(meta, data)
    if corr is not None:
        print(f"\n  correlator C(tau) from {corr['n_beads']} beads, "
              f"global mean <x> = {corr['global_mean']:+.6f}")
        print(f"  energy gap E1-E0 = {corr['plateau']:.4f} +- {corr['plateau_err']:.4f}"
              f"   (weighted plateau, slices {corr['fit_range'][0]}-{corr['fit_range'][1]}, "
              f"{corr['n_fit']} pts, chi2/dof = {corr['chi2']:.2f})")
        if corr["chi2"] > 2.0:
            print("    chi2/dof > 2: the window is not a clean plateau, "
                  "set it by hand after looking at the effective mass plot")
        if lam != 0.0:
            print(f"  first order perturbation theory: {meta['omega'] + 3 * lam:.4f}")
    return en, sh, corr


def exact_lattice(meta):
    """Exact discretised harmonic oscillator, valid only at lambda = 0."""
    n, beta = int(meta["N"]), meta["beta"]
    m, w = meta["mass_param"], meta["omega"]

    def x2(b):
        dt = b / n
        q = np.arange(n)
        h = (2 * m / dt) * (1 - np.cos(2 * np.pi * q / n)) + dt * m * w * w
        return np.mean(1.0 / h)

    e = m * w * w * x2(beta)
    h = 1e-5 * beta
    cv = -beta ** 2 * (m * w * w * (x2(beta + h) - x2(beta - h))) / (2 * h)
    return dict(E=e, x2=x2(beta), Cv=cv)


def plot(meta, data, en, sh, corr, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    hist = position_histogram(meta, data)
    if hist is not None:
        x, p = hist
        ax[0, 0].plot(x, p, lw=1.5, label="PIMC")
        g = np.exp(-x ** 2 / (2 * sh["x2"])) / np.sqrt(2 * np.pi * sh["x2"])
        ax[0, 0].plot(x, g, "--", lw=1.2,
                      label=f"gaussian, same $\\langle x^2\\rangle$")
        ax[0, 0].set_xlabel("$x$")
        ax[0, 0].set_ylabel(r"$|\psi_0(x)|^2$")
        ax[0, 0].set_title(f"position distribution, excess kurtosis "
                           f"{sh['kurtosis'] - 3:+.4f}")
        ax[0, 0].legend()

        ax[0, 1].semilogy(x, np.maximum(p, 1e-12), lw=1.5, label="PIMC")
        ax[0, 1].semilogy(x, np.maximum(g, 1e-12), "--", lw=1.2, label="gaussian")
        ax[0, 1].set_ylim(1e-6, None)
        ax[0, 1].set_xlabel("$x$")
        ax[0, 1].set_title("same, log scale (tails)")
        ax[0, 1].legend()

    if corr is not None:
        t = np.arange(corr["n_beads"]) * corr["dtau"]
        half = corr["n_beads"] // 2
        ax[1, 0].errorbar(t[:half], corr["c"][:half], yerr=corr["c_err"][:half],
                          fmt="o", ms=2.5, lw=0.8)
        ax[1, 0].set_yscale("log")
        ax[1, 0].set_xlabel(r"$\tau$")
        ax[1, 0].set_ylabel(r"$C(\tau)=\langle x(0)x(\tau)\rangle$")
        ax[1, 0].set_title("imaginary time correlator")

        gap = corr["gap"]
        ax[1, 1].errorbar(t[:len(gap)], gap, yerr=corr["gap_err"], fmt="o", ms=2.5, lw=0.8)
        ax[1, 1].axvspan(corr["fit_range"][0] * corr["dtau"],
                         corr["fit_range"][1] * corr["dtau"], color="C1", alpha=0.12)
        ax[1, 1].axhline(corr["plateau"], color="C1", ls="--",
                         label=f"plateau {corr['plateau']:.3f}")
        if meta["lambda"] != 0:
            ax[1, 1].axhline(meta["omega"] + 3 * meta["lambda"], color="C2", ls=":",
                             label=f"1st order PT {meta['omega'] + 3 * meta['lambda']:.3f}")
        else:
            ax[1, 1].axhline(meta["omega"], color="C2", ls=":", label="$\\omega$")
        ax[1, 1].set_ylim(0, 3 * max(corr["plateau"], meta["omega"]))
        ax[1, 1].set_xlabel(r"$\tau$")
        ax[1, 1].set_ylabel(r"$E_1-E_0$")
        ax[1, 1].set_title("effective mass")
        ax[1, 1].legend()

    fig.suptitle(f"anharmonic oscillator PIMC: "
                 f"$\\lambda$={meta['lambda']}, $\\beta$={meta['beta']}, N={meta['N']},  "
                 f"$E$={en['E']:.5f}$\\pm${en['E_err']:.5f}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    meta, data = load(args.stem)
    en, sh, corr = report(meta, data, args)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["N", "beta", "lambda", "E", "E_err", "Cv", "Cv_err",
                        "x2", "x2_err", "kurtosis", "kurtosis_err",
                        "Rg2", "Rg2_err", "gap", "gap_err"])
            w.writerow([meta["N"], meta["beta"], meta["lambda"],
                        en["E"], en["E_err"], en["Cv"], en["Cv_err"],
                        sh["x2"], sh["x2_err"], sh["kurtosis"], sh["kurtosis_err"],
                        sh["Rg2"], sh["Rg2_err"],
                        corr["plateau"] if corr else "", corr["plateau_err"] if corr else ""])
        print(f"wrote {args.csv}")

    if args.plot:
        out = args.out or (os.path.splitext(args.stem)[0] + ".png")
        plot(meta, data, en, sh, corr, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())