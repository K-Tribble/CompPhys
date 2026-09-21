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
import warnings

import numpy as np

SCALARS = ("E_vir", "E_prim", "E_vir_E_prim", "E_vir_sq",
           "m2", "m4", "xbar", "Rg2")

def load(stem):
    if stem.endswith(".json") or stem.endswith(".npy"):
        stem = os.path.splitext(stem)[0]
    with open(stem + ".json") as fh:
        meta = json.load(fh)
    data = np.load(stem + ".npy", mmap_mode="r")
    return meta, data


def column(meta, data, name):
    return np.asarray(data[:, :, meta["observables"].index(name)])


def path_block(meta, data):
    n_scalars = int(meta.get("num_scalars", len(SCALARS)))
    if data.shape[2] <= n_scalars:
        return None
    return data[:, :, n_scalars:]

def blocks_of(arrays, block_len):
    stacked = np.stack([np.asarray(a) for a in arrays], axis=-1)
    n_chains, n_draws, k = stacked.shape
    n_blocks = n_draws // block_len
    if n_blocks < 2:
        raise ValueError("not enough draws for two blocks at this tau")
    trimmed = stacked[:, : n_blocks * block_len, :]
    sums = trimmed.reshape(n_chains, n_blocks, block_len, k).sum(axis=2)
    return sums.reshape(n_chains * n_blocks, k), block_len

def jackknife(sums, count, func):
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

def energy_and_cv(meta, data):
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

def spectra_by_block(paths, block_len, global_mean):
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
    a = np.abs(z)
    return a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)


def effective_mass(c, n_beads, dtau):
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


def plateau_fit(mass, mass_err, mass_reps, lo, hi):
    sel = np.arange(lo, hi)
    ok = (np.isfinite(mass[sel]) & np.isfinite(mass_err[sel]) & (mass_err[sel] > 0)
          & np.all(np.isfinite(mass_reps[:, sel]), axis=0))
    sel = sel[ok]
    nan = float("nan")
    if sel.size < 4:
        return dict(value=nan, error=nan, naive=nan, drift=nan, drift_err=nan, n=int(sel.size))

    n_blocks = mass_reps.shape[0]
    scale = (n_blocks - 1) / n_blocks

    def weighted(idx):
        w = 1.0 / mass_err[idx] ** 2
        w /= w.sum()
        full = float(w @ mass[idx])
        reps = mass_reps[:, idx] @ w
        return full, reps

    value, reps = weighted(sel)
    error = float(np.sqrt(scale * np.sum((reps - reps.mean()) ** 2)))
    naive = float(1.0 / np.sqrt(np.sum(1.0 / mass_err[sel] ** 2)))

    half = sel.size // 2
    fa, ra = weighted(sel[:half])
    fb, rb = weighted(sel[half:])
    drift = fb - fa
    drift_err = float(np.sqrt(scale * np.sum(((rb - ra) - (rb - ra).mean()) ** 2)))

    return dict(value=value, error=error, naive=naive,
                drift=drift, drift_err=drift_err, n=int(sel.size))


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

    # leave-one-block-out correlators, then an effective mass for each
    total_spec = spec_sums.sum(axis=0)
    n_blocks = spec_sums.shape[0]
    n_total = bl * n_blocks
    reps = (total_spec - spec_sums) / (n_total - bl)
    c_reps = np.fft.irfft(reps / n_beads, n=n_beads, axis=-1)

    mass = effective_mass(c_full, n_beads, dtau)
    mass_reps = effective_mass(c_reps, n_beads, dtau)

    # far-end slices can be NaN in every replicate once the signal is gone
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmean(mass_reps, axis=0)
        mass_err = np.sqrt((n_blocks - 1) / n_blocks
                           * np.nansum((mass_reps - centre) ** 2, axis=0))
    mass_err = np.where(np.isfinite(mass), mass_err, np.nan)

    if window is None:
        # start past the near-end transient, stop before the signal is
        # swamped: keep slices whose relative error is under 2 percent
        with np.errstate(invalid="ignore", divide="ignore"):
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

    fit = plateau_fit(mass, mass_err, mass_reps, lo, hi)

    return dict(c=c_full, c_err=c_err, global_mean=global_mean,
                gap=mass, gap_err=mass_err,
                plateau=fit["value"], plateau_err=fit["error"],
                plateau_naive_err=fit["naive"],
                drift=fit["drift"], drift_err=fit["drift_err"], n_fit=fit["n"],
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

def report(meta, data, args):
    beta = meta["beta"]
    print(f"N={meta['N']}  beta={beta}  dtau={meta['dtau']:.4g}  "
          f"omega={meta['omega']}  lambda={meta['lambda']}  m={meta['mass_param']}")
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

    corr = correlator(meta, data)
    if corr is not None:
        print(f"\n  correlator C(tau) from {corr['n_beads']} beads, "
              f"global mean <x> = {corr['global_mean']:+.6f}")
        print(f"  energy gap E1-E0 = {corr['plateau']:.4f} +- {corr['plateau_err']:.4f}"
              f"   (slices {corr['fit_range'][0]}-{corr['fit_range'][1]}, {corr['n_fit']} pts)")
        if np.isfinite(corr["plateau_naive_err"]) and corr["plateau_naive_err"] > 0:
            print(f"    error is jackknifed over the whole plateau; treating the slices as "
                  f"independent would give +- {corr['plateau_naive_err']:.4f}, "
                  f"{corr['plateau_err'] / corr['plateau_naive_err']:.1f}x too small")
        if np.isfinite(corr["drift_err"]) and corr["drift_err"] > 0:
            z = corr["drift"] / corr["drift_err"]
            print(f"    plateau drift (2nd half - 1st half): {corr['drift']:+.4f} +- "
                  f"{corr['drift_err']:.4f}  ({z:+.1f} sigma)"
                  f"{'   <-- not flat, choose the window by hand' if abs(z) > 2 else ''}")

    ex = None
    if not args.no_exact:
        ex = exact_lattice(meta)
        print(f"\n  exact reference for THIS lattice (transfer matrix, "
              f"{ex['grid']} grid points):")
        rows = [("E", en["E"], en["E_err"], ex["E"]),
                ("<x^2>", sh["x2"], sh["x2_err"], ex["x2"]),
                ("<x^4>", sh["x4"], sh["x4_err"], ex["x4"]),
                ("kurtosis", sh["kurtosis"], sh["kurtosis_err"], ex["kurtosis"]),
                ("<Rg^2>", sh["Rg2"], sh["Rg2_err"], None),
                ("Cv", en["Cv"], en["Cv_err"], ex["Cv"])]
        if corr is not None:
            rows.append(("gap", corr["plateau"], corr["plateau_err"], ex["gap"]))
        for name, val, err, ref in rows:
            if ref is None:
                continue
            z = (val - ref) / err if err > 0 else float("nan")
            print(f"    {name:9s} exact {ref:.7f}   sampled {val:.7f} +- {err:.7f}   "
                  f"dev {z:+.2f} sigma")
    return en, sh, corr, ex


def transfer_matrix(meta, beta):
    n = int(meta["N"])
    m, w, lam = meta["mass_param"], meta["omega"], meta["lambda"]
    dt = beta / n
    half_width = max(6.0, 10.0 / np.sqrt(2.0 * m * w))
    dx_target = np.sqrt(dt / m) / 15.0
    grid = int(min(max(np.ceil(2 * half_width / dx_target), 400), 2500))
    x = np.linspace(-half_width, half_width, grid)
    dx = x[1] - x[0]
    v = 0.5 * m * w * w * x ** 2 + lam * x ** 4
    t = np.sqrt(m / (2 * np.pi * dt)) * np.exp(
        -m * (x[:, None] - x[None, :]) ** 2 / (2 * dt)
        - 0.5 * dt * (v[:, None] + v[None, :])) * dx
    vals, vecs = np.linalg.eigh(t)
    order = np.argsort(vals)[::-1][:40]
    return x, dx, vals[order], vecs[:, order]


def _thermal(meta, beta):
    n = int(meta["N"])
    m, w, lam = meta["mass_param"], meta["omega"], meta["lambda"]
    x, dx, t, v = transfer_matrix(meta, beta)
    keep = t > 0
    t, v = t[keep], v[:, keep]
    wts = np.exp(n * (np.log(t) - np.log(t[0])))
    wts /= wts.sum()
    prob = v ** 2                                   # each column sums to 1
    x2 = float(wts @ (prob.T @ x ** 2))
    x4 = float(wts @ (prob.T @ x ** 4))
    energy = m * w * w * x2 + 3.0 * lam * x4        # exact lattice virial identity
    density = (prob @ wts) / dx
    gap = float(-np.log(t[1] / t[0]) / (beta / n))
    return dict(x=x, density=density, x2=x2, x4=x4, E=energy, gap=gap, grid=len(x))


def exact_lattice(meta):
    beta = meta["beta"]
    ref = _thermal(meta, beta)
    h = 1e-3 * beta
    ep = _thermal(meta, beta + h)["E"]
    em = _thermal(meta, beta - h)["E"]
    ref["Cv"] = -beta * beta * (ep - em) / (2 * h)
    ref["kurtosis"] = ref["x4"] / ref["x2"] ** 2
    return ref


def plot(meta, data, en, sh, corr, ex, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    hist = position_histogram(meta, data)
    if hist is not None:
        x, p = hist
        g = np.exp(-x ** 2 / (2 * sh["x2"])) / np.sqrt(2 * np.pi * sh["x2"])
        for a, logy in ((ax[0, 0], False), (ax[0, 1], True)):
            a.plot(x, np.maximum(p, 1e-12), lw=1.5, label="PIMC")
            a.plot(x, np.maximum(g, 1e-12), "--", lw=1.2,
                   label=r"gaussian, same $\langle x^2\rangle$")
            if ex is not None:
                a.plot(ex["x"], np.maximum(ex["density"], 1e-12), ":", lw=1.6, color="k",
                       label="exact (transfer matrix)")
            a.set_xlabel("$x$")
            a.set_xlim(x[0], x[-1])
            if logy:
                a.set_yscale("log")
                a.set_ylim(1e-6, None)
            a.legend(fontsize=8)
        ax[0, 0].set_ylabel(r"$\rho(x)$")
        ax[0, 0].set_title(f"position distribution, excess kurtosis "
                           f"{sh['kurtosis'] - 3:+.4f}")
        ax[0, 1].set_title("same, log scale (tails)")

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
                         label=f"plateau {corr['plateau']:.4f} $\\pm$ {corr['plateau_err']:.4f}")
        if ex is not None:
            ax[1, 1].axhline(ex["gap"], color="k", ls=":", label=f"exact {ex['gap']:.4f}")
        ax[1, 1].set_ylim(0, 2 * max(corr["plateau"], meta["omega"]))
        ax[1, 1].set_xlabel(r"$\tau$")
        ax[1, 1].set_ylabel(r"$E_1-E_0$")
        ax[1, 1].set_title("effective mass")
        ax[1, 1].legend(fontsize=8)

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
    ap.add_argument("--no-exact", action="store_true",
                    help="skip the transfer-matrix exact reference")
    args = ap.parse_args()

    meta, data = load(args.stem)
    en, sh, corr, ex = report(meta, data, args)

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
        plot(meta, data, en, sh, corr, ex, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())