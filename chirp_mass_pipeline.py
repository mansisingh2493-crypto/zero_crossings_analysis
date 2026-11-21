
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.constants import G, c, pi

from pycbc.psd.analytical import aLIGOAPlusDesignSensitivityT1800042
from pycbc.noise import noise_from_psd
from pycbc.waveform import get_td_waveform
from pycbc.filter.matchedfilter import sigma
from pycbc.types import TimeSeries as PycbcTimeSeries
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

# =============================== CONFIG ===================================
PARAM_FILE = sys.argv[1] if len(sys.argv) > 1 else "zca_results.xlsx"
OUT_DIR = "results"

SAMPLE_RATE = 4096          # Hz
DURATION = 128              # s
FLOW = 15.0                 # Hz
APPROXIMANT = "IMRPhenomXPHM"

TARGET_SNR = 50.0           # rescale distance to this optimal SNR; None = use table distance
N_SEEDS = 50                # noise realisations per parameter set
SEED_OFFSET = 1000          # seeds = SEED_OFFSET + 0..N_SEEDS-1
N_WORKERS = os.cpu_count()

# signal conditioning (as in the notebook)
WHITEN_FFTLEN, WHITEN_OVERLAP = 2, 1
BAND = (35.0, 350.0)        # Hz
CROP = 2.0                  # s removed at each edge after whitening

# automatic window selection
PEAK_SEARCH = 0.05          # s, look for envelope peak within |t| < this (injection peak is at t=0)
ENV_SMOOTH = 0.005          # s, moving-average length for the envelope
ENV_K = 2.0                 # window starts where envelope > ENV_K * median off-source envelope
OFF_SOURCE = (-20.0, -5.0)  # s, signal-free stretch used for the noise level
MAX_LOOKBACK = 2.0          # s, never start earlier than this before the peak
END_PAD = 0.0               # s, end window this long after the peak
F_MAX_MULT = None           # e.g. 1.5 -> drop points above 1.5*f_ISCO (None = keep up to the peak)

# frequency-track cleaning and fit
MAX_JUMP = 2.0              # max allowed ratio between consecutive frequencies
MIN_POINTS = 4              # minimum points needed for a fit
WEIGHTED_FIT = True         # weight by sigma_y ~ f^(-5/3) (propagated timing error)

M_SUN = 1.98847e30
# ==========================================================================


# ----------------------------- physics helpers ----------------------------
def chirp_mass(m1, m2):
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def f_isco(m_total):
    """GW frequency (Hz) at the Schwarzschild ISCO, m_total in solar masses."""
    return c ** 3 / (6 ** 1.5 * pi * G * m_total * M_SUN)


def chi_eff(p):
    return (p["mass1"] * p["spin1z"] + p["mass2"] * p["spin2z"]) / (p["mass1"] + p["mass2"])


def chi_p(p):
    m1, m2 = p["mass1"], p["mass2"]
    s1p = np.hypot(p["spin1x"], p["spin1y"])
    s2p = np.hypot(p["spin2x"], p["spin2y"])
    if m2 > m1:
        m1, m2, s1p, s2p = m2, m1, s2p, s1p
    q = m2 / m1
    return max(s1p, (4 * q + 3) / (4 + 3 * q) * q * s2p)


# ------------------------------ data generation ---------------------------
_PSD = None


def get_psd():
    global _PSD
    if _PSD is None:
        n = int(DURATION * SAMPLE_RATE)
        _PSD = aLIGOAPlusDesignSensitivityT1800042(n // 2 + 1, 1.0 / DURATION, FLOW)
    return _PSD


def bbh(p):
    """Same as the notebook: waveform padded so the peak sits at t = 0."""
    fs = SAMPLE_RATE
    N = fs * DURATION
    hp, hc = get_td_waveform(
        approximant=APPROXIMANT,
        mass1=p["mass1"], mass2=p["mass2"],
        spin1x=p["spin1x"], spin1y=p["spin1y"], spin1z=p["spin1z"],
        spin2x=p["spin2x"], spin2y=p["spin2y"], spin2z=p["spin2z"],
        distance=p["distance"], delta_t=1 / fs, f_lower=FLOW,
    )
    # low-mass signals can be longer than the DURATION/2 s available before the
    # peak: keep only the last (DURATION/2 - 4) s and taper the cut edge
    if float(hp.sample_times[0]) < -(DURATION / 2 - 4):
        hp = hp.time_slice(-(DURATION / 2 - 4), float(hp.sample_times[-1]) + hp.delta_t)
        n_tap = int(1.0 * fs)                     # 1 s half-Hann taper
        hp.data[:n_tap] *= 0.5 * (1 - np.cos(np.pi * np.arange(n_tap) / n_tap))
    num_prepend = int(N / 2) - int(abs(hp.sample_times[0] * fs))
    num_append = int(N / 2) - int(abs(hp.sample_times[-1] * fs))
    hp.prepend_zeros(num_prepend)
    hp.append_zeros(num_append)
    return hp


def prepare_signal(p):
    """Waveform rescaled to TARGET_SNR. Returns (h, snr, distance)."""
    psd = get_psd()
    h = bbh(p)
    snr = float(sigma(h, psd=psd, low_frequency_cutoff=FLOW))
    dist = p["distance"]
    if TARGET_SNR:
        scale = TARGET_SNR / snr
        h = h * scale
        dist = dist / scale
        snr = TARGET_SNR
    return h, snr, dist


def make_data(h, seed):
    psd = get_psd()
    noise = noise_from_psd(len(h), h.delta_t, psd, seed=seed)
    noise.start_time = h.start_time
    d = h + noise
    return d.time_slice(-60, 60)


def condition(d):
    """Whiten + bandpass + crop, exactly as in the notebook."""
    s = GwpyTimeSeries(np.asarray(d), dt=float(d.delta_t), t0=float(d.start_time))
    t0 = s.t0.value
    w = s.whiten(fftlength=WHITEN_FFTLEN, overlap=WHITEN_OVERLAP).bandpass(*BAND)
    w = w.crop(start=t0 + CROP, end=t0 + s.duration.value - CROP)
    return w.times.value, w.value


# ------------------------------ analysis steps ----------------------------
def auto_window(t, x):
    """Start: last point before the peak where the smoothed envelope is below
    ENV_K x noise level. End: envelope peak (+ END_PAD)."""
    dt = t[1] - t[0]
    env = np.abs(hilbert(x))
    k = max(1, int(ENV_SMOOTH / dt))
    env_s = np.convolve(env, np.ones(k) / k, mode="same")

    off = (t > OFF_SOURCE[0]) & (t < OFF_SOURCE[1])
    thr = ENV_K * np.median(env_s[off])

    near = np.where(np.abs(t) < PEAK_SEARCH)[0]
    i_pk = near[np.argmax(env_s[near])]
    t_peak = t[i_pk]

    i_min = np.searchsorted(t, t_peak - MAX_LOOKBACK)
    below = np.where(env_s[i_min:i_pk] < thr)[0]
    i_start = i_min + below[-1] if below.size else i_min
    return t[i_start], t_peak + END_PAD, t_peak, thr, env_s


def zero_crossings(t, h):
    """Linear interpolation between samples (as in the notebook)."""
    idx = np.where(np.diff(np.sign(h)) != 0)[0]
    t1, t2, h1, h2 = t[idx], t[idx + 1], h[idx], h[idx + 1]
    return t1 - h1 * (t2 - t1) / (h2 - h1)


def freq_track(zc):
    """Full-period estimate f = 1/(z[i+2]-z[i]) at z[i+1]; cancels the
    half-cycle asymmetry of the half-period estimate."""
    if len(zc) < 3:
        return np.array([]), np.array([])
    return zc[1:-1], 1.0 / (zc[2:] - zc[:-2])


def clean_track(tf, f, m_total):
    """Keep in-band points; walking back from the peak, keep only a strictly
    increasing track without jumps larger than MAX_JUMP."""
    f_hi = BAND[1]
    if F_MAX_MULT is not None:
        f_hi = min(f_hi, F_MAX_MULT * f_isco(m_total))
    m = (f >= BAND[0]) & (f <= f_hi)
    tf, f = tf[m], f[m]
    if len(f) == 0:
        return tf, f
    keep = [len(f) - 1]
    for i in range(len(f) - 2, -1, -1):
        f_next = f[keep[-1]]
        if f[i] < f_next and f_next / f[i] < MAX_JUMP:
            keep.append(i)
    keep = keep[::-1]
    return tf[keep], f[keep]


def fit_chirp(tf, f):
    """Linear fit of f^(-8/3) vs t -> chirp mass (+ 1-sigma from fit covariance)."""
    y = f ** (-8 / 3)
    w = f ** (5 / 3) if WEIGHTED_FIT else None
    (slope, icpt), cov = np.polyfit(tf, y, 1, w=w, cov=True)
    if slope >= 0:
        raise ValueError("non-negative slope")
    A, dA = abs(slope), np.sqrt(cov[0, 0])
    mc = (c ** 3 / G) * ((5 / 256) * A / pi ** (8 / 3)) ** (3 / 5) / M_SUN
    dmc = 0.6 * mc * dA / A
    tc = -icpt / slope
    return mc, dmc, slope, icpt, tc


def clean_reference(h, t_start, m_total):
    """Same zero-crossing fit on the noise-free injected waveform, over the same
    window as the noisy runs (median start -> peak). Measures the bias of the
    method itself (Newtonian approximation near merger), without noise."""
    try:
        t, x = h.sample_times.numpy(), np.asarray(h)
        t_peak = t[np.argmax(np.abs(x))]
        sel = (t >= t_start) & (t <= t_peak)
        tf, f = clean_track(*freq_track(zero_crossings(t[sel], x[sel])), m_total)
        return fit_chirp(tf, f)[0] if len(f) >= MIN_POINTS else np.nan
    except Exception:
        return np.nan


def run_single(job):
    """One noise realisation. job = (case, p, h_array, delta_t, epoch, seed, keep_diag)."""
    case, p, h_arr, dt, epoch, seed, keep_diag = job
    h = PycbcTimeSeries(h_arr, delta_t=dt, epoch=epoch)
    m_total = p["mass1"] + p["mass2"]
    out = dict(case=case, seed=seed, Mc_est=np.nan, Mc_fit_err=np.nan, tc_est=np.nan,
               n_points=0, t_start=np.nan, t_end=np.nan, ok=False, reason="")
    diag = None
    try:
        t, x = condition(make_data(h, seed))
        t_start, t_end, t_peak, thr, env_s = auto_window(t, x)
        win = (t >= t_start) & (t <= t_end)
        zc = zero_crossings(t[win], x[win])
        tf_raw, f_raw = freq_track(zc)
        tf, f = clean_track(tf_raw, f_raw, m_total)
        out.update(t_start=t_start, t_end=t_end, n_points=len(f))
        if keep_diag:
            sel = (t > t_start - 0.1) & (t < t_end + 0.05)
            diag = dict(t=t[sel], x=x[sel], env=env_s[sel], thr=thr, t_start=t_start,
                        t_end=t_end, zc=zc, tf_raw=tf_raw, f_raw=f_raw, tf=tf, f=f)
        if len(f) < MIN_POINTS:
            out["reason"] = f"only {len(f)} points"
            return out, diag
        mc, dmc, slope, icpt, tc = fit_chirp(tf, f)
        out.update(Mc_est=mc, Mc_fit_err=dmc, tc_est=tc, ok=True)
        if diag is not None:
            diag.update(slope=slope, icpt=icpt, mc=mc, dmc=dmc)
    except Exception as e:  # keep the batch running
        out["reason"] = f"{type(e).__name__}: {e}"
    return out, diag


# ---------------------------------- plots ---------------------------------
def plot_diagnostic(diag, row, path):
    fig, ax = plt.subplots(3, 1, figsize=(10, 11))
    a = ax[0]
    a.plot(diag["t"], diag["x"], lw=0.8, label="whitened strain")
    a.plot(diag["t"], diag["env"], lw=1, color="gray", label="smoothed envelope")
    a.axhline(diag["thr"], ls=":", color="gray", label="window threshold")
    a.axvspan(diag["t_start"], diag["t_end"], color="C2", alpha=0.15, label="auto window")
    for z in diag["zc"]:
        a.axvline(z, color="C1", alpha=0.3, lw=0.6)
    a.set_xlabel("t - t_peak [s]")
    a.set_title(f"{row['case']}: auto-selected cycles (seed {SEED_OFFSET})")
    a.legend(fontsize=8, loc="upper left")

    a = ax[1]
    a.plot(diag["tf_raw"], diag["f_raw"], "x", color="gray", label="all crossings")
    a.plot(diag["tf"], diag["f"], "o", color="C1", label="kept")
    if "mc" in diag:
        tt = np.linspace(diag["tf"].min(), diag["tf"].max(), 200)
        yy = diag["slope"] * tt + diag["icpt"]
        a.plot(tt[yy > 0], yy[yy > 0] ** (-3 / 8), "--", color="navy", label="fit")
    a.set_xlabel("t [s]")
    a.set_ylabel("f [Hz]")
    a.legend(fontsize=8)

    a = ax[2]
    if len(diag["f"]):
        a.plot(diag["tf"], diag["f"] ** (-8 / 3), "o", color="C1", label="data")
    if "mc" in diag:
        a.plot(tt, yy, "--", color="navy",
               label=rf"fit: $M_c$={diag['mc']:.2f}$\pm${diag['dmc']:.2f} (true {row['Mc_true']:.2f})")
    a.set_xlabel("t [s]")
    a.set_ylabel(r"$f^{-8/3}$")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_hist(vals, row, path):
    fig, a = plt.subplots(figsize=(6, 4))
    a.hist(vals, bins=20, color="C0", alpha=0.7)
    a.axvline(row["Mc_true"], color="k", lw=2, label="true")
    a.axvline(np.mean(vals), color="C3", ls="--", label="mean")
    a.set_xlabel(r"$M_c$ estimate [$M_\odot$]")
    a.set_ylabel("count")
    a.set_title(f"{row['case']} ({len(vals)} noise realisations)")
    a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_summary(summ, runs, out):
    ok = summ.dropna(subset=["Mc_mean"])

    fig, a = plt.subplots(figsize=(6.5, 6))
    a.errorbar(ok["Mc_true"], ok["Mc_mean"], yerr=ok["Mc_std"], fmt="o", capsize=4)
    lim = [0, 1.1 * max(ok["Mc_true"].max(), (ok["Mc_mean"] + ok["Mc_std"]).max())]
    a.plot(lim, lim, "k--", lw=1, label="perfect recovery")
    for _, r in ok.iterrows():
        a.annotate(r["case"], (r["Mc_true"], r["Mc_mean"]), fontsize=7,
                   xytext=(4, 4), textcoords="offset points")
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel(r"true $M_c$ [$M_\odot$]")
    a.set_ylabel(r"estimated $M_c$ (mean $\pm$ std) [$M_\odot$]")
    a.set_title(f"Recovered vs injected chirp mass (SNR={TARGET_SNR})")
    a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "summary_recovered_vs_true.png"), dpi=200); plt.close(fig)

    fig, a = plt.subplots(figsize=(max(6, 0.6 * len(ok)), 4.5))
    a.errorbar(range(len(ok)), ok["bias_pct"], yerr=ok["scatter_pct"], fmt="o", capsize=4)
    a.axhline(0, color="k", lw=1)
    a.set_xticks(range(len(ok))); a.set_xticklabels(ok["case"], rotation=45, ha="right")
    a.set_ylabel(r"relative error (mean $\pm$ std) [%]")
    a.set_title("Chirp-mass bias and scatter per case")
    fig.tight_layout(); fig.savefig(os.path.join(out, "summary_bias_per_case.png"), dpi=200); plt.close(fig)

    good = runs[runs["ok"]]
    if len(good):
        cases = list(ok["case"])
        data = [good.loc[good.case == cs, "rel_err_pct"].values for cs in cases]
        fig, a = plt.subplots(figsize=(max(6, 0.6 * len(cases)), 4.5))
        a.boxplot(data)
        a.set_xticks(range(1, len(cases) + 1)); a.set_xticklabels(cases)
        a.axhline(0, color="k", lw=1)
        a.set_ylabel("relative error [%]")
        a.set_title("Distribution over noise realisations")
        plt.setp(a.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout(); fig.savefig(os.path.join(out, "summary_boxplot.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for a, col, lab in zip(ax, ["M_total", "q", "chi_eff", "cycles_mean"],
                           [r"$M_{tot}$ [$M_\odot$]", "q = m2/m1", r"$\chi_{eff}$", "mean # cycles used"]):
        sc = a.errorbar(ok[col], ok["bias_pct"], yerr=ok["scatter_pct"], fmt="o", capsize=3)
        a.axhline(0, color="k", lw=1)
        a.set_xlabel(lab)
    ax[0].set_ylabel("relative error [%]")
    fig.suptitle("Bias trends")
    fig.tight_layout(); fig.savefig(os.path.join(out, "summary_trends.png"), dpi=200); plt.close(fig)


# ---------------------------------- main ----------------------------------
PARAM_KEYS = ["mass1", "mass2", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z"]
# column names used in zca_results.xlsx -> pipeline names
XLSX_MAP = {"m1": "mass1", "m2": "mass2", "sz1": "spin1z", "sy1": "spin1y", "sx1": "spin1x",
            "sz2": "spin2z", "sy2": "spin2y", "sx2": "spin2x",
            "chirp mass obtained": "manual_Mc", "% error in chirp mass": "manual_err_pct",
            "no. of cycles used for zca": "manual_cycles"}


def load_params(path):
    """Reads either a plain table (mass1, mass2, spin..., distance) or the
    zca_results.xlsx layout (group label in col 0, masses forward-filled
    within each group, blank spins = 0). Returns one row per table row."""
    if path.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(path, sheet_name=None, header=None)
        raw = max(sheets.values(), key=len)          # use the fullest sheet
        header = [str(h).strip().lower() if pd.notna(h) else "group" for h in raw.iloc[0]]
        df = raw.iloc[1:].copy()
        df.columns = header
    else:
        df = pd.read_csv(path)
        df.columns = [str(col).strip().lower() for col in df.columns]
    df = df.rename(columns=XLSX_MAP)
    if "group" not in df:
        df["group"] = "all"
    df["group"] = df["group"].ffill().fillna("all")
    for col in PARAM_KEYS:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # spin-only rows inherit the masses of the row above (same group)
    df[["mass1", "mass2"]] = df.groupby("group")[["mass1", "mass2"]].ffill()
    df = df.dropna(subset=["mass1", "mass2"])
    df[PARAM_KEYS[2:]] = df[PARAM_KEYS[2:]].fillna(0.0)
    df["distance"] = 1000.0    # irrelevant when TARGET_SNR is set (distance is rescaled)
    for col in ["manual_Mc", "manual_err_pct", "manual_cycles"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    return df.reset_index(drop=True)


def plot_groups(table, out):
    """One panel per group: bias +- scatter vs the parameter that varies in it,
    with the original manual single-seed errors for comparison."""
    groups = list(dict.fromkeys(table["group"]))
    cand = [("M_total", r"$M_{tot}$ [$M_\odot$]"), ("q", "q = m2/m1"),
            ("chi_eff", r"$\chi_{eff}$"), ("chi_p", r"$\chi_p$")]
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.3), squeeze=False)
    for a, g in zip(axes[0], groups):
        d = table[(table.group == g) & table["bias_pct"].notna()]
        if d.empty:
            a.set_title(f"{g}: no results"); continue
        nun = {col: d[col].round(4).nunique() for col, _ in cand}
        col, lab = max(cand, key=lambda cl: nun[cl[0]])
        if nun[col] < len(d):       # mixed variation -> plot against case label
            x, lab = np.arange(len(d)), "case"
            a.set_xticks(x); a.set_xticklabels(d["case"], rotation=45, ha="right", fontsize=8)
        else:
            x = d[col].values
        a.errorbar(x, d["bias_pct"], yerr=d["scatter_pct"], fmt="o", capsize=4,
                   label=f"automated ({N_SEEDS} seeds, mean $\\pm$ std)")
        a.plot(x, d["clean_err_pct"], "^", color="C2", label="noise-free, same window")
        if d["manual_err_pct"].notna().any():
            a.plot(x, d["manual_err_pct"], "s", mfc="none", color="C3", label="manual (1 seed, |err|)")
        a.axhline(0, color="k", lw=1)
        a.set_xlabel(lab); a.set_title(g)
    axes[0][0].set_ylabel("chirp-mass error [%]")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "summary_by_group.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "per_case"), exist_ok=True)
    table = load_params(PARAM_FILE)
    uniq = table.drop_duplicates(subset=PARAM_KEYS)[PARAM_KEYS + ["distance"]].reset_index(drop=True)
    print(f"{len(table)} table rows -> {len(uniq)} unique parameter sets x {N_SEEDS} seeds, "
          f"target SNR = {TARGET_SNR}")

    all_runs, summary = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, row in uniq.iterrows():
            p = {k: float(row[k]) for k in PARAM_KEYS + ["distance"]}
            case = f"C{i + 1}"
            m_total = p["mass1"] + p["mass2"]
            info = dict(case=case, **{k: p[k] for k in PARAM_KEYS},
                        M_total=m_total, q=min(p["mass1"], p["mass2"]) / max(p["mass1"], p["mass2"]),
                        chi_eff=chi_eff(p), chi_p=chi_p(p), f_ISCO=f_isco(m_total),
                        Mc_true=chirp_mass(p["mass1"], p["mass2"]))
            try:
                h, snr, dist = prepare_signal(p)
            except Exception as e:
                print(f"  {case}: waveform failed ({e})")
                summary.append({**info, "status": f"waveform failed: {e}"})
                continue
            info.update(SNR=snr, distance_Mpc=dist)

            jobs = [(case, p, np.asarray(h), float(h.delta_t), float(h.start_time),
                     SEED_OFFSET + s, s == 0) for s in range(N_SEEDS)]
            results = list(ex.map(run_single, jobs))
            runs = pd.DataFrame([r for r, _ in results])
            runs["Mc_true"] = info["Mc_true"]
            runs["rel_err_pct"] = 100 * (runs["Mc_est"] - info["Mc_true"]) / info["Mc_true"]
            all_runs.append(runs)

            good = runs[runs.ok]
            n_ok = len(good)
            clean_mc = clean_reference(h, good["t_start"].median() if n_ok else -0.1, m_total)
            mc = good["Mc_est"].values
            s = dict(info,
                     N_ok=f"{n_ok}/{N_SEEDS}",
                     cycles_mean=((good["n_points"] + 1) / 2).mean() if n_ok else np.nan,
                     Mc_mean=mc.mean() if n_ok else np.nan,
                     Mc_std=mc.std(ddof=1) if n_ok > 1 else np.nan,
                     Mc_median=np.median(mc) if n_ok else np.nan,
                     Mc_fit_err_mean=good["Mc_fit_err"].mean() if n_ok else np.nan,
                     bias_pct=good["rel_err_pct"].mean() if n_ok else np.nan,
                     scatter_pct=good["rel_err_pct"].std(ddof=1) if n_ok > 1 else np.nan,
                     abs_err_pct=good["rel_err_pct"].abs().mean() if n_ok else np.nan,
                     Mc_clean=clean_mc,
                     clean_err_pct=100 * (clean_mc - info["Mc_true"]) / info["Mc_true"],
                     window_ms_mean=1e3 * (good["t_end"] - good["t_start"]).mean() if n_ok else np.nan,
                     status="ok" if n_ok else "not enough cycles (" + runs["reason"].iloc[0] + ")")
            summary.append(s)
            print(f"  {case} ({p['mass1']:g}+{p['mass2']:g}, s1z={p['spin1z']:g}, s2z={p['spin2z']:g}): "
                  f"Mc_true={info['Mc_true']:.2f} est={s['Mc_mean']:.2f} +- {s['Mc_std']:.2f} "
                  f"bias={s['bias_pct']:.1f}% ok={s['N_ok']}", flush=True)

            diag = results[0][1]
            if diag is not None:
                plot_diagnostic(diag, s, os.path.join(OUT_DIR, "per_case", f"{case}_diagnostic.png"))
            if n_ok > 1:
                plot_hist(mc, s, os.path.join(OUT_DIR, "per_case", f"{case}_hist.png"))

    summ = pd.DataFrame(summary)
    runs = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
    runs.to_csv(os.path.join(OUT_DIR, "all_runs.csv"), index=False)

    # map results back onto every row of the input table (keeps groups and order)
    full = table.merge(summ, on=PARAM_KEYS, how="left")
    full.to_csv(os.path.join(OUT_DIR, "results_table_full.csv"), index=False)

    # table in the same layout as zca_results.xlsx (+ extra columns)
    out = pd.DataFrame({
        "": full["group"].where(full["group"] != full["group"].shift()),
        "case": full["case"],
        "m1": full["mass1"], "m2": full["mass2"],
        "sz1": full["spin1z"], "sy1": full["spin1y"], "sx1": full["spin1x"],
        "sz2": full["spin2z"], "sy2": full["spin2y"], "sx2": full["spin2x"],
        "Distance": full["distance_Mpc"], "SNR": full["SNR"],
        "No. of cycles used for zca (mean)": full["cycles_mean"],
        "chirp mass obtained (mean)": full["Mc_mean"],
        "chirp mass std": full["Mc_std"],
        "chir mass theoretical": full["Mc_true"],
        "% error in chirp mass (mean |err|)": full["abs_err_pct"],
        "% bias (signed mean)": full["bias_pct"],
        "% scatter (std)": full["scatter_pct"],
        "successful runs": full["N_ok"],
        "noise-free % error (same window)": full["clean_err_pct"],
        "manual chirp mass (old)": full["manual_Mc"],
        "manual % error (old)": full["manual_err_pct"],
        "M_total": full["M_total"], "q": full["q"], "chi_eff": full["chi_eff"], "chi_p": full["chi_p"],
        "f_ISCO [Hz]": full["f_ISCO"], "window [ms]": full["window_ms_mean"],
        "status": full["status"],
    })
    out.round(3).to_excel(os.path.join(OUT_DIR, "results_table.xlsx"), index=False)
    out.round(3).to_csv(os.path.join(OUT_DIR, "results_table.csv"), index=False)
    show = ["", "m1", "m2", "sz1", "sz2", "sx1", "sx2", "Distance", "chir mass theoretical",
            "chirp mass obtained (mean)", "chirp mass std", "% bias (signed mean)", "% scatter (std)",
            "noise-free % error (same window)", "manual % error (old)", "successful runs"]
    try:
        md = out[show].fillna("").to_markdown(index=False, floatfmt=".2f")
    except ImportError:
        md = out[show].round(2).to_string(index=False)
    with open(os.path.join(OUT_DIR, "results_table.md"), "w") as fh:
        fh.write(md + "\n")
    print("\n" + md)

    if len(runs) and summ["Mc_mean"].notna().any():
        plot_summary(summ, runs, OUT_DIR)
        plot_groups(full, OUT_DIR)
    print(f"\nAll outputs in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
