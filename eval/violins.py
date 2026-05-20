import json
from collections import defaultdict
from pathlib import Path

import dask
dask.config.set(scheduler='synchronous')

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# =========================
# CONFIG
# =========================
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "training_paths.json"
with open(_CONFIG_PATH) as _f:
    CFG = json.load(_f)

domain = 'ALPS'

PREDS_PATH  = CFG["preds"]
DATA_PATH = Path(CFG["data_cordexbench"]).expanduser().resolve() / domain / f"{domain}_domain"

# --- Experiment configuration ---
training_experiment = 'ESD_pseudo_reality'  # (ESD_pseudo_reality, Emulator_hist_future)
target_var          = 'pr'                  # (tasmax, pr)


if domain == 'ALPS':
    gcm_name    = 'CNRM-CM5'
    spatial_dims = ('x', 'y')
elif domain == 'NZ':
    gcm_name    = 'ACCESS-CM2'
    spatial_dims = ('lat', 'lon')

if training_experiment == 'ESD_pseudo_reality':
    period_training = '1961-1980'
elif training_experiment == 'Emulator_hist_future':
    period_training = '1961-1980_2080-2099'
else:
    raise ValueError('Provide a valid training experiment: ESD_pseudo_reality or Emulator_hist_future')

# --- Load data ---
predictor_filename  = DATA_PATH / 'train' / training_experiment / 'predictors' / f'{gcm_name}_{period_training}.nc'
predictand_filename = DATA_PATH / 'train' / training_experiment / 'target' / f'pr_tasmax_{gcm_name}_{period_training}.nc'
FIGURES_DIR = Path(CFG["figures"])

TEST_PERIOD  = ("1975", "1981")


# =========================
# HELPERS
# =========================

# Canonical loss-function order within each architecture
_LOSS_ORDER = ["MSE", "ASYM", "BerGamma", "CRPS", "CRPS_spectral"]
# Architecture order: deep first, then vit
_ARCH_ORDER = ["deepesd", "vit"]

# Base color per loss function
_LOSS_COLOR = {
    "MSE":           "#1f77b4",  # blue
    "ASYM":          "#d62728",  # red
    "BerGamma":      "#9467bd",  # purple
    "CRPS":          "#2ca02c",  # green
    "CRPS_spectral": "#7f7f7f",  # gray
}

# Lightness blend range [lo, hi] per architecture.
# Factor=1 → original color, factor=0 → white.
# deepesd models are lighter; vit models are closer to the base color.
_ARCH_L_RANGE = {
    "deepesd": (0.30, 0.50),
    "vit":     (0.82, 0.97),
}


def _lighten(hex_color, factor):
    """Blend *hex_color* toward white. factor=1 → original, factor=0 → white."""
    r, g, b = mc.to_rgb(hex_color)
    return (1 - (1 - r) * factor, 1 - (1 - g) * factor, 1 - (1 - b) * factor)


def _parse_model(stem):
    """Return (arch, loss, suffix) for a model filename stem."""
    for arch in _ARCH_ORDER:
        if stem.startswith(arch + "_"):
            rest = stem[len(arch) + 1:]
            # check longer names first so CRPS_spectral isn't caught by CRPS
            for loss in sorted(_LOSS_ORDER, key=len, reverse=True):
                if rest == loss or rest.startswith(loss + "_"):
                    suffix = rest[len(loss):].lstrip("_")
                    return arch, loss, suffix
    return "unknown", "unknown", stem


def _sort_key(stem):
    arch, loss, suffix = _parse_model(stem)
    ai = _ARCH_ORDER.index(arch) if arch in _ARCH_ORDER else 99
    li = _LOSS_ORDER.index(loss) if loss in _LOSS_ORDER else 99
    return (ai, li, suffix)


def discover_models(preds_path):
    stems = [Path(f).stem for f in Path(preds_path).glob("*.nc")]
    return sorted(stems, key=_sort_key)


def make_color_map(model_list):
    """Assign colors: same hue per loss function, lighter for deepesd / darker for vit.
    Multiple variants of the same arch+loss are spread across the architecture's lightness range."""
    groups = defaultdict(list)
    for stem in model_list:
        arch, loss, _ = _parse_model(stem)
        groups[(arch, loss)].append(stem)

    color_map = {}
    for (arch, loss), members in groups.items():
        base_hex = _LOSS_COLOR.get(loss, "#555555")
        lo, hi = _ARCH_L_RANGE.get(arch, (0.6, 0.8))
        n = len(members)
        for i, stem in enumerate(members):
            factor = (lo + hi) / 2 if n == 1 else lo + i * (hi - lo) / (n - 1)
            color_map[stem] = _lighten(base_hex, factor)

    return color_map


def _needs_scale(pr_da):
    return float(pr_da.isel(time=0).max().values) < 1



def open_obs():
    try:
        ds = xr.open_dataset(predictand_filename, chunks={"time": 365})
    except (ValueError, ImportError):
        ds = xr.open_dataset(predictand_filename)
    pr = ds.pr
    if _needs_scale(pr):
        pr = pr * 86400
    return pr.sel(time=slice(*TEST_PERIOD))


def load_pr_spatial(model):
    try:
        ds = xr.open_dataset(f"{PREDS_PATH}/{model}.nc", chunks={"time": 365})
    except (ValueError, ImportError):
        ds = xr.open_dataset(f"{PREDS_PATH}/{model}.nc")
    pr = ds.pr
    if _needs_scale(pr):
        pr = pr * 86400
    pr = pr.sel(time=slice(*TEST_PERIOD))
    if "member" in ds.dims:
        pr = pr.isel(member=0)
    result = pr.load()
    ds.close()
    return result


# =========================
# FIGURA: Violin plots (todas las métricas)
# =========================
def build_violins(obs_pr, model_list, color_map):
    print("  Procesando: violin plots (todas las métricas)...")

    threshold   = 1.0
    obs_mean    = obs_pr.mean("time", skipna=True)
    obs_q99     = obs_pr.quantile(0.99, "time", skipna=True)
    obs_interan = obs_pr.groupby("time.year").mean("time").std("year")
    obs_wet_frac = (obs_pr >= threshold).mean("time")
    obs_sdii    = obs_pr.where(obs_pr >= threshold).mean("time", skipna=True)
    obs_rx1     = obs_pr.groupby("time.year").max("time").mean("year")

    metric_keys = [
        "Relative Bias (%)",
        "RMSE (mm/day)",
        "Interann. Var. Ratio",
        "P99 bias (mm/day)",
        "R01day bias (%)",
        "Rx1day bias (%)",
        "SDII bias (%)",
    ]
    metric_data = {k: [] for k in metric_keys}
    labels = []

    for model in model_list:
        pr = load_pr_spatial(model)
        labels.append(model)

        rb = ((pr.mean("time") - obs_mean) / obs_mean * 100).values.flatten()
        metric_data["Relative Bias (%)"].append(rb[np.isfinite(rb)])

        rmse = np.sqrt(((pr - obs_pr) ** 2).mean("time", skipna=True)).values.flatten()
        metric_data["RMSE (mm/day)"].append(rmse[np.isfinite(rmse)])

        ianv = (pr.groupby("time.year").mean("time").std("year") / obs_interan).values.flatten()
        metric_data["Interann. Var. Ratio"].append(ianv[np.isfinite(ianv)])

        p99b = (pr.quantile(0.99, "time") - obs_q99).values.flatten()
        metric_data["P99 bias (mm/day)"].append(p99b[np.isfinite(p99b)])

        r01 = ((pr >= threshold).mean("time") - obs_wet_frac) / obs_wet_frac * 100
        metric_data["R01day bias (%)"].append(r01.values.flatten()[np.isfinite(r01.values.flatten())])

        rx1 = ((pr.groupby("time.year").max("time").mean("year") - obs_rx1) / obs_rx1 * 100).values.flatten()
        metric_data["Rx1day bias (%)"].append(rx1[np.isfinite(rx1)])

        sdii = ((pr.where(pr >= threshold).mean("time", skipna=True) - obs_sdii) / obs_sdii * 100).values.flatten()
        metric_data["SDII bias (%)"].append(sdii[np.isfinite(sdii)])

        del pr

    colors_list = [color_map[m] for m in model_list]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    # Índices de modelos que NO son BerGamma (para calcular ylim)
    non_bergamma_idx = [j for j, m in enumerate(labels)
                        if _parse_model(m)[1] != "BerGamma"]

    for i, key in enumerate(metric_keys):
        ax = axes[i]
        parts = ax.violinplot(metric_data[key], positions=range(len(labels)),
                              showmedians=True, showextrema=True)
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors_list[j])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("navy")
        parts["cmedians"].set_linewidth(2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(key, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        ref = 1.0 if "Ratio" in key else 0.0
        ax.axhline(ref, color="black", linestyle="--", linewidth=1, alpha=0.7)

        # Ylim basado solo en modelos no-BerGamma para que los violines no se aplanen
        if non_bergamma_idx:
            vals = np.concatenate([metric_data[key][j] for j in non_bergamma_idx])
            if len(vals):
                lo, hi = np.nanpercentile(vals, [1, 99])
                margin = (hi - lo) * 0.15 or 0.5
                ax.set_ylim(lo - margin, hi + margin)

    axes[-1].set_visible(False)
    fig.suptitle(
        f"Model skill — Test period {TEST_PERIOD[0]}–{TEST_PERIOD[1]}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    return fig


# =========================
# MAIN
# =========================
def main():
    model_list = discover_models(PREDS_PATH)
    color_map  = make_color_map(model_list)

    print(f"Modelos encontrados: {model_list}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    obs_pr = open_obs()

    png_path = FIGURES_DIR / f"Evaluation_violins_{TEST_PERIOD[0]}-{TEST_PERIOD[1]}.png"
    print(f"\nGuardando: {png_path}")
    fig = build_violins(obs_pr, model_list, color_map)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Listo.")


if __name__ == "__main__":
    main()
