import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
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

PREDS_PATH = CFG["preds"]
DATA_PATH  = Path(CFG["data_cordexbench"]).expanduser().resolve() / domain / f"{domain}_domain"

training_experiment = 'ESD_pseudo_reality'
target_var          = 'pr'

if domain == 'ALPS':
    gcm_name     = 'CNRM-CM5'
    spatial_dims = ('x', 'y')
elif domain == 'NZ':
    gcm_name     = 'ACCESS-CM2'
    spatial_dims = ('lat', 'lon')

if training_experiment == 'ESD_pseudo_reality':
    period_training = '1961-1980'
elif training_experiment == 'Emulator_hist_future':
    period_training = '1961-1980_2080-2099'
else:
    raise ValueError('Provide a valid training experiment')

predictand_filename = DATA_PATH / 'train' / training_experiment / 'target' / f'pr_tasmax_{gcm_name}_{period_training}.nc'
FIGURES_DIR = Path(CFG["figures"])

TEST_PERIOD = ("1975", "1981")
N_MAX       = 2
N_SPREAD    = 3

USE_CARTOPY = (spatial_dims == ('x', 'y'))


# =========================
# HELPERS
# =========================
def discover_models(preds_path):
    return sorted(Path(f).stem for f in Path(preds_path).glob("*.nc"))


def is_ensemble(ds):
    return "member" in ds.dims


def load_pred(model, day_ts):
    ds = xr.open_dataset(f"{PREDS_PATH}/{model}.nc")
    needs_scale = float(ds.pr.isel(time=0).max().values) < 1
    ds = ds.sel(time=slice(*TEST_PERIOD))
    if is_ensemble(ds):
        ds = ds.isel(member=0)
    result = ds.pr.sel(time=day_ts, method="nearest").load()
    ds.close()
    if needs_scale:
        result = result * 86400
    # Forzar mismo orden de dims espaciales que obs (y, x)
    result = result.transpose(*spatial_dims[::-1])   # ('y', 'x')
    return result


def load_obs(obs_ds, day_ts):
    da = obs_ds.pr.sel(time=day_ts, method="nearest")
    return da.transpose(*spatial_dims[::-1])   # ('y', 'x')


def get_latlon(da):
    """
    Extrae coordenadas 2D lat/lon del DataArray.
    Para ALPS usa las variables auxiliares lat/lon del NetCDF CORDEX.
    Para NZ construye meshgrid desde las dims lat/lon 1D.
    """
    if 'lat' in da.coords and 'lon' in da.coords:
        lat2d = da['lat'].values
        lon2d = da['lon'].values
        if lat2d.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon2d, lat2d)
        return lat2d, lon2d
    raise ValueError(
        "No se encontraron coordenadas 'lat'/'lon' en el DataArray. "
        f"Coordenadas disponibles: {list(da.coords)}"
    )


def get_domain_extent(ds):
    """
    Calcula [lon_min, lon_max, lat_min, lat_max] directamente
    de las coordenadas lat/lon del dataset, con un pequeño margen.
    """
    if 'lat' in ds.coords and 'lon' in ds.coords:
        lat = ds['lat'].values
        lon = ds['lon'].values
    else:
        raise ValueError(
            f"No se encontraron lat/lon en el dataset. Coords: {list(ds.coords)}"
        )
    margin = 0.5
    return [
        float(lon.min()) - margin,
        float(lon.max()) + margin,
        float(lat.min()) - margin,
        float(lat.max()) + margin,
    ]


# =========================
# SELECCIÓN DE DÍAS
# =========================
def select_days(obs_ds, n_max=N_MAX, n_spread=N_SPREAD):
    pr = obs_ds.pr

    spatial_sum = pr.sum(dim=list(spatial_dims))
    top_idx  = spatial_sum.argsort(axis=0)[-n_max:].values
    top_days = obs_ds.time.values[top_idx]

    wet_values = pr.values[pr.values > 0.1]
    threshold  = np.percentile(wet_values, 75)
    coverage   = (pr > threshold).sum(dim=list(spatial_dims))

    all_times       = obs_ds.time.values
    remaining_mask  = ~np.isin(all_times, top_days)
    coverage_remain = coverage.values[remaining_mask]
    remaining_times = all_times[remaining_mask]

    spread_idx  = np.argsort(coverage_remain)[-n_spread:]
    spread_days = remaining_times[spread_idx]

    selected = np.concatenate([top_days, spread_days])
    labels = (
        [f"MAX #{i+1} | {str(d)[:10]}" for i, d in enumerate(top_days)] +
        [f"Wide #{i+1} | {str(d)[:10]}" for i, d in enumerate(spread_days)]
    )
    return selected, labels


# =========================
# FIGURA POR DÍA
# =========================
def build_day_figure(obs_ds, day_ts, day_label, model_list, domain_extent):
    print(f"  Procesando: {day_label}")

    obs   = load_obs(obs_ds, day_ts)
    preds = {m: load_pred(m, day_ts) for m in model_list}

    items = [("OBS (Ground Truth)", obs)] + list(preds.items())
    ncols = 4
    nrows = int(np.ceil(len(items) / ncols))

    vmin = 0
    all_vals = np.concatenate([
        v.values.flatten() if hasattr(v, 'values') else v.flatten()
        for v in [obs] + list(preds.values())
    ])
    vmax = max(float(np.nanmax(all_vals)) * 1.1, 1.0)

    if USE_CARTOPY:
        lat2d, lon2d = get_latlon(obs)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
        )
        axes = np.array(axes).reshape(-1)

        im = None
        for k, (title, data) in enumerate(items):
            ax = axes[k]
            ax.set_extent(domain_extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS,   linestyle=":", linewidth=0.6)
            ax.add_feature(cfeature.LAND,       facecolor="#1a1a2e", zorder=0)
            data_vals = data.values if hasattr(data, 'values') else data
            im = ax.pcolormesh(
                lon2d, lat2d, data_vals,
                transform=ccrs.PlateCarree(),
                shading="auto",
                cmap="turbo",
                vmin=vmin, vmax=vmax,
            )
            ax.set_title(title, fontsize=9, fontweight="bold")

        for k in range(len(items), len(axes)):
            axes[k].axis("off")

        cbar = fig.colorbar(im, ax=axes.tolist(), orientation="vertical",
                            fraction=0.025, pad=0.02)
        cbar.set_label("Precipitation (mm/day)", fontsize=11, fontweight="bold")

    else:
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            constrained_layout=True,
        )
        axes = np.array(axes).reshape(-1)

        im = None
        for k, (title, data) in enumerate(items):
            ax = axes[k]
            data_vals = data.values if hasattr(data, 'values') else data
            im = ax.imshow(
                data_vals,
                aspect='auto',
                cmap='turbo',
                vmin=vmin, vmax=vmax,
                origin='lower',
            )
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.axis('off')

        for k in range(len(items), len(axes)):
            axes[k].axis("off")

    tipo = "Maximum total precipitation" if "MAX" in day_label else "Wide spatial coverage"
    fig.suptitle(
        f"{tipo}  |  {day_label}  —  Test period {TEST_PERIOD[0]}–{TEST_PERIOD[1]}",
        fontsize=12, fontweight="bold",
    )
    return fig


# =========================
# MAIN
# =========================
def main():
    model_list = discover_models(PREDS_PATH)
    print(f"Modelos encontrados: {model_list}")

    obs_full = xr.open_dataset(predictand_filename).load()
    obs_full = obs_full.sel(time=slice(*TEST_PERIOD))

    domain_extent = get_domain_extent(obs_full)
    print(f"Extent del dominio: {domain_extent}")

    selected_days, labels = select_days(obs_full)
    print("\nDías seleccionados:")
    for lbl in labels:
        print(f"  {lbl}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGuardando PNGs en:", FIGURES_DIR)
    for i, (day_ts, day_label) in enumerate(zip(selected_days, labels)):
        fig = build_day_figure(obs_full, day_ts, day_label, model_list, domain_extent)
        safe_label = day_label.replace(" ", "_").replace("|", "").replace("#", "").strip()
        png_path = FIGURES_DIR / f"map_{i+1:02d}_{safe_label}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {png_path}")

    print("\nTodo listo.")


if __name__ == "__main__":
    main()