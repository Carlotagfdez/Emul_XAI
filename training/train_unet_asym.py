import json
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_cfg_path = (
    Path(os.environ["TRAINING_CONFIG"]).expanduser().resolve()
    if os.environ.get("TRAINING_CONFIG")
    else _repo_root / "config" / "training_paths.json"
)
with _cfg_path.open(encoding="utf-8") as f:
    _cfg = json.load(f)

_d4 = _cfg["deep4downscaling_repo"]
if _d4 not in sys.path:
    sys.path.insert(0, _d4)

MODELS_PATH = Path(_cfg["models"]).expanduser().resolve()
PREDS_PATH = Path(_cfg["preds"]).expanduser().resolve()
EXPERIMENTS_PATH = Path(_cfg["experiments"]).expanduser().resolve()
ASYM_PATH = Path(_cfg["asym_parameters"]).expanduser().resolve()
predictor_filename = Path(_cfg["predictor_nc"]).expanduser().resolve()
predictand_filename = Path(_cfg["predictand_nc"]).expanduser().resolve()

# Import libraries
import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import deep4downscaling.trans
import deep4downscaling.deep.loss
import deep4downscaling.deep.utils
import deep4downscaling.deep.models
import deep4downscaling.deep.train
import deep4downscaling.deep.tracker
import deep4downscaling.deep.pred

# Set device
device = 'cuda'

# Load predictors
predictor = xr.open_dataset(predictor_filename)
predictor = predictor.load()

# Subset predictors
predictor = predictor.sel(lon=slice(-24, 22.5))

# Extend latitude to 32 grid points
lat = predictor.lat.values
dlat = np.diff(lat).mean()
n_extra = 32 - lat.size
new_lat = np.concatenate([lat, lat[-1] + dlat * np.arange(1, n_extra + 1)])
predictor = predictor.reindex(lat=new_lat, method='nearest')

# Load predictand
predictand = xr.open_dataset(predictand_filename)
predictand = predictand.load()

# Subset predictand
predictand = predictand.sel(lon=slice(-9.425, 3.375))

# Extend latitude to 256 grid points
lat = predictand.lat.values
dlat = np.diff(lat).mean()
n_extra = 256 - lat.size
new_lat = np.concatenate([lat, lat[-1] + dlat * np.arange(1, n_extra + 1)])
predictand = predictand.reindex(lat=new_lat)

# Remove days with nans in the predictor
predictor = deep4downscaling.trans.remove_days_with_nans(predictor)

# Align both datasets in time
predictor, predictand = deep4downscaling.trans.align_datasets(predictor, predictand, 'time')

# Split data into training and test sets
years_train = ('1980', '2010')
years_test = ('2011', '2020')

x_train = predictor.sel(time=slice(*years_train))
y_train = predictand.sel(time=slice(*years_train))

x_test = predictor.sel(time=slice(*years_test))
y_test = predictand.sel(time=slice(*years_test))

# Standardize the predictors
x_train_stand = deep4downscaling.trans.standardize(data_ref=x_train, data=x_train)

# Set valid mask for the predictand
y_mask = deep4downscaling.trans.compute_valid_mask(y_train)
y_spatial_mask = deep4downscaling.trans.xarray_to_numpy(y_mask)

# Stack the predictand (full grid, no filtering — UnetPr has fixed spatial output)
y_train_stack = y_train.stack(gridpoint=('lat', 'lon'))

# Convert the data to numpy arrays
x_train_stand_arr = deep4downscaling.trans.xarray_to_numpy(x_train_stand)
y_train_arr = deep4downscaling.trans.xarray_to_numpy(y_train_stack)

# Create Dataset
train_dataset = deep4downscaling.deep.utils.StandardDataset(x=x_train_stand_arr,
                                                            y=y_train_arr)

# Split into training and validation sets
train_dataset, valid_dataset = random_split(train_dataset, [0.9, 0.1])

# Create DataLoaders
batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# Set model name
model_name = 'unet_ASYM'

# Create model
# input_padding=(left, right, top, bottom) pads the spatial input before the encoder;
# adjust if the predictor spatial size is not a multiple of 8 (3 maxpool operations)
model = deep4downscaling.deep.models.UnetPr(x_shape=x_train_stand_arr.shape,
                                            y_shape=y_train_arr.shape,
                                            stochastic=False,
                                            input_padding=(0, 0, 0, 0),
                                            kernel_size=3,
                                            padding=1,
                                            batch_norm=True,
                                            trans_conv=True)

# Wrap the model for multi-GPU training if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model.to(device)

# Set hyperparameters
num_epochs = 10000
learning_rate = 0.0001
patience_early_stopping = 100

# Set loss function
loss_function = deep4downscaling.deep.loss.Asym(ignore_nans=True, asym_path=ASYM_PATH)

# Compute ASYM parameters from training data if not already saved
if not loss_function.parameters_exist():
    loss_function.compute_parameters(y_train, var_target='pr')

loss_function.load_parameters()
loss_function.prepare_parameters(device)

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Create tracker for monitoring training progress
# tracker = deep4downscaling.deep.tracker.TrainingTracker(
#     experiment_dir=EXPERIMENTS_PATH,
#     experiment_name=model_name,
#     log_every=5,
#     num_samples=4,
#     spatial_mask=y_spatial_mask,
#     flip_ud=True)

# # Train the model
# train_loss, val_loss = deep4downscaling.deep.train.standard_training_loop(
#     model=model, model_name=model_name, model_path=MODELS_PATH,
#     device=device, num_epochs=num_epochs,
#     loss_function=loss_function, optimizer=optimizer,
#     train_data=train_dataloader, valid_data=valid_dataloader,
#     patience_early_stopping=patience_early_stopping,
#     mixed_precision=True, tracker=tracker)

# Load the best model weights
model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_name}.pt', weights_only=True))

# Standardize the test data
x_test_stand = deep4downscaling.trans.standardize(data_ref=x_train, data=x_test)

# Compute predictions
pred_test = deep4downscaling.deep.pred.compute_preds_standard(
    x_data=x_test_stand, model=model, device=device,
    var_target='pr', mask=y_mask, batch_size=16)

# Save the predictions
pred_test.to_netcdf(f'{PREDS_PATH}/{model_name}.nc')

# =========================
# GCM PROJECTIONS
# =========================
gcm_name      = _cfg["gcm_name"]
gcm_raw       = Path(_cfg["gcm_raw"]).expanduser().resolve()
GCM_PROJ_PATH = Path(_cfg["gcm_proj"]).expanduser().resolve()

gcm_hist = xr.open_dataset(gcm_raw / f'{gcm_name}_r1i1p1f1_historical.nc')
gcm_fut  = xr.open_dataset(gcm_raw / f'{gcm_name}_r1i1p1f1_ssp370.nc')

gcm_hist = gcm_hist.sel(lon=slice(-24, 22.5))
gcm_fut  = gcm_fut.sel(lon=slice(-24, 22.5))

lat = gcm_hist.lat.values
dlat = np.diff(lat).mean()
n_extra = 32 - lat.size
new_lat = np.concatenate([lat, lat[-1] + dlat * np.arange(1, n_extra + 1)])
gcm_hist = gcm_hist.reindex(lat=new_lat, method='nearest')

lat = gcm_fut.lat.values
dlat = np.diff(lat).mean()
n_extra = 32 - lat.size
new_lat = np.concatenate([lat, lat[-1] + dlat * np.arange(1, n_extra + 1)])
gcm_fut = gcm_fut.reindex(lat=new_lat, method='nearest')

gcm_hist_corrected = deep4downscaling.trans.scaling_delta_correction(
    data=gcm_hist, gcm_hist=gcm_hist, obs_hist=x_train)
gcm_fut_corrected = deep4downscaling.trans.scaling_delta_correction(
    data=gcm_fut, gcm_hist=gcm_hist, obs_hist=x_train)

gcm_hist_stand = deep4downscaling.trans.standardize(data_ref=x_train, data=gcm_hist_corrected)
gcm_fut_stand  = deep4downscaling.trans.standardize(data_ref=x_train, data=gcm_fut_corrected)

gcm_hist_stand = gcm_hist_stand.astype('float32')
gcm_fut_stand  = gcm_fut_stand.astype('float32')

proj_hist = deep4downscaling.deep.pred.compute_preds_standard(
    x_data=gcm_hist_stand, model=model, device=device,
    var_target='pr', mask=y_mask, batch_size=16)

proj_fut = deep4downscaling.deep.pred.compute_preds_standard(
    x_data=gcm_fut_stand, model=model, device=device,
    var_target='pr', mask=y_mask, batch_size=16)

proj_hist.to_netcdf(GCM_PROJ_PATH / f'GCM_proj_historical_{model_name}_{gcm_name}.nc')
proj_fut.to_netcdf(GCM_PROJ_PATH  / f'GCM_proj_future_{model_name}_{gcm_name}.nc')
