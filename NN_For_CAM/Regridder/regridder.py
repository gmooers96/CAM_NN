import os
import glob
import xarray as xr
import xesmf as xe

# Paths to input directory and destination file
src_dir = '/ocean/projects/ees240018p/gmooers/GM_Data/'
dst_file = '/ocean/projects/ees240018p/gmooers/CAM/aqua_sst_YOG_f09.cam.h0.0001-04-01-00000.nc'

# List all .nc4 files in the source directory
src_files = glob.glob(os.path.join(src_dir, '*.nc4'))

# Open destination (reference grid) file
dst_ds = xr.open_dataset(dst_file)

# Path to store weight file
weights_file = 'weights.nc'

# Create regridder and save weights if not already saved
if not os.path.exists(weights_file):
    print("Creating weight file...")
    src_ds_for_weights = xr.open_dataset(src_files[0])  # Use the first file to compute weights
    regridder = xe.Regridder(src_ds_for_weights, dst_ds, 'bilinear', filename=weights_file)
    regridder.clean_weight_file()  # Clean intermediate weight file if necessary
else:
    print(f"Reusing existing weight file: {weights_file}")

# Loop through each source file
for src_file in src_files:
    print(f"Processing file: {src_file}")
    
    src_ds = xr.open_dataset(src_file)
    
    regridder = xe.Regridder(src_ds, dst_ds, 'bilinear', filename=weights_file, reuse_weights=True)
    regridded = regridder(src_ds)
    
    # Copy non-spatial variables
    for var_name in src_ds.data_vars:
        if 'lat' not in src_ds[var_name].dims or 'lon' not in src_ds[var_name].dims:
            regridded[var_name] = src_ds[var_name]
    
    # Preserve metadata
    regridded.attrs.update(src_ds.attrs)
    for var_name in regridded.data_vars:
        if var_name in src_ds.data_vars:
            regridded[var_name].attrs.update(src_ds[var_name].attrs)
    
    # Create consistent output file name
    src_basename = os.path.basename(src_file)
    output_filename = f"Regridded_Data/regridded_{src_basename}"
    os.makedirs("Regridded_Data", exist_ok=True)
    regridded.to_netcdf(output_filename, format="NETCDF4")
    
    print(f"Regridded data saved to {output_filename}")

