import xarray as xr
import glob
import os
import time
import numpy as np
import xesmf as xe

# ---------- Paths ----------
input_dir = "/ocean/projects/ees240018p/gmooers/gsam_data/"
output_dir = "/ocean/projects/ees240018p/gmooers/gsam_data/cam_resolution_gsam_v2/"
weights_file = "/ocean/projects/ees240018p/gmooers/weights_GSAM_to_CAM.nc"

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# ---------- Load CAM Grid (Target) ----------
cam_file = "/ocean/projects/ees240018p/gmooers/CAM/aqua_sst_YOG_f09.cam.h0.0001-04-01-00000.nc"
cam_ds = xr.open_dataset(cam_file)
target_lat = cam_ds['lat'].data
target_lon = cam_ds['lon'].data

target_grid = xr.Dataset({
    'lat': (['lat'], target_lat),
    'lon': (['lon'], target_lon),
})

# ---------- Regridder Helper ----------
def prepare_regridder(weights_file, ds_with_mask, target_grid, method='conservative_normed'):
    print(f"🔧 Preparing regridder with weights file: {weights_file}")
    regridder = xe.Regridder(ds_with_mask, target_grid,
                             method=method,
                             filename=weights_file,
                             reuse_weights=os.path.exists(weights_file))
    return regridder

# ---------- Find Files ----------
file_patterns = ["*.2DC_atm.nc", "*.atm.3DC.nc"]
all_files = []
for pattern in file_patterns:
    all_files.extend(sorted(glob.glob(os.path.join(input_dir, pattern))))

total_files = len(all_files)
print(f"📦 Found {total_files} files to process.")

# ---------- Begin Processing Loop ----------
total_start = time.time()

for idx, file in enumerate(all_files, start=1):
    file_start = time.time()
    print(f"\n🚀 Processing file {idx} of {total_files}: {file}")

    ds = xr.open_dataset(file)
    var_list = [v for v in ds.data_vars if 'time' in ds[v].dims]

    # ---------- Choose Initial Mask for 2D variables ----------
    if 'LANDMASK' in ds.data_vars:
        print("📌 Using LANDMASK for masking (2D file)")
        landmask = ds['LANDMASK'].isel(time=0).squeeze()
        mask_da = xr.where(landmask > 0.5, 0, 1)  # 0 = land, 1 = ocean
        ds["mask"] = mask_da
        regridder = prepare_regridder(weights_file, ds, target_grid)
    else:
        regridder = None  # Will be created per level for 3D vars

    # ---------- Regrid Each Variable ----------
    ds_regridded = xr.Dataset()

    for var in var_list:
        da = ds[var]
        var_dims = da.dims

        if 'lat' in var_dims and 'lon' in var_dims:
            # 3D Variable
            if 'z' in var_dims or 'zi' in var_dims:
                vert_dim = 'z' if 'z' in var_dims else 'zi'
                print(f"🔹 Regridding 3D variable: {var}")
                regridded_levels = []

                # Determine which terrain mask to use
                use_zi = vert_dim == 'zi'
                if use_zi and 'TERRAW' in ds.data_vars:
                    terrain_mask = ds['TERRAW'].isel(time=0)
                    print(f"🧭 Variable {var} uses {vert_dim}; using TERRAW")
                elif not use_zi and 'TERRA' in ds.data_vars:
                    terrain_mask = ds['TERRA'].isel(time=0)
                    print(f"🧭 Variable {var} uses {vert_dim}; using TERRA")
                else:
                    print(f"⚠ No appropriate terrain mask found for variable {var}; using fallback")
                    terrain_mask = xr.ones_like(da.isel(time=0, **{vert_dim: 0}))

                for level in range(da.sizes[vert_dim]):
                    da_slice = da.isel(time=0, **{vert_dim: level})
                    mask_at_level = xr.where(terrain_mask.isel({vert_dim: level}) > 0.5, 0, 1)

                    ds_temp = xr.Dataset({
                        "data": da_slice,
                        "mask": mask_at_level
                    })

                    regridder_level = prepare_regridder(weights_file, ds_temp, target_grid)
                    regridded_slice = regridder_level(ds_temp["data"])
                    regridded_levels.append(regridded_slice)

                regridded_stack = xr.concat(regridded_levels, dim=vert_dim)
                regridded_stack = regridded_stack.assign_coords({vert_dim: da[vert_dim]})
                ds_regridded[var] = regridded_stack

            # 2D Variable
            else:
                print(f"🔹 Regridding 2D variable: {var}")
                if regridder is None:
                    print("⚠ Regridder not prepared for 2D variable. Skipping.")
                    continue
                regridded_da = regridder(da.isel(time=0))
                ds_regridded[var] = regridded_da

        else:
            print(f"⚠ Skipping variable (no lat/lon): {var}")

    # ---------- Save Regridded Output ----------
    filename = os.path.basename(file).replace('.nc', '_camres.nc')
    out_file = os.path.join(output_dir, filename)

    print(f"💾 Saving regridded output to: {out_file}")
    ds_regridded.to_netcdf(out_file)

    file_end = time.time()
    print(f"✅ Finished {filename} (Elapsed time: {file_end - file_start:.2f} seconds)")

# ---------- Wrap Up ----------
total_end = time.time()
total_elapsed = total_end - total_start
print(f"\n🎉 All {total_files} files processed! Total elapsed time: {total_elapsed / 60:.2f} minutes.")


