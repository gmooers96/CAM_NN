### section to force print statement in slurm logging
import sys
print("Job started")
sys.stdout.flush()

import yaml
import numpy as np
import glob
import atmos_physics as atmos_physics
import sys
import random 
import pdb
import math
import xarray as xr
from dask.distributed import Client, LocalCluster
import dask.array as da
import os

from train_test_generator_helper_functions import  create_specific_data_string_desc,  calculate_renormalization_factors_sample


def build_training_dataset(config_file):
    """Builds training and testing datasets."""
    #config_id=config.get('id')
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    #where the coarse-grained simulation data will come from
    filepath=config.get('filepath')
    # where to save the preprocessed netcdf data
    savepath=config.get('savepath')
    my_label=config.get('my_name')
    #what time index in the file to start at (hour index)
    filestart=config.get('file_start')
    # what time index in the file to end at (typically start + 160 steps (hours))
    fileend=config.get('file_end')
    # what vertical level to cut off at
    n_z_input=config.get('levels')
    # input to turn the land fraction into a scalar instead of a column
    ground_levels=config.get('ground_levels') 
    flag_dict=config.get('flag_dict')  
    rewight_outputs=config.get('rewight_outputs')
    shuffle=config.get('shuffle')
    subset=config.get('subset')

    # this collects EVERY coarse-grained data file
    all_files = sorted(glob.glob(filepath))
    
    # Select the subset of the data we want to preprocess
    if subset is True:
        all_files = all_files[filestart:fileend]
    # shuffles the order of the preprocessed data files for opening temporally if set to True
    if shuffle is True:
        random.shuffle(all_files)
    variables = xr.open_mfdataset(all_files)
    print("filepath loaded")
  
        
    x = variables.lon  # m
    y = variables.lat  # m
    z = variables.z  # m
    p = variables.p  # hPa
    rho = variables.rho  # kg/m^3
    n_x = x.size
    n_y = y.size
    n_z = z.size
    n_files = len(variables.time) #typically 160

    print("dimensions in script")

    #default on
    #TODO:@gmooers Customize for gSAM -- right now this only works for CAM
    if flag_dict['grid_cell_area']:
        area = np.load("/ocean/projects/ees240018p/gmooers/Githubs/Neural_nework_parameterization/NN_training/src/CAM_HPO/Testing_log/temp_data/cam_grid_area_sizes.npy")

        
    if flag_dict['land_frac']:
        #terra = xr.DataArray.squeeze(variables.TERRA[:,:ground_levels])
        terra = np.load("/ocean/projects/ees240018p/gmooers/Githubs/Neural_nework_parameterization/NN_training/src/CAM_HPO/Testing_log/temp_data/lat_lon_grid_regridded_for_CAM.npy")

    #default on
    if flag_dict['sfc_pres']:
        #SFC_PRES = variables.SFC_REFERENCE_P
        sfc_pres = np.load("/ocean/projects/ees240018p/gmooers/Githubs/Neural_nework_parameterization/NN_training/src/CAM_HPO/Testing_log/temp_data/z_height_of_terrain.npy")

    # default off
    if flag_dict['skt']:
        SKT = variables.SKT

    #default off
    if flag_dict['cos_lat']:
        cos_lat = np.zeros((n_files, n_y, n_x))
        cos_lat[:, :, :] = xr.ufuncs.cos(xr.ufuncs.radians(y.values[None, :, None]))

    #code adapted from Janni
    adz = xr.zeros_like(z[:n_z_input]) 
    dz = 0.5*(z[0]+z[1]) 
    adz[0] = 1.

    for k in range(1,n_z_input-1): # range doesn't include stopping number
        adz[k] = 0.5*(z[k+1]-z[k-1])/dz

    adz[n_z_input-1] = (z[n_z_input-1]-z[n_z_input-2])/dz
    rho_dz = adz*dz*rho
    
    Tin = variables.TABS_SIGMA[:,:n_z_input] #originally just called tabs
    Qrad = variables.QRAD_SIGMA[:,:n_z_input] / 86400.

    print("Tin, Qin in the script")

    #default on 
    if flag_dict['qin_feature']:
        qt = (variables.QV_SIGMA[:,:n_z_input] + variables.QC_SIGMA[:,:n_z_input] + variables.QI_SIGMA[:,:n_z_input]) / 1000.0 
    
    qp = variables.QP_SIGMA[:,:n_z_input] / 1000.0
    q_auto_out = -1.0*variables.QP_MICRO_SIGMA[:,:n_z_input] / 1000.0
    qpflux_z_coarse = variables.RHOQPW_SIGMA[:,:n_z_input] / 1000.0
    T_adv_out = variables.T_FLUX_Z_OUT_SUBGRID_SIGMA[:,:n_z_input]     #originally tflux_z
    q_adv_out = variables.Q_FLUX_Z_OUT_SUBGRID_SIGMA[:,:n_z_input] / 1000.0 #originally qtflux_z
    qpflux_z = variables.QP_FLUX_Z_OUT_SUBGRID_SIGMA[:,:n_z_input] / 1000.0 
    w = variables.W[:,:n_z_input]  # m/s
    precip = variables.PREC_SIGMA[:,:n_z_input]  # precipitation flux kg/m^2/s
    cloud_qt_flux = variables.SED_SIGMA[:,:n_z_input] / 1000.0
    cloud_lat_heat_flux = variables.LSED_SIGMA[:,:n_z_input] 
    qpflux_diff_coarse_z = variables.RHOQPS_SIGMA[:,:n_z_input] / 1000.0  # SGS qp flux kg/m^2/s Note that I need this variable
    
    a_pr = 1.0 / (atmos_physics.tprmax - atmos_physics.tprmin)
    omp = np.maximum(0.0, np.minimum(1.0, (Tin - atmos_physics.tprmin) * a_pr))
    fac = (atmos_physics.L + atmos_physics.Lf * (1.0 - omp)) / atmos_physics.cp
    
    q_sed_fluxc_out = ((atmos_physics.L + atmos_physics.Lf) * cloud_qt_flux + cloud_lat_heat_flux) / atmos_physics.Lf
    q_sed_fluxi_out = - (atmos_physics.L * cloud_qt_flux + cloud_lat_heat_flux) / atmos_physics.Lf
    q_sed_flux_tot  = cloud_qt_flux
    
    dfac_dz = np.zeros((n_files, n_z_input, n_y, n_x))
    for k in range(n_z_input - 1):
        kb = max(0, k - 1)
        dfac_dz[:, k, :, :] = (fac[:, k + 1, :, :] - fac[:, k, :, :]) / rho_dz[k, :] * rho[:, k]
        
    Tout = dfac_dz * (qpflux_z_coarse + qpflux_diff_coarse_z - precip) / rho

    print("Variables Created")
    
    data_specific_description = create_specific_data_string_desc(flag_dict)
    
    my_dict_train = {}

    # code below reshapes the data into (z column, sample) where sample = lat*time*lon
    # Janni's version reshaped to (z, lat, sample) where sample = time*lon
    # at this point in the code (via .values) the data is transformed from an xarray/DataArray object into a numpy array in memory
    if flag_dict['Tin_feature']:
        Tin = Tin.transpose("z","lat","time","lon").values
        Tin = np.reshape(Tin, (n_z_input, n_y*n_files*n_x))
        my_dict_train["Tin"] = (("z","sample"), Tin)
        del Tin
    
    if flag_dict['qin_feature']:
        qin = qt.transpose("z","lat","time","lon").values
        qin = np.reshape(qin, (n_z_input, n_y*n_files*n_x))
        my_dict_train["qin"] = (("z","sample"), qin)
        del qin
    
    if flag_dict['predict_tendencies']:
        Tout = Tout.transpose("z","lat","time","lon").values
        Tout = np.reshape(Tout, (n_z_input, n_y*n_files*n_x))
        my_dict_train["Tout"] = (("z","sample"), Tout)

        T_adv_out = T_adv_out.transpose("z","lat","time","lon").values
        T_adv_out = np.reshape(T_adv_out, (n_z_input, n_y*n_files*n_x)) 
        my_dict_train["T_adv_out"] = (("z","sample"), T_adv_out)
        
        q_adv_out = q_adv_out.transpose("z","lat","time","lon").values
        q_adv_out = np.reshape(q_adv_out, (n_z_input, n_y*n_files*n_x)) 
        my_dict_train["q_adv_out"] = (("z","sample"), q_adv_out)
        
        q_auto_out = q_auto_out.transpose("z","lat","time","lon").values
        q_auto_out = np.reshape(q_auto_out, (n_z_input, n_y*n_files*n_x))
        my_dict_train["q_auto_out"] = (("z","sample"), q_auto_out)

        q_sed_flux_tot = q_sed_flux_tot.transpose("z","lat","time","lon").values
        q_sed_flux_tot = np.reshape(q_sed_flux_tot, (n_z_input, n_y*n_files*n_x))
        my_dict_train["q_sed_flux_tot"] = (("z","sample"), q_sed_flux_tot)
    
    if flag_dict['land_frac']:
        terra = np.expand_dims(terra, axis=0).repeat(n_files, axis=0)
        terra = np.reshape(terra, (n_y*n_files*n_x))
        my_dict_train["terra"] = (("sample"), terra)
        del terra
    
    if flag_dict['sfc_pres']:
        sfc_pres = np.expand_dims(sfc_pres, axis=0).repeat(n_files, axis=0)
        sfc_pres = np.reshape(sfc_pres, (n_y*n_files*n_x))
        my_dict_train["sfc_pres"] = (("sample"), sfc_pres)
        del sfc_pres
    
    if flag_dict['skt']:
        skt = SKT.transpose("lat","time","lon").values
        skt = np.reshape(skt, (n_y*n_files*n_x))
        my_dict_train["skt"] = (("sample"), skt)
        del skt

    if flag_dict['grid_cell_area']:
        area = np.expand_dims(area, axis=0).repeat(n_files, axis=0)
        area = np.reshape(area, (n_y*n_files*n_x))
        my_dict_train["area"] = (("sample"), area)
        del area

    print("Variables in Dict")
    my_weight_dict = {}
    # code from Janni -- calculates std / min(std) of output vars for normailzation
    norm_list = calculate_renormalization_factors_sample(Tout,
                                                       T_adv_out,
                                                       q_adv_out,
                                                       q_auto_out,
                                                       q_sed_flux_tot,
                                                       rho_dz[:,0].values,
                                                      ) 
     
    del Tout, T_adv_out, q_adv_out, q_auto_out, q_sed_flux_tot, rho_dz
    my_weight_dict["norms"] = (("norm"), norm_list)

    print("Norms in Dict")
    
    ds_weight = xr.Dataset(
        my_weight_dict,
            coords={
                "norm":np.arange(1,6,1),
            },
        )
    
    print("Starting Save of weights")
    
    ds_weight.to_netcdf(savepath + my_label + data_specific_description + "file_"+str(filestart)+"_to_"+str(fileend)+"_w8s.nc")

    print("Weights Saved")
    
    ds_train = xr.Dataset(
    my_dict_train,
    coords={
        "z": z[:n_z_input].values,
        "lat": y.values,
        "lon": x.values,
        "z_profile": z.values,
        "rho": rho[0,:].values,
        "p": p[0,:].values,
        "sample": np.arange(0,n_files*len(x.values)*len(y.values), 1),
    },)
    
    print("starting save of main data")
    ds_train.to_netcdf(savepath + my_label + data_specific_description + "file_"+str(filestart)+"_to_"+str(fileend)+".nc")
    print("Finished Save of main data")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python input_output_data_preparation.py <config_file.yaml>")
    else:
        build_training_dataset(sys.argv[1])    