### section to force print statement in slurm logging
import sys
print("Job started")
#sys.stdout.flush()
import yaml
import numpy as np
import glob
import atmos_physics as atmos_physics
import random 
import pdb
import xarray as xr

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
    polar_mask=config.get('polar_mask')
    land_mask=config.get('land_mask')
    gSAM = config.get('gSAM')

    # this collects EVERY coarse-grained data file
    all_files_original = sorted(glob.glob(filepath))
    
    # Select the subset of the data we want to preprocess
    all_files = all_files_original[filestart:fileend]
    # shuffles the order of the preprocessed data files for opening temporally if set to True
    if shuffle is True:
        random.shuffle(all_files)

    print("File names shuffled")

    
    # original code
    variables = xr.open_mfdataset(all_files)
    print("filepath loaded")
  
        
    x = variables.lon  # m
    y = variables.lat  # m
    z = variables.z  # m
    #p = variables.p  # hPa
    #rho = variables.rho  # kg/m^3
    n_x = x.size
    n_y = y.size
    n_z = z.size
    n_files = len(variables.time) #typically 160

    hr_variables = xr.open_dataset('/ocean/projects/ees240018p/gmooers/GM_Data/POG_Correction/DYAMOND2_coars_9216x4608x74_10s_4608_20200301000000_0000354240.atm.3D_resolved.nc4')

    p = hr_variables.p  # hPa
    rho = hr_variables.rho  # kg/m^3

    print("dimensions in script")

    #default on
    #TODO:@gmooers Customize for gSAM -- right now this only works for CAM
    if flag_dict['grid_cell_area']:
        if gSAM == True:
            area = np.load("/ocean/projects/ees240018p/gmooers/Githubs/Neural_nework_parameterization/NN_training/src/CAM_HPO/Testing_log/temp_data/gsam_grid_area_sizes.npy")
        else:
            area = np.load("/ocean/projects/ees240018p/gmooers/Githubs/Neural_nework_parameterization/NN_training/src/CAM_HPO/Testing_log/temp_data/cam_grid_area_sizes.npy")

        
    if flag_dict['land_frac']:
        if gSAM == True:
            land_mask_path = '/ocean/projects/ees240018p/gmooers/gsam_data/DYAMOND2_coars_9216x4608x74_10s_4608_20200301000000_0000354240.2DC_atm.nc'
            land_mask_ds = xr.open_dataset(land_mask_path)
            terra = np.squeeze(land_mask_ds.LANDMASK.values)
        else:
            land_mask_path = '/ocean/projects/ees240018p/gmooers/CAM/aqua_sst_YOG_f09.cam.h0.0001-04-01-00000.nc'
            land_mask_ds = xr.open_dataset(land_mask_path)
            terra = np.squeeze(land_mask_ds.LANDFRAC.values)[0,:,:]

        # new code for land mask
        terra_expanded = np.expand_dims(terra, axis=0).repeat(n_files, axis=0)  # time, lat, lon
        terra_flat = terra_expanded.flatten()  # (time * lat * lon)
        ocean_mask_flat = terra_flat < 0.5  # True = ocean, False = land


    if flag_dict['sfc_pres']:
        sfc_pres = variables.SFC_REFERENCE_P.values
        # shape is time, lat, lon


    if flag_dict['skt']:
        SKT = variables.SKT
        # shape is time, lat, lon

    #code adapted from Janni
    adz = xr.zeros_like(z[:n_z_input]) 
    dz = 0.5*(z[0]+z[1]) 
    adz[0] = 1.

    for k in range(1,n_z_input-1): # range doesn't include stopping number
        adz[k] = 0.5*(z[k+1]-z[k-1])/dz

    adz[n_z_input-1] = (z[n_z_input-1]-z[n_z_input-2])/dz
    rho_dz = adz*dz*rho
    
    Tin = variables.TABS_SIGMA[:,:n_z_input] #originally just called tabs
    print("Tin raw mean:", Tin.mean().values, 
      "min:", Tin.min().values, 
      "max:", Tin.max().values)
    Qrad = variables.QRAD_SIGMA[:,:n_z_input] / 86400. # need to include for radiation
    

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
        dfac_dz[:, k, :, :] = (fac[:, k + 1, :, :] - fac[:, k, :, :]) / rho_dz[k] * rho[k]
            

            
        
    Tout = dfac_dz * (qpflux_z_coarse + qpflux_diff_coarse_z - precip) / rho # could add in qrad here for radation

    print("Variables Created")
    
    data_specific_description = create_specific_data_string_desc(flag_dict)
    
    my_dict_train = {}

    # code below reshapes the data into (z column, sample) where sample = lat*time*lon
    # Janni's version reshaped to (z, lat, sample) where sample = time*lon
    # at this point in the code (via .values) the data is transformed from an xarray/DataArray object into a numpy array in memory


    if flag_dict['Tin_feature']:
        # shape is originally t, z, lat, lon
        Tin = Tin.transpose("z","lat","time","lon").values
        Tin = np.reshape(Tin, (n_z_input, n_y*n_files*n_x))
        print("Tin reshaped mean:", Tin.mean(), "min:", Tin.min(), "max:", Tin.max())
        my_dict_train["Tin"] = (("z","sample"), Tin)
        del Tin
    
    if flag_dict['qin_feature']:
        qin = qt.transpose("z","lat","time","lon").values
        qin = np.reshape(qin, (n_z_input, n_y*n_files*n_x))
        my_dict_train["qin"] = (("z","sample"), qin)
        del qin

    if flag_dict['Terra_feature']:
        TERRA = variables.TERRA_SIGMA[:,:n_z_input] #originally just called tabs
        TERRA = TERRA.transpose("z","lat","time","lon").values
        TERRA = np.reshape(TERRA, (n_z_input, n_y*n_files*n_x))
        my_dict_train["terra"] = (("z","sample"), TERRA)
        del TERRA

    if flag_dict['Terraw_feature']:
        TERRAW = variables.TERRAW_SIGMA[:,:n_z_input] #originally just called tabs
        TERRAW = TERRAW.transpose("z","lat","time","lon").values
        TERRAW = np.reshape(TERRAW, (n_z_input, n_y*n_files*n_x))
        my_dict_train["terraw"] = (("z","sample"), TERRAW)
        del TERRAW
 
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
        my_dict_train["land_frac"] = (("sample"), terra)
        del terra

    if flag_dict['sfc_pres']:
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

    if flag_dict['land_ice']:
        land_ice = variables.LANDICEM
        land_ice = land_ice.transpose("lat","time","lon").values
        land_ice = np.reshape(land_ice, (n_y*n_files*n_x))
        my_dict_train["land_ice"] = (("sample"), land_ice)
        del land_ice

    if flag_dict['sea_ice']:
        sea_ice = variables.SEAICEMA
        sea_ice = sea_ice.transpose("lat","time","lon").values
        sea_ice = np.reshape(sea_ice, (n_y*n_files*n_x))
        my_dict_train["sea_ice"] = (("sample"), sea_ice)
        del sea_ice

    if flag_dict['veg']:
        veg = variables.VEG
        veg = veg.transpose("lat","time","lon").values
        veg = np.reshape(veg, (n_y*n_files*n_x))
        my_dict_train["veg"] = (("sample"), veg)
        del veg

    if flag_dict['soil_temp']:
        soil_temp = variables.SOILT
        soil_temp = soil_temp.transpose("lat","time","lon").values
        soil_temp = np.reshape(soil_temp, (n_y*n_files*n_x))
        my_dict_train["soil_temp"] = (("sample"), soil_temp)
        del soil_temp

    if flag_dict['soil_water']:
        soil_water = variables.SOILW
        soil_water = soil_water.transpose("lat","time","lon").values
        soil_water = np.reshape(soil_water, (n_y*n_files*n_x))
        my_dict_train["soil_water"] = (("sample"), soil_water)
        del soil_water

    if flag_dict['tree_canopy']:
        tree_canopy = variables.TCANOP
        tree_canopy = tree_canopy.transpose("lat","time","lon").values
        tree_canopy = np.reshape(tree_canopy, (n_y*n_files*n_x))
        my_dict_train["tree_canopy"] = (("sample"), tree_canopy)
        del tree_canopy


    print("Variables in Dict")
    my_weight_dict = {}
    # code from Janni -- calculates std / min(std) of output vars for normailzation
    norm_list = calculate_renormalization_factors_sample(Tout,
                                                       T_adv_out,
                                                       q_adv_out,
                                                       q_auto_out,
                                                       q_sed_flux_tot,
                                                       rho_dz.values,
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
            "lat": y.values,  # note: lat/lon here are just full grid references, not per-sample
            "lon": x.values,
            "z_profile": z.values,
            "rho": rho.values,
            "p": p.values,
            "sample": np.arange(my_dict_train["Tin"][1].shape[1]),  # updated sample count
        },
    )
    for varname in ds_train.data_vars:
        data = ds_train[varname].values
        print(f"{varname} -- mean: {np.nanmean(data):.4f}, min: {np.nanmin(data):.4f}, max: {np.nanmax(data):.4f}")

        
    print("starting save of main data")
    ds_train.to_netcdf(savepath + my_label + data_specific_description + "file_"+str(filestart)+"_to_"+str(fileend)+".nc")
    print("Finished Save of main data")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python input_output_data_preparation.py <config_file.yaml>")
    else:
        build_training_dataset(sys.argv[1])    