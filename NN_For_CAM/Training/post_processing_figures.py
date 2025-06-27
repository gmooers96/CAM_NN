import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xarray as xr
import pdb
from plotting_functions import *
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy

fz = 15*1.5
lw = 4
siz = 100
XNNA = 1.25 # Abscissa where architecture-constrained network will be placed
XTEXT = 0.25 # Text placement
YTEXT = 0.3 # Text placement

plt.rc('text', usetex=False)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
#mpl.rcParams["font.serif"] = "STIX"
plt.rc('font', family='serif', size=fz)
matplotlib.rcParams['lines.linewidth'] = lw


def main_plotting(truth,
                 pred,
                  z_dim,
                 raw_data,
                 var_names,
                 save_path,
                 nametag,
                 do_poles,
                 ):

    raw_data = xr.open_dataset(raw_data)
    print("savepath", save_path)
    print("nametag", nametag)
    print("var_names", var_names)
    print('z_dim', z_dim)
    np.save('dummy_data/truths.npy', truth)
    np.save('dummy_data/preds.npy', pred)
    
    lon = raw_data.lon.values
    lat = raw_data.lat.values
    if do_poles == False:
        north_lat_idx = np.abs(lat - 80.0).argmin()
        south_lat_idx = np.abs(lat - (-1.0*80.0)).argmin()
        lat = lat[south_lat_idx:north_lat_idx+1] 
        
    z = raw_data.z.values[:z_dim]
    p = raw_data.p.values
    TERRA = raw_data.TERRA.values.squeeze()
    sigma = p / p[0]
    land_mask = TERRA[0]

    srf_height = np.load("/ocean/projects/ees240018p/gmooers/Githubs/Neural_nework_parameterization/NN_training/src/CAM_HPO/Testing_log/temp_data/high_res_z_height_of_terrain.npy")

    sigma_plot, lat_plot = np.meshgrid(sigma, lat)
    lat_mesh, lon_mesh = np.meshgrid(lat, lon)

    n_z_input = len(z)
    n_y = len(lat)
    n_x = len(lon)
    n_files = int(truth.shape[1]/(len(lat)*len(lon)))
    
    t_Tout = truth[:z_dim,:].reshape(n_z_input, n_y, n_files, n_x)
    t_T_adv_out = truth[z_dim:z_dim*2,:].reshape(n_z_input, n_y, n_files, n_x)
    t_q_adv_out = truth[z_dim*2:z_dim*3,:].reshape(n_z_input, n_y, n_files, n_x)
    t_q_auto_out = truth[z_dim*3:z_dim*4,:].reshape(n_z_input, n_y, n_files, n_x)
    t_q_sed_flux_tot = truth[z_dim*4:,:].reshape(n_z_input, n_y, n_files, n_x)

    p_Tout = pred[:z_dim,:].reshape(n_z_input, n_y, n_files, n_x)
    p_T_adv_out = pred[z_dim:z_dim*2,:].reshape(n_z_input, n_y, n_files, n_x)
    p_q_adv_out = pred[z_dim*2:z_dim*3,:].reshape(n_z_input, n_y, n_files, n_x)
    p_q_auto_out = pred[z_dim*3:z_dim*4,:].reshape(n_z_input, n_y, n_files, n_x)
    p_q_sed_flux_tot = pred[z_dim*4:,:].reshape(n_z_input, n_y, n_files, n_x)

    #average out longitude
    lon_axis = 3 # was 2 before
    lat_axis = 1
    time_axis=2
    z_axis=0

    #### beggining of precip figures

    truth_precip = p[0]*np.sum(t_q_auto_out, axis=0)
    predicted_precip = p[0]*np.sum(p_q_auto_out, axis=0)

    land_mask_bool = land_mask > 0.5
    ocean_mask_bool = ~land_mask_bool
    
    land_mask_3d = land_mask_bool[:, np.newaxis, :]  # shape (lat, 1, lon)
    ocean_mask_3d = ocean_mask_bool[:, np.newaxis, :]

    truth_land_precip = np.where(land_mask_3d, truth_precip, np.nan)
    pred_land_precip = np.where(land_mask_3d, predicted_precip, np.nan)
    
    truth_ocean_precip = np.where(ocean_mask_3d, truth_precip, np.nan)
    pred_ocean_precip = np.where(ocean_mask_3d, predicted_precip, np.nan)

    time_axis_precip = 1

    sse_precip = ((truth_precip - predicted_precip)**2).sum(axis=time_axis_precip)
    svar_precip = np.sum((truth_precip - truth_precip.mean(axis=time_axis_precip)[:,None,:])**2, axis=time_axis_precip)
    R2_Precip = 1- (sse_precip/svar_precip)
    
    sse_land_precip = np.nansum((truth_land_precip - pred_land_precip)**2, axis=time_axis_precip)
    svar_land_precip = np.nansum((truth_land_precip - np.nanmean(truth_land_precip, axis=time_axis_precip)[:,None,:])**2, axis=time_axis_precip)
    R2_land_Precip = 1- (sse_land_precip/svar_land_precip)
    
    sse_ocean_precip = np.nansum((truth_ocean_precip - pred_ocean_precip)**2, axis=time_axis_precip)
    svar_ocean_precip = np.nansum((truth_ocean_precip - np.nanmean(truth_ocean_precip, axis=time_axis_precip)[:,None,:])**2, axis=time_axis_precip)
    R2_ocean_Precip = 1- (sse_ocean_precip/svar_ocean_precip)

    print("Global R2 is", R2_Precip.mean())
    print("Land R2 is", np.nanmean(R2_land_Precip))
    print("Ocean R2 is", np.nanmean(R2_ocean_Precip))

    single_geo_plotter(lats=lat_mesh, 
                   lons=lon_mesh, 
                   field_data=R2_Precip.T, 
                   title=r'Global Precipitation $R^2$', 
                   units="Skill Score", 
                   cmap = "Blues", 
                   logcbar=False, 
                   save=False, 
                   vmin = 0, 
                   vmax = 1.0, 
                   base_dir=save_path+nametag,
                   save_dir="/Precip_Figures",
                  )

    my_scatter_plot(
        truth_array=truth_precip, 
        pred_array=predicted_precip, 
        y_label='NN Predictions', 
        x_label='gSAM Targets', 
        title='Global Precipitation Truth vs. NN Predictions', 
        save=True, 
        base_dir=save_path+nametag,
        save_dir="/Precip_Figures",
    )

    my_scatter_plot(
        truth_array=truth_land_precip, 
        pred_array=pred_land_precip, 
        y_label='NN Predictions', 
        x_label='gSAM Targets', 
        title='Land Precipitation Truth vs. NN Predictions', 
        save=True, 
        base_dir=save_path+nametag,
        save_dir="/Precip_Figures",
    )

    my_scatter_plot(
        truth_array=truth_ocean_precip, 
        pred_array=pred_ocean_precip, 
        y_label='NN Predictions', 
        x_label='gSAM Targets', 
        title='Ocean Precipitation Truth vs. NN Predictions', 
        save=True, 
        base_dir=save_path+nametag,
        save_dir="/Precip_Figures",
    )

    truth_precip_1d = truth_precip.ravel()
    predicted_precip_1d = predicted_precip.ravel()
    
    truth_land_precip_1d = truth_land_precip.ravel()
    pred_land_precip_1d = pred_land_precip.ravel()
    
    truth_ocean_precip_1d = truth_ocean_precip.ravel()
    pred_ocean_precip_1d = pred_ocean_precip.ravel()

    num_bins = 100
    bins = np.logspace(np.log10(0.02), np.log10(np.nanmax(truth_precip_1d)), num_bins)

    precip_distribution_plot(
        global_truth=truth_precip_1d, 
        global_nn=predicted_precip_1d, 
        land_truth=truth_land_precip_1d, 
        land_nn=pred_land_precip_1d,
        ocean_truth=truth_ocean_precip_1d, 
        ocean_nn=pred_ocean_precip_1d, 
        title_all='Global', 
        title_land='Land Only', 
        title_ocean='Ocean Only', 
        label_truth='gSAM Targets', 
        label_nn='NN Predictions',
        suptitle='Precipitation PDFs', 
        x_label='Frequency', 
        y_label=r"$mm \ day^{-1}$",
        save=True, 
        base_dir=save_path+nametag,
        save_dir="/Precip_Figures",
        savename='Precip_Dist',
        color_true='blue', 
        color_nn='green', 
        bins=bins, 
        x_log = False, 
        y_log = False, 
        x_limits=None,
    )

    truth_precip_lat = np.mean(truth_precip, axis=(1,2))
    predicted_precip_lat = np.mean(predicted_precip, axis=(1,2))
    
    truth_land_precip_lat = np.nanmean(truth_land_precip, axis=(1,2))
    pred_land_precip_lat = np.nanmean(pred_land_precip, axis=(1,2))
    
    truth_ocean_precip_lat = np.nanmean(truth_ocean_precip, axis=(1,2))
    pred_ocean_precip_lat = np.nanmean(pred_ocean_precip, axis=(1,2))

    precip_with_latitude(
        precip_truth=truth_precip_lat, 
        precip_nn=predicted_precip_lat, 
        truth_label='gSAM Target Data', 
        nn_label='NN Predictions', 
        title='Precipitation with latitude', 
        lats=lat, 
        color_truth='Blue', 
        color_nn='Green', 
        y_label='Precipitation', 
        x_label='Latitude',
        vmin=None, 
        vmax=None, 
        save=True, 
        base_dir=save_path+nametag,
        save_dir="/Precip_Figures",
        savename='Precip_vs_Lat',
    )

    precip_with_latitude(
        precip_truth=truth_land_precip_lat, 
        precip_nn=pred_land_precip_lat, 
        truth_label='gSAM Target Data', 
        nn_label='NN Predictions', 
        title='Precipitation with latitude (Land)', 
        lats=lat, 
        color_truth='Blue', 
        color_nn='Green', 
        y_label='Precipitation', 
        x_label='Latitude',
        vmin=None, 
        vmax=None, 
        save=True, 
        base_dir=save_path+nametag,
        save_dir="/Precip_Figures",
        savename='Precip_vs_Lat_land',
    )

    precip_with_latitude(
        precip_truth=truth_ocean_precip_lat, 
        precip_nn=pred_ocean_precip_lat, 
        truth_label='gSAM Target Data', 
        nn_label='NN Predictions', 
        title='Precipitation with latitude (Ocean)', 
        lats=lat, 
        color_truth='Blue', 
        color_nn='Green', 
        y_label='Precipitation', 
        x_label='Latitude',
        vmin=None, 
        vmax=None, 
        save=True, 
        base_dir=save_path+nametag,
        save_dir="/Precip_Figures",
        savename='Precip_vs_Lat_ocean',
    )

    truth_precip_lat_lon = np.mean(truth_precip, axis=(1))
    predicted_precip_lat_lon = np.mean(predicted_precip, axis=(1))

    ssim_map = ssim(truth_precip_lat_lon, predicted_precip_lat_lon, data_range=truth_precip_lat_lon.max() - truth_precip_lat_lon.min())
    print('SSIM:', ssim_map)

    plot_side_by_side_with_diff(
        A=truth_precip_lat_lon.T, 
        B=predicted_precip_lat_lon.T, 
        lons=lon_mesh, 
        lats=lat_mesh, 
        units=r"$mm \ day^{-1}$",
        cmap="Greens",
        diff_cmap="seismic",
        title_A="gSAM Targets",
        title_B="NN Predictions",
        title_diff="Difference",
        suptitle='Gridcell Precipitation Means',
        projection=ccrs.Robinson(central_longitude=180),
        linthresh=None,
        norm=False,
        save=True, 
        base_dir=save_path+nametag,
        save_dir="/Precip_Figures",
        savename='Precip_mean_lat_lat',
        vmin=None,
        vmax=None,
        vmax_clip=None,
        vmin_clip=None,
        diff_mode="raw", # 'raw' or 'relative'
    )

    rmse_map = np.sqrt(np.nanmean((predicted_precip - truth_precip)**2, axis=1))

    global_rmse = compute_rmse(truth_precip, predicted_precip)
    global_rmse_land = compute_rmse(truth_land_precip, pred_land_precip)
    global_rmse_ocean = compute_rmse(truth_ocean_precip, pred_ocean_precip)

    
    global_mae = compute_mae(truth_precip, predicted_precip)
    global_mae_land = compute_mae(truth_land_precip, pred_land_precip)
    global_mae_ocean = compute_mae(truth_ocean_precip, pred_ocean_precip)

    print("Global RMSE is:", global_rmse)
    print("Land RMSE is:", global_rmse_land)
    print("Ocean RMSE is:", global_rmse_ocean)
    
    print("Global MAE is:", global_mae)
    print("Land MAE is:", global_mae_land)
    print("Ocean MAE is:", global_mae_ocean)

    single_geo_plotter(
        lats=lat_mesh, 
        lons=lon_mesh, 
        field_data=rmse_map.T, 
        title='RMSE of Precip', 
        units=r"$mm \ day^{-1}$", 
        cmap = "Reds", 
        logcbar=False, 
        save=True, 
        vmin = None, 
        vmax=None, 
        base_dir=save_path+nametag,
        save_dir="/Precip_Figures",
        savename='Precip_RMSE'
    )

    truth_precip_time = np.mean(truth_precip, axis=(0,2))
    predicted_precip_time = np.mean(predicted_precip, axis=(0,2))
    
    truth_land_precip_time = np.nanmean(truth_land_precip, axis=(0,2))
    pred_land_precip_time = np.nanmean(pred_land_precip, axis=(0,2))
    
    truth_ocean_precip_time = np.nanmean(truth_ocean_precip, axis=(0,2))
    pred_ocean_precip_time = np.nanmean(pred_ocean_precip, axis=(0,2))

    time_array = np.arange(truth_precip_time.size)

    precip_with_time(
        truth_all=truth_precip_time, 
        nn_all=predicted_precip_time, 
        truth_land=truth_land_precip_time, 
        nn_land=pred_land_precip_time,
        truth_ocean=truth_ocean_precip_time, 
        nn_ocean=pred_ocean_precip_time, 
        times=time_array,       
        truth_all_label='All gSAM Targets', 
        nn_all_label='All NN Predictions', 
        truth_land_label='Land gSAM Targets', 
        nn_land_label='Land NN Predcitions', 
        truth_ocean_label='Ocean gSAM Targets', 
        nn_ocean_label='Ocean NN Predictions', 
        all_color='purple', 
        land_color='green', 
        ocean_color='blue',
        y_label=r"$mm \ day^{-1}$",  
        x_label='Time in Hours', 
        global_title='Precipitation Though Time',
        vmin=None, 
        vmax=None, 
        save=True, 
        base_dir=save_path+nametag,
        save_dir="/Precip_Figures",
        savename='Precip_Time',
    )

    #### end of precip figures
    
    t_Tout_lon_mean = np.mean(t_Tout, axis=lon_axis)
    t_T_adv_out_lon_mean = np.mean(t_T_adv_out, axis=lon_axis)
    t_q_adv_out_lon_mean = np.mean(t_q_adv_out, axis=lon_axis)
    t_q_auto_out_lon_mean = np.mean(t_q_auto_out, axis=lon_axis)
    t_q_sed_flux_tot_lon_mean = np.mean(t_q_sed_flux_tot, axis=lon_axis)

    p_Tout_lon_mean = np.mean(p_Tout, axis=lon_axis)
    p_T_adv_out_lon_mean = np.mean(p_T_adv_out, axis=lon_axis)
    p_q_adv_out_lon_mean = np.mean(p_q_adv_out, axis=lon_axis)
    p_q_auto_out_lon_mean = np.mean(p_q_auto_out, axis=lon_axis)
    p_q_sed_flux_tot_lon_mean = np.mean(p_q_sed_flux_tot, axis=lon_axis)

    #avewrage out Z for lat/lon maps

    t_Tout_z_mean = np.mean(t_Tout, axis=z_axis)
    t_T_adv_out_z_mean = np.mean(t_T_adv_out, axis=z_axis)
    t_q_adv_out_z_mean = np.mean(t_q_adv_out, axis=z_axis)
    t_q_auto_out_z_mean = np.mean(t_q_auto_out, axis=z_axis)
    t_q_sed_flux_tot_z_mean = np.mean(t_q_sed_flux_tot, axis=z_axis)

    p_Tout_z_mean = np.mean(p_Tout, axis=z_axis)
    p_T_adv_out_z_mean = np.mean(p_T_adv_out, axis=z_axis)
    p_q_adv_out_z_mean = np.mean(p_q_adv_out, axis=z_axis)
    p_q_auto_out_z_mean = np.mean(p_q_auto_out, axis=z_axis)
    p_q_sed_flux_tot_z_mean = np.mean(p_q_sed_flux_tot, axis=z_axis)

    #apply land and sea masks to data
    # origiunal code was [:,None,:,None]
    t_Tout_ocean = np.where(land_mask[None,:,None,:] == 1.0, t_Tout, np.nan)
    t_Tout_land = np.where(land_mask[None,:,None,:] == 0.0, t_Tout, np.nan)
    t_T_adv_out_ocean = np.where(land_mask[None,:,None,:] == 1.0, t_T_adv_out, np.nan)
    t_T_adv_out_land = np.where(land_mask[None,:,None,:] == 0.0, t_T_adv_out, np.nan)
    t_q_adv_out_ocean = np.where(land_mask[None,:,None,:] == 1.0, t_q_adv_out, np.nan)
    t_q_adv_out_land = np.where(land_mask[None,:,None,:] == 0.0, t_q_adv_out, np.nan)
    t_q_auto_out_ocean = np.where(land_mask[None,:,None,:] == 1.0, t_q_auto_out, np.nan)
    t_q_auto_out_land = np.where(land_mask[None,:,None,:] == 0.0, t_q_auto_out, np.nan)
    t_q_sed_flux_tot_ocean = np.where(land_mask[None,:,None,:] == 1.0, t_q_sed_flux_tot, np.nan)
    t_q_sed_flux_tot_land = np.where(land_mask[None,:,None,:] == 0.0, t_q_sed_flux_tot, np.nan)

    p_Tout_ocean = np.where(land_mask[None,:,None,:] == 1.0, p_Tout, np.nan)
    p_Tout_land = np.where(land_mask[None,:,None,:] == 0.0, p_Tout, np.nan)
    p_T_adv_out_ocean = np.where(land_mask[None,:,None,:] == 1.0, p_T_adv_out, np.nan)
    p_T_adv_out_land = np.where(land_mask[None,:,None,:] == 0.0, p_T_adv_out, np.nan)
    p_q_adv_out_ocean = np.where(land_mask[None,:,None,:] == 1.0, p_q_adv_out, np.nan)
    p_q_adv_out_land = np.where(land_mask[None,:,None,:] == 0.0, p_q_adv_out, np.nan)
    p_q_auto_out_ocean = np.where(land_mask[None,:,None,:] == 1.0, p_q_auto_out, np.nan)
    p_q_auto_out_land = np.where(land_mask[None,:,None,:] == 0.0, p_q_auto_out, np.nan)
    p_q_sed_flux_tot_ocean = np.where(land_mask[None,:,None,:] == 1.0, p_q_sed_flux_tot, np.nan)
    p_q_sed_flux_tot_land = np.where(land_mask[None,:,None,:] == 0.0, p_q_sed_flux_tot, np.nan)

    #get ocean and land lon means for R2 figures -- longitude

    t_Tout_lon_mean_ocean = np.nanmean(t_Tout_ocean, axis=lon_axis)
    t_T_adv_out_lon_mean_ocean = np.nanmean(t_T_adv_out_ocean, axis=lon_axis)
    t_q_adv_out_lon_mean_ocean = np.nanmean(t_q_adv_out_ocean, axis=lon_axis)
    t_q_auto_out_lon_mean_ocean = np.nanmean(t_q_auto_out_ocean, axis=lon_axis)
    t_q_sed_flux_tot_lon_mean_ocean = np.nanmean(t_q_sed_flux_tot_ocean, axis=lon_axis)

    p_Tout_lon_mean_ocean = np.nanmean(p_Tout_ocean, axis=lon_axis)
    p_T_adv_out_lon_mean_ocean = np.nanmean(p_T_adv_out_ocean, axis=lon_axis)
    p_q_adv_out_lon_mean_ocean = np.nanmean(p_q_adv_out_ocean, axis=lon_axis)
    p_q_auto_out_lon_mean_ocean = np.nanmean(p_q_auto_out_ocean, axis=lon_axis)
    p_q_sed_flux_tot_lon_mean_ocean = np.nanmean(p_q_sed_flux_tot_ocean, axis=lon_axis)

    t_Tout_lon_mean_land = np.nanmean(t_Tout_land, axis=lon_axis)
    t_T_adv_out_lon_mean_land = np.nanmean(t_T_adv_out_land, axis=lon_axis)
    t_q_adv_out_lon_mean_land = np.nanmean(t_q_adv_out_land, axis=lon_axis)
    t_q_auto_out_lon_mean_land = np.nanmean(t_q_auto_out_land, axis=lon_axis)
    t_q_sed_flux_tot_lon_mean_land = np.nanmean(t_q_sed_flux_tot_land, axis=lon_axis)

    p_Tout_lon_mean_land = np.nanmean(p_Tout_land, axis=lon_axis)
    p_T_adv_out_lon_mean_land = np.nanmean(p_T_adv_out_land, axis=lon_axis)
    p_q_adv_out_lon_mean_land = np.nanmean(p_q_adv_out_land, axis=lon_axis)
    p_q_auto_out_lon_mean_land = np.nanmean(p_q_auto_out_land, axis=lon_axis)
    p_q_sed_flux_tot_lon_mean_land = np.nanmean(p_q_sed_flux_tot_land, axis=lon_axis)


    #all 
    t_Tout_vertical_mean = np.mean(t_Tout, axis=(lon_axis,lat_axis,time_axis))
    t_T_adv_out_vertical_mean = np.mean(t_T_adv_out, axis=(lon_axis,lat_axis,time_axis))
    t_q_adv_out_vertical_mean = np.mean(t_q_adv_out, axis=(lon_axis,lat_axis,time_axis))
    t_q_auto_out_vertical_mean = np.mean(t_q_auto_out, axis=(lon_axis,lat_axis,time_axis))
    t_q_sed_flux_tot_vertical_mean = np.mean(t_q_sed_flux_tot, axis=(lon_axis,lat_axis,time_axis))
    p_Tout_vertical_mean = np.mean(p_Tout, axis=(lon_axis,lat_axis,time_axis))
    p_T_adv_out_vertical_mean = np.mean(p_T_adv_out, axis=(lon_axis,lat_axis,time_axis))
    p_q_adv_out_vertical_mean = np.mean(p_q_adv_out, axis=(lon_axis,lat_axis,time_axis))
    p_q_auto_out_vertical_mean = np.mean(p_q_auto_out, axis=(lon_axis,lat_axis,time_axis))
    p_q_sed_flux_tot_vertical_mean = np.mean(p_q_sed_flux_tot, axis=(lon_axis,lat_axis,time_axis))

    t_Tout_vertical_std = np.std(t_Tout, axis=(lon_axis,lat_axis,time_axis))
    t_T_adv_out_vertical_std = np.std(t_T_adv_out, axis=(lon_axis,lat_axis,time_axis))
    t_q_adv_out_vertical_std = np.std(t_q_adv_out, axis=(lon_axis,lat_axis,time_axis))
    t_q_auto_out_vertical_std = np.std(t_q_auto_out, axis=(lon_axis,lat_axis,time_axis))
    t_q_sed_flux_tot_vertical_std = np.std(t_q_sed_flux_tot, axis=(lon_axis,lat_axis,time_axis))
    p_Tout_vertical_std = np.std(p_Tout, axis=(lon_axis,lat_axis,time_axis))
    p_T_adv_out_vertical_std = np.std(p_T_adv_out, axis=(lon_axis,lat_axis,time_axis))
    p_q_adv_out_vertical_std = np.std(p_q_adv_out, axis=(lon_axis,lat_axis,time_axis))
    p_q_auto_out_vertical_std = np.std(p_q_auto_out, axis=(lon_axis,lat_axis,time_axis))
    p_q_sed_flux_tot_vertical_std = np.std(p_q_sed_flux_tot, axis=(lon_axis,lat_axis,time_axis))

    #ocean
    t_Tout_vertical_mean_ocean = np.nanmean(t_Tout_ocean, axis=(lon_axis,lat_axis,time_axis))
    t_T_adv_out_vertical_mean_ocean = np.nanmean(t_T_adv_out_ocean, axis=(lon_axis,lat_axis,time_axis))
    t_q_adv_out_vertical_mean_ocean = np.nanmean(t_q_adv_out_ocean, axis=(lon_axis,lat_axis,time_axis))
    t_q_auto_out_vertical_mean_ocean = np.nanmean(t_q_auto_out_ocean, axis=(lon_axis,lat_axis,time_axis))
    t_q_sed_flux_tot_vertical_mean_ocean = np.nanmean(t_q_sed_flux_tot_ocean, axis=(lon_axis,lat_axis,time_axis))
    p_Tout_vertical_mean_ocean = np.nanmean(p_Tout_ocean, axis=(lon_axis,lat_axis,time_axis))
    p_T_adv_out_vertical_mean_ocean = np.nanmean(p_T_adv_out_ocean, axis=(lon_axis,lat_axis,time_axis))
    p_q_adv_out_vertical_mean_ocean = np.nanmean(p_q_adv_out_ocean, axis=(lon_axis,lat_axis,time_axis))
    p_q_auto_out_vertical_mean_ocean = np.nanmean(p_q_auto_out_ocean, axis=(lon_axis,lat_axis,time_axis))
    p_q_sed_flux_tot_vertical_mean_ocean = np.nanmean(p_q_sed_flux_tot_ocean, axis=(lon_axis,lat_axis,time_axis))

    t_Tout_vertical_std_ocean = np.nanstd(t_Tout_ocean, axis=(lon_axis,lat_axis,time_axis))
    t_T_adv_out_vertical_std_ocean = np.nanstd(t_T_adv_out_ocean, axis=(lon_axis,lat_axis,time_axis))
    t_q_adv_out_vertical_std_ocean = np.nanstd(t_q_adv_out_ocean, axis=(lon_axis,lat_axis,time_axis))
    t_q_auto_out_vertical_std_ocean = np.nanstd(t_q_auto_out_ocean, axis=(lon_axis,lat_axis,time_axis))
    t_q_sed_flux_tot_vertical_std_ocean = np.nanstd(t_q_sed_flux_tot_ocean, axis=(lon_axis,lat_axis,time_axis))
    p_Tout_vertical_std_ocean = np.nanstd(p_Tout_ocean, axis=(lon_axis,lat_axis,time_axis))
    p_T_adv_out_vertical_std_ocean = np.nanstd(p_T_adv_out_ocean, axis=(lon_axis,lat_axis,time_axis))
    p_q_adv_out_vertical_std_ocean = np.nanstd(p_q_adv_out_ocean, axis=(lon_axis,lat_axis,time_axis))
    p_q_auto_out_vertical_std_ocean = np.nanstd(p_q_auto_out_ocean, axis=(lon_axis,lat_axis,time_axis))
    p_q_sed_flux_tot_vertical_std_ocean = np.nanstd(p_q_sed_flux_tot_ocean, axis=(lon_axis,lat_axis,time_axis))

    #land
    t_Tout_vertical_mean_land = np.nanmean(t_Tout_land, axis=(lon_axis,lat_axis,time_axis))
    t_T_adv_out_vertical_mean_land = np.nanmean(t_T_adv_out_land, axis=(lon_axis,lat_axis,time_axis))
    t_q_adv_out_vertical_mean_land = np.nanmean(t_q_adv_out_land, axis=(lon_axis,lat_axis,time_axis))
    t_q_auto_out_vertical_mean_land = np.nanmean(t_q_auto_out_land, axis=(lon_axis,lat_axis,time_axis))
    t_q_sed_flux_tot_vertical_mean_land = np.nanmean(t_q_sed_flux_tot_land, axis=(lon_axis,lat_axis,time_axis))
    p_Tout_vertical_mean_land = np.nanmean(p_Tout_land, axis=(lon_axis,lat_axis,time_axis))
    p_T_adv_out_vertical_mean_land = np.nanmean(p_T_adv_out_land, axis=(lon_axis,lat_axis,time_axis))
    p_q_adv_out_vertical_mean_land = np.nanmean(p_q_adv_out_land, axis=(lon_axis,lat_axis,time_axis))
    p_q_auto_out_vertical_mean_land = np.nanmean(p_q_auto_out_land, axis=(lon_axis,lat_axis,time_axis))
    p_q_sed_flux_tot_vertical_mean_land = np.nanmean(p_q_sed_flux_tot_land, axis=(lon_axis,lat_axis,time_axis))

    t_Tout_vertical_std_land = np.nanstd(t_Tout_land, axis=(lon_axis,lat_axis,time_axis))
    t_T_adv_out_vertical_std_land = np.nanstd(t_T_adv_out_land, axis=(lon_axis,lat_axis,time_axis))
    t_q_adv_out_vertical_std_land = np.nanstd(t_q_adv_out_land, axis=(lon_axis,lat_axis,time_axis))
    t_q_auto_out_vertical_std_land = np.nanstd(t_q_auto_out_land, axis=(lon_axis,lat_axis,time_axis))
    t_q_sed_flux_tot_vertical_std_land = np.nanstd(t_q_sed_flux_tot_land, axis=(lon_axis,lat_axis,time_axis))
    p_Tout_vertical_std_land = np.nanstd(p_Tout_land, axis=(lon_axis,lat_axis,time_axis))
    p_T_adv_out_vertical_std_land = np.nanstd(p_T_adv_out_land, axis=(lon_axis,lat_axis,time_axis))
    p_q_adv_out_vertical_std_land = np.nanstd(p_q_adv_out_land, axis=(lon_axis,lat_axis,time_axis))
    p_q_auto_out_vertical_std_land = np.nanstd(p_q_auto_out_land, axis=(lon_axis,lat_axis,time_axis))
    p_q_sed_flux_tot_vertical_std_land = np.nanstd(p_q_sed_flux_tot_land, axis=(lon_axis,lat_axis,time_axis))
  
    #organize the data
    mean_data_list_truth = [t_Tout_vertical_mean, t_T_adv_out_vertical_mean, t_q_adv_out_vertical_mean,
                       t_q_auto_out_vertical_mean, t_q_sed_flux_tot_vertical_mean]
    mean_data_list_pred = [p_Tout_vertical_mean, p_T_adv_out_vertical_mean, p_q_adv_out_vertical_mean,
                       p_q_auto_out_vertical_mean, p_q_sed_flux_tot_vertical_mean]
    std_data_list_truth = [t_Tout_vertical_std, t_T_adv_out_vertical_std, t_q_adv_out_vertical_std,
                       t_q_auto_out_vertical_std, t_q_sed_flux_tot_vertical_std]
    std_data_list_pred = [p_Tout_vertical_std, p_T_adv_out_vertical_std, p_q_adv_out_vertical_std,
                       p_q_auto_out_vertical_std, p_q_sed_flux_tot_vertical_std]

    mean_data_list_truth_ocean = [t_Tout_vertical_mean_ocean, t_T_adv_out_vertical_mean_ocean, 
                              t_q_adv_out_vertical_mean_ocean,
                       t_q_auto_out_vertical_mean_ocean, t_q_sed_flux_tot_vertical_mean_ocean]
    mean_data_list_pred_ocean = [p_Tout_vertical_mean_ocean, p_T_adv_out_vertical_mean_ocean, 
                             p_q_adv_out_vertical_mean_ocean,
                       p_q_auto_out_vertical_mean_ocean, p_q_sed_flux_tot_vertical_mean_ocean]
    std_data_list_truth_ocean = [t_Tout_vertical_std_ocean, t_T_adv_out_vertical_std_ocean, 
                             t_q_adv_out_vertical_std_ocean,
                       t_q_auto_out_vertical_std_ocean, t_q_sed_flux_tot_vertical_std_ocean]
    std_data_list_pred_ocean = [p_Tout_vertical_std_ocean, p_T_adv_out_vertical_std_ocean, 
                            p_q_adv_out_vertical_std_ocean,
                       p_q_auto_out_vertical_std_ocean, p_q_sed_flux_tot_vertical_std_ocean]

    mean_data_list_truth_land = [t_Tout_vertical_mean_land, t_T_adv_out_vertical_mean_land, 
                              t_q_adv_out_vertical_mean_land,
                       t_q_auto_out_vertical_mean_land, t_q_sed_flux_tot_vertical_mean_land]
    mean_data_list_pred_land = [p_Tout_vertical_mean_land, p_T_adv_out_vertical_mean_land, 
                             p_q_adv_out_vertical_mean_land,
                       p_q_auto_out_vertical_mean_land, p_q_sed_flux_tot_vertical_mean_land]
    std_data_list_truth_land = [t_Tout_vertical_std_land, t_T_adv_out_vertical_std_land, 
                             t_q_adv_out_vertical_std_land,
                       t_q_auto_out_vertical_std_land, t_q_sed_flux_tot_vertical_std_land]
    std_data_list_pred_land = [p_Tout_vertical_std_land, p_T_adv_out_vertical_std_land, 
                            p_q_adv_out_vertical_std_land,
                       p_q_auto_out_vertical_std_land, p_q_sed_flux_tot_vertical_std_land]


    #plot vertical profiles
    mean_std_vertical_level_comparison(mean_data_list_truth=mean_data_list_truth, 
                                   mean_data_list_pred=mean_data_list_pred, 
                                   std_data_list_truth=std_data_list_truth, 
                                   std_data_list_pred=std_data_list_pred,
                                       name_list=var_names, 
                                   z=sigma[:49], 
                                   super_title="Offline Column Means and Stds", 
                                        x_labels=['','','','',''], 
                                   y_label='hPa',
                                   base_dir=save_path+nametag,
                                   save_dir="/1D_Statistics",
                                   landsea=True, 
                                   mean_land_truth=mean_data_list_truth_land, 
                                   mean_land_pred=mean_data_list_pred_land,
                                       mean_ocean_truth=mean_data_list_truth_ocean, 
                                   mean_ocean_pred=mean_data_list_pred_ocean,
                                       std_land_truth=std_data_list_truth_land, 
                                   std_land_pred=std_data_list_pred_land, 
                                       std_ocean_truth=std_data_list_truth_ocean, 
                                   std_ocean_pred=std_data_list_pred_ocean,
                                       xlim_min=None, 
                                   xlim_max=None,
                                        ylim_min=None, 
                                   ylim_max=None, 
                                   inverted_y=True)

    #Calculate R2
    sse_Tout_lon_mean = ((t_Tout_lon_mean - p_Tout_lon_mean)**2).sum(axis=time_axis)
    sse_T_adv_out_lon_mean = ((t_T_adv_out_lon_mean - p_T_adv_out_lon_mean)**2).sum(axis=time_axis)
    sse_q_adv_out_lon_mean = ((t_q_adv_out_lon_mean - p_q_adv_out_lon_mean)**2).sum(axis=time_axis)
    sse_q_auto_out_lon_mean = ((t_q_auto_out_lon_mean - p_q_auto_out_lon_mean)**2).sum(axis=time_axis)
    sse_q_sed_flux_tot_lon_mean = ((t_q_sed_flux_tot_lon_mean - p_q_sed_flux_tot_lon_mean)**2).sum(axis=time_axis)

    sse_Tout_lon_mean_ocean = np.nansum(((t_Tout_lon_mean_ocean - p_Tout_lon_mean_ocean)**2), axis=time_axis)
    sse_T_adv_out_lon_mean_ocean = np.nansum(((t_T_adv_out_lon_mean_ocean - p_T_adv_out_lon_mean_ocean)**2), axis=time_axis)
    sse_q_adv_out_lon_mean_ocean = np.nansum(((t_q_adv_out_lon_mean_ocean - p_q_adv_out_lon_mean_ocean)**2), axis=time_axis)
    sse_q_auto_out_lon_mean_ocean = np.nansum(((t_q_auto_out_lon_mean_ocean - p_q_auto_out_lon_mean_ocean)**2), axis=time_axis)
    sse_q_sed_flux_tot_lon_mean_ocean = np.nansum(((t_q_sed_flux_tot_lon_mean_ocean - p_q_sed_flux_tot_lon_mean_ocean)**2), axis=time_axis)

    sse_Tout_lon_mean_land = np.nansum(((t_Tout_lon_mean_land - p_Tout_lon_mean_land)**2), axis=time_axis)
    sse_T_adv_out_lon_mean_land = np.nansum(((t_T_adv_out_lon_mean_land - p_T_adv_out_lon_mean_land)**2), axis=time_axis)
    sse_q_adv_out_lon_mean_land = np.nansum(((t_q_adv_out_lon_mean_land - p_q_adv_out_lon_mean_land)**2), axis=time_axis)
    sse_q_auto_out_lon_mean_land = np.nansum(((t_q_auto_out_lon_mean_land - p_q_auto_out_lon_mean_land)**2), axis=time_axis)
    sse_q_sed_flux_tot_lon_mean_land = np.nansum(((t_q_sed_flux_tot_lon_mean_land - p_q_sed_flux_tot_lon_mean_land)**2), axis=time_axis)

    #was [:,None]
    svar_Tout_lon_mean = np.sum((t_Tout_lon_mean - t_Tout_lon_mean.mean(axis=time_axis)[:,:,None])**2, axis=time_axis)
    svar_T_adv_out_lon_mean = np.sum((t_T_adv_out_lon_mean - t_T_adv_out_lon_mean.mean(axis=time_axis)[:,:,None])**2, axis=time_axis)
    svar_q_adv_out_lon_mean = np.sum((t_q_adv_out_lon_mean - t_q_adv_out_lon_mean.mean(axis=time_axis)[:,:,None])**2, axis=time_axis)
    svar_q_auto_out_lon_mean = np.sum((t_q_auto_out_lon_mean - t_q_auto_out_lon_mean.mean(axis=time_axis)[:,:,None])**2, axis=time_axis)
    svar_q_sed_flux_tot_lon_mean = np.sum((t_q_sed_flux_tot_lon_mean - t_q_sed_flux_tot_lon_mean.mean(axis=time_axis)[:,:,None])**2, axis=time_axis)

    svar_Tout_lon_mean_ocean = np.nansum((t_Tout_lon_mean_ocean - np.nanmean(t_Tout_lon_mean_ocean, axis=time_axis)[:,:,None])**2, axis=time_axis)
    svar_T_adv_out_lon_mean_ocean = np.nansum((t_T_adv_out_lon_mean_ocean - np.nanmean(t_T_adv_out_lon_mean_ocean, axis=time_axis)[:,:,None])**2, axis=time_axis)
    svar_q_adv_out_lon_mean_ocean = np.nansum((t_q_adv_out_lon_mean_ocean - np.nanmean(t_q_adv_out_lon_mean_ocean, axis=time_axis)[:,:,None])**2, axis=time_axis)
    svar_q_auto_out_lon_mean_ocean = np.nansum((t_q_auto_out_lon_mean_ocean - np.nanmean(t_q_auto_out_lon_mean_ocean, axis=time_axis)[:,:,None])**2, axis=time_axis)
    svar_q_sed_flux_tot_lon_mean_ocean = np.nansum((t_q_sed_flux_tot_lon_mean_ocean - np.nanmean(t_q_sed_flux_tot_lon_mean_ocean, axis=time_axis)[:,:,None])**2, axis=time_axis)

    svar_Tout_lon_mean_land = np.nansum((t_Tout_lon_mean_land - np.nanmean(t_Tout_lon_mean_land, axis=time_axis)[:,:,None])**2, axis=time_axis)
    svar_T_adv_out_lon_mean_land = np.nansum((t_T_adv_out_lon_mean_land - np.nanmean(t_T_adv_out_lon_mean_land, axis=time_axis)[:,:,None])**2, axis=time_axis)
    svar_q_adv_out_lon_mean_land = np.nansum((t_q_adv_out_lon_mean_land - np.nanmean(t_q_adv_out_lon_mean_land, axis=time_axis)[:,:,None])**2, axis=time_axis)
    svar_q_auto_out_lon_mean_land = np.nansum((t_q_auto_out_lon_mean_land - np.nanmean(t_q_auto_out_lon_mean_land, axis=time_axis)[:,:,None])**2, axis=time_axis)
    svar_q_sed_flux_tot_lon_mean_land = np.nansum((t_q_sed_flux_tot_lon_mean_land - np.nanmean(t_q_sed_flux_tot_lon_mean_land, axis=time_axis)[:,:,None])**2, axis=time_axis)


    R2_Tout_lon_mean = 1- (sse_Tout_lon_mean/svar_Tout_lon_mean)
    R2_T_adv_out_lon_mean = 1 - (sse_T_adv_out_lon_mean/svar_T_adv_out_lon_mean)
    R2_q_adv_out_lon_mean = 1 - (sse_q_adv_out_lon_mean/svar_q_adv_out_lon_mean)
    R2_q_auto_out_lon_mean = 1 - (sse_q_auto_out_lon_mean/svar_q_auto_out_lon_mean) 
    R2_q_sed_flux_tot_lon_mean = 1 - (sse_q_sed_flux_tot_lon_mean/svar_q_sed_flux_tot_lon_mean)

    R2_Tout_lon_mean_ocean = 1- (sse_Tout_lon_mean_ocean/svar_Tout_lon_mean_ocean)
    R2_T_adv_out_lon_mean_ocean = 1 - (sse_T_adv_out_lon_mean_ocean/svar_T_adv_out_lon_mean_ocean)
    R2_q_adv_out_lon_mean_ocean = 1 - (sse_q_adv_out_lon_mean_ocean/svar_q_adv_out_lon_mean_ocean)
    R2_q_auto_out_lon_mean_ocean = 1 - (sse_q_auto_out_lon_mean_ocean/svar_q_auto_out_lon_mean_ocean) 
    R2_q_sed_flux_tot_lon_mean_ocean = 1 - (sse_q_sed_flux_tot_lon_mean_ocean/svar_q_sed_flux_tot_lon_mean_ocean)

    R2_Tout_lon_mean_land = 1- (sse_Tout_lon_mean_land/svar_Tout_lon_mean_land)
    R2_T_adv_out_lon_mean_land = 1 - (sse_T_adv_out_lon_mean_land/svar_T_adv_out_lon_mean_land)
    R2_q_adv_out_lon_mean_land = 1 - (sse_q_adv_out_lon_mean_land/svar_q_adv_out_lon_mean_land)
    R2_q_auto_out_lon_mean_land = 1 - (sse_q_auto_out_lon_mean_land/svar_q_auto_out_lon_mean_land) 
    R2_q_sed_flux_tot_lon_mean_land = 1 - (sse_q_sed_flux_tot_lon_mean_land/svar_q_sed_flux_tot_lon_mean_land)


    field_list_all = [R2_Tout_lon_mean, R2_T_adv_out_lon_mean, R2_q_adv_out_lon_mean, 
              R2_q_auto_out_lon_mean, R2_q_sed_flux_tot_lon_mean]

    field_list_ocean = [R2_Tout_lon_mean_ocean, R2_T_adv_out_lon_mean_ocean, R2_q_adv_out_lon_mean_ocean, 
              R2_q_auto_out_lon_mean_ocean, R2_q_sed_flux_tot_lon_mean_ocean]

    field_list_land = [R2_Tout_lon_mean_land, R2_T_adv_out_lon_mean_land, R2_q_adv_out_lon_mean_land,
              R2_q_auto_out_lon_mean_land, R2_q_sed_flux_tot_lon_mean_land]


    five_panel_lat_pressure_cross_section(field_list=field_list_all, 
                                      x_values=lat_plot, 
                                      y_values=sigma_plot,  
                                        xlabel = "Latitudes", 
                                      ylabel="Pressure (sigma)", 
                                      title_list = var_names, 
                                    cbar_label= r"$R^2$", 
                                      super_title= "Offline Fit", 
                                      base_dir=save_path+nametag,
                                      save_dir="/R2_Figures",
                                      identifier="All_R2_",
                                      vertical_splice = z_dim,  
                                      )

    five_panel_lat_pressure_cross_section(field_list=field_list_ocean, 
                                      x_values=lat_plot, 
                                      y_values=sigma_plot,  
                                        xlabel = "Latitudes", 
                                      ylabel="Pressure (sigma)", 
                                      title_list = var_names, 
                                    cbar_label= r"$R^2$", 
                                      super_title= "Offline Fit", 
                                      base_dir=save_path+nametag,
                                      save_dir="/R2_Figures",
                                      identifier="Ocean_R2_",
                                      vertical_splice = z_dim,  
                                      )


    five_panel_lat_pressure_cross_section(field_list=field_list_land, 
                                      x_values=lat_plot, 
                                      y_values=sigma_plot,  
                                        xlabel = "Latitudes", 
                                      ylabel="Pressure (sigma)", 
                                      title_list = var_names, 
                                    cbar_label= r"$R^2$", 
                                      super_title= "Offline Fit", 
                                      base_dir=save_path+nametag,
                                      save_dir="/R2_Figures",
                                      identifier="Land_R2_",
                                      vertical_splice = z_dim,  
                                      )

    #Calculate R2 with no z
    sse_Tout_z_mean = ((t_Tout_z_mean - p_Tout_z_mean)**2).sum(axis=1)
    sse_T_adv_out_z_mean = ((t_T_adv_out_z_mean - p_T_adv_out_z_mean)**2).sum(axis=1)
    sse_q_adv_out_z_mean = ((t_q_adv_out_z_mean - p_q_adv_out_z_mean)**2).sum(axis=1)
    sse_q_auto_out_z_mean = ((t_q_auto_out_z_mean - p_q_auto_out_z_mean)**2).sum(axis=1)
    sse_q_sed_flux_tot_z_mean = ((t_q_sed_flux_tot_z_mean - p_q_sed_flux_tot_z_mean)**2).sum(axis=1)

    #was [:,None]
    svar_Tout_z_mean = np.sum((t_Tout_z_mean - t_Tout_z_mean.mean(axis=time_axis)[:,:,None])**2, axis=1)
    svar_T_adv_out_z_mean = np.sum((t_T_adv_out_z_mean - t_T_adv_out_z_mean.mean(axis=time_axis)[:,:,None])**2, axis=1)
    svar_q_adv_out_z_mean = np.sum((t_q_adv_out_z_mean - p_q_adv_out_z_mean.mean(axis=time_axis)[:,:,None])**2, axis=1)
    svar_q_auto_out_z_mean = np.sum((t_q_auto_out_z_mean - p_q_auto_out_z_mean.mean(axis=time_axis)[:,:,None])**2, axis=1)
    svar_q_sed_flux_tot_z_mean = np.sum((t_q_sed_flux_tot_z_mean - p_q_sed_flux_tot_z_mean.mean(axis=time_axis)[:,:,None])**2, axis=1)

    R2_Tout_z_mean = 1- (sse_Tout_z_mean/svar_Tout_z_mean)
    R2_T_adv_out_z_mean = 1 - (sse_T_adv_out_z_mean/svar_T_adv_out_z_mean)
    R2_q_adv_out_z_mean = 1 - (sse_q_adv_out_z_mean/svar_q_adv_out_z_mean)
    R2_q_auto_out_z_mean = 1 - (sse_q_auto_out_z_mean/svar_q_auto_out_z_mean) 
    R2_q_sed_flux_tot_z_mean = 1 - (sse_q_sed_flux_tot_z_mean/svar_q_sed_flux_tot_z_mean)

    print("shape of R2_Tout_z_mean", R2_Tout_z_mean.shape)
    print("shape of R2_T_adv_out_z_mean", R2_T_adv_out_z_mean.shape)
    print("shape of R2_q_adv_out_z_mean", R2_q_adv_out_z_mean.shape)
    print("shape of R2_q_auto_out_z_mean", R2_q_auto_out_z_mean.shape)
    print("shape of R2_q_sed_flux_tot_z_mean", R2_q_sed_flux_tot_z_mean.shape)
    print("shape of lon_mesh", lon_mesh.shape)
    print("shape of lat_mesh", lat_mesh.shape)


    field_list_all_z = [ R2_Tout_z_mean, R2_T_adv_out_z_mean, R2_q_adv_out_z_mean,
              R2_q_auto_out_z_mean, R2_q_sed_flux_tot_z_mean]



    five_panel_lat_pressure_cross_section(field_list=field_list_all_z, 
                                      x_values=lon_mesh, 
                                      y_values=lat_mesh,  
                                        xlabel = "Longitudes", 
                                      ylabel="Latitudes", 
                                      title_list = var_names, 
                                    cbar_label= r"$R^2$", 
                                      super_title= "Offline Fit", 
                                      base_dir=save_path+nametag,
                                      save_dir="/R2_Figures",
                                      identifier="XY_All_R2_",
                                      vertical_splice = 1000,  # a bit hacky but should work here
                                      )



    np.save("dummy_data/R2_T_adv_out_z_mean.npy", R2_T_adv_out_z_mean)
    print(save_path + nametag)
    
    plot_r2_vs_surface_features(
        R2_2D=R2_T_adv_out_z_mean,
        land_fraction=land_mask,
        surface_height=srf_height,
        base_dir=save_path + nametag,
        save_dir="/R2_Figures",
        identifier="Lat_Lon_T_Adv_Out_R2_",
        filename_prefix="T advection out",
        title_suffix="(T_Adv_Out)"
    )

    plot_r2_vs_surface_features(
        R2_2D=R2_q_adv_out_z_mean,
        land_fraction=land_mask,
        surface_height=srf_height,
        save_dir="/R2_Figures",
        identifier="Lat_Lon_Q_Adv_Out_R2_",
        base_dir=save_path + nametag,
        filename_prefix="Q advection out",
        title_suffix="(Q_Adv_Out)"
    )



    #latitude/pressure variance plots
    Tout_mean_truth = t_Tout.mean(axis=(lon_axis,time_axis))
    Tout_mean_prediction = p_Tout.mean(axis=(lon_axis,time_axis))
    T_adv_out_mean_truth = t_T_adv_out.mean(axis=(lon_axis,time_axis))
    T_adv_out_mean_prediction = p_T_adv_out.mean(axis=(lon_axis,time_axis))
    q_adv_out_mean_truth = t_q_adv_out.mean(axis=(lon_axis,time_axis))
    q_adv_out_mean_prediction = p_q_adv_out.mean(axis=(lon_axis,time_axis))
    q_auto_out_mean_truth = t_q_auto_out.mean(axis=(lon_axis,time_axis))
    q_auto_out_mean_prediction = p_q_auto_out.mean(axis=(lon_axis,time_axis))
    q_sed_flux_tot_mean_truth = t_q_sed_flux_tot.mean(axis=(lon_axis,time_axis))
    q_sed_flux_tot_mean_prediction = p_q_sed_flux_tot.mean(axis=(lon_axis,time_axis))

    Tout_std_truth = t_Tout.std(axis=(lon_axis,time_axis))
    Tout_std_prediction = p_Tout.std(axis=(lon_axis,time_axis))
    T_adv_out_std_truth = t_T_adv_out.std(axis=(lon_axis,time_axis))
    T_adv_out_std_prediction = p_T_adv_out.std(axis=(lon_axis,time_axis))
    q_adv_out_std_truth = t_q_adv_out.std(axis=(lon_axis,time_axis))
    q_adv_out_std_prediction = p_q_adv_out.std(axis=(lon_axis,time_axis))
    q_auto_out_std_truth = t_q_auto_out.std(axis=(lon_axis,time_axis))
    q_auto_out_std_prediction = p_q_auto_out.std(axis=(lon_axis,time_axis))
    q_sed_flux_tot_std_truth = t_q_sed_flux_tot.std(axis=(lon_axis,time_axis))
    q_sed_flux_tot_std_prediction = p_q_sed_flux_tot.std(axis=(lon_axis,time_axis))

    # ocean means and stds
    Tout_mean_truth_ocean = np.nanmean(t_Tout_ocean, axis=(lon_axis,time_axis))
    Tout_mean_prediction_ocean = np.nanmean(p_Tout_ocean, axis=(lon_axis,time_axis))
    T_adv_out_mean_truth_ocean = np.nanmean(t_T_adv_out_ocean, axis=(lon_axis,time_axis))
    T_adv_out_mean_prediction_ocean = np.nanmean(p_T_adv_out_ocean, axis=(lon_axis,time_axis))
    q_adv_out_mean_truth_ocean = np.nanmean(t_q_adv_out_ocean, axis=(lon_axis,time_axis))
    q_adv_out_mean_prediction_ocean = np.nanmean(p_q_adv_out_ocean, axis=(lon_axis,time_axis))
    q_auto_out_mean_truth_ocean = np.nanmean(t_q_auto_out_ocean, axis=(lon_axis,time_axis))
    q_auto_out_mean_prediction_ocean = np.nanmean(p_q_auto_out_ocean, axis=(lon_axis,time_axis))
    q_sed_flux_tot_mean_truth_ocean = np.nanmean(t_q_sed_flux_tot_ocean, axis=(lon_axis,time_axis))
    q_sed_flux_tot_mean_prediction_ocean = np.nanmean(p_q_sed_flux_tot_ocean, axis=(lon_axis,time_axis))

    Tout_std_truth_ocean = np.nanstd(t_Tout_ocean, axis=(lon_axis,time_axis))
    Tout_std_prediction_ocean = np.nanstd(p_Tout_ocean, axis=(lon_axis,time_axis))
    T_adv_out_std_truth_ocean = np.nanstd(t_T_adv_out_ocean, axis=(lon_axis,time_axis))
    T_adv_out_std_prediction_ocean = np.nanstd(p_T_adv_out_ocean, axis=(lon_axis,time_axis))
    q_adv_out_std_truth_ocean = np.nanstd(t_q_adv_out_ocean, axis=(lon_axis,time_axis))
    q_adv_out_std_prediction_ocean = np.nanstd(p_q_adv_out_ocean, axis=(lon_axis,time_axis))
    q_auto_out_std_truth_ocean = np.nanstd(t_q_auto_out_ocean, axis=(lon_axis,time_axis))
    q_auto_out_std_prediction_ocean = np.nanstd(p_q_auto_out_ocean, axis=(lon_axis,time_axis))
    q_sed_flux_tot_std_truth_ocean = np.nanstd(t_q_sed_flux_tot_ocean, axis=(lon_axis,time_axis))
    q_sed_flux_tot_std_prediction_ocean = np.nanstd(p_q_sed_flux_tot_ocean, axis=(lon_axis,time_axis))

    # land means and stds
    Tout_mean_truth_land = np.nanmean(t_Tout_land, axis=(lon_axis,time_axis))
    Tout_mean_prediction_land = np.nanmean(p_Tout_land, axis=(lon_axis,time_axis))
    T_adv_out_mean_truth_land = np.nanmean(t_T_adv_out_land, axis=(lon_axis,time_axis))
    T_adv_out_mean_prediction_land = np.nanmean(p_T_adv_out_land, axis=(lon_axis,time_axis))
    q_adv_out_mean_truth_land = np.nanmean(t_q_adv_out_land, axis=(lon_axis,time_axis))
    q_adv_out_mean_prediction_land = np.nanmean(p_q_adv_out_land, axis=(lon_axis,time_axis))
    q_auto_out_mean_truth_land = np.nanmean(t_q_auto_out_land, axis=(lon_axis,time_axis))
    q_auto_out_mean_prediction_land = np.nanmean(p_q_auto_out_land, axis=(lon_axis,time_axis))
    q_sed_flux_tot_mean_truth_land = np.nanmean(t_q_sed_flux_tot_land, axis=(lon_axis,time_axis))
    q_sed_flux_tot_mean_prediction_land = np.nanmean(p_q_sed_flux_tot_land, axis=(lon_axis,time_axis))
    
    Tout_std_truth_land = np.nanstd(t_Tout_land, axis=(lon_axis,time_axis))
    Tout_std_prediction_land = np.nanstd(p_Tout_land, axis=(lon_axis,time_axis))
    T_adv_out_std_truth_land = np.nanstd(t_T_adv_out_land, axis=(lon_axis,time_axis))
    T_adv_out_std_prediction_land = np.nanstd(p_T_adv_out_land, axis=(lon_axis,time_axis))
    q_adv_out_std_truth_land = np.nanstd(t_q_adv_out_land, axis=(lon_axis,time_axis))
    q_adv_out_std_prediction_land = np.nanstd(p_q_adv_out_land, axis=(lon_axis,time_axis))
    q_auto_out_std_truth_land = np.nanstd(t_q_auto_out_land, axis=(lon_axis,time_axis))
    q_auto_out_std_prediction_land = np.nanstd(p_q_auto_out_land, axis=(lon_axis,time_axis))
    q_sed_flux_tot_std_truth_land = np.nanstd(t_q_sed_flux_tot_land, axis=(lon_axis,time_axis))
    q_sed_flux_tot_std_prediction_land = np.nanstd(p_q_sed_flux_tot_land, axis=(lon_axis,time_axis))

    single_variable_mean_std_plot_all(truth_all=Tout_mean_truth, 
                                  truth_ocean=Tout_mean_truth_ocean, 
                                  truth_land=Tout_mean_truth_land,
                                  pred_all=Tout_mean_prediction, 
                                  pred_ocean=Tout_mean_prediction_ocean, 
                                  pred_land=Tout_mean_prediction_land, 
                                  X=lat_plot[:,:z_dim], 
                                  Z=sigma_plot[:,:z_dim], 
                                  variable_name="Tout", 
                                  y_units='sigma', 
                                  variable_units='K',
                                  x_units='latitude', 
                                  colormap="Spectral_r", 
                                  cmap_diff='bwr', 
                                  measure='mean', 
                                  suptitle='Tout Mean Statistics', 
                                  base_dir=save_path+nametag,
                                  save_dir="/2D_Statistics",
                                  data_percentile=95,
                                  diff_percentile=90,
                                 )

    single_variable_mean_std_plot_all(truth_all=Tout_std_truth, 
                                  truth_ocean=Tout_std_truth_ocean, 
                                  truth_land=Tout_std_truth_land,
                                  pred_all=Tout_std_prediction, 
                                  pred_ocean=Tout_std_prediction_ocean, 
                                  pred_land=Tout_std_prediction_land, 
                                  X=lat_plot[:,:z_dim], 
                                  Z=sigma_plot[:,:z_dim], 
                                  variable_name="Tout", 
                                  y_units='sigma', 
                                  variable_units='K',
                                  x_units='latitude', 
                                  colormap="Spectral_r", 
                                  cmap_diff='bwr', 
                                  measure='std', 
                                  suptitle='Tout std Statistics',
                                  base_dir=save_path+nametag,
                                  save_dir="/2D_Statistics",
                                  data_percentile=95,
                                  diff_percentile=90,
                                 )

    single_variable_mean_std_plot_all(truth_all=T_adv_out_mean_truth, 
                                  truth_ocean=T_adv_out_mean_truth_ocean, 
                                  truth_land=T_adv_out_mean_truth_land,
                                  pred_all=T_adv_out_mean_prediction, 
                                  pred_ocean=T_adv_out_mean_prediction_ocean, 
                                  pred_land=T_adv_out_mean_prediction_land, 
                                  X=lat_plot[:,:z_dim], 
                                  Z=sigma_plot[:,:z_dim], 
                                  variable_name="Tadv", 
                                  y_units='sigma', 
                                  variable_units='K/s',
                                  x_units='latitude', 
                                  colormap="Spectral_r", 
                                  cmap_diff='bwr', 
                                  measure='mean', 
                                  suptitle='Tadv Mean Statistics',
                                  base_dir=save_path+nametag,
                                  save_dir="/2D_Statistics",
                                  data_percentile=95,
                                  diff_percentile=90,
                                 )

    single_variable_mean_std_plot_all(truth_all=T_adv_out_std_truth, 
                                  truth_ocean=T_adv_out_std_truth_ocean, 
                                  truth_land=T_adv_out_std_truth_land,
                                  pred_all=T_adv_out_std_prediction, 
                                  pred_ocean=T_adv_out_std_prediction_ocean, 
                                  pred_land=T_adv_out_std_prediction_land, 
                                  X=lat_plot[:,:z_dim], 
                                  Z=sigma_plot[:,:z_dim], 
                                  variable_name="Tadv", 
                                  y_units='sigma', 
                                  variable_units='K/s',
                                  x_units='latitude', 
                                  colormap="Spectral_r", 
                                  cmap_diff='bwr', 
                                  measure='std', 
                                  suptitle='Tadv std Statistics', 
                                  base_dir=save_path+nametag,
                                  save_dir="/2D_Statistics",
                                  data_percentile=95,
                                  diff_percentile=90,
                                 )

    single_variable_mean_std_plot_all(truth_all=q_adv_out_mean_truth, 
                                  truth_ocean=q_adv_out_mean_truth_ocean, 
                                  truth_land=q_adv_out_mean_truth_land,
                                  pred_all=q_adv_out_mean_prediction, 
                                  pred_ocean=q_adv_out_mean_prediction_ocean, 
                                  pred_land=q_adv_out_mean_prediction_land, 
                                  X=lat_plot[:,:z_dim], 
                                  Z=sigma_plot[:,:z_dim], 
                                  variable_name="Qadv", 
                                  y_units='sigma', 
                                  variable_units='kg/kg/s',
                                  x_units='latitude', 
                                  colormap="Spectral_r", 
                                  cmap_diff='bwr', 
                                  measure='mean', 
                                  suptitle='Qadv Mean Statistics', 
                                  base_dir=save_path+nametag,
                                  save_dir="/2D_Statistics",
                                  data_percentile=95,
                                  diff_percentile=90,
                                 )


    single_variable_mean_std_plot_all(truth_all=q_adv_out_std_truth, 
                                  truth_ocean=q_adv_out_std_truth_ocean, 
                                  truth_land=q_adv_out_std_truth_land,
                                  pred_all=q_adv_out_std_prediction, 
                                  pred_ocean=q_adv_out_std_prediction_ocean, 
                                  pred_land=q_adv_out_std_prediction_land, 
                                  X=lat_plot[:,:z_dim], 
                                  Z=sigma_plot[:,:z_dim], 
                                  variable_name="Qadv", 
                                  y_units='sigma', 
                                  variable_units='kg/kg/s',
                                  x_units='latitude', 
                                  colormap="Spectral_r", 
                                  cmap_diff='bwr', 
                                  measure='std', 
                                  suptitle='Qadv std Statistics', 
                                  base_dir=save_path+nametag,
                                  save_dir="/2D_Statistics",
                                  data_percentile=95,
                                  diff_percentile=90,
                                 )

    single_variable_mean_std_plot_all(truth_all=q_auto_out_mean_truth, 
                                  truth_ocean=q_auto_out_mean_truth_ocean, 
                                  truth_land=q_auto_out_mean_truth_land,
                                  pred_all=q_auto_out_mean_prediction, 
                                  pred_ocean=q_auto_out_mean_prediction_ocean, 
                                  pred_land=q_auto_out_mean_prediction_land, 
                                  X=lat_plot[:,:z_dim], 
                                  Z=sigma_plot[:,:z_dim], 
                                  variable_name="Qauto", 
                                  y_units='sigma', 
                                  variable_units='kg/kg/s',
                                  x_units='latitude', 
                                  colormap="Spectral_r", 
                                  cmap_diff='bwr', 
                                  measure='mean', 
                                  suptitle='Qauto Mean Statistics', 
                                  base_dir=save_path+nametag,
                                  save_dir="/2D_Statistics",
                                  data_percentile=95,
                                  diff_percentile=90,
                                 )


    single_variable_mean_std_plot_all(truth_all=q_auto_out_std_truth, 
                                  truth_ocean=q_auto_out_std_truth_ocean, 
                                  truth_land=q_auto_out_std_truth_land,
                                  pred_all=q_auto_out_std_prediction, 
                                  pred_ocean=q_auto_out_std_prediction_ocean, 
                                  pred_land=q_auto_out_std_prediction_land, 
                                  X=lat_plot[:,:z_dim], 
                                  Z=sigma_plot[:,:z_dim], 
                                  variable_name="Qauto", 
                                  y_units='sigma', 
                                  variable_units='kg/kg/s',
                                  x_units='latitude', 
                                  colormap="Spectral_r", 
                                  cmap_diff='bwr', 
                                  measure='std', 
                                  suptitle='Qauto std Statistics',
                                  base_dir=save_path+nametag,
                                  save_dir="/2D_Statistics",
                                  data_percentile=95,
                                  diff_percentile=90,
                                 )



    single_variable_mean_std_plot_all(truth_all=q_sed_flux_tot_mean_truth, 
                                  truth_ocean=q_sed_flux_tot_mean_truth_ocean, 
                                  truth_land=q_sed_flux_tot_mean_truth_land,
                                  pred_all=q_sed_flux_tot_mean_prediction, 
                                  pred_ocean=q_sed_flux_tot_mean_prediction_ocean, 
                                  pred_land=q_sed_flux_tot_mean_prediction_land, 
                                  X=lat_plot[:,:z_dim], 
                                  Z=sigma_plot[:,:z_dim], 
                                  variable_name="Qsed", 
                                  y_units='sigma', 
                                  variable_units='kg/kg/s',
                                  x_units='latitude', 
                                  colormap="Spectral_r", 
                                  cmap_diff='bwr', 
                                  measure='mean', 
                                  suptitle='Qsed Mean Statistics', 
                                  base_dir=save_path+nametag,
                                  save_dir="/2D_Statistics",
                                  data_percentile=95,
                                  diff_percentile=90,
                                 )

    single_variable_mean_std_plot_all(truth_all=q_sed_flux_tot_std_truth, 
                                  truth_ocean=q_sed_flux_tot_std_truth_ocean, 
                                  truth_land=q_sed_flux_tot_std_truth_land,
                                  pred_all=q_sed_flux_tot_std_prediction, 
                                  pred_ocean=q_sed_flux_tot_std_prediction_ocean, 
                                  pred_land=q_sed_flux_tot_std_prediction_land, 
                                  X=lat_plot[:,:z_dim], 
                                  Z=sigma_plot[:,:z_dim], 
                                  variable_name="Qsed", 
                                  y_units='sigma', 
                                  variable_units='kg/kg/s',
                                  x_units='latitude', 
                                  colormap="Spectral_r", 
                                  cmap_diff='bwr', 
                                  measure='std', 
                                  suptitle='Qsed std Statistics', 
                                  base_dir=save_path+nametag,
                                  save_dir="/2D_Statistics",
                                  data_percentile=95,
                                  diff_percentile=90,
                                 )


    frequency_tadv_truth, pd_tadv_truth = spectral_analysis_numpy(
        signal=t_T_adv_out,
        step=1,
        dim=time_axis,
    )

    frequency_tadv_pred, pd_tadv_pred = spectral_analysis_numpy(
        signal=p_T_adv_out,
        step=1,
        dim=time_axis,
    )

    frequency_qadv_truth, pd_qadv_truth = spectral_analysis_numpy(
        signal=t_q_adv_out,
        step=1,
        dim=time_axis,
    )

    frequency_qadv_pred, pd_qadv_pred = spectral_analysis_numpy(
        signal=p_q_adv_out,
        step=1,
        dim=time_axis,
    )


    frequency_tadv_truth_ocean, pd_tadv_truth_ocean = spectral_analysis_numpy(
        signal=t_T_adv_out_ocean,
        step=1,
        dim=time_axis,
    )

    frequency_tadv_pred_ocean, pd_tadv_pred_ocean = spectral_analysis_numpy(
        signal=p_T_adv_out_ocean,
        step=1,
        dim=time_axis,
    )

    frequency_qadv_truth_ocean, pd_qadv_truth_ocean = spectral_analysis_numpy(
        signal=t_q_adv_out_ocean,
        step=1,
        dim=time_axis,
    )

    frequency_qadv_pred_ocean, pd_qadv_pred_ocean = spectral_analysis_numpy(
        signal=p_q_adv_out_ocean,
        step=1,
        dim=time_axis,
    )


    frequency_tadv_truth_land, pd_tadv_truth_land = spectral_analysis_numpy(
        signal=t_T_adv_out_land,
        step=1,
        dim=time_axis,
    )

    frequency_tadv_pred_land, pd_tadv_pred_land = spectral_analysis_numpy(
        signal=p_T_adv_out_land,
        step=1,
        dim=time_axis,
    )

    frequency_qadv_truth_land, pd_qadv_truth_land = spectral_analysis_numpy(
        signal=t_q_adv_out_land,
        step=1,
        dim=time_axis,
    )

    frequency_qadv_pred_land, pd_qadv_pred_land = spectral_analysis_numpy(
        signal=p_q_adv_out_land,
        step=1,
        dim=time_axis,
    )


    pd_tadv_truth_mean = pd_tadv_truth.mean(axis=(lat_axis,lon_axis,z_axis))
    pd_tadv_pred_mean = pd_tadv_pred.mean(axis=(lat_axis,lon_axis,z_axis))
    pd_qadv_truth_mean = pd_qadv_truth.mean(axis=(lat_axis,lon_axis,z_axis))
    pd_qadv_pred_mean = pd_qadv_pred.mean(axis=(lat_axis,lon_axis,z_axis))

    pd_tadv_truth_mean_ocean = np.nanmean(pd_tadv_truth_ocean, axis=(lat_axis,lon_axis,z_axis))
    pd_tadv_pred_mean_ocean = np.nanmean(pd_tadv_pred_ocean, axis=(lat_axis,lon_axis,z_axis))
    pd_qadv_truth_mean_ocean = np.nanmean(pd_qadv_truth_ocean, axis=(lat_axis,lon_axis,z_axis))
    pd_qadv_pred_mean_ocean = np.nanmean(pd_qadv_pred_ocean, axis=(lat_axis,lon_axis,z_axis))

    pd_tadv_truth_mean_land = np.nanmean(pd_tadv_truth_land, axis=(lat_axis,lon_axis,z_axis))
    pd_tadv_pred_mean_land = np.nanmean(pd_tadv_pred_land, axis=(lat_axis,lon_axis,z_axis))
    pd_qadv_truth_mean_land = np.nanmean(pd_qadv_truth_land, axis=(lat_axis,lon_axis,z_axis))
    pd_qadv_pred_mean_land = np.nanmean(pd_qadv_pred_land, axis=(lat_axis,lon_axis,z_axis))


    plot_spectral_analysis_all(truth_var_one_all=pd_tadv_truth_mean, 
                           pred_var_one_all=pd_tadv_pred_mean, 
                           truth_var_two_all=pd_qadv_truth_mean, 
                           pred_var_two_all=pd_qadv_pred_mean, 
                           truth_var_one_ocean=pd_tadv_truth_mean_ocean, 
                           pred_var_one_ocean=pd_tadv_pred_mean_ocean, 
                           truth_var_two_ocean=pd_qadv_truth_mean_ocean, 
                           pred_var_two_ocean=pd_qadv_pred_mean_ocean, 
                           truth_var_one_land=pd_tadv_truth_mean_land, 
                           pred_var_one_land=pd_tadv_pred_mean_land, 
                           truth_var_two_land=pd_qadv_truth_mean_land, 
                           pred_var_two_land=pd_qadv_pred_mean_land, 
                           freq_one=frequency_tadv_truth, 
                           freq_two=frequency_qadv_truth, 
                           x_label="Time in Hours",
                           y_one_label=r"$\frac{k^2}{s^2 hour}$", 
                           y_two_label=r"$\frac{kg^2*s^2}{kg^2*hour}$", 
                           variable_name_one="T Advection",
                           variable_name_two="Q Advection",
                           suptitle="Spectral Analysis", 
                           color_truth="blue", 
                           color_prediction="green",
                           base_dir=save_path+nametag,
                           save_dir="/Spectral_Analysis",
                          )

    my_sigma = [0, 14, 29]
    #my_sigma = [0, 14]
    precise = [None, None, None]
    for i in range(len(my_sigma)):
        animation_generator_gif(
            x_whirl=p_Tout[my_sigma[i],:,:,:].squeeze(),
            x_wb=t_Tout[my_sigma[i],:,:,:].squeeze(),
            lat=lat,
            lon=lon,
            unit_labels="K",
            var_name="Tout",
            base_dir=save_path+nametag,
            save_dir='Animations',
            elevation=str(sigma[my_sigma[i]]),
            time_axis=1,
            data_percentile=95,
            diff_percentile=95,
            contour_levels=12,
            pos_neg_colorbar=True,
            contour_precision=precise[i],
        )


    # q adv 
    # sfc, 250, 500
    my_sigma = [0, 32, 42]
    # was [:,:,:,my_sigma[i]]
    for i in range(len(my_sigma)):
        animation_generator_gif(
            x_whirl=p_q_adv_out[my_sigma[i],:,:,:].squeeze(),
            x_wb=t_q_adv_out[my_sigma[i],:,:,:].squeeze(),
            lat=lat,
            lon=lon,
            unit_labels=r"$\frac{kg}{kg*s}$",
            var_name="Qadv",
            base_dir=save_path+nametag,
            save_dir='Animations',
            elevation=str(sigma[my_sigma[i]]),
            time_axis=1,
            data_percentile=95,
            diff_percentile=95,
            contour_levels=10,
            cmap="PiYG",
            pos_neg_colorbar=True,
        )


    # t adv 
    # sfc, 850, 500
    my_sigma = [0, 19, 32]
    for i in range(len(my_sigma)):
        animation_generator_gif(
            x_whirl=p_T_adv_out[my_sigma[i],:,:,:].squeeze(),
            x_wb=t_T_adv_out[my_sigma[i],:,:,:].squeeze(),
            lat=lat,
            lon=lon,
            unit_labels=r"$\frac{K}{s}$",
            var_name="Tadv",
            base_dir=save_path+nametag,
            save_dir='Animations',
            elevation=str(sigma[my_sigma[i]]),
            time_axis=1,
            data_percentile=95,
            diff_percentile=95,
            contour_levels=12,
            alpha=0.2,
        )


    # q auto out 
    my_sigma = [0, 32, 39]
    for i in range(len(my_sigma)):
        animation_generator_gif(
            x_whirl=p_q_auto_out[my_sigma[i],:,:,:].squeeze(),
            x_wb=t_q_auto_out[my_sigma[i],:,:,:].squeeze(),
            lat=lat,
            lon=lon,
            unit_labels=r"$\frac{kg}{kg*s}$",
            var_name="Qout",
            base_dir=save_path+nametag,
            save_dir='Animations',
            elevation=str(sigma[my_sigma[i]]),
            time_axis=1,
            data_percentile=95,
            diff_percentile=95,
            contour_levels=12,
            pos_neg_colorbar=True,
            cmap="PiYG",
        )


    # q sed 
    # sfc, 500, 250
    my_sigma = [15, 32, 42]
    for i in range(len(my_sigma)):
        animation_generator_gif(
            x_whirl=p_q_sed_flux_tot[my_sigma[i],:,:,:].squeeze(),
            x_wb=t_q_sed_flux_tot[my_sigma[i],:,:,:].squeeze(),
            lat=lat,
            lon=lon,
            unit_labels=r"$\frac{kg}{kg*s}$",
            var_name="Qsed",
            base_dir=save_path+nametag,
            save_dir='Animations',
            elevation=str(sigma[my_sigma[i]]),
            time_axis=1,
            data_percentile=95,
            diff_percentile=95,
            contour_levels=12,
        )

