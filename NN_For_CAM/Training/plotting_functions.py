import os
import numpy as np
import math
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import matplotlib

#Animations
import tempfile
import shutil
import glob
from PIL import Image

# environment issue -- need to fix
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import seaborn as sns

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


def single_geo_plotter(lats, lons, field_data, title, units, cmap = "RdYlBu_r", logcbar=False, save=False, 
                       vmin = None, vmax=None, base_dir=None, save_dir=None, savename=None): 

    if os.path.isdir(base_dir+save_dir):
        pass
    else:
        os.mkdir(base_dir+save_dir)
        
    fig, ax = plt.subplots(figsize=(20,12),subplot_kw={'projection':ccrs.Robinson(central_longitude=180)})
    
    if logcbar is False:
        contour_plot = ax.pcolormesh(lons, lats, field_data, cmap=cmap, transform=ccrs.PlateCarree(),
                                 rasterized=True, vmin=vmin, vmax=vmax)
    else:
        contour_plot = ax.pcolormesh(lons, lats, field_data, cmap=cmap, transform=ccrs.PlateCarree(),
                                 rasterized=True, vmin=vmin, vmax=vmax, norm=matplotlib.colors.LogNorm())
    ax.coastlines(linewidth=0.5,edgecolor='0.25') 
    ax.gridlines()
    ax.add_feature(cfeature.BORDERS,linewidth=0.5,edgecolor='0.25')
    ax.set_title(title, fontsize = 1.5*fz)

    p0 = ax.get_position().get_points().flatten()
    p2 = ax.get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[0]+0.137, 0.09, p0[1]+0.350, 0.03])
    cbar = fig.colorbar(contour_plot, cax=ax_cbar, orientation='horizontal')
    cbar.set_label(units, fontsize=fz*1.25)
    cbar.ax.tick_params(labelsize=fz*1.25)
    
    if save is True:
        savepath = base_dir+save_dir+'/'+savename+'.png'
        plt.savefig(savepath)


def my_scatter_plot(truth_array, pred_array, y_label, x_label, title, save=False, base_dir=None, save_dir=None, savename=None):

    if os.path.isdir(base_dir+save_dir):
        pass
    else:
        os.mkdir(base_dir+save_dir)


    plt.figure(figsize=(16, 16))
    plt.scatter(truth_array, pred_array, alpha=0.6, s=10)
    plt.plot(
        [truth_array.min(), truth_array.max()],
        [truth_array.min(), truth_array.max()],
        'r--', label='1:1 Line'
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save is True:
        savepath = base_dir+save_dir+'/'+savename+'.png'
        plt.savefig(savepath)


def jensen_shannon(p, q):
    p = np.array(p, dtype=np.float64) + 1e-12
    q = np.array(q, dtype=np.float64) + 1e-12
    p /= np.sum(p)
    q /= np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


def precip_distribution_plot(global_truth, global_nn, land_truth, land_nn,
                            ocean_truth, ocean_nn, title_all, title_land, title_ocean, label_truth, label_nn,
                             suptitle, x_label, y_label, save, base_dir, save_dir, savename,
                             color_true, color_nn, bins, x_log = False, y_log = False, x_limits=None,
                            ):

    if os.path.isdir(base_dir+save_dir):
        pass
    else:
        os.mkdir(base_dir+save_dir)
    
    fig, ax = plt.subplots(3, 1, figsize=(15,20))

    shared_bins = np.histogram_bin_edges(global_truth, bins=bins, range=(0, global_truth.max()))

    global_freq_truth, global_edges_truth = np.histogram(global_truth, bins = shared_bins, density=True)
    global_freq_nn, global_edges_nn = np.histogram(global_nn, bins = shared_bins, density=True)

    js_dist = jensen_shannon(global_freq_truth, global_freq_nn)
    print("Global Jensen-Shannon Distance is:", js_dist)

    ax[0].plot(global_edges_truth[:-1], global_freq_truth/np.sum(global_freq_truth),  label = label_truth, color = color_true)
    ax[0].plot(global_edges_nn[:-1], global_freq_nn/np.sum(global_freq_nn),  label = label_nn, color = color_nn)
    ax[0].set_ylim(1e-8, 1)
    ax[0].set_title(title_all, fontsize = fz)
    ax[0].legend(loc = 'best')
    ax[0].set_ylabel(y_label)
    ax[0].grid(True)
    if x_limits:
        ax[0].set_xlim(x_limits)

    land_freq_truth, land_edges_truth = np.histogram(land_truth, bins = shared_bins, density=True)
    land_freq_nn, land_edges_nn = np.histogram(land_nn, bins = shared_bins, density=True)

    js_dist = jensen_shannon(land_freq_truth, land_freq_nn)
    print("Land Jensen-Shannon Distance is:", js_dist)

    ax[1].plot(land_edges_truth[:-1], land_freq_truth/np.sum(land_freq_truth),  label = label_truth, color = color_true)
    ax[1].plot(land_edges_nn[:-1], land_freq_nn/np.sum(land_freq_nn),  label = label_nn, color = color_nn)
    ax[1].set_ylim(1e-8, 1)
    ax[1].set_title(title_land, fontsize = fz)
    ax[1].set_ylabel(y_label)
    ax[1].grid(True)
    if x_limits:
        ax[1].set_xlim(x_limits)

    ocean_freq_truth, ocean_edges_truth = np.histogram(ocean_truth, bins = shared_bins, density=True)
    ocean_freq_nn, ocean_edges_nn = np.histogram(ocean_nn, bins = shared_bins, density=True)

    js_dist = jensen_shannon(ocean_freq_truth, ocean_freq_nn)
    print("Ocean Jensen-Shannon Distance is:", js_dist)
   
    ax[2].plot(ocean_edges_truth[:-1], ocean_freq_truth/np.sum(ocean_freq_truth),  label = label_truth, color = color_true)
    ax[2].plot(ocean_edges_nn[:-1], ocean_freq_nn/np.sum(ocean_freq_nn),  label = label_nn, color = color_nn)
    ax[2].set_ylim(1e-8, 1)
    ax[2].set_title(title_ocean, fontsize = fz)
    ax[2].set_ylabel(y_label)
    ax[2].grid(True)
    ax[2].set_xlabel(x_label)
    if x_limits:
        ax[2].set_xlim(x_limits)

    if x_log == True:
        ax[2].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].set_xscale('log')

    if y_log == True:
        ax[2].set_yscale('log')
        ax[1].set_yscale('log')
        ax[0].set_yscale('log')


    plt.suptitle(suptitle)

    if save is True:
        savepath = base_dir+save_dir+'/'+savename+'.png'
        plt.savefig(savepath)
        plt.show()


def precip_with_latitude(precip_truth, precip_nn, truth_label, nn_label, title, lats, 
                         color_truth, color_nn, y_label, x_label,
                         vmin, vmax, save=False, base_dir=None, save_dir=None, savename=None,
                        ):

    if os.path.isdir(base_dir+save_dir):
        pass
    else:
        os.mkdir(base_dir+save_dir)

    fig, ax = plt.subplots(figsize=(25,10))

    if vmin is None:
        vmin = min(
            np.nanmin(precip_truth), np.nanmin(precip_nn),
        )
    
    if vmax is None:
        vmax = max(
            np.nanmax(precip_truth), np.nanmax(precip_nn),
        )

    ax.plot(lats, precip_truth, label = truth_label, color=color_truth)
    ax.plot(lats, precip_nn, label = nn_label, color=color_nn)
    ax.set_ylim(vmin, vmax)
    ax.legend()
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)

    ax.set_xlabel(x_label)

    if save is True:
        savepath = base_dir+save_dir+'/'+savename+'.png'
        plt.savefig(savepath)
        plt.show()


def plot_side_by_side_with_diff(A, B, lons, lats, units,
                                 cmap="Reds",
                                 diff_cmap="bwr",
                                 title_A="A",
                                 title_B="B",
                                 title_diff="Difference",
                                 suptitle='Comparison with Difference',
                                 projection=ccrs.Robinson(central_longitude=180),
                                 linthresh=None,
                                 save=False,
                                 norm=True,
                                 base_dir=None,
                                 save_dir=None,
                                 savename=None,
                                 vmin=None,
                                 vmax=None,
                                 vmax_clip=None,
                                 vmin_clip=None,
                                 diff_mode="raw", # 'raw' or 'relative'
                                ):
    """
    Plot two arrays side by side plus a difference plot, with Cartopy and two colorbars.

    Args:
        A (2D array): True field
        B (2D array): Predicted or comparison field
        lons, lats (2D arrays): longitude and latitude grids
        units (str): units for colorbar
        diff_mode (str): 'raw' for True - Pred, 'relative' for (True - Pred)/True * 100.
    """

    if os.path.isdir(base_dir+save_dir):
        pass
    else:
        os.mkdir(base_dir+save_dir)
    
    if vmin is None:
        vmin = min(np.nanmin(A), np.nanmin(B))
    if vmax is None:
        vmax = max(np.nanmax(A), np.nanmax(B))

    if (vmin_clip is not None) and (vmax_clip is not None):
        vmax = min(vmax, vmax_clip)
        vmin = max(vmin, vmin_clip)

    if norm == True:
        if linthresh is None:
            linthresh = max(1e-2, 0.01 * vmax)
        print(f"\u26a1 Using SymLogNorm for A/B with linthresh={linthresh:.2e}")
        norm_main = matplotlib.colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
    else:
        norm_main = None

    # Compute difference
    if diff_mode == "raw":
        diff = A - B
        diff_units = units
    elif diff_mode == "relative":
        diff = 100.0 * (A - B) / (A + 1e-10)  # small value to avoid 0-div
        diff_units = "% Error"
    else:
        raise ValueError("diff_mode must be 'raw' or 'relative'")

    fig, axs = plt.subplots(1, 3, figsize=(30, 6),
                            subplot_kw={'projection': projection})

    # Plot A and B
    for ax, data, title in zip(axs[:2], [A, B], [title_A, title_B]):
        mesh = ax.pcolormesh(lons, lats, data,
                             transform=ccrs.PlateCarree(),
                             cmap=cmap, rasterized=True,
                             norm=norm_main if norm_main else None,
                             vmin=None if norm_main else vmin,
                             vmax=None if norm_main else vmax)
        ax.coastlines(linewidth=0.5, edgecolor='0.25')
        ax.gridlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='0.25')
        ax.set_title(title, fontsize=1.5*fz)

    # Plot difference
    diff_vmax = np.nanmax(np.abs(diff))
    diff_norm = matplotlib.colors.CenteredNorm(vcenter=0)


    vmax_diff = np.nanpercentile(diff, 95)
    vmin_diff = np.nanpercentile(diff, 5)
    bound = max(np.abs(vmax_diff), np.abs(vmin_diff))

    

    mesh_diff = axs[2].pcolormesh(lons, lats, diff, vmin=-bound, vmax=bound,
                                  transform=ccrs.PlateCarree(),
                                  cmap=diff_cmap, rasterized=True,
                                  norm=None)
    axs[2].coastlines(linewidth=0.5, edgecolor='0.25')
    axs[2].gridlines()
    axs[2].add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='0.25')
    axs[2].set_title(title_diff, fontsize=1.5*fz)

    # Main colorbar
    ax_cbar_main = fig.add_axes([0.18, 0.05, 0.4, 0.05])  # left, bottom, width, height
    ax_cbar_diff = fig.add_axes([0.69, 0.05, 0.18, 0.05])
    
    cbar_main = fig.colorbar(mesh, cax=ax_cbar_main, orientation='horizontal')
    cbar_main.set_label(units, fontsize=fz*1.25)
    cbar_main.ax.tick_params(labelsize=fz*1.25)

    # Difference colorbar
    cbar_diff = fig.colorbar(mesh_diff, cax=ax_cbar_diff, orientation='horizontal')
    cbar_diff.set_label(diff_units, fontsize=fz*1.25)
    cbar_diff.ax.tick_params(labelsize=fz*1.25)

    plt.suptitle(suptitle)
    fig.subplots_adjust(wspace=0.1)

    if save is True:
        savepath = base_dir+save_dir+'/'+savename+'.png'
        plt.savefig(savepath)
        plt.show()


def precip_with_time(truth_all, nn_all, truth_land, nn_land,
                     truth_ocean, nn_ocean, times,
                     truth_all_label, nn_all_label, truth_land_label, nn_land_label, 
                     truth_ocean_label, nn_ocean_label, all_color, land_color, ocean_color,
                     y_label, x_label, global_title,
                      vmin, vmax, save=False, base_dir=None, save_dir=None, savename=None,
                        ):

    if os.path.isdir(base_dir+save_dir):
        pass
    else:
        os.mkdir(base_dir+save_dir)

    fig, ax = plt.subplots(figsize=(25,15))

    if vmin is None:
        vmin = min(
            np.nanmin(truth_all), np.nanmin(nn_all), np.nanmin(truth_land),
            np.nanmin(nn_land), np.nanmin(truth_ocean), np.nanmin(nn_ocean),
        )
    
    if vmax is None:
        vmax = max(
            np.nanmax(truth_all), np.nanmax(nn_all), np.nanmax(truth_land),
            np.nanmax(nn_land), np.nanmax(truth_ocean), np.nanmax(nn_ocean),
        )

    ax.plot(times, truth_all, label = truth_all_label, color=all_color)
    ax.plot(times, nn_all, label = nn_all_label, color=all_color, linestyle='dashed')
    ax.plot(times, truth_land, label = truth_land_label, color=land_color)
    ax.plot(times, nn_land, label = nn_land_label, color=land_color, linestyle='dashed')
    ax.plot(times, truth_ocean, label = truth_ocean_label, color=ocean_color)
    ax.plot(times, nn_ocean, label = nn_ocean_label, color=ocean_color, linestyle='dashed')
    ax.set_ylim(vmin, vmax)
    ax.legend()
    ax.set_ylabel(y_label)
    ax.set_title(global_title)
    ax.grid(True)

    ax.set_xlabel(x_label)

    if save is True:
        savepath = base_dir+save_dir+'/'+savename+'.png'
        plt.savefig(savepath)
        plt.show()


def compute_rmse(y_true, y_pred, ignore_nan=True):
    """
    Compute Root Mean Squared Error (RMSE) between y_true and y_pred.
    
    Parameters:
        y_true (np.ndarray): Ground truth array.
        y_pred (np.ndarray): Predicted array.
        ignore_nan (bool): If True, ignore NaNs in the input.

    Returns:
        float: RMSE value.
    """
    if ignore_nan:
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
    else:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true, y_pred, ignore_nan=True):
    """
    Compute Mean Absolute Error (MAE) between y_true and y_pred.

    Parameters:
        y_true (np.ndarray): Ground truth array.
        y_pred (np.ndarray): Predicted array.
        ignore_nan (bool): If True, ignore NaNs in the input.

    Returns:
        float: MAE value.
    """
    if ignore_nan:
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        return np.mean(np.abs(y_true[mask] - y_pred[mask]))
    else:
        return np.mean(np.abs(y_true - y_pred))
        

def compute_R2_aligned(true, pred, axis):
    """
    Computes R² score along a given axis with proper broadcasting of the mean.
    Equivalent to original approach used in hand-coded method.
    """
    true_mean = np.nanmean(true, axis=axis, keepdims=True)
    sse = np.nansum((true - pred)**2, axis=axis)
    svar = np.nansum((true - true_mean)**2, axis=axis)
    return 1 - sse / svar


def compute_R2_2D_aligned(true, pred, axis):
    """
    Computes R² across a specified axis (e.g., time), ensuring correct shape for broadcasting.
    """
    true_mean = np.nanmean(true, axis=axis, keepdims=True)
    sse = np.nansum((true - pred)**2, axis=axis)
    svar = np.nansum((true - true_mean)**2, axis=axis)
    return 1 - sse / svar



def find_exp(number) -> int:
    base10 = math.log10(abs(number))
    return abs(math.floor(base10))

def min_max_helper_1D(true_all_data, true_land_data, true_ocean_data,
                     pred_all_data, pred_land_data, pred_ocean_data,):
    
    vmax = max([true_all_data.max(), true_land_data.max(), true_ocean_data.max(),
               pred_all_data.max(), pred_land_data.max(), pred_ocean_data.max(),])
    vmin = min([true_all_data.min(), true_land_data.min(), true_ocean_data.min(),
               pred_all_data.min(), pred_land_data.min(), pred_ocean_data.min(),])
    
    return vmax, vmin


def plot_r2_vs_surface_features(R2_2D, land_fraction, surface_height, identifier, base_dir, save_dir, filename_prefix="R2", title_suffix=""):

    # Flatten to 1D
    R2_flat = R2_2D.flatten()
    land_flat = land_fraction.flatten()
    height_flat = surface_height.flatten()

    # Mask NaNs from R²
    valid = ~np.isnan(R2_flat)
    R2_flat = R2_flat[valid]
    land_flat = land_flat[valid]
    height_flat = height_flat[valid]

    # Create DataFrame
    df = pd.DataFrame({
        "R²": R2_flat,
        "Land Fraction": land_flat,
        "Surface Reference Pressure": height_flat
    })

    # Plot
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    sns.regplot(data=df, x="Land Fraction", y="R²", scatter_kws={"s": 10}, line_kws={"color": "red"})
    plt.title(f"R² vs Land Fraction {title_suffix}")
    plt.xlabel("Land Fraction")
    plt.ylabel("R²")

    plt.subplot(1, 2, 2)
    sns.regplot(data=df, x="Surface Reference Pressure", y="R²", scatter_kws={"s": 10}, line_kws={"color": "red"})
    plt.title(f"R² vs Surface Height {title_suffix}")
    plt.xlabel("mb")
    plt.ylabel("R²")

    plt.tight_layout()
    savepath = base_dir+save_dir+'/'+identifier+'.png'
    plt.savefig(savepath)
    plt.close()

def mean_std_vertical_level_comparison(mean_data_list_truth, 
                                       mean_data_list_pred, 
                                       std_data_list_truth, 
                                       std_data_list_pred,
                                       name_list, 
                                       z, 
                                       super_title, 
                                        x_labels, 
                                       y_label,
                                       base_dir,
                                       save_dir: str='1D_Statistics',
                                       landsea=False, 
                                       mean_land_truth=None, 
                                       mean_land_pred=None,
                                       mean_ocean_truth=None, 
                                       mean_ocean_pred=None,
                                       std_land_truth=None, 
                                       std_land_pred=None, 
                                       std_ocean_truth=None, 
                                       std_ocean_pred=None,
                                       xlim_min=None, 
                                       xlim_max=None,
                                       ylim_min=None, 
                                       ylim_max=None, 
                                       inverted_y=False):
    
    
    if os.path.isdir(base_dir+save_dir):
        pass
    else:
        os.mkdir(base_dir+save_dir)
    
    fig, ax = plt.subplots(len(mean_data_list_truth), 2, figsize=(20,5.5*len(mean_data_list_truth)))
    for i in range(len(mean_data_list_truth)):
        ax[i,0].plot(mean_data_list_truth[i], z, c='purple', label='all (truth)')
        ax[i,0].plot(mean_data_list_pred[i], z, c='purple', label='all (prediction)', linestyle="dashed")
        ax[i,0].set_title(name_list[i] + " (mean)", fontsize=fz*0.8)
        ax[i,0].set_ylabel(y_label, fontsize=fz)
        ax[i,1].plot(std_data_list_truth[i], z, c='purple', label='all (truth)')
        ax[i,1].plot(std_data_list_pred[i], z, c='purple', label='all (prediction)', linestyle="dashed")
        ax[i,1].set_yticks([])
        ax[i,1].set_title(name_list[i] + " (std)", fontsize=fz*0.8)
        ax[i,0].tick_params(axis='y', which='both', labelsize=fz)
        ax[i,0].tick_params(axis='x', which='both', labelsize=fz)
        ax[i,1].tick_params(axis='y', which='both', labelsize=fz)
        ax[i,1].tick_params(axis='x', which='both', labelsize=fz)
        ax[i,0].set_xlim(xlim_min, xlim_max)
        ax[i,0].set_ylim(ylim_min, ylim_max)
        ax[i,1].set_xlim(xlim_min, xlim_max)
        ax[i,1].set_ylim(ylim_min, ylim_max)
        ax[i,0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax[i,1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        if landsea == True:
            ax[i,0].plot(mean_land_truth[i], z, c='green', label='land')
            ax[i,0].plot(mean_ocean_truth[i], z, c='blue', label='ocean')
            ax[i,1].plot(std_land_truth[i], z, c='green', label='land')
            ax[i,1].plot(std_ocean_truth[i], z, c='blue', label='ocean')
            ax[i,0].plot(mean_land_pred[i], z, c='green', label='land', linestyle="dashed")
            ax[i,0].plot(mean_ocean_pred[i], z, c='blue', label='ocean', linestyle="dashed")
            ax[i,1].plot(std_land_pred[i], z, c='green', label='land', linestyle="dashed")
            ax[i,1].plot(std_ocean_pred[i], z, c='blue', label='ocean', linestyle="dashed")
            
            vmax_mean, vmin_mean = min_max_helper_1D(true_all_data=mean_data_list_truth[i], 
                                           true_land_data=mean_land_truth[i], 
                                           true_ocean_data=mean_ocean_truth[i],
                                           pred_all_data=mean_data_list_pred[i], 
                                           pred_land_data=mean_land_pred[i], 
                                           pred_ocean_data=mean_ocean_pred[i],
                                          )
            
            vmax_std, vmin_std = min_max_helper_1D(true_all_data=std_data_list_truth[i], 
                                           true_land_data=std_land_truth[i], 
                                           true_ocean_data=std_ocean_truth[i],
                                           pred_all_data=std_data_list_pred[i], 
                                           pred_land_data=std_land_pred[i], 
                                           pred_ocean_data=std_ocean_pred[i],
                                          )
                
            ax[i,0].set_xlim(vmin_mean, vmax_mean)
            ax[i,1].set_xlim(vmin_std, vmax_std)
            
        
        if i == 0:
            ax[i,0].legend(fontsize=fz*0.75)
            

        ax[i,0].set_xlabel(x_labels[i], fontsize=fz)
        ax[i,1].set_xlabel(x_labels[i], fontsize=fz)
            
        if inverted_y == True:
            ax[i,0].invert_yaxis()
            ax[i,1].invert_yaxis()
    
    plt.suptitle(super_title, y =0.92)
    savepath = base_dir+save_dir+'/Means_Stds.png'
    plt.savefig(savepath)


def five_panel_lat_pressure_cross_section(field_list: list, 
                                          x_values: np.array, 
                                          y_values: np.array,  
                                          xlabel: str, 
                                          ylabel: str, 
                                          title_list: list, 
                                          cbar_label: str, 
                                          super_title: str, 
                                          base_dir: str, 
                                          save_dir: str = "R2_Figures",
                                          identifier: str = "R2_lon_averaged_before_calc",
                                          cmap: str =  "Blues", 
                                          vertical_splice: int = None, 
                                          vmin: int = 0, 
                                          vmax: int = 1,
                                          ):
    """Plot a classic latitude/pressure cross section for two variables or one prediction and truth"""
    
    if os.path.isdir(base_dir+save_dir):
        pass
    else:
        os.mkdir(base_dir+save_dir)
    
    fig, ax = plt.subplots(2,3, figsize=(25,15))
    ax[0,0].pcolor(x_values[:, :vertical_splice], y_values[:, :vertical_splice], 
                 field_list[0].T, cmap = cmap, vmin = vmin, 
                 vmax = vmax, rasterized=True)
    ax[0,0].set_title(title_list[0], fontsize = fz*0.9)
    ax[0,0].set_ylim(ax[0,0].get_ylim()[::-1])
    ax[0,0].set_ylabel(ylabel)
    ax[0,0].set_xticks([])

    contour_plot = ax[0,1].pcolor(x_values[:, :vertical_splice], y_values[:, :vertical_splice], 
                                field_list[1].T, cmap = cmap, vmin = vmin, 
                                vmax = vmax, rasterized=True)
    ax[0,1].set_title(title_list[1], fontsize = fz*0.9)
    ax[0,1].set_ylim(ax[0,1].get_ylim()[::-1])
    ax[0,1].set_yticks([])
    ax[0,1].set_xticks([])
    
    ax[1,0].pcolor(x_values[:, :vertical_splice], y_values[:, :vertical_splice], 
                 field_list[2].T, cmap = cmap, vmin = vmin, 
                 vmax = vmax, rasterized=True)
    ax[1,0].set_title(title_list[2], fontsize = fz*0.9)
    ax[1,0].set_ylim(ax[1,0].get_ylim()[::-1])
    ax[1,0].set_ylabel(ylabel)
    ax[1,0].set_xlabel(xlabel)
    
    ax[1,1].pcolor(x_values[:, :vertical_splice], y_values[:, :vertical_splice], 
                 field_list[3].T, cmap = cmap, vmin = vmin, 
                 vmax = vmax, rasterized=True)
    ax[1,1].set_title(title_list[3], fontsize = fz*0.9)
    ax[1,1].set_ylim(ax[1,1].get_ylim()[::-1])
    ax[1,1].set_yticks([])
    ax[1,1].set_xlabel(xlabel)
    
    ax[0,2].pcolor(x_values[:, :vertical_splice], y_values[:, :vertical_splice], 
                 field_list[4].T, cmap = cmap, vmin = vmin, 
                 vmax = vmax, rasterized=True)
    ax[0,2].set_title(title_list[4], fontsize = fz*0.9)
    ax[0,2].set_ylim(ax[0,2].get_ylim()[::-1])
    ax[0,2].set_yticks([])
    ax[0,2].set_xlabel(xlabel)


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.12, 0.05, 0.76])
    fig.colorbar(contour_plot, label=cbar_label, cax=cbar_ax)
    plt.suptitle(super_title, y = 0.95, x=0.47)
    plt.subplots_adjust(hspace=0.07, wspace=0.01)
    
    fig.delaxes(ax[1,2])
    savepath = base_dir+save_dir+'/'+identifier+'.png'
    plt.savefig(savepath)


def min_max_getter(array, diff_percentile=95):
        return np.percentile(array.flatten(), diff_percentile), np.percentile(array.flatten(), 100-diff_percentile)

def min_max_getter_diff(array, data_percentile=90):
        positive_array = np.abs(array)
        positive = np.percentile(positive_array, data_percentile)
        return positive, positive*(-1.0)


def single_variable_mean_std_plot_all(truth_all, truth_ocean, truth_land,
                                      pred_all, pred_ocean, pred_land, X, Z, 
                                      variable_name, y_units, variable_units,
                                  x_units, colormap, cmap_diff, measure, suptitle,
                                      base_dir, save_dir:str='2D_Statistics',
                                      data_percentile=90, diff_percentile=90, 
                                  vmin=None, vmax=None,
                                 ):
    
    if os.path.isdir(base_dir+save_dir):
        pass
    else:
        os.mkdir(base_dir+save_dir)
    
    fig, ax = plt.subplots(3, 3, figsize=(20,15)) 
    
    if (vmin == None) and (vmax == None):
        vmax, vmin = min_max_getter(truth_all, data_percentile)

    cp = ax[0,0].pcolormesh(X, Z, truth_all.T, cmap = colormap, vmin=vmin, vmax=vmax)
    ax[0,0].set_title("(a) true "+measure+" (all)", fontsize = fz)
    ax[0,0].set_ylim(ax[0,0].get_ylim()[::-1])
    ax[0,0].set_xticks([])
    ax[0,0].set_ylabel(y_units)
    
    cbar_ax = fig.add_axes([0.065, 0.125, 0.01, 0.754])
    cbar = fig.colorbar(cp, cax=cbar_ax)
    cbar.set_label(label=variable_units, labelpad=-1.5)
    cbar.ax.tick_params(labelsize=fz*0.8)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.get_offset_text().set(size=fz*0.65)
    
    cp = ax[0,1].pcolormesh(X, Z, pred_all.T, cmap = colormap, vmin=vmin, vmax=vmax)
    ax[0,1].set_title("(b) pred. "+measure+" (all)", fontsize = fz)
    ax[0,1].set_ylim(ax[0,1].get_ylim()[::-1])
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    
    diff_all = truth_all - pred_all
    vmax_diff, vmin_diff = min_max_getter_diff(diff_all, diff_percentile)
    
    cp = ax[0,2].pcolormesh(X, Z, diff_all.T, cmap = cmap_diff, vmin=vmin_diff, vmax=vmax_diff)
    ax[0,2].set_title("(c) "+measure+" difference (all)", fontsize = fz)
    ax[0,2].set_ylim(ax[0,2].get_ylim()[::-1])
    ax[0,2].set_xticks([])
    ax[0,2].set_yticks([])
    
    cbar_ax = fig.add_axes([0.905, 0.125, 0.01, 0.754])
    cbar = fig.colorbar(cp, cax=cbar_ax)
    cbar.set_label(label=variable_units, labelpad=-1.5)
    cbar.ax.tick_params(labelsize=fz*0.8)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.get_offset_text().set(size=fz*0.65)
    
    cp = ax[1,0].pcolormesh(X, Z, truth_ocean.T, cmap = colormap, vmin=vmin, vmax=vmax)
    ax[1,0].set_title("(d) true "+measure+" (ocean)", fontsize = fz)
    ax[1,0].set_ylim(ax[1,0].get_ylim()[::-1])
    ax[1,0].set_xticks([])
    ax[1,0].set_ylabel(y_units)
    
    cp = ax[1,1].pcolormesh(X, Z, pred_ocean.T, cmap = colormap, vmin=vmin, vmax=vmax)
    ax[1,1].set_title("(e) pred. "+measure+" (ocean)", fontsize = fz)
    ax[1,1].set_ylim(ax[1,1].get_ylim()[::-1])
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    
    diff_ocean = truth_ocean - pred_ocean
    cp = ax[1,2].pcolormesh(X, Z, diff_ocean.T, cmap = cmap_diff, vmin=vmin_diff, vmax=vmax_diff)
    ax[1,2].set_title("(f) "+measure+" difference (ocean)", fontsize = fz)
    ax[1,2].set_ylim(ax[1,2].get_ylim()[::-1])
    ax[1,2].set_xticks([])
    ax[1,2].set_yticks([])
    
    cp = ax[2,0].pcolormesh(X, Z, truth_land.T, cmap = colormap, vmin=vmin, vmax=vmax)
    ax[2,0].set_title("(g) true "+measure+" (land)", fontsize = fz)
    ax[2,0].set_ylim(ax[2,0].get_ylim()[::-1])
    ax[2,0].set_xlabel(x_units)
    ax[2,0].set_ylabel(y_units)
    
    cp = ax[2,1].pcolormesh(X, Z, pred_land.T, cmap = colormap, vmin=vmin, vmax=vmax)
    ax[2,1].set_title("(h) pred. "+measure+" (land)", fontsize = fz)
    ax[2,1].set_ylim(ax[2,1].get_ylim()[::-1])
    ax[2,1].set_yticks([])
    ax[2,1].set_xlabel(x_units)
    
    diff_land = truth_land - pred_land
    cp = ax[2,2].pcolormesh(X, Z, diff_land.T, cmap = cmap_diff, vmin=vmin_diff, vmax=vmax_diff)
    ax[2,2].set_title("(i) "+measure+" difference (land)", fontsize = fz)
    ax[2,2].set_ylim(ax[2,2].get_ylim()[::-1])
    ax[2,2].set_yticks([])
    ax[2,2].set_xlabel(x_units)
    
    plt.suptitle(suptitle)
    
    my_dir = base_dir+save_dir+'/'+measure+'_'+variable_name+'.png'
    plt.savefig(my_dir)


def spectral_analysis_numpy(
    signal: np.array,
    step: int or float,
    dim: str,) -> np.array:
    
    """Calculate a global spatial or temporal fast fourier transform."""  
    midpoint_index = signal.shape[dim] // 2
    positive_half = np.arange(0, midpoint_index, 1)
    frequency_values = np.fft.fftfreq(signal.shape[dim], d=step)[positive_half] # Get frequencies.
    fft_signal = np.fft.fft(signal, axis=dim)
    fft_signal_positive_half = fft_signal.take(positive_half, axis=dim)
    power_spectrum = (2 * np.conjugate(fft_signal_positive_half) *
    fft_signal_positive_half / signal.shape[dim]**2)
    power_density = power_spectrum * step * signal.shape[dim]
    return frequency_values, power_density



def plot_spectral_analysis_all(truth_var_one_all, pred_var_one_all, truth_var_two_all, 
                           pred_var_two_all, truth_var_one_ocean, pred_var_one_ocean, truth_var_two_ocean, 
                           pred_var_two_ocean, truth_var_one_land, pred_var_one_land, truth_var_two_land, 
                           pred_var_two_land, freq_one, freq_two, x_label,
                           y_one_label, y_two_label, variable_name_one, variable_name_two, 
                               suptitle, color_truth, color_prediction, base_dir, save_dir: str="Spectral_Analysis",
                          x_scale=False, linewidth=4):
    
    if os.path.isdir(base_dir+save_dir):
        pass
    else:
        os.mkdir(base_dir+save_dir)
    
    fig, ax = plt.subplots(3, 2 ,figsize=(20,25))
    ax[0,0].plot(1/freq_one, truth_var_one_all, color=color_truth, linewidth=linewidth, label="Truth")
    ax[0,0].plot(1/freq_one, pred_var_one_all, color=color_prediction, linewidth=linewidth, label="Prediction")
    ax[0,0].set_ylabel(y_one_label)
    ax[0,0].legend()
    ax[0,0].set_title("All: "+variable_name_one, fontsize=fz)
    ax[0,0].set_xticks([])
    
    ax[0,1].plot(1/freq_two, truth_var_two_all, color=color_truth, linewidth=linewidth, label="Truth")
    ax[0,1].plot(1/freq_two, pred_var_two_all, color=color_prediction, linewidth=linewidth, label="Prediction")
    ax[0,1].set_ylabel(y_two_label)
    ax[0,1].set_title("All: "+variable_name_two, fontsize=fz)
    ax[0,1].set_xticks([])
    
    ax[1,0].plot(1/freq_one, truth_var_one_ocean, color=color_truth, linewidth=linewidth, label="Truth")
    ax[1,0].plot(1/freq_one, pred_var_one_ocean, color=color_prediction, linewidth=linewidth, label="Prediction")
    ax[1,0].set_ylabel(y_one_label)
    ax[1,0].legend()
    ax[1,0].set_title("Ocean: "+variable_name_one, fontsize=fz)
    ax[1,0].set_xticks([])
    
    ax[1,1].plot(1/freq_two, truth_var_two_ocean, color=color_truth, linewidth=linewidth, label="Truth")
    ax[1,1].plot(1/freq_two, pred_var_two_ocean, color=color_prediction, linewidth=linewidth, label="Prediction")
    ax[1,1].set_ylabel(y_two_label)
    ax[1,1].set_title("Ocean: "+variable_name_two, fontsize=fz)
    ax[1,1].set_xticks([])
    
    ax[2,0].plot(1/freq_one, truth_var_one_land, color=color_truth, linewidth=linewidth, label="Truth")
    ax[2,0].plot(1/freq_one, pred_var_one_land, color=color_prediction, linewidth=linewidth, label="Prediction")
    ax[2,0].set_xlabel(x_label)
    ax[2,0].set_ylabel(y_one_label)
    ax[2,0].legend()
    ax[2,0].set_title("Land: "+variable_name_one, fontsize=fz)
    
    ax[2,1].plot(1/freq_two, truth_var_two_land, color=color_truth, linewidth=linewidth, label="Truth")
    ax[2,1].plot(1/freq_two, pred_var_two_land, color=color_prediction, linewidth=linewidth, label="Prediction")
    ax[2,1].set_xlabel(x_label)
    ax[2,1].set_ylabel(y_two_label)
    ax[2,1].set_title("Land: "+variable_name_two, fontsize=fz)
    
    if x_scale == True:
        ax[0,0].set_xscale('log')
        ax[0,1].set_xscale('log')
        ax[1,0].set_xscale('log')
        ax[1,1].set_xscale('log')
        ax[2,0].set_xscale('log')
        ax[2,1].set_xscale('log')
    
    plt.suptitle(suptitle, y=0.92)
    
    my_dir = base_dir+save_dir+'/'+variable_name_one+"_"+variable_name_two+"_"+'Spectral_Analysis'+'.png'
    plt.savefig(my_dir)



def animation_generator_gif_Paul(
    x_whirl: np.ndarray,
    x_wb: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    unit_labels: str,
    var_name: str,
    elevation: str,
    time_axis: int,
    base_dir: str,
    save_dir: str='Animations',
    title_whirl: str = 'Prediction',
    title_wb: str = 'Target',
    title_difference: str = 'Target - Prediction',
    alpha: float=0.1,
    alpha_diff: float=0.25,
    data_percentile: int = 95,
    diff_percentile: int = 95,
    pos_neg_colorbar:bool=False,
    vmin: float = None,
    vmax: float = None,
    vmin_diffs: float = None,
    vmax_diffs: float = None,
    contour_levels: int = 16,
    contour_precision: int = None,
    cmap: str = 'RdBu',
    cmap_diffs: str = 'seismic',
    fps: int = 2, # about 2-3 second rendering in html per gif
    time_interval: str = 'hour',
    ):
    
    if pos_neg_colorbar == False:
        if (vmax is None) and (vmin is None):
            vmax, vmin = min_max_getter(x_wb, diff_percentile=data_percentile)
    
    else:
        if (vmax is None) and (vmin is None):
            vmax, vmin = min_max_getter_diff(x_wb, data_percentile=data_percentile)
    
    if (vmax_diffs is None) and (vmin_diffs is None):
        vmax_diffs, vmin_diffs = min_max_getter_diff(x_wb - x_whirl, data_percentile=diff_percentile)
    
    if contour_precision is None:
        contour_precision = find_exp(x_wb.mean()) + 1
     
    levels = np.round(np.arange(vmin, vmax + (
        vmax - vmin) / contour_levels, (
        vmax - vmin) / contour_levels), decimals=contour_precision)
    
    levels_diffs = np.round(np.arange(vmin_diffs, vmax_diffs + (
        vmax_diffs - vmin_diffs) / contour_levels, (
        vmax_diffs - vmin_diffs) / contour_levels), decimals=contour_precision)
    
    y_grids, x_grids = np.meshgrid(lat, lon)
    temp_directory = tempfile.mkdtemp('temp')
    
    if os.path.isdir(base_dir+save_dir):
        pass
    else:
        os.mkdir(base_dir+save_dir)
        
    for i in range(x_wb.shape[time_axis]):
        fig, ax = plt.subplots(
        1, 2, figsize=(25, 9), subplot_kw=dict(projection=cartopy.crs.Robinson(central_longitude=180)))
        
        
        temp_truth = np.take(x_wb, i, axis=time_axis)
        im_a = ax[0].pcolormesh(x_grids, y_grids, temp_truth.T, cmap=cmap,
                            vmin=vmin, vmax=vmax, transform=cartopy.crs.PlateCarree(), alpha=alpha, animated=True)
        ax[0].set_title(title_wb, fontsize=fz)
        ax[0].set_global()
        ax[0].coastlines(linewidth=1, edgecolor="black")
        ax[0].gridlines()
        ax[0].add_feature(cartopy.feature.BORDERS, linewidth=1, 
                          edgecolor="black")
        cax = fig.add_axes([0.20, 0.13, 0.63, 0.03])
        cb = fig.colorbar(im_a, orientation='horizontal', cax=cax, ticks=levels[::2], boundaries=levels)
        cb.set_label(unit_labels, fontsize=fz*1.5)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        cb.solids.set(alpha=1)
        
        temp_prediction = np.take(x_whirl, i, axis=time_axis)
        ax[1].pcolormesh(x_grids, y_grids, temp_prediction.T, cmap=cmap, vmin=vmin,
            vmax=vmax, transform=cartopy.crs.PlateCarree(), alpha=alpha, animated=True)
        ax[1].set_title(title_whirl, fontsize=fz)
        ax[1].coastlines(linewidth=1, edgecolor='black')
        ax[1].gridlines()
        ax[1].add_feature(cartopy.feature.BORDERS, linewidth=1, 
                          edgecolor="black")

        plt.suptitle(var_name + ' (sigma='+elevation+') at timestep ' + 
                 str(i), y=0.92,
                 x=0.51, fontsize=fz*1.2)
        plt.subplots_adjust(wspace=0.03)
        fig.patch.set_facecolor('xkcd:white')
    
        if i < 10:
            plt.savefig(temp_directory + '/00' + str(i) + '.png', transparent=True,
                    bbox_inches='tight', pad_inches=0.02)
        if (i >= 10) and (i < 100):
            plt.savefig(temp_directory + '/0' + str(i) + '.png', transparent=True,
                    bbox_inches='tight', pad_inches=0.02)
        if i >= 100:
            plt.savefig(temp_directory + '/' + str(i) + '.png', transparent=True, 
                    bbox_inches='tight',pad_inches=0.02)

        plt.close()
    imgs = sorted(glob.glob(temp_directory + '/*.png'))
    if len(imgs) < 24:
        duration = 300
    else:
        duration = int(300.*30./len(imgs))
    my_dir = base_dir+save_dir+'/'+var_name+'_sigma_of_'+elevation+'_'+'.gif'
    img, *my_imgs = [Image.open(f) for f in imgs]
    img.save(fp=my_dir, format='GIF', append_images=my_imgs,
             save_all=True, duration=duration, loop=0)
    shutil.rmtree(temp_directory)


def animation_generator_gif(
    x_whirl: np.ndarray,
    x_wb: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    unit_labels: str,
    var_name: str,
    elevation: str,
    time_axis: int,
    base_dir: str,
    save_dir: str='Animations',
    title_whirl: str = 'Prediction',
    title_wb: str = 'Target',
    title_difference: str = 'Target - Prediction',
    alpha: float=0.1,
    alpha_diff: float=0.25,
    data_percentile: int = 95,
    diff_percentile: int = 95,
    pos_neg_colorbar:bool=False,
    vmin: float = None,
    vmax: float = None,
    vmin_diffs: float = None,
    vmax_diffs: float = None,
    contour_levels: int = 16,
    contour_precision: int = None,
    cmap: str = 'RdBu',
    cmap_diffs: str = 'seismic',
    fps: int = 2, # about 2-3 second rendering in html per gif
    time_interval: str = 'hour',
    ):
    # Make it here
    if pos_neg_colorbar == False:
        if (vmax is None) and (vmin is None):
            vmax, vmin = min_max_getter(x_wb, diff_percentile=data_percentile)
            if (vmax == 0.0) and (vmin == 0.0):
                vmax = np.max(np.abs(x_wb))
                vmin = -1.0*np.max(np.abs(x_wb))
    
    else:
        if (vmax is None) and (vmin is None):
            vmax, vmin = min_max_getter_diff(x_wb, data_percentile=data_percentile)
            if (vmax == 0.0) and (vmin == 0.0):
                vmax = np.max(np.abs(x_wb))
                vmin = -1.0*np.max(np.abs(x_wb))
    
    if (vmax_diffs is None) and (vmin_diffs is None):
        vmax_diffs, vmin_diffs = min_max_getter_diff(x_wb - x_whirl, data_percentile=diff_percentile)
    
    #if contour_precision is None:
    #    contour_precision = find_exp(x_wb.mean()) + 1

    #new code
    data_range = vmax - vmin

    # Set precision based on the data range and contour levels
    if data_range > 0:
        contour_step = data_range / contour_levels
        if contour_step >= 1:
            contour_precision = 0
        else:
            # Compute the number of decimal places to avoid all zero levels
            contour_precision = int(np.ceil(-np.log10(contour_step)))
    else:
        contour_precision = 0

    #end new code
    levels = np.round(np.arange(vmin, vmax + (
        vmax - vmin) / contour_levels, (
        vmax - vmin) / contour_levels), decimals=contour_precision)
    
    levels_diffs = np.round(np.arange(vmin_diffs, vmax_diffs + (
        vmax_diffs - vmin_diffs) / contour_levels, (
        vmax_diffs - vmin_diffs) / contour_levels), decimals=contour_precision)

    y_grids, x_grids = np.meshgrid(lat, lon)
    temp_directory = tempfile.mkdtemp('temp')
    if os.path.isdir(base_dir+"/"+save_dir):
        pass
    else:
        os.mkdir(base_dir+"/"+save_dir)
        
    for i in range(x_wb.shape[time_axis]):
        fig, ax = plt.subplots(
        1, 3, figsize=(25, 7), subplot_kw=dict(projection=cartopy.crs.Robinson(central_longitude=180)))
        
        
        temp_truth = np.take(x_wb, i, axis=time_axis)
        im_a = ax[0].pcolormesh(x_grids, y_grids, temp_truth.T, cmap=cmap,
                            vmin=vmin, vmax=vmax, transform=cartopy.crs.PlateCarree(), alpha=alpha, animated=True)
        ax[0].set_title(title_wb, fontsize=fz)
        ax[0].set_global()
        ax[0].coastlines(linewidth=1, edgecolor="black")
        ax[0].gridlines()
        ax[0].add_feature(cartopy.feature.BORDERS, linewidth=1, 
                          edgecolor="black")
        cax = fig.add_axes([0.17, 0.13, 0.43, 0.03])
        cb = fig.colorbar(im_a, orientation='horizontal', cax=cax, ticks=levels[::2], boundaries=levels)
        cb.set_label(unit_labels, fontsize=fz*1.5)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        cb.solids.set(alpha=1)
        
        temp_prediction = np.take(x_whirl, i, axis=time_axis)
        ax[1].pcolormesh(x_grids, y_grids, temp_prediction.T, cmap=cmap, vmin=vmin,
            vmax=vmax, transform=cartopy.crs.PlateCarree(), alpha=alpha, animated=True)
        ax[1].set_title(title_whirl, fontsize=fz)
        ax[1].coastlines(linewidth=1, edgecolor='black')
        ax[1].gridlines()
        ax[1].add_feature(cartopy.feature.BORDERS, linewidth=1, 
                          edgecolor="black")

        diffs = temp_truth - temp_prediction

        im_c = ax[2].pcolormesh(x_grids, y_grids, diffs.T, cmap=cmap_diffs, vmin=vmin_diffs, 
                            vmax=vmax_diffs, animated=True, alpha=alpha_diff, transform=cartopy.crs.PlateCarree())
        ax[2].set_title(title_difference, fontsize=fz)
        ax[2].coastlines(linewidth=1, edgecolor="black")
        ax[2].gridlines()
        ax[2].add_feature(cartopy.feature.BORDERS, linewidth=1, edgecolor="black")
        
        cax_2 = fig.add_axes([0.66, 0.13, 0.23, 0.03])
        cb_2 = fig.colorbar(
        im_c, orientation='horizontal', cax=cax_2, ticks=levels_diffs[::2], boundaries=levels_diffs)
        cb_2.set_label(unit_labels, fontsize=fz*1.5)
        cb_2.formatter.set_powerlimits((0, 0))
        cb_2.update_ticks()
        cb_2.solids.set(alpha=1)
        
        plt.suptitle(var_name + ' (sigma='+elevation+') at timestep ' + 
                 str(i), y=0.92,
                 x=0.51, fontsize=fz*1.2)
        plt.subplots_adjust(wspace=0.03)
        fig.patch.set_facecolor('xkcd:white')
    
        if i < 10:
            plt.savefig(temp_directory + '/00' + str(i) + '.png', transparent=True,
                    bbox_inches='tight', pad_inches=0.02)
        if (i >= 10) and (i < 100):
            plt.savefig(temp_directory + '/0' + str(i) + '.png', transparent=True,
                    bbox_inches='tight', pad_inches=0.02)
        if i >= 100:
            plt.savefig(temp_directory + '/' + str(i) + '.png', transparent=True, 
                    bbox_inches='tight',pad_inches=0.02)

        plt.close()
    imgs = sorted(glob.glob(temp_directory + '/*.png'))
    if len(imgs) < 24:
        duration = 300
    else:
        duration = int(300.*30./len(imgs))
    my_dir = base_dir+"/"+save_dir+'/'+var_name+'_sigma_of_'+elevation+'_'+'.gif'
    img, *my_imgs = [Image.open(f) for f in imgs]
    img.save(fp=my_dir, format='GIF', append_images=my_imgs,
             save_all=True, duration=duration, loop=0)
    shutil.rmtree(temp_directory)