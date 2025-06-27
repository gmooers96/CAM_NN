import numpy as np
from sklearn import preprocessing, metrics
import sklearn
import scipy.stats
import pickle
import warnings
import atmos_physics as atmos_physics
import pandas as pd
from netCDF4 import Dataset
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from torch import nn, optim
import xarray as xr
import pdb
import math
import matplotlib.pyplot as plt

def symlog_transform(x, linthresh):
    return np.sign(x) * np.log10(1 + np.abs(x / linthresh))

def symlog_inverse(x, linthresh):
    return np.sign(x) * linthresh * (10**np.abs(x) - 1)

def compute_linthresh(x, quantile=5):
    """Computes linthresh from a quantile of the non-zero abs values."""
    x_nonzero = np.abs(x[np.abs(x) > 1e-8])  # avoid log(0)
    return np.percentile(x_nonzero, quantile)

def standardize_inputs(arr, mean=None, std=None):
    """
    Standardizes each vertical level separately.

    Parameters:
        arr (numpy.ndarray): Input array of shape (49, sample).
        mean (numpy.ndarray or None): Precomputed mean values (shape: (49,)).
        std (numpy.ndarray or None): Precomputed standard deviation values (shape: (49,)).

    Returns:
        standardized_arr (numpy.ndarray): Standardized array with the same shape as `arr`.
        mean (numpy.ndarray): Mean values computed for each vertical level.
        std (numpy.ndarray): Standard deviation values computed for each vertical level.
    """
    if mean is None:
        mean = arr.mean(axis=1, keepdims=True)  # Compute mean for each vertical level
    if std is None:
        std = arr.std(axis=1, keepdims=True)  # Compute std for each vertical level

    standardized_arr = (arr - mean) / std
    return standardized_arr, mean, std

def standardize_outputs(arr, mean=None, std=None):
    if mean is None:
        mean = arr.mean()
    if std is None:
        std = arr.std()
    return (arr - mean) / std, mean, std


def train_percentile_calc(percent, ds):
    sample = ds.sample
    lon = ds.lon
    lat = ds.lat
    times = int(len(sample) / (len(lon)*len(lat)))
    proportion = math.floor(times*percent/100.)
    splicer = int(proportion*len(lon)*len(lat))
    return splicer



# TODO:@gmooers add in a z splicer and a pole splicer
def LoadDataStandardScaleData_v3(traindata,
                                 testdata,
                                 input_vert_vars,
                                 output_vert_vars,
                                 training_data_volume,
                                 test_data_volume,
                                 chunk={'sample': 1024, 'lat': 426, 'lon': 768, 'z': 49},
                                 weights=None,
                                 sym_log = False,
                                 t_adv_vert_scale= False,
                                 save_data_figs = False,
                                 save_path=None,
                                ):
    """

    TODO: gmooers
    """

    # open the xarray data file
    train_variables = xr.open_mfdataset(traindata, chunks=chunk, combine='nested', concat_dim='sample')
    test_variables = xr.open_dataset(testdata, chunks=chunk)

    # calculate a training data percentage
    # i think this is wrong -- function needs to include lat -- would explain the error in test data
    training_data_percentage = train_percentile_calc(training_data_volume, train_variables)
    test_data_percentage = train_percentile_calc(test_data_volume, test_variables)

    # get the apropriate slice -- this is temporarily modified
    train_variables = train_variables.isel(sample=slice(0, training_data_percentage))
    test_variables = test_variables.isel(sample=slice(0, test_data_percentage))
    print("Training data sample size", len(train_variables.sample))
    print("Test data sample size", len(test_variables.sample))
    #train_variables = train_variables.isel(sample=slice(0, 1000))
    #test_variables = test_variables.isel(sample=slice(0, 1000))
    #print("hi")


    # not sure what this line does or why necessary
    my_train_variables = train_variables.variables 
    my_test_variables = test_variables.variables

    # TODO:@gmooers -- make the below a stronger code that can work for None type
    if weights is not None:
        weight_variables = xr.open_dataset(weights, chunks=chunk)
        my_weight_variables = weight_variables.norms.values

    scaled_inputs = {
        'train': {},
        'mean': {},
        'std': {},
        'test': {},
    }

    scaled_outputs = {
        'train': {},
        'mean': {},
        'std': {},
        'test': {},
    }

    # deal with the training data -- loop over each variable in the input vector
    for i in range(len(input_vert_vars)):
        print(input_vert_vars[i])
        # extract a given input variable
        my_train_data = my_train_variables[input_vert_vars[i]]
        my_test_data = my_test_variables[input_vert_vars[i]]

        # if it is a scalar (sfc_pres, land_frac), add a vertical dim of 1 in front of the array
        scalar = False
        if len(my_train_data.shape) == 1:
            my_train_data = xr.DataArray(my_train_data).expand_dims(dim={"z": 1})
            my_train_data = my_train_data.transpose("z", ...)

            my_test_data = xr.DataArray(my_test_data).expand_dims(dim={"z": 1})
            my_test_data = my_test_data.transpose("z", ...)
            scalar = True
        else:
            scalar = False

        #Normalize data at each vertical level (dimension 0)

        # original code
        #standardized_arr_inputs, mean, std = standardize_inputs(my_train_data.values)
        #scaled_inputs['train'][input_vert_vars[i]] = standardized_arr_inputs

        #scaled_inputs['mean'][input_vert_vars[i]] = np.squeeze(mean)
        #scaled_inputs['std'][input_vert_vars[i]] = np.squeeze(std)

        #standardized_arr_inputs, mean, std = standardize_inputs(my_test_data.values, mean=mean, std=std)
        #scaled_inputs['test'][input_vert_vars[i]] = standardized_arr_inputs

        # changed to avoid scaling the inputs already 0-1

        # Compute quick summary stats
        data_min = np.nanmin(my_train_data.values)
        data_max = np.nanmax(my_train_data.values)
        is_bounded_0_1 = (data_min >= 0.0) and (data_max <= 1.0)
        
        if (is_bounded_0_1) and (scalar == True):
            print(f"Skipping normalization for {input_vert_vars[i]} (bounded [0,1])")
            scaled_inputs['train'][input_vert_vars[i]] = my_train_data.values
            scaled_inputs['mean'][input_vert_vars[i]] = 0.0
            scaled_inputs['std'][input_vert_vars[i]] = 1.0
            standardized_arr_inputs = my_test_data.values
            scaled_inputs['test'][input_vert_vars[i]] = standardized_arr_inputs
            
        else:
            # Normalize data at each vertical level (dimension 0)
            standardized_arr_inputs, mean, std = standardize_inputs(my_train_data.values)
            scaled_inputs['train'][input_vert_vars[i]] = standardized_arr_inputs
        
            scaled_inputs['mean'][input_vert_vars[i]] = np.squeeze(mean)
            scaled_inputs['std'][input_vert_vars[i]] = np.squeeze(std)
        
            standardized_arr_inputs, mean, std = standardize_inputs(my_test_data.values, mean=mean, std=std)
            scaled_inputs['test'][input_vert_vars[i]] = standardized_arr_inputs

        if save_data_figs == True:
                plot_histograms_before_after_norm(
                    array_before_norm=my_test_data.values.flatten(), 
                    array_after_norm=standardized_arr_inputs.flatten(), 
                    var_name=input_vert_vars[i], 
                    savepath=save_path,
                )
    
    print('Outputs') 
    
    for i in range(len(output_vert_vars)):
        print(output_vert_vars[i])
        # extract a given output variable
        my_train_data = my_train_variables[output_vert_vars[i]]
        my_test_data = my_test_variables[output_vert_vars[i]]


        if (sym_log == True) and output_vert_vars[i] == 'T_adv_out':
            linthresh = compute_linthresh(my_train_data.values)
            print(f"Using symlog for {output_vert_vars[i]} with linthresh = {linthresh}")

            train_symlog = symlog_transform(my_train_data, linthresh)
            test_symlog = symlog_transform(my_test_data, linthresh)

            mean = train_symlog.mean()
            std = train_symlog.std()

            scaled_outputs['train'][output_vert_vars[i]] = (train_symlog - mean) / std
            scaled_outputs['test'][output_vert_vars[i]] = (test_symlog - mean) / std
            scaled_outputs['mean'][output_vert_vars[i]] = mean
            scaled_outputs['std'][output_vert_vars[i]] = std
            scaled_outputs['linthresh'] = {output_vert_vars[i]: linthresh}  # Save for postprocessing
            
        elif (t_adv_vert_scale == True) and output_vert_vars[i] == 'T_adv_out':
            print("Using vertical scaling on T_adv")
            standardized_arr_inputs, mean, std = standardize_inputs(my_train_data.values)
            scaled_outputs['train'][output_vert_vars[i]] = standardized_arr_inputs

            scaled_outputs['mean'][output_vert_vars[i]] = np.squeeze(mean)
            scaled_outputs['std'][output_vert_vars[i]] = np.squeeze(std)

            standardized_arr_inputs, mean, std = standardize_inputs(my_test_data.values, mean=mean, std=std)
            scaled_outputs['test'][input_vert_vars[i]] = standardized_arr_inputs
            
        else:
            #Normalize data for each entire vertical column
            standardized_arr_outputs, mean, std = standardize_outputs(my_train_data.values)
            scaled_outputs['train'][output_vert_vars[i]] = standardized_arr_outputs

            scaled_outputs['mean'][output_vert_vars[i]] = np.squeeze(mean)
            scaled_outputs['std'][output_vert_vars[i]] = np.squeeze(std)

            standardized_arr_outputs, mean, std = standardize_outputs(my_test_data.values, mean=mean, std=std)
            scaled_outputs['test'][output_vert_vars[i]] = standardized_arr_outputs

        if save_data_figs == True:
            plot_histograms_before_after_norm(
                array_before_norm=my_test_data.values.flatten(), 
                array_after_norm=standardized_arr_outputs.flatten(), 
                var_name=output_vert_vars[i], 
                savepath=save_path,
            )
            


    # build a numpy array for training/test inputs/outputs of shape (sample, features)
   
    train_inputs = np.concatenate(
        [scaled_inputs['train'][x] for x in scaled_inputs['train']], 0)
    test_inputs = np.concatenate(
        [scaled_inputs['test'][x] for x in scaled_inputs['test']], 0)
    train_outputs = np.concatenate(
        [scaled_outputs['train'][x] for x in scaled_outputs['train']], 0)
    test_outputs = np.concatenate(
        [scaled_outputs['test'][x] for x in scaled_outputs['test']], 0)

    ### Debugging code -- comment in if reduced data size

    #has_nans = np.isnan(train_inputs).any()
    #print("Any NaNs in train_inputs?", has_nans)
    #has_nans = np.isnan(test_inputs).any()
    #print("Any NaNs in test_inputs?", has_nans)
    #has_nans = np.isnan(train_outputs).any()
    #print("Any NaNs in train_outputs?", has_nans)
    #has_nans = np.isnan(test_outputs).any()
    #print("Any NaNs in test_outputs?", has_nans)

    #num_nans = np.isnan(train_inputs).sum()
    #print("Number of NaNs in train_inputs:", num_nans)
    #num_nans = np.isnan(test_inputs).sum()
    #print("Number of NaNs in test_inputs:", num_nans)
    #num_nans = np.isnan(train_outputs).sum()
    #print("Number of NaNs in train_outputs:", num_nans)
    #num_nans = np.isnan(test_outputs).sum()
    #print("Number of NaNs in test_outputs:", num_nans)

    train_inputs = np.nan_to_num(train_inputs, nan=0.0)
    test_inputs = np.nan_to_num(test_inputs, nan=0.0)
    train_outputs = np.nan_to_num(train_outputs, nan=0.0)
    test_outputs = np.nan_to_num(test_outputs, nan=0.0)

    ### End of debugging code
    
    return train_inputs.T, test_inputs.T, train_outputs.T, test_outputs.T, scaled_inputs, scaled_outputs


def undo_scaling_predictions(scaled_array, scaler_dict, vertical_dimension, output_variables, sym_log=False, t_adv_vert_scale=False):
    """
    Undo the scaling of the given scaled numpy array using the provided scalers.

    Returns:
    np.ndarray: The unscaled numpy array with the same shape as the input.
    """
    num_samples = scaled_array.shape[0]

    # Initialize the unscaled array with the same shape as the input
    unscaled_array = np.zeros_like(scaled_array)

    array_start_index = 0
    array_end_index = vertical_dimension

    for key, value in scaler_dict['mean'].items():

        scaled_array_slice = scaled_array[:,array_start_index:vertical_dimension + array_start_index]
        mean = scaler_dict['mean'][key]
        std = scaler_dict['std'][key]
        
        if key == 'T_adv_out' and sym_log == True:
            linthresh = scaler_dict['linthresh'][key]
            symlog_unscaled = scaled_array_slice * std + mean
            unscaled_array[:, array_start_index:array_start_index+vertical_dimension] = symlog_inverse(symlog_unscaled, linthresh)
            
            
        else:
            unscaled_array[:,array_start_index:vertical_dimension + array_start_index] = ((scaled_array_slice * std) 
                                                               + mean)
        array_start_index = array_start_index + vertical_dimension


    return unscaled_array


def undo_scaling_targets(scaled_array, scaler_dict, vertical_dimension, output_variables, sym_log=False, t_adv_vert_scale=False):
    """
    Undo the scaling of the given scaled numpy array using the provided scalers.

    Returns:
    np.ndarray: The unscaled numpy array with the same shape as the input.
    """
    num_samples = scaled_array.shape[0]

    # Initialize the unscaled array with the same shape as the input
    unscaled_array = np.zeros_like(scaled_array)

    array_start_index = 0
    array_end_index = vertical_dimension

    for key, value in scaler_dict['mean'].items():
        scaled_array_slice = scaler_dict['test'][key].T
        mean = scaler_dict['mean'][key]
        std = scaler_dict['std'][key]

        if key == 'T_adv_out' and sym_log == True:
            linthresh = scaler_dict['linthresh'][key]
            symlog_unscaled = scaled_array_slice * std + mean
            unscaled_array[:, array_start_index:array_start_index+vertical_dimension] = symlog_inverse(symlog_unscaled, linthresh)
             
        else:
            unscaled_array[:,array_start_index:vertical_dimension + array_start_index] = ((scaled_array_slice * std) 
                                                               + mean)

        array_start_index = array_start_index + vertical_dimension
        return unscaled_array

def plot_histograms_before_after_norm(array_before_norm, array_after_norm, var_name, savepath):
    """
    Plots histograms of the data before and after normalization, and saves them as a .png file.
    
    Args:
        array_before_norm (numpy.ndarray): Data array before normalization.
        array_after_norm (numpy.ndarray): Data array after normalization.
        var_name (str): Name of the variable, used in the save file name.
        savepath (str): Path where the histogram image will be saved.
    """
    # Create the histogram plots

    before_percentiles = np.percentile(array_before_norm, [1, 5, 50, 95, 99])
    after_percentiles = np.percentile(array_after_norm, [1, 5, 50, 95, 99])

    print(f"{var_name} - Before normalization percentiles (1, 5, 50, 95, 99): {before_percentiles}")
    print(f"{var_name} - After normalization percentiles (1, 5, 50, 95, 99): {after_percentiles}")
    
    plt.figure(figsize=(26, 12))

    # Plot before normalization
    plt.subplot(1, 2, 1)
    plt.hist(array_before_norm, bins=100, alpha=0.7, color='blue')
    plt.title(f'{var_name} - Before Normalization')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.yscale("log")

    # Plot after normalization
    plt.subplot(1, 2, 2)
    plt.hist(array_after_norm, bins=100, alpha=0.7, color='green')
    plt.title(f'{var_name} - After Normalization')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.yscale("log")

    # Save the figure with a name based on the variable name
    plt.tight_layout()
    save_file = f"{savepath}/{var_name}_histograms.png"
    plt.savefig(save_file)

    # Close the plot to free memory
    plt.close()

    print(f"Histograms saved to {save_file}")






