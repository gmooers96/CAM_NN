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


def train_percentile_calc(percent, ds):
    sample = ds.sample
    lon = ds.lon
    times = int(len(sample) / len(lon))
    proportion = math.floor(times*percent/100.)
    splicer = int(proportion*len(lon))
    return splicer

# TODO:@gmooers add in a z splicer and a pole splicer
def LoadDataStandardScaleData_v2(traindata,
                                 testdata,
                                 input_vert_vars,
                                 output_vert_vars,
                                 poles,
                                 training_data_volume,
                                 chunk={'sample': 1024, 'lat': 426, 'lon': 768, 'z': 49},
                                 weights=None,
                                ):
    """

    TODO: gmooers
    """

    # open the xarray data file
    # works on the safe janni data
    #train_variables = xr.open_mfdataset(traindata, chunks=chunk)

    train_variables = xr.open_mfdataset(traindata, chunks=chunk, combine='nested', concat_dim='sample')

    # calculate a training data percentage
    training_data_percentage = train_percentile_calc(training_data_volume, train_variables)

    # get the apropriate slice -- this is temporarily modified
    train_variables = train_variables.isel(sample=slice(0, training_data_percentage))
    #train_variables = train_variables.isel(sample=slice(0, 1000)) # temporary

    # not sure what this line does or why necessary
    my_train_variables = train_variables.variables 

    # repeat above but for the test data
    test_variables = xr.open_mfdataset(testdata, chunks=chunk, combine='nested', concat_dim='sample')
    #test_variables = test_variables.isel(sample=slice(0, 1000)) # temporary
    my_test_variables = test_variables.variables

    # TODO:@gmooers -- make the below a stronger code that can work for None type
    if weights is not None:
        weight_variables = xr.open_dataset(weights, chunks=chunk)
        my_weight_variables = weight_variables.norms.values

    # build dictionaries for the inputs and outputs fro train/test data
    train_input_variable_dict = dict()
    train_output_variable_dict = dict()
    test_input_variable_dict = dict()
    test_output_variable_dict = dict()

    # I think preprocessing dicts to hold a standard scalar
    train_inputs_pp_dict = dict()
    train_outputs_pp_dict = dict()

    # new dictionaries for the transformed (scaled) data 
    train_inputs_transformed_data = dict()
    test_inputs_transformed_data = dict()
    train_outputs_transformed_data = dict()
    test_outputs_transformed_data = dict()
    
    print('Inputs')

    # deal with the training data -- loop over each variable in the input vector
    for i in range(len(input_vert_vars)):
        print(input_vert_vars[i])
        # extract a given input variable
        my_train_data = my_train_variables[input_vert_vars[i]]
        my_test_data = my_test_variables[input_vert_vars[i]]

        # if it is a scalar (sfc_pres, land_frac), add a vertical dim of 1 in front of the array
        if len(my_train_data.shape) == 1:
            my_train_data = xr.DataArray(my_train_data).expand_dims(dim={"z": 1})
            my_train_data = my_train_data.transpose("z", ...)

            my_test_data = xr.DataArray(my_test_data).expand_dims(dim={"z": 1})
            my_test_data = my_test_data.transpose("z", ...)


        # put the data in the form (sample, shape {e.g. z height}) to correctly pass into StandardScaler
        # convert to numpy here for simplicity -- this may need to change later for memory
        train_input_variable_dict[input_vert_vars[i]] = my_train_data.transpose('sample', 'z').values
        test_input_variable_dict[input_vert_vars[i]] = my_test_data.transpose('sample', 'z').values

        # put standard scalars in a dictionary to process each input separately
        train_inputs_pp_dict[input_vert_vars[i]] = sklearn.preprocessing.StandardScaler()


        #fit the standard scalar to the training data
        
        # case of vertical column -- e.g. temp or humidity -- need to scale each level separately
        if train_input_variable_dict[input_vert_vars[i]].shape[1] > 1:

            # fit the standard scalar to the training data; 0 ensures it is applied colm
            np.apply_along_axis(
                lambda x: train_inputs_pp_dict[input_vert_vars[i]].fit(
                    x.reshape(-1, 1)), 0, train_input_variable_dict[input_vert_vars[i]])
            
            # scale the training data based on the fit applied above
            # the numpy squeeze gets rid of the extra dimension created by the level by level scaling
            train_inputs_transformed_data[input_vert_vars[i]] = np.apply_along_axis(
                lambda x: train_inputs_pp_dict[input_vert_vars[i]].transform(
                    x.reshape(-1, 1)), 0, train_input_variable_dict[input_vert_vars[i]]).squeeze()

            # scale the test data based on the fit above
            test_inputs_transformed_data[input_vert_vars[i]] = np.apply_along_axis(
                lambda x: train_inputs_pp_dict[input_vert_vars[i]].transform(
                    x.reshape(-1, 1)), 0, test_input_variable_dict[input_vert_vars[i]]).squeeze()
            
        # case of scalar with z of 1
        else:
            # fit the standard scalar to the training data
            train_inputs_pp_dict[input_vert_vars[i]].fit(train_input_variable_dict[input_vert_vars[i]])

            # scale the training data based on the fit applied above
            train_inputs_transformed_data[input_vert_vars[i]] = train_inputs_pp_dict[input_vert_vars[i]].transform(
                train_input_variable_dict[input_vert_vars[i]])

            # scale the test data based on the fit applied above
            test_inputs_transformed_data[input_vert_vars[i]] = train_inputs_pp_dict[input_vert_vars[i]].transform(
                test_input_variable_dict[input_vert_vars[i]])
    
    print('Outputs')  
    
    for i in range(len(output_vert_vars)):
        print(output_vert_vars[i])
        # extract a given output variable
        my_train_data = my_train_variables[output_vert_vars[i]]
        my_test_data = my_test_variables[output_vert_vars[i]]

        # put the data in the form (sample, shape {e.g. z height}) to correctly pass into StandardScaler
        # convert to numpy here for simplicity -- this may need to change later for memory
        train_output_variable_dict[output_vert_vars[i]] = my_train_data.transpose('sample', 'z').values
        test_output_variable_dict[output_vert_vars[i]] = my_test_data.transpose('sample', 'z').values

        # put standard scalars in a dictionary to process each output separately
        train_outputs_pp_dict[output_vert_vars[i]] = sklearn.preprocessing.StandardScaler()

        # fit the standard scalar to the training data
        train_outputs_pp_dict[output_vert_vars[i]].fit(train_output_variable_dict[output_vert_vars[i]])

        # scale the training data based on the fit applied above
        train_outputs_transformed_data[output_vert_vars[i]] = train_outputs_pp_dict[output_vert_vars[i]].transform(
            train_output_variable_dict[output_vert_vars[i]])

        # scale the test data based on the fit applied above
        test_outputs_transformed_data[output_vert_vars[i]] = train_outputs_pp_dict[output_vert_vars[i]].transform(
            test_output_variable_dict[output_vert_vars[i]])

        


    # build a numpy array for training/test inputs/outputs of shape (sample, features)
   
    train_inputs = np.concatenate(
        [train_inputs_transformed_data[x] for x in train_inputs_transformed_data], 1)
    test_inputs = np.concatenate(
        [test_inputs_transformed_data[x] for x in test_inputs_transformed_data], 1)
    train_outputs = np.concatenate(
        [train_outputs_transformed_data[x] for x in train_outputs_transformed_data], 1)
    test_outputs = np.concatenate(
        [test_outputs_transformed_data[x] for x in test_outputs_transformed_data], 1)

    # this may not be necessary
    train_inputs_original = np.concatenate(
        [train_input_variable_dict[x] for x in train_input_variable_dict], 1)
    test_inputs_original = np.concatenate(
        [test_input_variable_dict[x] for x in test_input_variable_dict], 1)
    train_outputs_original = np.concatenate(
        [train_output_variable_dict[x] for x in train_output_variable_dict], 1)
    test_outputs_original = np.concatenate(
        [test_output_variable_dict[x] for x in test_output_variable_dict], 1)

    # TODO:@gmooers -- chanbge the outputs to only what is necessary
    return (train_inputs, test_inputs, train_outputs, test_outputs, test_outputs_original, train_inputs_pp_dict, train_outputs_pp_dict, "Std_Scalar")


def reshape_cos_ys(z, ind_z, y, is_sfc=False):
    if is_sfc:
        z = z.swapaxes(0, 1)
        z2 = np.empty((0))
    else:
        z = z[ind_z, :, :]
        z = z.swapaxes(0, 2)
        z2 = np.empty((0, sum(ind_z)))
    n_ex = z.shape[0]
    for i, yval in enumerate(y):
        # cosine of pseudo latitude
        Ninds = int(n_ex * np.cos((yval-np.mean(y))/6.37122e6))
        if is_sfc:
            z2 = np.concatenate((z2, z[0: Ninds, i]), axis=0)
        else:
            z2 = np.concatenate((z2, z[0:Ninds, i, :]), axis=0)
    return z2


def     reshape_all_ys(z, ind_z):
    # Expects data to be n_z n_y n_samples and returns
    # (n_y*n_samp n_z)
    z = z[ind_z, :, :]
    z = z.swapaxes(0, 2)
    return np.reshape(z, (-1, sum(ind_z)))

def reshape_all_ys_4d(z, ind_z):
    # Expects data to be n_z n_y n_samples and returns
    # (n_y*n_samp n_z)
    if len(z.shape) == 4:
        z = z[ind_z, :, :, :]
        z = z.swapaxes(0, 3)
    elif len(z.shape) == 3:
        z = z[:, :, :]
        z = np.transpose(z, axes=(2, 0, 1))
    else:
        TypeError('Cannot reshape this becuse dealing only with 3 and 4D arrays')

    return np.reshape(z, (-1, sum(ind_z)))


def reshape_one_y(z, ind_z, ind_y):
    # Expects data to be (n_z n_y n_samples) and returns (n_samp n_z)
    if len(z.shape) == 3 and ind_z.shape[0] > 1:
        z = z[ind_z, ind_y, :]
        z = z.swapaxes(0, 1)
    elif len(z.shape) == 3 and ind_z.shape[0] == 1:
        z = z[ind_y, :, :]
        # z = z.swapaxes(0, 1)
        z = np.reshape(z,(-1,1))
    elif len(z.shape) == 2:
        z = z[ind_y, :]
        z = np.reshape(z,(z.shape[0],1))
        # z = z.swapaxes(0, 1)
    elif len(z.shape) == 4:
        z = z[ind_z, ind_y,:,  :]
        z = np.reshape(z, (z.shape[0], -1))
        z = z.swapaxes(0, 1)
    else:
        raise TypeError('number of dimensions is unexpected')
    return z

def pack_f(T, q, axis=1):
    """Combines input profiles"""
    return np.concatenate((T, q), axis=axis)

def pack_list(v, vars_list ,axis=1):
    """gets a dictionary and makes it a large array"""
    inp_array = v[vars_list[0]] #initialize the array
    for var in vars_list[1:]:
        inp_array = np.concatenate((inp_array, v[var]), axis)
    return inp_array

def unpack_list(l_array, vars_list, vars_z_size ,axis=1):
    """Takes a large array, and give back a dictionary with the relevant fields"""
    v = dict()
    curr_dim = 0
    if sum(vars_z_size) >1:
        for name, dim in zip(vars_list, vars_z_size):
            v[name] = l_array[:, curr_dim:dim + curr_dim]
            curr_dim = curr_dim + dim
    else: #The case I only have one dimention....
        v[vars_list[0]] = l_array[:,None]
    return v


def pack_f_extended(T, q, u, v, w, axis=1):
    """Combines input profiles"""
    return np.concatenate((T, q, u, v, w), axis=axis)

def unpack_f(data, vari, axis=1):
    """Reverse pack operation"""
    N = int(data.shape[axis]/2)
    varipos = {'T': np.arange(N), 'q': np.arange(N,2*N)}
    out = np.take(data, varipos[vari], axis=axis)
    return out

def unpack_f_extended(data, vari, axis=1, wind_input=False):
    """Reverse pack operation"""
    if wind_input:
        Num_vars = int(data.shape[axis]/48)
        N = int(data.shape[axis]/Num_vars)
    else:
        N = int(data.shape[axis] / 2)

    varipos = {'T': np.arange(N), 'q': np.arange(N,2*N)}
    out = np.take(data, varipos[vari], axis=axis)
    return out

def pack_o(d1, d2, axis=1):
    """Combines T & q profiles"""
    return np.concatenate((d1, d2), axis=axis)


def choose_output_from_dic():
    """Gets an output from dictionary of outputs"""

def unpack_o(data, vari, axis=1):
    """Reverse pack operation"""
    N = int(data.shape[axis]/2)
    varipos = {'T': np.arange(N), 'q': np.arange(N, 2*N)}
    out = np.take(data, varipos[vari], axis=axis)
    return out

# Initialize & fit scaler Modified by Yani to fit for each generalized feature together
def init_pp_generalized(ppi, dict_data, input_vert_vars,scale_per_column):
    # Initialize list of scaler objects
    pp_dict = dict()
    for name in input_vert_vars:
        if ppi['name'] == 'MinMax':
            pp_dict[name] = preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0))
            pp_dict[name].fit(np.reshape(dict_data[name],(-1,1)))
        elif ppi['name'] == 'MaxAbs':
            pp_dict[name] = preprocessing.MaxAbsScaler()
            pp_dict[name].fit(np.reshape(dict_data[name],(-1,1)))
        elif ppi['name'] == 'StandardScaler':
            pp_dict[name] = preprocessing.StandardScaler()
            if scale_per_column: #If yes it should scale every feature differently!
                pp_dict[name].fit(dict_data[name])
            else:
                pp_dict[name].fit(np.reshape(dict_data[name],(-1,1)))
        elif  ppi['name'] == 'F_stscl_add':
            pp_dict[name] = preprocessing.StandardScaler()
            if scale_per_column:  # Should scle each column seperately - to verify!
                pp_dict[name].fit(dict_data[name])
                std_add = 0.00001
                X_std = np.std(dict_data[name], axis=0, dtype=np.float64) + std_add
                pp_dict[name].mean_ = np.mean(dict_data[name], axis=0, dtype=np.float64)
                pp_dict[name].var_ = X_std*X_std
            else:
                raise TypeError('Choosing F_stscl_add was coded to assume we scale features for each column1')

        elif ppi['name'] == 'RobustScaler':
            pp_dict[name] = preprocessing.RobustScaler()
            pp_dict[name].fit(np.reshape(dict_data[name],(-1,1)))
        elif ppi['name'] == 'SimpleO':
            if len(input_vert_vars) !=2:
                print('Note that all variables but the first two are not normalized with 1!')
                # raise ValueError('Incorrect scaler name')
            pp_dict[name] = [atmos_physics.cp, atmos_physics.L]
            for i in range(len(input_vert_vars) - 2):
                pp_dict[name].append(1)
        elif ppi['name'] == 'SimpleO_expz':
            if len(input_vert_vars) !=2:
                # raise ValueError('Incorrect scaler name')
                print('Note that all variables but the first two are not normalized with 1!')
            pp_dict[name] = [atmos_physics.cp, atmos_physics.L]
            for i in range(len(input_vert_vars) - 2):
                pp_dict[name].append(1)
            else:
                pp_dict[name] = [atmos_physics.cp, atmos_physics.L]
        elif ppi['name'] == 'NoScaler':
            pp_dict[name] = []
        else:
            raise ValueError('Incorrect scaler name')

    return pp_dict



# Initialize & fit scaler
def init_pp(ppi, raw_data):
    # Initialize list of scaler objects
    if ppi['name'] == 'MinMax':
        pp = preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0))
        pp.fit(raw_data)
    elif ppi['name'] == 'MaxAbs':
        pp = preprocessing.MaxAbsScaler() 
        pp.fit(raw_data)
    elif ppi['name'] == 'StandardScaler':
        pp = preprocessing.StandardScaler() 
        pp.fit(raw_data)
    elif ppi['name'] == 'RobustScaler':
        pp = preprocessing.RobustScaler()
        pp.fit(raw_data)
    elif ppi['name'] == 'SimpleO':
        pp = [atmos_physics.cp, atmos_physics.L]  
    elif ppi['name'] == 'SimpleO_expz':
        pp = [atmos_physics.cp, atmos_physics.L]  
    elif ppi['name'] == 'NoScaler':
        pp = []
    else:
        raise ValueError('Incorrect scaler name')

    return pp


# Transform data using initialized scaler
def transform_data_generalized(ppi, f_pp_dict, f_dict, input_vert_vars, z,scale_per_column=False,rewight_outputs=False,weight_list=[1,1]):
    if ppi['name'] == 'SimpleO':
        trans_data_dic = dict()
        for (index, name) in enumerate(input_vert_vars):
            trans_data_dic[name]= f_dict[name]*f_pp_dict[name][index]
    elif ppi['name'] == 'SimpleO_expz':
        trans_data_dic = dict()
        for (index, name) in enumerate(input_vert_vars):
            trans_data_dic[name]= f_dict[name]*f_pp_dict[name][index]*np.exp(-z/7000.0)
    elif ppi['name'] == 'NoScaler':
        trans_data_dic = f_dict
    elif ppi['name'] == 'F_stscl_add': 
        trans_data_dic = dict()
        for name in input_vert_vars:
            if scale_per_column:  
                trans_data_dic[name] = (f_dict[name] - f_pp_dict[name].mean_)/np.sqrt(f_pp_dict[name].var_)
            else:
                raise TypeError('Choosing F_stscl_add was coded to assume we scale features for each column')

    else: #Using standard scalar to renormalize
        trans_data_dic = dict()
        for name in input_vert_vars:
            if scale_per_column:
                try:
                    trans_data_dic[name] = f_pp_dict[name].transform(f_dict[name])
                except ValueError:
                    trans_data_dic[name] = f_pp_dict[name].fit_transform(f_dict[name])
            else: 
                trans_data_dic[name] = np.reshape(f_pp_dict[name].transform(np.reshape(f_dict[name],(-1,1))),(f_dict[name].shape[0],f_dict[name].shape[1]))

    if rewight_outputs: 
        print('rescaling outputs')
        print('length of the weight list is:', len(weight_list))
        for ind, name in enumerate(input_vert_vars,start=0):
            trans_data_dic[name] = trans_data_dic[name]*weight_list[ind]

    return trans_data_dic


def inverse_transform_data_generalized(ppi, f_pp_dict, f_dict, input_vert_vars,
                                       z,scale_per_column=False,rewight_outputs=False,weight_list=[1,1]):

    if rewight_outputs: 
        for ind, name in enumerate(input_vert_vars,start=0):
            f_dict[name] = f_dict[name]/weight_list[ind]

    if ppi['name'] == 'SimpleO':
        trans_data_dic = dict()
        for (index, name) in enumerate(input_vert_vars):
            trans_data_dic[name]= f_dict[name]/f_pp_dict[name][index]
    elif ppi['name'] == 'SimpleO_expz':
        trans_data_dic = dict()
        for (index, name) in enumerate(input_vert_vars):
            trans_data_dic[name]= f_dict[name]/f_pp_dict[name][index]/np.exp(-z/7000.0)
    elif ppi['name'] == 'NoScaler':
        trans_data_dic = f_dict
    else:
        trans_data_dic = dict()
        for name in input_vert_vars:
            if scale_per_column: 
                trans_data_dic[name] = f_pp_dict[name].inverse_transform(f_dict[name])
            else: 
                trans_data_dic[name] = np.reshape(f_pp_dict[name].inverse_transform(np.reshape(f_dict[name],(-1,1))),(f_dict[name].shape[0],f_dict[name].shape[1]))
    return_data = pack_list(trans_data_dic,input_vert_vars)
    # Return a numpy array of the transformed data output
    return return_data


def inverse_transform_data_generalized_big_data(ppi, f_pp_dict, f_dict, input_vert_vars,
                                       z,scale_per_column=False,rewight_outputs=False,weight_list=[1,1]):

    trans_data_dic = dict()
    if rewight_outputs: 
        for ind, name in enumerate(input_vert_vars,start=0):
            f_dict[name] = f_dict[name]/weight_list[ind]

    
    for name in input_vert_vars:
        if scale_per_column: 
            trans_data_dic[name] = f_pp_dict[name].inverse_transform(f_dict[name])
        else: 
            trans_data_dic[name] = np.reshape(f_pp_dict[name].inverse_transform(np.reshape(f_dict[name],(-1,1))),(f_dict[name].shape[0],f_dict[name].shape[1]))
    return_data = pack_list(trans_data_dic,input_vert_vars)
    # Return a numpy array of the transformed data output
    return return_data


# Transform data using initialized scaler
def transform_data(ppi, pp, raw_data, z):
    if ppi['name'] == 'SimpleO':
        T_data = unpack_o(raw_data, 'T')*pp[0]
        q_data = unpack_o(raw_data, 'q')*pp[1]
        return_data = pack_o(T_data, q_data)
    elif ppi['name'] == 'SimpleO_expz':
        T_data = unpack_o(raw_data, 'T')*pp[0]*np.exp(-z/7000.0)
        q_data = unpack_o(raw_data, 'q')*pp[1]*np.exp(-z/7000.0)
        return_data = pack_o(T_data, q_data)
    elif ppi['name'] == 'NoScaler':
        return_data = raw_data
    else:
        return_data = pp.transform(raw_data)

    # Return single transformed array as output
    return return_data 


# Apply inverse transformation to unscale data
def inverse_transform_data(ppi, pp, trans_data, z):
    if ppi['name'] == 'SimpleO':
        T_data = unpack_o(trans_data, 'T')/pp[0]
        q_data = unpack_o(trans_data, 'q')/pp[1]
        return_data = pack_o(T_data, q_data)
    elif ppi['name'] == 'SimpleO_expz':
        T_data = unpack_o(trans_data, 'T')/pp[0]*np.exp(z/7000.0)
        q_data = unpack_o(trans_data, 'q')/pp[1]*np.exp(z/7000.0)
        return_data = pack_o(T_data, q_data)
    elif ppi['name'] == 'NoScaler':
        return_data = trans_data
    else:
        return_data = pp.inverse_transform(trans_data)
    return return_data



def load_one_y(f_ppi, o_ppi, f_pp, o_pp, est, ind_y, datafile, max_z, input_vert_vars, output_vert_vars, input_vert_dim, output_vert_dim,
                 n_trn_exs, rain_only, no_cos, use_rh, wind_input = False,scale_per_column=False,
                 rewight_outputs=False,weight_list=[1,1],do_nn=False):
    """Returns n_samples 2*n_z array of true and predicted values
       at a given y"""
    # Load data
    f, o, y, z, rho, p, weight_list = \
        LoadData(datafile, max_z, input_vert_vars, output_vert_vars, all_ys=False, ind_y=ind_y,
                 verbose=False, n_trn_exs=None, rain_only=rain_only, 
                 no_cos=no_cos, use_rh=use_rh, wind_input = wind_input, rewight_outputs =rewight_outputs )
    # Calculate predicted output

    f_dict = unpack_list(f, input_vert_vars,input_vert_dim)
    f_scl_dict = transform_data_generalized(f_ppi, f_pp, f_dict, input_vert_vars, z,scale_per_column, rewight_outputs=False)
    # f_scl = transform_data(f_ppi, f_pp, f, z)
    f_scl = pack_list(f_scl_dict, input_vert_vars)

    if do_nn:
        tmp_f_scl = torch.from_numpy(f_scl)
        est.eval()
        o_pred_scl = est(tmp_f_scl.float()) 
        o_pred_scl = o_pred_scl.detach().numpy()
    else:
        o_pred_scl = est.predict(f_scl)
    o_pred_scl_dict = unpack_list(o_pred_scl, output_vert_vars,output_vert_dim)
    o_pred = inverse_transform_data_generalized(o_ppi, o_pp, o_pred_scl_dict,output_vert_vars, z,scale_per_column
                                                ,rewight_outputs=rewight_outputs,weight_list=weight_list)
    o_pred_dict = unpack_list(o_pred, output_vert_vars,output_vert_dim)

    o_dict = unpack_list(o, output_vert_vars,output_vert_dim)


    return o_dict, o_pred_dict


def load_one_y_big_data(f_ppi, o_ppi, f_pp, o_pp, est, ind_y, datafile, max_z, input_vert_vars, output_vert_vars, input_vert_dim, output_vert_dim,
                 n_trn_exs, rain_only, no_cos, use_rh, wind_input = False,scale_per_column=False,
                 rewight_outputs=False,weight_list=[1,1],do_nn=False):
    """Returns n_samples 2*n_z array of true and predicted values
       at a given y"""
    
    
    f, o, y, z, rho, p, weight_list = \
        LoadDataStandardScaleData_v2(datafile, max_z, input_vert_vars, output_vert_vars, all_ys=False, ind_y=ind_y,
                 verbose=False, n_trn_exs=None, rain_only=rain_only, 
                 no_cos=no_cos, use_rh=use_rh, wind_input = wind_input, rewight_outputs =rewight_outputs )
    # Calculate predicted output

    f_dict = unpack_list(f, input_vert_vars,input_vert_dim)
    f_scl_dict = transform_data_generalized(f_ppi, f_pp, f_dict, input_vert_vars, z,scale_per_column, rewight_outputs=False)
    # f_scl = transform_data(f_ppi, f_pp, f, z)
    f_scl = pack_list(f_scl_dict, input_vert_vars)

    if do_nn:
        tmp_f_scl = torch.from_numpy(f_scl)
        est.eval()
        o_pred_scl = est(tmp_f_scl.float()) 
        o_pred_scl = o_pred_scl.detach().numpy()
    else:
        o_pred_scl = est.predict(f_scl)
    o_pred_scl_dict = unpack_list(o_pred_scl, output_vert_vars,output_vert_dim)
    o_pred = inverse_transform_data_generalized(o_ppi, o_pp, o_pred_scl_dict,output_vert_vars, z,scale_per_column
                                                ,rewight_outputs=rewight_outputs,weight_list=weight_list)
    o_pred_dict = unpack_list(o_pred, output_vert_vars,output_vert_dim)

    o_dict = unpack_list(o, output_vert_vars,output_vert_dim)


    return o_dict, o_pred_dict

def stats_by_yz(f_ppi, o_ppi, f_pp, o_pp, est, y, z, rho, datafile, n_trn_exs, input_vert_vars, output_vert_vars,
                input_vert_dim, output_vert_dim, rain_only, no_cos, use_rh, wind_input = False,scale_per_column=False,
                rewight_outputs=False,weight_list=[1,1],do_nn=False):
    # Initialize
    output_stat_dict = dict()
    feature_list = ['_mean','_var','_bias','_rmse','_r','_Rsq']
    for output_name,z_dim in zip(output_vert_vars,output_vert_dim):
        for feature in feature_list:
            output_stat_dict[output_name+feature] = np.zeros((len(y), z_dim))

    output_stat_dict['Pmean_true'] = np.zeros((len(y)))
    output_stat_dict['Pmean_pred']= np.zeros((len(y)))
    output_stat_dict['Pextreme_true']= np.zeros((len(y)))
    output_stat_dict['Pextreme_pred']= np.zeros((len(y)))
    #
    for i in range(len(y)):
        o_true_dict, o_pred_dict = \
            load_one_y_big_data(f_ppi, o_ppi, f_pp, o_pp, est, i, datafile,
                         np.max(z), input_vert_vars, output_vert_vars, input_vert_dim, output_vert_dim, n_trn_exs, rain_only,
                         no_cos, use_rh, wind_input = wind_input,scale_per_column = scale_per_column,
                         rewight_outputs=rewight_outputs,weight_list=weight_list,do_nn=do_nn)

        if i==0:
         print('size of test dataset for a given y and level', o_true_dict[output_vert_vars[0]].shape[0])

        for output_name,z_dim in zip(output_vert_vars,output_vert_dim):
            output_stat_dict[output_name+'_mean'][i,:] = np.mean(o_true_dict[output_name],axis=0)
            output_stat_dict[output_name+'_var'][i,:] = np.var(o_true_dict[output_name],axis=0)
            output_stat_dict[output_name+'_bias'][i,:] = np.mean(o_pred_dict[output_name],axis=0) - output_stat_dict[output_name+'_mean'][i,:]
            output_stat_dict[output_name+'_rmse'][i,:] = np.sqrt(
                metrics.mean_squared_error(
                o_true_dict[output_name], o_pred_dict[output_name],
                                           multioutput='raw_values'))
            for j in range(z_dim):
                if np.sum(o_true_dict[output_name][:, j]==0) >  o_true_dict[output_name][:, j].shape[0]*0.99 and output_name!='qpout': 
                    output_stat_dict[output_name + '_Rsq'][i, j] = np.nan
                    continue
                output_stat_dict[output_name +'_r'][i,j] = scipy.stats.pearsonr(
                    o_true_dict[output_name][:, j], o_pred_dict[output_name][:, j])[0]
                output_stat_dict[output_name + '_Rsq'][i,j] = metrics.r2_score(o_true_dict[output_name][:, j], o_pred_dict[output_name][:, j])
                if output_stat_dict[output_name + '_Rsq'][i, j] < -10:
                    output_stat_dict[output_name + '_Rsq'][i, j] = -10
            if output_name == 'qout':
                P_true = atmos_physics.calc_precip(o_true_dict['qout'], rho, z,output_vert_vars,o_true_dict)
                P_pred = atmos_physics.calc_precip(o_pred_dict['qout'], rho, z,output_vert_vars,o_pred_dict)
                output_stat_dict['Pmean_true'][i] = np.mean(P_true)
                output_stat_dict['Pmean_pred'][i] = np.mean(P_pred)
                output_stat_dict['Pextreme_true'][i] = np.percentile(P_true, 99.9)
                output_stat_dict['Pextreme_pred'][i] = np.percentile(P_pred, 99.9)




    return output_stat_dict

def GetDataPath(training_expt, wind_input = False,is_cheyenne=False,full_data_separate=False):
    if is_cheyenne == False:  # On aimsir/esker
        #base_dir = '/net/aimsir/archive1/janniy/'
        base_dir = '/ocean/projects/ees220005p/gmooers/GM_Data'
    else:
        #base_dir = '/glade/scratch/janniy/'
        base_dir = '/ocean/projects/ees220005p/gmooers/GM_Data'

    if wind_input:
        datadir = base_dir + '/training_data/'
    else:
        datadir = base_dir + '/training_data/'
    # practice_flag = False
    if full_data_separate:
        trainfile = datadir + training_expt + '_training_short.pkl'
        testfile = datadir + training_expt + '_testing_short.pkl'
    else:
        trainfile = datadir + training_expt + '_training.pkl'
        testfile = datadir + training_expt + '_testing.pkl'

    pp_str = training_expt + '_'

    print(trainfile)
    print(testfile)
    return datadir, trainfile, testfile, pp_str


def get_f_o_pred_true(est_str, datadir, training_expt, training_file, testing_file, zdim, max_z, input_vert_vars, output_vert_vars,input_vert_dim,output_vert_dim,
                      all_ys=True, ind_y=None, 
                      n_trn_exs=None, rain_only=False,  
                      no_cos=False, use_rh=False, wind_input = False, scale_per_column=False,
                      rewight_outputs=False,weight_list=[1,1],is_cheyenne=False, do_nn =False):
    # Load model and preprocessors
    base_dir = '/ocean/projects/ees220005p/gmooers/GM_Data/'
    if rewight_outputs == True:
        my_weights = datadir + training_expt + "_weight.nc"
        weight_variables = xr.open_dataset(my_weights)
        weights = weight_variables.norms.values
    else:
        weights=[1,1]
        my_weights=None
    
    est, _, errors, f_ppi, o_ppi, f_pp, o_pp, y, z, _, _ = \
        pickle.load(open(base_dir + 'mldata_tmp/regressors/' + est_str + '.pkl', 'rb'))
    # Load raw data from file
    
    _, f_scl, _, otrue_scl, _, f, _, otrue, _, _, _ = LoadDataStandardScaleData_v2(training_file,
                                 testing_file,
                                 input_vert_vars,
                              output_vert_vars,
                              zdim,
                              weights=my_weights
                              
                                 
    )

    if do_nn:
        tmp_f_scl = torch.from_numpy(f_scl)
        est.eval()
        opred_scl = est(tmp_f_scl.float()) 
        opred_scl=opred_scl.detach().numpy()
    else:
        opred_scl = est.predict(f_scl) 
    
    opred_scl_dict = unpack_list(opred_scl, output_vert_vars, output_vert_dim)
    opred = inverse_transform_data_generalized_big_data(o_ppi, o_pp, opred_scl_dict,output_vert_vars, z, scale_per_column,
                                               rewight_outputs=rewight_outputs, weight_list=weights)
    return f_scl, opred_scl, otrue_scl, f, opred, otrue


def load_error_history(est_str,is_cheyenne=False):
    if is_cheyenne == False:  # On aimsir/esker
        base_dir = '/net/aimsir/archive1/janniy/'
    else:
        base_dir = '/glade/scratch/janniy/'
    _, _, err, _, _, _, _, _, _, _ = pickle.load(open(base_dir + 'mldata_tmp/regressors/' +
                                                      est_str, + 'pkl', 'rb'))
    return err



def GetDataPath_nn(training_expt, wind_input = False,is_cheyenne=False,full_data_separate=False):
    if is_cheyenne == False:  # On aimsir/esker
        base_dir = '/ocean/projects/ees220005p/gmooers/GM_Data/'
    else:
        base_dir = '/glade/scratch/janniy/'

    if wind_input:
        datadir = base_dir + '/training_data/'
    else:
        datadir = base_dir + '/training_data/'

    if full_data_separate:
        trainfile = datadir + training_expt + '_training.pkl'
        testfile = datadir + training_expt + '_testing.pkl'
    else:
        trainfile = datadir + training_expt + '_training.pkl'  
        testfile = datadir + training_expt + '_testing.pkl' 

    pp_str = training_expt + '_'

    print(trainfile)
    print(testfile)
    return datadir, trainfile, testfile, pp_str
