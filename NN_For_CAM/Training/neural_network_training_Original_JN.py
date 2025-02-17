import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
import data_scalar_and_reshaper_original as ml_load
import pickle
import post_processing_figures
import ml_plot_nn_older as ml_plot_nn
import os
import math
import yaml

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torchview import draw_graph
import torchvision
from torch import nn, optim
from netCDF4 import Dataset
import netCDF4
from sklearn.metrics import r2_score
import pdb
import xarray as xr
import math

import matplotlib.pyplot as plt


# ---  build random forest or neural net  ---
def train_wrapper(config_file):

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    #open the dictionaries from the config with info to populate the training script
    #open the dicts
    name_dict=config.get('name_dict')
    data_dict=config.get('data_dict')
    nn_dict=config.get('nn_dict')
    post_processing_dict=config.get('post_processing_dict')
    misc_dict=config.get("misc_dict")

    #unpack the naming dict for information about the experiment
    config_id = name_dict['id']
    nametag = name_dict['nametag']
    save_name = name_dict['save_name']
    f_ppi = name_dict['f_ppi']
    config_id = name_dict['id']

    #unpack the data specific calls for the experiment
    training_data_path = data_dict['training_expt']
    test_data_path = data_dict['test_expt']
    weights_path = data_dict['weights_path']
    save_path = data_dict['save_path']
    single_file = data_dict['single_file']
    levels = data_dict['levels']
    do_poles = data_dict['do_poles']
    input_vert_vars = data_dict['input_vert_vars']
    output_vert_vars = data_dict['output_vert_vars']
    rewight_outputs = data_dict['rewight_outputs']
    training_data_volume = data_dict['training_data_volume']
    data_chunks=data_dict['data_chunks']
    
    # get the neural network specifications by unpack the nn dict
    do_nn = nn_dict['do_nn']
    batch_norm = nn_dict['batch_norm']
    n_layers = nn_dict['n_layers']
    epochs = nn_dict['epochs']
    lr = nn_dict['lr']
    min_lr = nn_dict['min_lr']
    max_lr = nn_dict['max_lr']
    step_size_up = nn_dict['step_size_up']
    batch_size = nn_dict['batch_size']
    n_trn_exs = nn_dict['n_trn_exs']
    dropoff = nn_dict['dropoff']

    # define what will be plotted in the post-processing dict
    plotting_path=post_processing_dict['plotting_path']
    only_plot=post_processing_dict['only_plot']

    # generate unique tag to save the experiment
    nametag = "EXPERIMENT_"+str(config_id)+"_"+nametag + "_use_poles_"+str(do_poles)+"_physical_weighting_"+str(rewight_outputs)+"_epochs_"+str(epochs)+"_tr_data_percent_"+str(int(training_data_volume))
    
    if os.path.isdir(save_path+nametag):
        pass
    else:
        os.mkdir(save_path+nametag)

    if only_plot is False:

        #########################################
        # Scale and reshape the data
        #########################################
        train_inputs_scaled, test_inputs_scaled, train_outputs_scaled, test_outputs_scaled, test_outputs_original, train_inputs_pp_dict,  train_outputs_pp_dict, pp_str= ml_load.LoadDataStandardScaleData_v2(
                                                       traindata=training_data_path,
                                                       testdata=test_data_path,
                                                       input_vert_vars=input_vert_vars,
                                                       output_vert_vars=output_vert_vars,
                                                        poles=do_poles,
                                                        training_data_volume=training_data_volume,
                                                       weights=weights_path,
                                                       chunk=data_chunks,
        )
        
        # get the size of features of the input_vector
        input_feature_length = train_inputs_scaled.shape[1]
        # get the size of features of the output_vector
        output_feature_length = train_outputs_scaled.shape[1]
        
        ######################################
        #### Construct the Neural Network
        ######################################
            
        # TODO:gmooers -- make this customizable with all hyperparameters
        model, model_str = BuildNN(pp_str=pp_str, 
                                n_in=input_feature_length, 
                                n_out=output_feature_length, 
                                n_layers=n_layers, 
                                dropoff=dropoff, 
                                batch_norm=batch_norm,
                                  ) 


        # Use DataParallel for multiple GPUs -- this allows for the option of future paralleization
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        model = model.to(device)

        if do_nn:

            # https://github.com/mert-kurttutan/torchview
            if os.path.isdir(save_path+nametag+"/Design/"):
                pass
            else:
                os.mkdir(save_path+nametag+"/Design/")
            path_for_design = save_path + "/"+nametag+"/Design/"
            model_graph = draw_graph(model, 
                                input_size=(batch_size, input_feature_length),
                                graph_name=nametag,
                                save_graph=True,
                                directory=path_for_design,
                                    )

            ###########################################
            ### Train the constructed Neural Network
            ############################################

            model = train_nn(
                net=model,
                                                        savepath=save_path,
                                                        nametag=nametag,
                                                        save_name=save_name,
                                                    est_str=model_str, 
                                                    f_scl=train_inputs_scaled, 
                                                    o_scl=train_outputs_scaled, 
                                                    tf_scl=test_inputs_scaled, 
                                                    to_scl=test_outputs_scaled, 
                                                    output_vert_dim=levels, 
                                                    output_vert_vars=output_vert_vars,
                                                    epochs=epochs,
                                                    min_lr=min_lr,
                                                    max_lr=max_lr,
                                                    step_size_up=step_size_up,
                                                    batch_size=batch_size,
                                                    lr=lr,
                                                   )

            # bring in the data for info in the function below
            # TODO:@gmooers -- check how much of this is really necessary
            #variables = xr.open_mfdataset(training_data_path, chunks=data_chunks)
            variables = xr.open_mfdataset(training_data_path, combine='nested', concat_dim='sample')
            
            save_nn(net=model,
                    output_filename=nametag, 
                    base_dir=save_path, 
                    est_str=model_str, 
                    n_layers=n_layers, 
                    o_pp=train_outputs_pp_dict, 
                    f_pp=train_inputs_pp_dict, 
                    f_ppi=f_ppi, # type of scaling done to inputs
                    o_ppi=f_ppi, # type of scaling done to outputs
                    y=variables.lat.values, 
                    z=variables.z.values, 
                    p=variables.p.values, 
                    rho=variables.rho.values, 
                    batch_norm=batch_norm)


    # Calculate the predictions of the trained model

    else:
        
        save_location = save_path+nametag+"/Saved_Models/"+"stage_2_"+save_name
        model.load_state_dict(torch.load(save_location))


    scaled_predictions = get_model_predictions_from_numpy(model=model, #trained model
                                     test_data=test_inputs_scaled, #scaled test data
                                     batch_size=batch_size,
                                                         )
    # unscale the predictions to compare to the outputs
    unscaled_predictions = undo_scaling(
                                scaled_array=scaled_predictions, 
                                scaler_dict=train_outputs_pp_dict, 
                                vertical_dimension=levels, 
                                output_variables=output_vert_vars,
                                )

    
    post_processing_figures.main_plotting(truth=np.transpose(test_outputs_original), # put z dimension first for plotting code
                                              pred=np.transpose(unscaled_predictions), #put z dimension first for plotting code 
                                              raw_data=single_file,
                                              z_dim=levels,
                                             var_names=output_vert_vars,
                                              save_path=save_path,
                                              nametag=nametag,
                                             )

    return "Offline Neural Model Trained and Test"



def BuildNN(pp_str, n_in, n_out, n_layers =2, dropoff=0.0, batch_norm = True):

    """Builds an NN using pytorch
    Currently only two options - 2 layers or 5 layers
    """
   
    est = Net_ANN_5_no_BN(n_in, n_out, neurons=128)

    # Construct name
    est_str = pp_str
    # Add the number of iterations too
    est_str = est_str + 'NN_layers' + str(n_layers) + 'in'+str(n_in) + 'out' + str(n_out)+ '_BN_' + str(batch_norm)[0]
    return est, est_str


############################
# Function added by Griffin
############################


def plot_training_testing_errors(train_losses, test_losses, epochs, plot_save_path=None):
    plt.figure(figsize=(10, 6))
    
    # Plot for Stage 1
    plt.plot(np.arange(0,len(train_losses[:epochs]))+1, train_losses[:epochs], label='Training Error (Stage 1)', color='green')
    #plt.plot(np.arange(0,len(test_losses[:epochs]))+1, [-1 * x for x in test_losses[:epochs]], label='Testing Error (Stage 1)', color='blue')
    plt.plot(np.arange(0,len(test_losses[:epochs]))+1, test_losses[:epochs], label='Testing Error (Stage 1)', color='blue')
    
    # Plot for Stage 2
    plt.plot(range(epochs, epochs + (epochs-2)), train_losses[epochs:], label='Training Error (Stage 2)', linestyle='--', color='green')
    plt.plot(range(epochs, epochs + (epochs-2)),  test_losses[epochs:], label='Testing Error (Stage 2)', linestyle='--', color='blue')
    
    # Mark the learning rate change point
    plt.axvline(x=epochs, color='red', linestyle=':', label='Learning Rate Change')
    
    plt.legend()
    plt.title('Training and Testing Error per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Error')

    if plot_save_path:
        plt.savefig(plot_save_path)
    
    plt.show()


def get_model_predictions_from_numpy(model, test_data, batch_size):
    """
    Get predictions from a trained neural network model given a NumPy array of test data.

    Parameters:
    model (torch.nn.Module): Trained neural network model.
    test_data (np.ndarray): NumPy array of test data (samples, features).
    batch_size (int): Number of samples per batch.

    Returns:
    np.ndarray: Model predictions for the entire test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    num_samples = test_data.shape[0]
    
    # Loop through the test data in batches
    with torch.no_grad():  # Disable gradient calculations for efficiency
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = test_data[start_idx:end_idx]

            # Convert the NumPy array batch to a PyTorch tensor
            batch_data_tensor = torch.tensor(batch_data, dtype=torch.float32)
            
            # Get the model predictions
            outputs = model(batch_data_tensor)
            
            # Move predictions back to CPU, convert to NumPy, and store them
            all_predictions.append(outputs.cpu().numpy())
    
    # Concatenate all the batches into a single NumPy array
    return np.concatenate(all_predictions, axis=0)


def undo_scaling(scaled_array, scaler_dict, vertical_dimension, output_variables):
    """
    Undo the scaling of the given scaled numpy array using the provided scalers.

    Returns:
    np.ndarray: The unscaled numpy array with the same shape as the input.
    """
    num_samples = scaled_array.shape[0]

    # Initialize the unscaled array with the same shape as the input
    unscaled_array = np.zeros_like(scaled_array)

    # Iterate over the variables, extract the corresponding features, and apply inverse scaling
    for i, var in enumerate(output_variables):
        start_idx = i * vertical_dimension
        end_idx = start_idx + vertical_dimension
        
        # Extract the part of the scaled array corresponding to this variable
        scaled_part = scaled_array[:, start_idx:end_idx]
        
        # Get the appropriate scaler and apply inverse transform
        unscaled_part = scaler_dict[var].inverse_transform(scaled_part)
        
        # Place the unscaled part back into the unscaled_array
        unscaled_array[:, start_idx:end_idx] = unscaled_part

    return unscaled_array

############################
# End of function added by Griffin
############################

def train_nn(net, savepath, nametag, save_name, est_str, f_scl, o_scl, tf_scl, to_scl, output_vert_dim, output_vert_vars,epochs =7, min_lr = 2e-4, max_lr = 2e-3, step_size_up=4000,batch_size = 1024, lr=1e-7):
   
    y_train_small_py = torch.from_numpy(o_scl.reshape(-1, o_scl.shape[1])).float()

    X_norm_py = torch.from_numpy(f_scl.reshape(-1, f_scl.shape[1])).float()
    
    X_train_val_norm_py = torch.from_numpy(tf_scl.reshape(-1, tf_scl.shape[1])).float()

    y_test_small_py = torch.from_numpy(to_scl.reshape(-1, to_scl.shape[1])).float()
    
    torch_dataset = Data.TensorDataset(X_norm_py, y_train_small_py)
    
    torch_dataset_test = Data.TensorDataset(X_train_val_norm_py, y_test_small_py)

    # Griffin changed num workers from 4 to 1

    # this is train loader
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True, num_workers=1,)

    # I am making test loader here
    test_loader = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=batch_size,
        shuffle=True, num_workers=1,)


    # Griffin change
    #optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.Adam(net.parameters(), lr=min_lr)
    loss_func = torch.nn.MSELoss()
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr= max_lr, step_size_up=step_size_up, cycle_momentum=False)
    torch.set_num_threads(10)

    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):  
        train_score = train_model_cyclic(net, loss_func, loader, optimizer, scheduler)
        test_score = train_model_cyclic(net, loss_func, test_loader, optimizer, scheduler)
        #test_score = test_model(net, X_train_val_norm_py, to_scl, output_vert_dim, output_vert_vars)

        train_losses.append(train_score)
        test_losses.append(test_score)

    save_location = savepath+nametag+"/Saved_Models/"+"stage_1_"+save_name 

    os.makedirs(os.path.dirname(save_location), exist_ok=True)
    torch.save(net.state_dict(), save_location)

    #Run a few epochs with lower learning rate.
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr/10, max_lr= max_lr/10, step_size_up=step_size_up, cycle_momentum=False)
    for epoch in range(epochs-2):
        train_score = train_model_cyclic(net, loss_func, loader, optimizer, scheduler)
        test_score = train_model_cyclic(net, loss_func, test_loader, optimizer, scheduler)
        #test_score = test_model(net, X_train_val_norm_py, to_scl, output_vert_dim, output_vert_vars)

        train_losses.append(train_score)
        test_losses.append(test_score)
        
    test_score = test_model(net, X_train_val_norm_py, to_scl, output_vert_dim, output_vert_vars)
    train_score = test_model(net, X_norm_py[0:500000,:], o_scl[0:500000,:], output_vert_dim, output_vert_vars)

    save_location = savepath+nametag+"/Saved_Models/"+"stage_2_"+save_name  
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_location), exist_ok=True)
    torch.save(net.state_dict(), save_location)

    plot_save_path = os.path.join(savepath, nametag, "Loss_Curves", "Model_Losses.pdf")
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    print("plotting NN performance")
    plot_training_testing_errors(
        train_losses=train_losses, 
        test_losses=test_losses, 
        epochs=epochs, 
        plot_save_path=plot_save_path,
    )

    return net

def rmse(x, y): return math.sqrt(((x - y) ** 2).mean())

def test_model(net,X_train_val_norm_py,y_train_val, output_vert_dim, output_vert_vars):
    net.eval()
    pred_val = net(Variable(X_train_val_norm_py))
    print('RMSE: ',rmse(pred_val.data.numpy(),y_train_val), ' R2:' ,r2_score(y_train_val[:,:], pred_val.data.numpy()[:,:],multioutput='variance_weighted'))

    for i in range(len(output_vert_vars)):
        start = int(output_vert_dim*i)
        end = int(output_vert_dim*(i+1))
        print(output_vert_vars[i] + 'R2:',r2_score(y_train_val[:, start:end], pred_val.data.numpy()[:, start:end], multioutput='variance_weighted'))
    #idim_now = 0
    #for dim1,name in zip(output_vert_dim,output_vert_vars):
    #    print(name + 'R2:',r2_score(y_train_val[:, idim_now:idim_now+dim1], pred_val.data.numpy()[:, idim_now:idim_now+dim1], multioutput='variance_weighted'))
    #    idim_now = idim_now  + dim1
    return r2_score(y_train_val[:,:], pred_val.data.numpy()[:,:],multioutput='variance_weighted')


def train_model_cyclic(net,criterion,trainloader,optimizer,scheduler):
    net.train()
    test_loss = 0
    for step, (batch_x, batch_y) in enumerate(trainloader):  # for each training step
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        prediction = net(b_x)  # input x and predict based on x
        loss = criterion(prediction, b_y)  # must be (1. nn output, 2. target)
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        optimizer.zero_grad()
        test_loss = test_loss + loss.data.numpy()
        scheduler.step() #defined I think on the epoches...
    print('the loss in this Epoch',test_loss)
    return test_loss


def save_nn(net,
            base_dir,
            output_filename,
            est_str, 
            n_layers, 
            o_pp, 
            f_pp, 
            f_ppi, 
            o_ppi, 
            y, 
            z, 
            p, 
            rho,
            save_name = "NC_Models/",
            batch_norm = True,
           ):
    '''Write nn as nc file - only equipped to deal with 2 and 5 layers at the moment'''
    # Set output filename
    #base_dir = '/ocean/projects/ees220005p/gmooers/GM_Data/'
    #output_filename = base_dir + 'gcm_regressors/'+est_str+'.nc'

    file_path = os.path.join(base_dir, output_filename, save_name)
    full_dir_path = os.path.join(file_path, "My_NN.nc")

    # Ensure the directory exists, and create it if not
    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path)
    
    net2 = net
    net2.eval()

    full_dir_path = full_dir_path + "My_NN.nc"

    in_dim = net2.linear1.weight.T.shape[0]
    X_mean = np.zeros(in_dim)
    X_std = np.zeros(in_dim)
    ind_now = 0
    print('check that I am iterating correctly over variables')
    for key, value in f_pp.items():  
        ind_tmp = ind_now + value.mean_.shape[0]
        X_mean[ind_now:ind_tmp] = value.mean_
        X_std[ind_now:ind_tmp] = np.sqrt(value.var_)  
        ind_now = ind_tmp


    Y_mean = np.zeros(len(o_pp))
    Y_std = np.zeros(len(o_pp))
    ind_now = 0
    print('check that I am iterating correctly over outputs')
    # Commented out to see if necessary
    for key, value in o_pp.items():  # Iterate over the different features mean and std.
        Y_mean[ind_now] = value.mean_.mean()
        Y_std[ind_now] = np.sqrt(value.var_).mean()
        ind_now = ind_now +1

    if n_layers == 2:
        ncfile = Dataset(full_dir_path, 'w', format="NETCDF3_CLASSIC")
        ncfile.createDimension('single', 1)
        ncfile.createDimension('N_in', net2.linear1.weight.T.shape[0])
        ncfile.createDimension('N_h1', net2.linear1.weight.T.shape[1])
        ncfile.createDimension('N_out', net2.linear2.weight.T.shape[1])
        ncfile.createDimension('N_out_dim', len(o_pp))
        # Create variable entries in the file
        nc_w1 = ncfile.createVariable('w1', np.dtype('float32').char,
                                      ('N_h1', 'N_in'))  # Reverse dims
        nc_w2 = ncfile.createVariable('w2', np.dtype('float32').char,
                                      ('N_out', 'N_h1'))
        nc_b1 = ncfile.createVariable('b1', np.dtype('float32').char,
                                      ('N_h1'))
        nc_b2 = ncfile.createVariable('b2', np.dtype('float32').char,
                                      ('N_out'))

        if batch_norm:
            nc_batch_mean = ncfile.createVariable('batch_mean',
                                                  np.dtype('float32').char, ('N_h1'))
            nc_batch_stnd = ncfile.createVariable('batch_stnd',
                                                  np.dtype('float32').char, ('N_h1'))

            nc_batch_weight = ncfile.createVariable('batch_weight',
                                                    np.dtype('float32').char, ('N_h1'))
            nc_batch_bias = ncfile.createVariable('batch_bias',
                                              np.dtype('float32').char, ('N_h1'))

        nc_oscale_mean = ncfile.createVariable('oscale_mean',
                                               np.dtype('float32').char, ('N_out_dim'))
        nc_oscale_stnd = ncfile.createVariable('oscale_stnd',
                                               np.dtype('float32').char, ('N_out_dim'))

        nc_fscale_mean = ncfile.createVariable('fscale_mean',
                                               np.dtype('float32').char, ('N_in'))
        nc_fscale_stnd = ncfile.createVariable('fscale_stnd',
                                               np.dtype('float32').char, ('N_in'))

        nc_w1[:] = net2.linear1.weight.data.numpy()
        nc_w2[:] = net2.linear2.weight.data.numpy()
        nc_b1[:] = net2.linear1.bias.data.numpy()
        nc_b2[:] = net2.linear2.bias.data.numpy()

        if batch_norm:
            print('saving NC NN with BN')
            nc_batch_mean[:] = net2.dense1_bn.running_mean.data.numpy()
            nc_batch_stnd[:] = torch.sqrt(net2.dense1_bn.running_var).data.numpy()
            nc_batch_weight[:] = net2.dense1_bn.weight.data.numpy()
            nc_batch_bias[:] = net2.dense1_bn.bias.data.numpy()

        nc_oscale_mean[:] = Y_mean
        nc_oscale_stnd[:] = Y_std

        nc_fscale_mean[:] = X_mean
        nc_fscale_stnd[:] = X_std

        ncfile.description = 'NN flux Created with ml_train_nn'
        ncfile.close()

    elif n_layers == 3:
        ncfile = Dataset(full_dir_path, 'w', format="NETCDF3_CLASSIC")
        ncfile.createDimension('single', 1)
        ncfile.createDimension('N_in', net2.linear1.weight.T.shape[0])
        ncfile.createDimension('N_h1', net2.linear1.weight.T.shape[1])
        ncfile.createDimension('N_h2', net2.linear2.weight.T.shape[1])
        ncfile.createDimension('N_out', net2.linear3.weight.T.shape[1])
        ncfile.createDimension('N_out_dim', len(o_pp))
        # Create variable entries in the file
        nc_w1 = ncfile.createVariable('w1', np.dtype('float32').char,
                                      ('N_h1', 'N_in'))  # Reverse dims

        nc_w2 = ncfile.createVariable('w2', np.dtype('float32').char,
                                      ('N_h2', 'N_h1'))  # Reverse dims
        nc_w3 = ncfile.createVariable('w3', np.dtype('float32').char,
                                      ('N_out', 'N_h2'))  # Reverse dims

        nc_b1 = ncfile.createVariable('b1', np.dtype('float32').char,
                                      ('N_h1'))
        nc_b2 = ncfile.createVariable('b2', np.dtype('float32').char,
                                      ('N_h2'))
        nc_b3 = ncfile.createVariable('b3', np.dtype('float32').char,
                                      ('N_out'))

        if batch_norm:
            nc_batch_mean = ncfile.createVariable('batch_mean',
                                                  np.dtype('float32').char, ('N_h1'))
            nc_batch_stnd = ncfile.createVariable('batch_stnd',
                                                  np.dtype('float32').char, ('N_h1'))

            nc_batch_weight = ncfile.createVariable('batch_weight',
                                                    np.dtype('float32').char, ('N_h1'))
            nc_batch_bias = ncfile.createVariable('batch_bias',
                                                  np.dtype('float32').char, ('N_h1'))

            nc_batch_mean2 = ncfile.createVariable('batch_mean2',
                                                   np.dtype('float32').char, ('N_h2'))
            nc_batch_stnd2 = ncfile.createVariable('batch_stnd2',
                                                   np.dtype('float32').char, ('N_h2'))
            nc_batch_weight2 = ncfile.createVariable('batch_weight2',
                                                     np.dtype('float32').char, ('N_h2'))
            nc_batch_bias2 = ncfile.createVariable('batch_bias2',
                                                   np.dtype('float32').char, ('N_h2'))

        nc_oscale_mean = ncfile.createVariable('oscale_mean',
                                               np.dtype('float32').char, ('N_out_dim'))
        nc_oscale_stnd = ncfile.createVariable('oscale_stnd',
                                               np.dtype('float32').char, ('N_out_dim'))

        nc_fscale_mean = ncfile.createVariable('fscale_mean',
                                               np.dtype('float32').char, ('N_in'))
        nc_fscale_stnd = ncfile.createVariable('fscale_stnd',
                                               np.dtype('float32').char, ('N_in'))

        nc_w1[:] = net2.linear1.weight.data.numpy()
        nc_w2[:] = net2.linear2.weight.data.numpy()
        nc_w3[:] = net2.linear3.weight.data.numpy()

        nc_b1[:] = net2.linear1.bias.data.numpy()
        nc_b2[:] = net2.linear2.bias.data.numpy()
        nc_b3[:] = net2.linear3.bias.data.numpy()

        if batch_norm:
            print('saving NC NN with BN')
            nc_batch_mean[:] = net2.dense1_bn.running_mean.data.numpy()
            nc_batch_stnd[:] = torch.sqrt(net2.dense1_bn.running_var).data.numpy()
            nc_batch_weight[:] = net2.dense1_bn.weight.data.numpy()
            nc_batch_bias[:] = net2.dense1_bn.bias.data.numpy()

            nc_batch_mean2[:] = net2.dense2_bn.running_mean.data.numpy()
            nc_batch_stnd2[:] = torch.sqrt(net2.dense2_bn.running_var).data.numpy()
            nc_batch_weight2[:] = net2.dense2_bn.weight.data.numpy()
            nc_batch_bias2[:] = net2.dense2_bn.bias.data.numpy()

        nc_oscale_mean[:] = Y_mean
        nc_oscale_stnd[:] = Y_std

        nc_fscale_mean[:] = X_mean
        nc_fscale_stnd[:] = X_std

        ncfile.description = 'NN flux Created with ml_train_nn'
        ncfile.close()


    #######
    elif n_layers == 4:
        ncfile = Dataset(full_dir_path, 'w', format="NETCDF3_CLASSIC")
        ncfile.createDimension('single', 1)
        ncfile.createDimension('N_in', net2.linear1.weight.T.shape[0])
        ncfile.createDimension('N_h1', net2.linear1.weight.T.shape[1])
        ncfile.createDimension('N_h2', net2.linear2.weight.T.shape[1])
        ncfile.createDimension('N_h3', net2.linear3.weight.T.shape[1])
        ncfile.createDimension('N_out', net2.linear4.weight.T.shape[1])
        ncfile.createDimension('N_out_dim', len(o_pp))
        # Create variable entries in the file
        nc_w1 = ncfile.createVariable('w1', np.dtype('float32').char,
                                      ('N_h1', 'N_in'))  # Reverse dims

        nc_w2 = ncfile.createVariable('w2', np.dtype('float32').char,
                                      ('N_h2', 'N_h1'))  # Reverse dims
        nc_w3 = ncfile.createVariable('w3', np.dtype('float32').char,
                                      ('N_h3', 'N_h2'))  # Reverse dims

        nc_w4 = ncfile.createVariable('w4', np.dtype('float32').char,
                                      ('N_out', 'N_h3'))  # Reverse dims

        nc_b1 = ncfile.createVariable('b1', np.dtype('float32').char,
                                      ('N_h1'))
        nc_b2 = ncfile.createVariable('b2', np.dtype('float32').char,
                                      ('N_h2'))
        nc_b3 = ncfile.createVariable('b3', np.dtype('float32').char,
                                      ('N_h3'))
        nc_b4 = ncfile.createVariable('b4', np.dtype('float32').char,
                                      ('N_out'))
        if batch_norm:
            nc_batch_mean = ncfile.createVariable('batch_mean',
                                                  np.dtype('float32').char, ('N_h1'))
            nc_batch_stnd = ncfile.createVariable('batch_stnd',
                                                  np.dtype('float32').char, ('N_h1'))

            nc_batch_weight = ncfile.createVariable('batch_weight',
                                                    np.dtype('float32').char, ('N_h1'))
            nc_batch_bias = ncfile.createVariable('batch_bias',
                                                  np.dtype('float32').char, ('N_h1'))

            nc_batch_mean2 = ncfile.createVariable('batch_mean2',
                                                   np.dtype('float32').char, ('N_h2'))
            nc_batch_stnd2 = ncfile.createVariable('batch_stnd2',
                                                   np.dtype('float32').char, ('N_h2'))
            nc_batch_weight2 = ncfile.createVariable('batch_weight2',
                                                     np.dtype('float32').char, ('N_h2'))
            nc_batch_bias2 = ncfile.createVariable('batch_bias2',
                                                   np.dtype('float32').char, ('N_h2'))

            nc_batch_mean3 = ncfile.createVariable('batch_mean3',
                                                   np.dtype('float32').char, ('N_h3'))
            nc_batch_stnd3 = ncfile.createVariable('batch_stnd3',
                                                   np.dtype('float32').char, ('N_h3'))
            nc_batch_weight3 = ncfile.createVariable('batch_weight3',
                                                     np.dtype('float32').char, ('N_h3'))
            nc_batch_bias3 = ncfile.createVariable('batch_bias3',
                                                   np.dtype('float32').char, ('N_h3'))

        nc_oscale_mean = ncfile.createVariable('oscale_mean',
                                               np.dtype('float32').char, ('N_out_dim'))
        nc_oscale_stnd = ncfile.createVariable('oscale_stnd',
                                               np.dtype('float32').char, ('N_out_dim'))


        nc_fscale_mean = ncfile.createVariable('fscale_mean',
                                               np.dtype('float32').char, ('N_in'))
        nc_fscale_stnd = ncfile.createVariable('fscale_stnd',
                                               np.dtype('float32').char, ('N_in'))

        nc_w1[:] = net2.linear1.weight.data.numpy()
        nc_w2[:] = net2.linear2.weight.data.numpy()
        nc_w3[:] = net2.linear3.weight.data.numpy()
        nc_w4[:] = net2.linear4.weight.data.numpy()

        nc_b1[:] = net2.linear1.bias.data.numpy()
        nc_b2[:] = net2.linear2.bias.data.numpy()
        nc_b3[:] = net2.linear3.bias.data.numpy()
        nc_b4[:] = net2.linear4.bias.data.numpy()

        if batch_norm:
            print('saving NC NN with BN')
            nc_batch_mean[:] = net2.dense1_bn.running_mean.data.numpy()
            nc_batch_stnd[:] = torch.sqrt(net2.dense1_bn.running_var).data.numpy()
            nc_batch_weight[:] = net2.dense1_bn.weight.data.numpy()
            nc_batch_bias[:] = net2.dense1_bn.bias.data.numpy()

            nc_batch_mean2[:] = net2.dense2_bn.running_mean.data.numpy()
            nc_batch_stnd2[:] = torch.sqrt(net2.dense2_bn.running_var).data.numpy()
            nc_batch_weight2[:] = net2.dense2_bn.weight.data.numpy()
            nc_batch_bias2[:] = net2.dense2_bn.bias.data.numpy()

            nc_batch_mean3[:] = net2.dense3_bn.running_mean.data.numpy()
            nc_batch_stnd3[:] = torch.sqrt(net2.dense3_bn.running_var).data.numpy()
            nc_batch_weight3[:] = net2.dense3_bn.weight.data.numpy()
            nc_batch_bias3[:] = net2.dense3_bn.bias.data.numpy()

        nc_oscale_mean[:] = Y_mean
        nc_oscale_stnd[:] = Y_std

        nc_fscale_mean[:] = X_mean
        nc_fscale_stnd[:] = X_std

        ncfile.description = 'NN flux Created with ml_train_nn'
        ncfile.close()

#########
    elif n_layers == 5:
        ncfile = Dataset(full_dir_path, 'w', format="NETCDF3_CLASSIC")
        ncfile.createDimension('single', 1)
        ncfile.createDimension('N_in', net2.linear1.weight.T.shape[0])
        ncfile.createDimension('N_h1', net2.linear1.weight.T.shape[1])
        ncfile.createDimension('N_h2', net2.linear2.weight.T.shape[1])
        ncfile.createDimension('N_h3', net2.linear3.weight.T.shape[1])
        ncfile.createDimension('N_h4', net2.linear4.weight.T.shape[1])
        ncfile.createDimension('N_out', net2.linear5.weight.T.shape[1])
        ncfile.createDimension('N_out_dim', len(o_pp))
        # Create variable entries in the file
        nc_w1 = ncfile.createVariable('w1', np.dtype('float32').char,
                                      ('N_h1', 'N_in'))  # Reverse dims

        nc_w2 = ncfile.createVariable('w2', np.dtype('float32').char,
                                      ('N_h2', 'N_h1'))  # Reverse dims
        nc_w3 = ncfile.createVariable('w3', np.dtype('float32').char,
                                      ('N_h3', 'N_h2'))  # Reverse dims
        nc_w4 = ncfile.createVariable('w4', np.dtype('float32').char,
                                      ('N_h4', 'N_h3'))  # Reverse dims
        nc_w5 = ncfile.createVariable('w5', np.dtype('float32').char,
                                      ('N_out', 'N_h4'))  # Reverse dims

        nc_b1 = ncfile.createVariable('b1', np.dtype('float32').char,
                                      ('N_h1'))
        nc_b2 = ncfile.createVariable('b2', np.dtype('float32').char,
                                      ('N_h2'))
        nc_b3 = ncfile.createVariable('b3', np.dtype('float32').char,
                                      ('N_h3'))
        nc_b4 = ncfile.createVariable('b4', np.dtype('float32').char,
                                      ('N_h4'))
        nc_b5 = ncfile.createVariable('b5', np.dtype('float32').char,
                                      ('N_out'))
        if batch_norm:
            nc_batch_mean = ncfile.createVariable('batch_mean',
                                                  np.dtype('float32').char, ('N_h1'))
            nc_batch_stnd = ncfile.createVariable('batch_stnd',
                                                  np.dtype('float32').char, ('N_h1'))

            nc_batch_weight = ncfile.createVariable('batch_weight',
                                                    np.dtype('float32').char, ('N_h1'))
            nc_batch_bias = ncfile.createVariable('batch_bias',
                                                  np.dtype('float32').char, ('N_h1'))

            nc_batch_mean2 = ncfile.createVariable('batch_mean2',
                                                   np.dtype('float32').char, ('N_h2'))
            nc_batch_stnd2 = ncfile.createVariable('batch_stnd2',
                                                   np.dtype('float32').char, ('N_h2'))
            nc_batch_weight2 = ncfile.createVariable('batch_weight2',
                                                     np.dtype('float32').char, ('N_h2'))
            nc_batch_bias2 = ncfile.createVariable('batch_bias2',
                                                   np.dtype('float32').char, ('N_h2'))

            nc_batch_mean3 = ncfile.createVariable('batch_mean3',
                                                   np.dtype('float32').char, ('N_h3'))
            nc_batch_stnd3 = ncfile.createVariable('batch_stnd3',
                                                   np.dtype('float32').char, ('N_h3'))
            nc_batch_weight3 = ncfile.createVariable('batch_weight3',
                                                     np.dtype('float32').char, ('N_h3'))
            nc_batch_bias3 = ncfile.createVariable('batch_bias3',
                                                   np.dtype('float32').char, ('N_h3'))

            nc_batch_mean4 = ncfile.createVariable('batch_mean4',
                                                   np.dtype('float32').char, ('N_h4'))
            nc_batch_stnd4 = ncfile.createVariable('batch_stnd4',
                                                   np.dtype('float32').char, ('N_h4'))
            nc_batch_weight4 = ncfile.createVariable('batch_weight4',
                                                     np.dtype('float32').char, ('N_h4'))
            nc_batch_bias4 = ncfile.createVariable('batch_bias4',
                                                   np.dtype('float32').char, ('N_h4'))

        nc_oscale_mean = ncfile.createVariable('oscale_mean',
                                               np.dtype('float32').char, ('N_out_dim'))
        nc_oscale_stnd = ncfile.createVariable('oscale_stnd',
                                               np.dtype('float32').char, ('N_out_dim'))


        nc_fscale_mean = ncfile.createVariable('fscale_mean',
                                               np.dtype('float32').char, ('N_in'))
        nc_fscale_stnd = ncfile.createVariable('fscale_stnd',
                                               np.dtype('float32').char, ('N_in'))

        nc_w1[:] = net2.linear1.weight.data.numpy()
        nc_w2[:] = net2.linear2.weight.data.numpy()
        nc_w3[:] = net2.linear3.weight.data.numpy()
        nc_w4[:] = net2.linear4.weight.data.numpy()
        nc_w5[:] = net2.linear5.weight.data.numpy()

        nc_b1[:] = net2.linear1.bias.data.numpy()
        nc_b2[:] = net2.linear2.bias.data.numpy()
        nc_b3[:] = net2.linear3.bias.data.numpy()
        nc_b4[:] = net2.linear4.bias.data.numpy()
        nc_b5[:] = net2.linear5.bias.data.numpy()

        if batch_norm:
            print('saving NC NN with BN')
            nc_batch_mean[:] = net2.dense1_bn.running_mean.data.numpy()
            nc_batch_stnd[:] = torch.sqrt(net2.dense1_bn.running_var).data.numpy()
            nc_batch_weight[:] = net2.dense1_bn.weight.data.numpy()
            nc_batch_bias[:] = net2.dense1_bn.bias.data.numpy()

            nc_batch_mean2[:] = net2.dense2_bn.running_mean.data.numpy()
            nc_batch_stnd2[:] = torch.sqrt(net2.dense2_bn.running_var).data.numpy()
            nc_batch_weight2[:] = net2.dense2_bn.weight.data.numpy()
            nc_batch_bias2[:] = net2.dense2_bn.bias.data.numpy()

            nc_batch_mean3[:] = net2.dense3_bn.running_mean.data.numpy()
            nc_batch_stnd3[:] = torch.sqrt(net2.dense3_bn.running_var).data.numpy()
            nc_batch_weight3[:] = net2.dense3_bn.weight.data.numpy()
            nc_batch_bias3[:] = net2.dense3_bn.bias.data.numpy()

            nc_batch_mean4[:] = net2.dense4_bn.running_mean.data.numpy()
            nc_batch_stnd4[:] = torch.sqrt(net2.dense4_bn.running_var).data.numpy()
            nc_batch_weight4[:] = net2.dense4_bn.weight.data.numpy()
            nc_batch_bias4[:] = net2.dense4_bn.bias.data.numpy()

        nc_oscale_mean[:] = Y_mean
        nc_oscale_stnd[:] = Y_std

        nc_fscale_mean[:] = X_mean
        nc_fscale_stnd[:] = X_std

        ncfile.description = 'NN flux Created with ml_train_nn'
        ncfile.close()

    elif n_layers == 6:
        ncfile = Dataset(full_dir_path, 'w', format="NETCDF3_CLASSIC")
        ncfile.createDimension('single', 1)
        ncfile.createDimension('N_in', net2.linear1.weight.T.shape[0])
        ncfile.createDimension('N_h1', net2.linear1.weight.T.shape[1])
        ncfile.createDimension('N_h2', net2.linear2.weight.T.shape[1])
        ncfile.createDimension('N_h3', net2.linear3.weight.T.shape[1])
        ncfile.createDimension('N_h4', net2.linear4.weight.T.shape[1])
        ncfile.createDimension('N_h5', net2.linear5.weight.T.shape[1])
        ncfile.createDimension('N_out', net2.linear6.weight.T.shape[1])
        ncfile.createDimension('N_out_dim', len(o_pp))
        # Create variable entries in the file
        nc_w1 = ncfile.createVariable('w1', np.dtype('float32').char,
                                      ('N_h1', 'N_in'))  # Reverse dims

        nc_w2 = ncfile.createVariable('w2', np.dtype('float32').char,
                                      ('N_h2', 'N_h1'))  # Reverse dims
        nc_w3 = ncfile.createVariable('w3', np.dtype('float32').char,
                                      ('N_h3', 'N_h2'))  # Reverse dims
        nc_w4 = ncfile.createVariable('w4', np.dtype('float32').char,
                                      ('N_h4', 'N_h3'))  # Reverse dims
        nc_w5 = ncfile.createVariable('w5', np.dtype('float32').char,
                                      ('N_h5', 'N_h4'))  # Reverse dims
        nc_w6 = ncfile.createVariable('w6', np.dtype('float32').char,
                                      ('N_out', 'N_h5'))  # Reverse dims

        nc_b1 = ncfile.createVariable('b1', np.dtype('float32').char,
                                      ('N_h1'))
        nc_b2 = ncfile.createVariable('b2', np.dtype('float32').char,
                                      ('N_h2'))
        nc_b3 = ncfile.createVariable('b3', np.dtype('float32').char,
                                      ('N_h3'))
        nc_b4 = ncfile.createVariable('b4', np.dtype('float32').char,
                                      ('N_h4'))
        nc_b5 = ncfile.createVariable('b5', np.dtype('float32').char,
                                      ('N_h5'))
        nc_b6 = ncfile.createVariable('b6', np.dtype('float32').char,
                                      ('N_out'))
        if batch_norm:
            raise Exception('No BN with 6 layers!')

        nc_oscale_mean = ncfile.createVariable('oscale_mean',
                                               np.dtype('float32').char, ('N_out_dim'))
        nc_oscale_stnd = ncfile.createVariable('oscale_stnd',
                                               np.dtype('float32').char, ('N_out_dim'))


        nc_fscale_mean = ncfile.createVariable('fscale_mean',
                                               np.dtype('float32').char, ('N_in'))
        nc_fscale_stnd = ncfile.createVariable('fscale_stnd',
                                               np.dtype('float32').char, ('N_in'))

        nc_w1[:] = net2.linear1.weight.data.numpy()
        nc_w2[:] = net2.linear2.weight.data.numpy()
        nc_w3[:] = net2.linear3.weight.data.numpy()
        nc_w4[:] = net2.linear4.weight.data.numpy()
        nc_w5[:] = net2.linear5.weight.data.numpy()
        nc_w6[:] = net2.linear6.weight.data.numpy()

        nc_b1[:] = net2.linear1.bias.data.numpy()
        nc_b2[:] = net2.linear2.bias.data.numpy()
        nc_b3[:] = net2.linear3.bias.data.numpy()
        nc_b4[:] = net2.linear4.bias.data.numpy()
        nc_b5[:] = net2.linear5.bias.data.numpy()
        nc_b6[:] = net2.linear6.bias.data.numpy()


        nc_oscale_mean[:] = Y_mean
        nc_oscale_stnd[:] = Y_std

        nc_fscale_mean[:] = X_mean
        nc_fscale_stnd[:] = X_std

        ncfile.description = 'NN flux Created with ml training'
        ncfile.close()

    else:
        raise Exception('Can only save DNN with 2 or 5 layers')


    """Save dictionary for nn rescaling and other properies"""

    #if not os.path.exists(base_dir + 'mldata_tmp/regressors/'):
    #    os.makedirs(base_dir + 'mldata_tmp/regressors/')
    #est_errors = 0 
    #pickle.dump([net, est_str, est_errors, f_ppi, o_ppi, f_pp, o_pp, y, z, p, rho],
    #            open(base_dir + 'mldata_tmp/regressors/' + est_str + '.pkl', 'wb'))

    print(f"Neural network and metadata saved to {full_dir_path}")
    



class Net_ANN_5_no_BN(nn.Module):
    def __init__(self,n_in, n_out, neurons = 128, dropoff=0.0):
        super(Net_ANN_5_no_BN, self).__init__()
        self.linear1 = nn.Linear(n_in, neurons)
        self.linear2 = nn.Linear(neurons, neurons)
        self.linear3 = nn.Linear(neurons, neurons)
        self.linear4 = nn.Linear(neurons, neurons)
        self.linear5 = nn.Linear(neurons, n_out)

        self.lin_drop = nn.Dropout(dropoff)  # regularization method to prevent overfitting.
        self.lin_drop2 = nn.Dropout(dropoff)
        self.lin_drop3 = nn.Dropout(dropoff)
        self.lin_drop4 = nn.Dropout(dropoff)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.lin_drop(x)
        x = F.relu(self.linear2(x))
        x = self.lin_drop2(x)
        x = F.relu(self.linear3(x))
        x = self.lin_drop3(x)
        x = F.relu(self.linear4(x))
        x = self.lin_drop4(x)
        x = self.linear5(x)
        return x




if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: neural_network_training.py <config_file.yaml>")
    else:
        train_wrapper(sys.argv[1])
