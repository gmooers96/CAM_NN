""" This script is an edited version of JY original training code. Most redundant or not relevant modules eliminated and some (though not complete) optimization made for Bridges2 system. Could have memory errors.
"""

### section to force print statement in slurm logging
import sys
print("Job started")
sys.stdout.flush()

# helper file imports
import data_scalar_and_reshaper_baseline as ml_load
import post_processing_figures
import Build_Neural_Networks as neural_networks

# file creation imports
import os
import yaml

# Data imports
import numpy as np
import math
from netCDF4 import Dataset
import xarray as xr
from sklearn.metrics import r2_score

# Model imports
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torchview import draw_graph
import torchvision
from torch import nn, optim

# debugging imports
import pdb

# plotting imports
import matplotlib.pyplot as plt

def vertical_grad_penalty(y_pred, lambda_smooth=0.01, z_dim=49):
    total_penalty = 0.0
    for i in range(5):  # 5 output variables, each length 49
        start = i * z_dim
        end = (i + 1) * z_dim
        y_slice = y_pred[:, start:end]  # (batch, 49)
        dz = y_slice[:, 1:] - y_slice[:, :-1]  # finite difference
        total_penalty += dz.pow(2).mean()
    return lambda_smooth * total_penalty


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
    config_id = name_dict['id']

    #unpack the data specific calls for the experiment
    training_data_path = data_dict['training_expt']
    test_data_path = data_dict['test_expt']
    weights_path = data_dict['weights_path']
    save_path = data_dict['save_path']
    single_file = data_dict['single_file']
    f_ppi = data_dict['f_ppi']
    o_ppi = data_dict['o_ppi']
    levels = data_dict['levels']
    do_poles = data_dict['do_poles']
    input_vert_vars = data_dict['input_vert_vars']
    output_vert_vars = data_dict['output_vert_vars']
    rewight_outputs = data_dict['rewight_outputs']
    training_data_volume = data_dict['training_data_volume']
    test_data_volume = data_dict['test_data_volume']
    data_chunks=data_dict['data_chunks']
    dtype_size=data_dict['dtype_size']
    memory_fraction=data_dict['memory_fraction']
    sym_log=data_dict['sym_log']
    t_adv_vert_scale=data_dict['t_adv_vert_scale']
    save_data_figs=data_dict['save_data_figs']
    
    # get the neural network specifications by unpack the nn dict
    do_nn = nn_dict['do_nn']
    layers = nn_dict['layer_sizes'] # added by Griffin
    activation_function = nn_dict['activation_function'] # added by Griffin
    neural_network_name = nn_dict['neural_network_name'] # added by Griffin 
    criterion=nn_dict['criterion']  # Name of the loss function class
    lambda_smooth=nn_dict['lambda_smooth']
    optimizer=nn_dict['optimizer']
    num_workers=nn_dict['num_workers']
    cycle_momentum=nn_dict['cycle_momentum']
    torch_thread_number=nn_dict['torch_thread_number']
    batch_norm = nn_dict['batch_norm']
    epochs = nn_dict['epochs']
    epochs_small_lr = nn_dict['epochs_small_lr']
    lr = nn_dict['lr']
    min_lr = nn_dict['min_lr']
    max_lr = nn_dict['max_lr']
    lr_redux_factor = nn_dict['lr_redux_factor']
    step_size_up = nn_dict['step_size_up']
    batch_size = nn_dict['batch_size']
    dropoff = nn_dict['dropoff']
    weighted_loss = nn_dict['weighted_loss']
    t_adv_weight = nn_dict['t_adv_weight']
    q_adv_weight = nn_dict['q_adv_weight']
    conv = nn_dict['conv']
    conv_layer_configs=nn_dict['conv_layer_configs']

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

        if os.path.isdir(save_path+nametag+'/Training_Data/'):
            pass
        else:
            os.mkdir(save_path+nametag+'/Training_Data/')
        
        train_inputs_scaled, test_inputs_scaled, train_outputs_scaled, test_outputs_scaled, scaled_inputs,  scaled_outputs = ml_load.LoadDataStandardScaleData_v3(
                                                       traindata=training_data_path,
                                                       testdata=test_data_path,
                                                       input_vert_vars=input_vert_vars,
                                                       output_vert_vars=output_vert_vars,
                                                        training_data_volume=training_data_volume,
                                                        test_data_volume=test_data_volume,
                                                       weights=weights_path,
                                                       chunk=data_chunks,
                                                        sym_log=sym_log,
                                                        t_adv_vert_scale=t_adv_vert_scale,
                                                        save_data_figs=save_data_figs,
                                                        save_path=save_path+nametag+'/Training_Data/',
        )
        
        # get the size of features of the input_vector
        input_feature_length = train_inputs_scaled.shape[1]
        # get the size of features of the output_vector
        output_feature_length = train_outputs_scaled.shape[1]

        #append the input and output size to the list
        layers.insert(0, input_feature_length)

        # Append an item to the end
        layers.append(output_feature_length)
        
        ######################################
        #### Construct the Neural Network
        ######################################

        activation_class = getattr(nn, activation_function)()
        NeuralNetClass = getattr(neural_networks, neural_network_name)

        if conv and neural_network_name == "ConvColumnNet1D":
            model = NeuralNetClass(
                conv_layer_configs=conv_layer_configs,
                fc_layer_sizes=layers,  # here 'layers' is your dense layer list
                activation_fn=activation_class,
                dropoff=dropoff,
                levels=levels
            )
        elif conv and neural_network_name == "ConvColumnNet2D":
            model = NeuralNetClass(
                conv_layer_configs=conv_layer_configs,
                fc_layer_sizes=layers,
                activation_fn=activation_class,
                dropoff=dropoff,
                levels=levels
            )
        else:
            model = NeuralNetClass(
                layer_sizes=layers,
                activation_fn=activation_class,
                dropoff=dropoff
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

            #choose optimal torch number if not preselected
            if isinstance(torch_thread_number, int):
                pass
            else:
                torch_thread_number = calculate_optimal_threads(
                                                                batch_size=batch_size, 
                                                                input_size=input_feature_length, 
                                                                output_size=output_feature_length, 
                                                                dtype_size=dtype_size, 
                                                                memory_fraction=memory_fraction,
                                                                )
    

            model = train_nn_griffin(
                                    net=model,
                                    savepath=save_path,
                                    nametag=nametag,
                                    save_name=save_name,
                                    train_inputs_scaled=train_inputs_scaled, 
                                    train_outputs_scaled=train_outputs_scaled, 
                                    test_inputs_scaled=test_inputs_scaled, 
                                    test_outputs_scaled=test_outputs_scaled, 
                                    output_vert_dim=levels, 
                                    output_vert_vars=output_vert_vars,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    num_workers=num_workers,
                                    cycle_momentum=cycle_momentum,
                                    torch_thread_number=torch_thread_number,
                                    epochs=epochs,
                                    epochs_small_lr=epochs_small_lr,
                                    min_lr=min_lr,
                                    max_lr=max_lr,
                                    lr_redux_factor=lr_redux_factor,
                                    step_size_up=step_size_up,
                                    batch_size=batch_size,
                                    lr=lr,
                                    weighted_loss=weighted_loss,
                                    t_adv_weight=t_adv_weight,
                                    q_adv_weight=q_adv_weight,
                                    lambda_smooth=lambda_smooth,
                                    )

            # bring in the data for info in the function below
            # TODO:@gmooers -- check how much of this is really necessary
            variables = xr.open_mfdataset(training_data_path, chunks=data_chunks, combine='nested', concat_dim='sample')

            save_nn_griffin_V3(
                net=model, 
                output_preprocess_dict=scaled_outputs, 
                input_preprocess_dict=scaled_inputs, 
                input_scaling_type=f_ppi,  # leftover label (in form of dict) parameter from JY -- not sure if necessary -- keep for now for code running
                output_scaling_type=o_ppi, # leftover label (in form of dict) parameter from JY -- not sure if necessary -- keep for now for code running
                y=variables.lat.values, 
                z=variables.z.values, 
                p=variables.p.values, 
                rho=variables.rho.values,  
                output_filename=nametag, 
                base_dir=save_path, 
                batch_norm=batch_norm,
                t_adv_vert_scale=t_adv_vert_scale,
            )



    # Calculate the predictions of the trained model

    else:
        
        save_location = save_path+nametag+"/Saved_Models/"+"stage_2_"+save_name
        model.load_state_dict(torch.load(save_location))


    scaled_predictions = get_model_predictions_from_numpy(model=model, #trained model
                                     test_data=test_inputs_scaled, #scaled test data
                                     batch_size=batch_size,
                                                         )
    # unscale the predictions to compare to the outputs
    unscaled_predictions = ml_load.undo_scaling_predictions(
                                scaled_array=scaled_predictions, 
                                scaler_dict=scaled_outputs, 
                                vertical_dimension=levels, 
                                output_variables=output_vert_vars,
                                sym_log=sym_log,
                                )

    test_outputs_original = ml_load.undo_scaling_targets(
                                scaled_array=scaled_predictions, 
                                scaler_dict=scaled_outputs, 
                                vertical_dimension=levels, 
                                output_variables=output_vert_vars,
                                sym_log=sym_log,
                                )

    post_processing_figures.main_plotting(truth=np.transpose(test_outputs_original), # put z dimension first for plotting code
                                              pred=np.transpose(unscaled_predictions), #put z dimension first for plotting code 
                                              raw_data=single_file,
                                              z_dim=levels,
                                             var_names=output_vert_vars,
                                              save_path=save_path,
                                              nametag=nametag,
                                              do_poles=do_poles,
                                             )

    return "Offline Neural Model Trained and Test"


############################
# Function added or edited by Griffin
############################

def rmse(x, y): return math.sqrt(((x - y) ** 2).mean())


def train_nn_griffin(
    net, 
    savepath, 
    nametag, 
    save_name, 
    train_inputs_scaled, 
    train_outputs_scaled, 
    test_inputs_scaled, 
    test_outputs_scaled,
    output_vert_dim, 
    output_vert_vars,
    criterion,
    optimizer,
    num_workers=1, # was 4, changed to 1 for improved performance on Bridges2 -- 4 gives a warning message
    cycle_momentum=False, # Default param from JN
    torch_thread_number=10, #set by Janni
    epochs =7, 
    epochs_small_lr=5,
    min_lr = 2e-4,
    max_lr = 2e-3,
    lr_redux_factor = 10,
    step_size_up=4000,
    batch_size = 1024, 
    lr=1e-7,
    weighted_loss=False,
    t_adv_weight=1.0,
    q_adv_weight=1.0,
    lambda_smooth=0.0,
    ):

    ###converts from type numpy to type tensor and .float() ensures data is 32 bit

    # training inputs
    train_outputs_scaled_tensor = torch.from_numpy(train_outputs_scaled).float()

    # training outputs
    train_inputs_scaled_tensor = torch.from_numpy(train_inputs_scaled).float()

    # test inputs
    test_inputs_scaled_tensor = torch.from_numpy(test_inputs_scaled).float()

    # test outputs
    test_outputs_scaled_tensor = torch.from_numpy(test_outputs_scaled).float()

    # build a combined training dataset 
    train_torch_dataset = Data.TensorDataset(train_inputs_scaled_tensor, train_outputs_scaled_tensor)
    test_torch_dataset = Data.TensorDataset(test_inputs_scaled_tensor, test_outputs_scaled_tensor)

    # build a training data loader object for the nn
    train_loader = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,  
        )

    test_loader = Data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,  
        )

    # load in the hyperparameter choices preset in the config file
    if criterion == "LogCoshLoss":
        loss_func = LogCoshLoss()
    else:
        loss_func = getattr(nn, criterion)()
    #loss_func = getattr(nn, criterion)()
    optimizer = getattr(optim, optimizer)(net.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr= max_lr, step_size_up=step_size_up, cycle_momentum=cycle_momentum)
    torch.set_num_threads(torch_thread_number)

    train_losses = []
    test_losses = []
    test_r2 = []
    test_r2_per_var = []
    test_rmse = []

    # first stge of training -- larger learning rate
    for epoch in range(epochs):
        print(" ")
        print("This is high LR epoch", int(epoch+1))
        print(" ")
        # training
        train_score, test_score = train_model_cyclic_griffin(net, loss_func, train_loader, test_loader, 
                                                             optimizer, scheduler, weighted_loss, t_adv_weight, q_adv_weight, lambda_smooth)
        # test model performance
        R2, RMSE, per_var_r2 = test_model(net, test_inputs_scaled_tensor, test_outputs_scaled, output_vert_dim, output_vert_vars)

        train_losses.append(train_score)
        test_losses.append(test_score)
        test_r2.append(R2)
        test_r2_per_var.append(per_var_r2)
        
        test_rmse.append(RMSE)

    # save the model after the first stage is completed
    save_location = savepath+nametag+"/Saved_Models/"+"stage_1_"+save_name 
    
    os.makedirs(os.path.dirname(save_location), exist_ok=True)
    torch.save(net.state_dict(), save_location)

    #Run a few epochs with lower learning rate.
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr/lr_redux_factor, max_lr= max_lr/lr_redux_factor, step_size_up=step_size_up, cycle_momentum=False)
    for epoch in range(epochs_small_lr):
        print(" ")
        print("This is low LR epoch", int(epoch+1))
        print(" ")
        train_score, test_score = train_model_cyclic_griffin(net, loss_func, train_loader, test_loader, 
                                                             optimizer, scheduler, weighted_loss, t_adv_weight, q_adv_weight, lambda_smooth)
        R2, RMSE, per_var_r2 = test_model(net, test_inputs_scaled_tensor, test_outputs_scaled, output_vert_dim, output_vert_vars)

        train_losses.append(train_score)
        test_losses.append(test_score)
        test_r2.append(R2)
        test_r2_per_var.append(per_var_r2)
        test_rmse.append(RMSE)

    #save the model after the second stage
    save_location = savepath+nametag+"/Saved_Models/"+"stage_2_"+save_name  
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_location), exist_ok=True)
    torch.save(net.state_dict(), save_location)

    # plot the model performance (training/test loss at each epoch across both stages)
    plot_save_path = os.path.join(savepath, nametag, "Loss_Curves", "Model_Losses.pdf")
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    
    print("plotting NN performance")

    plot_save_path = os.path.join(savepath, nametag, "Loss_Curves", "RMSE_R2_Curve.pdf")
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

    plot_rmse_r2(
        test_rmse=test_rmse,
        test_r2=test_r2,
        epochs=epochs,
        plot_save_path=plot_save_path,
    )

    # Save RMSE + T_adv_out R² plot
    plot_save_path_tadv = os.path.join(savepath, nametag, "Loss_Curves", "RMSE_Tadv_R2_Curve.pdf")
    os.makedirs(os.path.dirname(plot_save_path_tadv), exist_ok=True)
    plot_rmse_tadv_r2(
        test_rmse=test_rmse,
        test_r2_per_var=test_r2_per_var,
        epochs=epochs,
        plot_save_path=plot_save_path_tadv,
    )
    
    # Save RMSE + all 5 R² plot
    plot_save_path_all = os.path.join(savepath, nametag, "Loss_Curves", "RMSE_All_R2_Curve.pdf")
    os.makedirs(os.path.dirname(plot_save_path_all), exist_ok=True)
    plot_rmse_all_r2(
        test_rmse=test_rmse,
        test_r2_per_var=test_r2_per_var,
        output_vert_vars=output_vert_vars,
        epochs=epochs,
        plot_save_path=plot_save_path_all,
    )


    return net
    

def test_model(
            net,
            input_features,
            output_targets, 
            output_vert_dim, 
            output_vert_vars,
            ):

    # set neural network to evaluation mode for analysis
    net.eval()

    # Get the predictions from the neural net based on input training or test features
    nn_predictions = net(input_features).detach().cpu().numpy()

    # compare skill between "truth" and nn predictions
    RMSE = rmse(nn_predictions, output_targets)
    #R2 = r2_score(nn_predictions, output_targets,
                   #multioutput='variance_weighted')
    R2 = r2_score(output_targets.flatten(), nn_predictions.flatten())
    
    # print out the scores at each epoch for analysis statistics
    print('RMSE: ',RMSE, 
          ' R2:' , R2,
         )

    # Get the R2 scores for each individual output variable by doing a for loop over each variable in the output vector level. e.g. each 49 length column
    per_var_r2 = []
    for i in range(len(output_vert_vars)):
        start = int(output_vert_dim * i)
        end = int(output_vert_dim * (i + 1))
        r2_i = r2_score(output_targets[:, start:end], nn_predictions[:, start:end], multioutput='variance_weighted')
        per_var_r2.append(r2_i)
        print(f'{output_vert_vars[i]} R2: {r2_i:.4f}')

    return R2, RMSE, per_var_r2


def train_model_cyclic_griffin(net, criterion, trainloader, testloader, optimizer, scheduler, weighted_loss, t_adv_weight, q_adv_weight, lambda_smooth):
    # Training phase
    net.train()  # Set the model to training mode
    total_train_loss = 0

    # Initialize weights once
    weights = None
    if weighted_loss:
        # Assuming output dim is 245
        example_batch = next(iter(trainloader))[1]  # get shape from y_batch
        weights = torch.ones(example_batch.shape[1])
        weights[49:98] *= t_adv_weight  # Adjust weight for T_adv_out
        weights[98:147] *= q_adv_weight  # Adjust weight for T_adv_out

    # Loop over each batch in the training set
    for step, (batch_x, batch_y) in enumerate(trainloader):
        # Input and output batches
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        # Zero out gradients before backpropagation
        optimizer.zero_grad()

        # Forward pass: get predictions
        prediction = net(b_x)

        if weighted_loss:
            loss_fn = nn.MSELoss(reduction='none')  # per-element loss
            loss_raw = loss_fn(prediction, b_y)     # shape (batch, 245)

            # Move weights to device just once
            if weights.device != prediction.device:
                weights = weights.to(prediction.device)

            loss = (loss_raw * weights).mean()
        else:
            loss = criterion(prediction, b_y)

        # Backpropagation to compute gradients
        loss.backward()

        # Apply gradients to update model parameters
        optimizer.step()

        # Accumulate training loss
        total_train_loss += loss.item()

    # Update learning rate schedule after the epoch
    scheduler.step()

    print(f"Total training loss for the epoch: {total_train_loss}")

    # Test phase
    net.eval()  # Set the model to evaluation mode
    total_test_loss = 0

    # Turn off gradients for testing phase to save memory and computations
    with torch.no_grad():
        for batch_x, batch_y in testloader:
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)

            # Forward pass: get predictions
            prediction = net(b_x)

            # Compute test loss
            loss = criterion(prediction, b_y)

            if lambda_smooth != 0.0:
                smooth_penalty = vertical_grad_penalty(prediction, lambda_smooth=lambda_smooth)
                loss = loss + smooth_penalty

            # Accumulate test loss
            total_test_loss += loss.item()

    print(f"Total test loss for the epoch: {total_test_loss}")

    # Return both training and test loss
    #total_train_loss / len(train_loader.dataset) -- add in for plotting comparison
    return total_train_loss, total_test_loss


def plot_training_testing_errors(train_losses, test_losses, epochs, plot_save_path=None):
    plt.figure(figsize=(20, 12))
    
    # Plot for Stage 1
    plt.plot(np.arange(0,len(train_losses[:epochs]))+1, train_losses[:epochs], label='Training Error (Stage 1)', color='green')
    plt.plot(np.arange(0,len(test_losses[:epochs]))+1, [test_losses[:epochs]], label='Testing Error (Stage 1)', color='blue')
    
    # Plot for Stage 2_test
    plt.plot(range(epochs, epochs + (epochs-2)), train_losses[epochs:], label='Training Error (Stage 2)', linestyle='--', color='green')
    plt.plot(range(epochs, epochs + (epochs-2)),  [test_losses[epochs:]], label='Testing Error (Stage 2)', linestyle='--', color='blue')
    
    # Mark the learning rate change point
    plt.axvline(x=epochs, color='red', linestyle=':', label='Learning Rate Change')
    
    plt.legend()
    plt.title('Training and Testing Error per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Error')

    if plot_save_path:
        plt.savefig(plot_save_path)


def plot_rmse_r2(test_rmse, test_r2, epochs, plot_save_path=None):
    fig, ax1 = plt.subplots(figsize=(20, 12))

    # Plot RMSE on the left y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE', color='orange')
    
    # Stage 1 RMSE
    ax1.plot(np.arange(0, epochs) + 1, test_rmse[:epochs], label='Test RMSE (Stage 1)', color='orange')
    # Stage 2 RMSE
    ax1.plot(np.arange(epochs, epochs + len(test_rmse[epochs:])), test_rmse[epochs:], label='Test RMSE (Stage 2)', linestyle='--', color='orange')
    
    ax1.tick_params(axis='y', labelcolor='orange')

    # Create a second y-axis for R2
    ax2 = ax1.twinx()  
    ax2.set_ylabel(r'$R^2$', color='blue')
    
    # Stage 1 R2
    ax2.plot(np.arange(0, epochs) + 1, test_r2[:epochs], label=r'Global Test $R^2$ (Stage 1)', color='blue')
    # Stage 2 R2
    ax2.plot(np.arange(epochs, epochs + len(test_r2[epochs:])), test_r2[epochs:], label=r'Global Test $R^2$ (Stage 2)', linestyle='--', color='blue')
    
    ax2.tick_params(axis='y', labelcolor='blue')

    # Mark the learning rate change point
    ax1.axvline(x=epochs, color='red', linestyle=':', label='Learning Rate Change')

    # Add legends for both lines
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(r'Test RMSE and $R^2$ per Epoch')

    if plot_save_path:
        plt.savefig(plot_save_path)
        

def plot_rmse_tadv_r2(test_rmse, test_r2_per_var, epochs, plot_save_path=None):
    t_adv_r2 = [r2s[1] for r2s in test_r2_per_var]  # assuming index 1 is T_adv_out

    fig, ax1 = plt.subplots(figsize=(20, 12))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE', color='orange')
    ax1.plot(np.arange(len(test_rmse)), test_rmse, label='Test RMSE', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'T_adv_out $R^2$', color='blue')
    ax2.plot(np.arange(len(t_adv_r2)), t_adv_r2, label='T_adv_out R2', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.title(r'Test RMSE and T_adv_out $R^2$ per Epoch')
    plt.axvline(x=epochs, color='red', linestyle=':', label='Learning Rate Change')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    if plot_save_path:
        plt.savefig(plot_save_path)

def plot_rmse_all_r2(test_rmse, test_r2_per_var, output_vert_vars, epochs, plot_save_path=None):
    fig, ax1 = plt.subplots(figsize=(20, 12))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE', color='orange')
    ax1.plot(np.arange(len(test_rmse)), test_rmse, label='Test RMSE', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$R^2$', color='blue')

    for idx, var_name in enumerate(output_vert_vars):
        var_r2 = [r2s[idx] for r2s in test_r2_per_var]
        ax2.plot(np.arange(len(var_r2)), var_r2, label=f'{var_name} R2')

    ax2.tick_params(axis='y', labelcolor='blue')

    plt.title(r'Test RMSE and Individual $R^2$ per Epoch')
    plt.axvline(x=epochs, color='red', linestyle=':', label='Learning Rate Change')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    if plot_save_path:
        plt.savefig(plot_save_path)




def save_nn_griffin_V3(net, 
                       output_preprocess_dict, 
                       input_preprocess_dict, 
                       input_scaling_type, 
                       output_scaling_type, 
                       y, z, p, rho,
                       output_filename, base_dir,
                       save_name="NC_Models/",
                       batch_norm=True,
                       t_adv_vert_scale=False):
    """
    Save PyTorch NN weights (fully connected and conv layers) and scaling info to NetCDF.
    Compatible with both ConvColumnNet1D and ConvColumnNet2D.
    """
    net.eval()
    print(net)

    in_dim = net.layers[0].weight.shape[1]

    input_raw_dim = sum(len(np.atleast_1d(val)) for val in input_preprocess_dict['mean'].values())
    X_mean = np.zeros(input_raw_dim, dtype=np.float32)
    X_std = np.zeros(input_raw_dim, dtype=np.float32)

    ind_now = 0
    for value in input_preprocess_dict['mean'].values():
        val = np.atleast_1d(value).astype(np.float32)
        X_mean[ind_now:ind_now + len(val)] = val
        ind_now += len(val)

    ind_now = 0
    for value in input_preprocess_dict['std'].values():
        val = np.atleast_1d(value).astype(np.float32)
        X_std[ind_now:ind_now + len(val)] = val
        ind_now += len(val)

    if t_adv_vert_scale:
        Y_mean = np.zeros(len(output_preprocess_dict['mean']) + 48, dtype=np.float32)
        Y_std = np.zeros(len(output_preprocess_dict['mean']) + 48, dtype=np.float32)
    else:
        Y_mean = np.zeros(len(output_preprocess_dict['mean']), dtype=np.float32)
        Y_std = np.zeros(len(output_preprocess_dict['mean']), dtype=np.float32)

    ind_now = 0
    for value in output_preprocess_dict['mean'].values():
        val = np.atleast_1d(value).astype(np.float32)
        Y_mean[ind_now:ind_now + len(val)] = val
        ind_now += len(val)

    ind_now = 0
    for value in output_preprocess_dict['std'].values():
        val = np.atleast_1d(value).astype(np.float32)
        Y_std[ind_now:ind_now + len(val)] = val
        ind_now += len(val)

    data_dict = {}
    conv_coords = {}

    for i, layer in enumerate(net.layers):
        w = layer.weight.data.numpy().astype(np.float32)
        b = layer.bias.data.numpy().astype(np.float32)
        if i == 0:
            data_dict[f'w{i+1}'] = (['N_h1', 'N_fc_input'], w)
        else:
            data_dict[f'w{i+1}'] = (['N_h'+str(i+1), 'N_h'+str(i)], w)
        data_dict[f'b{i+1}'] = (['N_h'+str(i+1)], b)

    if hasattr(net, 'conv_layers'):
        conv_layer_counter = 1
        for module in net.conv_layers:
            if isinstance(module, torch.nn.Conv2d):
                w = module.weight.detach().cpu().numpy().astype(np.float32)
                b = module.bias.detach().cpu().numpy().astype(np.float32)
                
                # Save both PyTorch and Fortran-friendly layouts separately if needed
                # But here, save Fortran-friendly format directly
                w_fortran = np.transpose(w, (3, 2, 1, 0))  # (k_w, k_h, in_c, out_c)
                
                data_dict[f'conv{conv_layer_counter}_w'] = (
                    [f'conv{conv_layer_counter}_kernel_width',
                     f'conv{conv_layer_counter}_kernel_height',
                     f'conv{conv_layer_counter}_in_channels',
                     f'conv{conv_layer_counter}_out_channels'], w_fortran
                )
                
                data_dict[f'conv{conv_layer_counter}_b'] = (
                    [f'conv{conv_layer_counter}_out_channels'], b
                )
                
                conv_coords[f'conv{conv_layer_counter}_kernel_width'] = np.arange(w_fortran.shape[0], dtype=np.int32)
                conv_coords[f'conv{conv_layer_counter}_kernel_height'] = np.arange(w_fortran.shape[1], dtype=np.int32)
                conv_coords[f'conv{conv_layer_counter}_in_channels'] = np.arange(w_fortran.shape[2], dtype=np.int32)
                conv_coords[f'conv{conv_layer_counter}_out_channels'] = np.arange(w_fortran.shape[3], dtype=np.int32)


                conv_layer_counter += 1

    elif hasattr(net, 'temp_conv_layers') and hasattr(net, 'qin_conv_layers'):
        temp_conv = [m for m in net.temp_conv_layers if isinstance(m, torch.nn.Conv1d)][0]
        qin_conv = [m for m in net.qin_conv_layers if isinstance(m, torch.nn.Conv1d)][0]

        temp_w = temp_conv.weight.data.numpy().squeeze(1).astype(np.float32)  # Transposed
        qin_w = qin_conv.weight.data.numpy().squeeze(1).astype(np.float32)    # Transposed
        temp_b = temp_conv.bias.data.numpy().astype(np.float32)
        qin_b = qin_conv.bias.data.numpy().astype(np.float32)

        data_dict['temp_conv_w'] = (['conv_filters', 'kernel_size'], temp_w)
        data_dict['temp_conv_b'] = (['conv_filters'], temp_b)
        data_dict['qin_conv_w'] = (['conv_filters', 'kernel_size'], qin_w)
        data_dict['qin_conv_b'] = (['conv_filters'], qin_b)

        conv_coords['conv_filters'] = np.arange(temp_w.shape[0], dtype=np.int32)
        conv_coords['kernel_size'] = np.arange(temp_w.shape[1], dtype=np.int32)

    if batch_norm and hasattr(net, 'batch_norm_layers'):
        for i, bn_layer in enumerate(net.batch_norm_layers):
            data_dict[f'batch_mean{i+1}'] = (['N_h'+str(i+1)], bn_layer.running_mean.data.numpy().astype(np.float32))
            data_dict[f'batch_stnd{i+1}'] = (['N_h'+str(i+1)], np.sqrt(bn_layer.running_var.data.numpy()).astype(np.float32))
            data_dict[f'batch_weight{i+1}'] = (['N_h'+str(i+1)], bn_layer.weight.data.numpy().astype(np.float32))
            data_dict[f'batch_bias{i+1}'] = (['N_h'+str(i+1)], bn_layer.bias.data.numpy().astype(np.float32))

    data_dict['yscale_mean'] = (['N_out_dim'], Y_mean)
    data_dict['yscale_stnd'] = (['N_out_dim'], Y_std)
    data_dict['xscale_mean'] = (['N_input_raw'], X_mean.astype('float32'))
    data_dict['xscale_stnd'] = (['N_input_raw'], X_std.astype('float32'))

    ds = xr.Dataset(
        data_dict,
        coords={
            'N_fc_input': np.arange(in_dim, dtype=np.int32),
            'N_input_raw': np.arange(input_raw_dim, dtype=np.int32),
            'N_out': np.arange(net.layers[-1].weight.shape[0], dtype=np.int32),
            'N_out_dim': np.arange(len(Y_mean), dtype=np.int32),
            **{f'N_h{i+1}': np.arange(net.layers[i].weight.shape[0], dtype=np.int32) for i in range(len(net.layers))},
            **conv_coords
        }
    )

    ds.attrs['y'] = np.array(y, dtype=np.float32).tolist()
    ds.attrs['z'] = np.array(z, dtype=np.float32).tolist()
    ds.attrs['p'] = np.array(p, dtype=np.float32).tolist()
    ds.attrs['rho'] = np.array(rho, dtype=np.float32).tolist()
    ds.attrs['input_scaling_type'] = str(input_scaling_type)
    ds.attrs['output_scaling_type'] = str(output_scaling_type)

    full_dir_path = os.path.join(base_dir, output_filename, save_name)
    os.makedirs(full_dir_path, exist_ok=True)

    ds.to_netcdf(os.path.join(full_dir_path, 'My_NN.nc'), format='NETCDF3_CLASSIC')

    print(f"✅ Neural network and metadata saved to {os.path.join(full_dir_path, 'My_NN.nc')}")



def calculate_optimal_threads(batch_size, input_size, output_size, dtype_size=4, memory_fraction=0.75):
    """
    Calculate the optimal number of threads for torch.set_num_threads based on CPU resources and memory limits.
    
    Parameters:
    - batch_size (int): The batch size used during training
    - input_size (int): Number of elements in the input vector
    - output_size (int): Number of elements in the output vector
    - dtype_size (int): Size of each data element in bytes (default is 4 bytes for float32)
    - memory_fraction (float): Fraction of available memory to use (default is 0.75, i.e., use 75% of available memory)
    
    Returns:
    - optimal_threads (int): The optimal number of threads to use
    """
    # Estimate memory usage per batch (in bytes)
    memory_per_batch = batch_size * (input_size + output_size) * dtype_size
    
    # Get available memory
    available_memory = psutil.virtual_memory().available * memory_fraction  # Only use a fraction of total memory

    # Estimate the number of batches that can fit into available memory
    max_batches_fit = available_memory // memory_per_batch

    # Calculate the number of CPU cores
    available_cores = multiprocessing.cpu_count()

    # We will use the minimum between available cores and max batches fitting in memory to set optimal threads
    optimal_threads = min(available_cores, max_batches_fit)

    # Use at least 1 thread (in case memory is tight)
    return max(1, int(optimal_threads))


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



if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: neural_network_training.py <config_file.yaml>")
    else:
        train_wrapper(sys.argv[1])
