import torch
import time 
from utils.checkpoints import load_checkpoint, save_checkpoint
from vecopsciml.utils import TensOps

from .eval import loss_function
from .eval import loss_function_autoencoder


def train_epoch(model, optimizer, X_train, y_train, f_train, D):
    """
    Trains the model for one epoch.

    Parameters:
        model (torch.Model): The model being trained.
        optimizer (torch.Optimizer): The optimization algorithm used to update model parameters.
        X_train (torch.Tensor): Input of the model.
        y_train (TensOps): Solution of the equation.
        f_train (TensOps): Source term of the equation.
        D (torch.Tensor): Derivative filters.

    Returns:
        loss (float): Total computed loss for the training epoch.
        e_loss, pi1_loss, pi2_loss, pi3_loss (float): Specific components of the loss related to the model's performance (regularizations).
    """

    # Set the model to training mode and activate forward pass.
    model.train()
    y_pred, K_pred = model(X_train)

    # Compute the total loss and its components by passing the necessary arguments to the loss function.
    loss, e_loss, pi1_loss, pi2_loss, pi3_loss = loss_function(X_train, y_train, y_pred, K_pred, f_train, D)

    # Reset the gradients, perform backpropagation, and update model's parameters using optimizer and compute gradients.
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    # Return the total loss and its components.
    return loss, e_loss, pi1_loss, pi2_loss, pi3_loss



def test_epoch(model, X_test, y_test, f_test, D):
    """
    Evaluates the model on the test dataset for one epoch.

    Parameters:
        model (torch.Model): The model being trained.
        optimizer (torch.Optimizer): The optimization algorithm used to update model parameters.
        X_test (torch.Tensor): Input of the model.
        y_test (TensOps): Solution of the equation.
        f_test (TensOps): Source term of the equation.
        D (torch.Tensor): Derivative filters.

    Returns:
        loss (float): Total computed loss for the training epoch.
        e_loss, pi1_loss, pi2_loss, pi3_loss (float): Specific components of the loss related to the model's performance (regularizations).
    """

    # Perform a forward pass with the test data to generate predictions
    y_pred, K_pred = model(X_test)

    # Compute the total loss and its components for the test data.
    loss, e_loss, pi1_loss, pi2_loss, pi3_loss = loss_function(X_test, y_test, y_pred, K_pred, f_test, D)

    # Return the total loss and its components.
    return loss, e_loss, pi1_loss, pi2_loss, pi3_loss



def train_loop(model, optimizer, X_train, y_train, f_train, X_test, y_test, f_test, D, 
               n_checkpoints, start_epoch, n_epochs, batch_size, model_results_path, device, new_lr=None):
    """
    Main training loop for the model over multiple epochs.

    Parameters:
        model (torch.Model): The machine learning model to train.
        optimizer (torch.Optimizer): Optimizer used to update model parameters.
        n_checkpoints (int): Number of checkpoints to save during training.
        
        X_train (torch.Tensor): Training data (features).
        y_train (TensOps): Training labels or targets.
        X_test (torch.Tensor): Test data (features).
        y_test (TensOps): Test labels or targets.
        f_train (TensOps): Ground truth function or reference data for training.
        f_test (TensOps): Ground truth function or reference data for testing.
        D (torch.Tensor): Derivative filters.

        start_epoch (int): Epoch to start training from.
        n_epochs (int): Total number of epochs to train the model.
        batch_size (int): Number of samples per training batch.
        model_results_path (str): Path to save model checkpoints and results.
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').
        new_lr (int): Optional. New learning rate to set if resuming from a checkpoint.
    """

    # Handle case where training resumes from a saved checkpoint
    if start_epoch > 0:
        # Load the saved model, optimizer, and associated lists from checkpoint
        print(f'Starting training from a checkpoint. Epoch {start_epoch}.')
        resume_epoch = start_epoch
        model, optimizer, lists = load_checkpoint(model, optimizer, resume_epoch, model_results_path)

        # Update learning rate if a new one is specified
        if new_lr != None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        time_list = lists['time_list']

        train_total_loss_list = lists['train_total_loss_list']
        train_e_loss_list = lists['train_e_loss_list']
        train_pi1_loss_list = lists['train_pi1_loss_list']
        train_pi2_loss_list = lists['train_pi2_loss_list']
        train_pi3_loss_list = lists['train_pi3_loss_list']

        test_total_loss_list = lists['test_total_loss_list']
        test_e_loss_list = lists['test_e_loss_list']
        test_pi1_loss_list = lists['test_pi1_loss_list']
        test_pi2_loss_list = lists['test_pi2_loss_list']
        test_pi3_loss_list = lists['test_pi3_loss_list']

    # Handle case where training starts from scratch
    else:
        print("Starting training from scratch.")

        train_total_loss_list = []
        train_e_loss_list = []
        train_pi1_loss_list = []
        train_pi2_loss_list = []
        train_pi3_loss_list = []

        test_total_loss_list = []
        test_e_loss_list = []
        test_pi1_loss_list = []
        test_pi2_loss_list = []
        test_pi3_loss_list = []

        time_list = []

    # Get the number of samples in training and test datasets
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    # Start training loop for specified number of epochs
    for epoch_i in range(start_epoch, n_epochs):
        
        # Time instant when starts the training of the model for one epoch
        time_init = time.time()

        # Process training data in batches and prepare them in the needed format
        for batch_start in range(0, N_train, batch_size):

            X_batch = X_train[batch_start:(batch_start+batch_size)].to(device)
            y_batch = TensOps(y_train.values[batch_start:(batch_start+batch_size)].to(device), space_dimension=y_train.space_dim, contravariance=y_train.order[0], covariance=y_train.order[1])
            f_batch = TensOps(f_train.values[batch_start:(batch_start+batch_size)].to(device), space_dimension=f_train.space_dim, contravariance=f_train.order[0], covariance=f_train.order[1])

            # Train a batch
            loss, e_loss, pi1_loss, pi2_loss, pi3_loss = train_epoch(model, optimizer, X_batch, y_batch, f_batch, D)

        # Time instant when ends  the training of the model for one epoch and obtaining the total time
        time_end = time.time()
        train_time_epoch = time_end - time_init

        # Record average training losses after processing all batches
        time_list.append(train_time_epoch)
        train_total_loss_list.append(loss.item()/batch_size)
        train_e_loss_list.append(e_loss.item()/batch_size)
        train_pi1_loss_list.append(pi1_loss.item()/batch_size)
        train_pi2_loss_list.append(pi2_loss.item()/batch_size)
        train_pi3_loss_list.append(pi3_loss.item()/batch_size)
          
        # Evaluate model on the test dataset at the end of each epoch
        loss_test, e_loss_test, pi1_loss_test, pi2_loss_test, pi3_loss_test = test_epoch(model, X_test, y_test, f_test, D)

        # Record average test losses
        test_total_loss_list.append(loss_test.item()/N_test)
        test_e_loss_list.append(e_loss_test.item()/N_test)
        test_pi1_loss_list.append(pi1_loss_test.item()/N_test)
        test_pi2_loss_list.append(pi2_loss_test.item()/N_test)
        test_pi3_loss_list.append(pi3_loss_test.item()/N_test)

        # Print training and testing metrics periodically
        if epoch_i % (1 if n_epochs < 100 else (10 if n_epochs <= 1000 else 100)) == 0:
            print(f'Epoch {epoch_i}, Train loss: {loss.item()/batch_size:.3e}, Test loss: {loss_test.item()/N_test:.3e}, MSE(e): {e_loss.item()/batch_size:.3e}, MSE(pi1): {pi1_loss.item()/batch_size:.3e}, MSE(pi2): {pi2_loss.item()/batch_size:.3e}, MSE(pi3): {pi3_loss.item()/batch_size:.3e}')

        # Save a checkpoint at regular intervals
        if epoch_i % (int(n_epochs/n_checkpoints)) == 0:
            save_checkpoint(model, optimizer, epoch_i, model_results_path, time_list=time_list,
                            train_total_loss_list=train_total_loss_list, train_e_loss_list=train_e_loss_list, 
                            train_pi1_loss_list=train_pi1_loss_list, train_pi2_loss_list=train_pi2_loss_list, train_pi3_loss_list=train_pi3_loss_list,
                            test_total_loss_list=test_total_loss_list, test_e_loss_list=test_e_loss_list,
                            test_pi1_loss_list=test_pi1_loss_list, test_pi2_loss_list=test_pi2_loss_list, test_pi3_loss_list=test_pi3_loss_list)

    print("\nTraining process finished after", n_epochs, "epochs\n")

    save_checkpoint(model, optimizer, epoch_i, model_results_path, time_list=time_list,
                    train_total_loss_list=train_total_loss_list, train_e_loss_list=train_e_loss_list, 
                    train_pi1_loss_list=train_pi1_loss_list, train_pi2_loss_list=train_pi2_loss_list, train_pi3_loss_list=train_pi3_loss_list,
                    test_total_loss_list=test_total_loss_list, test_e_loss_list=test_e_loss_list,
                    test_pi1_loss_list=test_pi1_loss_list, test_pi2_loss_list=test_pi2_loss_list, test_pi3_loss_list=test_pi3_loss_list)
    
    save_checkpoint(model, optimizer, epoch_i, model_results_path, end_flag=True, time_list=time_list,
                    train_total_loss_list=train_total_loss_list, train_e_loss_list=train_e_loss_list, 
                    train_pi1_loss_list=train_pi1_loss_list, train_pi2_loss_list=train_pi2_loss_list, train_pi3_loss_list=train_pi3_loss_list,
                    test_total_loss_list=test_total_loss_list, test_e_loss_list=test_e_loss_list,
                    test_pi1_loss_list=test_pi1_loss_list, test_pi2_loss_list=test_pi2_loss_list, test_pi3_loss_list=test_pi3_loss_list)
    


def train_autoencoder_epoch(model, optimizer, X_train, y_train):
    """
    Performs one training epoch for an autoencoder model.

    Parameters:
        model (torch.nn.Module): The autoencoder model to train.
        optimizer (torch.optim.Optimizer): Optimizer used to update the model's parameters.
        X_train (torch.Tensor): Input data for training.
        y_train (torch.Tensor): Target data (ground truth) corresponding to the input data.

    Returns:
        loss (torch.Tensor): The computed loss for this epoch.
    """
    # Set the model to training mode and activate forward pass.
    model.train()
    y_pred = model(X_train)
    
    # Compute the total loss and its components by passing the necessary arguments to the loss function.
    loss = loss_function_autoencoder(y_train, y_pred)
    
    # Reset the gradients, perform backpropagation, and update model's parameters using optimizer and compute gradients.
    optimizer.zero_grad() 
    loss.backward(retain_graph=True)
    optimizer.step()
    
    # Return the computed loss for tracking purposes
    return loss


def test_autoencoder_epoch(model, X_test, y_test):
    """
    Evaluates the model on the test dataset for one epoch.

    Parameters:
        model (torch.Model): The model being trained.
        optimizer (torch.Optimizer): The optimization algorithm used to update model parameters.
        X_test (torch.Tensor): Input of the model.
        y_test (TensOps): Solution of the equation.

    Returns:
        loss (float): Total computed loss for the training epoch.
    """
    # Perform a forward pass with the test data to generate predictions and compute the loss for the test data
    y_pred = model(X_test)
    loss = loss_function_autoencoder(y_test, y_pred)

    # Return the total loss 
    return loss


def train_autoencoder_loop(model, optimizer, X_train, y_train, X_test, y_test,
                           n_checkpoints, start_epoch, n_epochs, batch_size, 
                           model_results_path, device, new_lr=None):
    """
    Main training loop for the model over multiple epochs.

    Parameters:
        model (torch.Model): The machine learning model to train.
        optimizer (torch.Optimizer): Optimizer used to update model parameters.
        n_checkpoints (int): Number of checkpoints to save during training.
        
        X_train (torch.Tensor): Training data (features).
        y_train (TensOps): Training labels or targets.
        X_test (torch.Tensor): Test data (features).
        y_test (TensOps): Test labels or targets.

        start_epoch (int): Epoch to start training from.
        n_epochs (int): Total number of epochs to train the model.
        batch_size (int): Number of samples per training batch.
        model_results_path (str): Path to save model checkpoints and results.
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').
        new_lr (int): Optional. New learning rate to set if resuming from a checkpoint.
    """
    # Handle case where training resumes from a saved checkpoint
    if start_epoch > 0:
        # Load the saved model, optimizer, and associated lists from checkpoint
        print(f'Starting training from a checkpoint. Epoch {start_epoch}.')

        resume_epoch = start_epoch
        model, optimizer, lists = load_checkpoint(model, optimizer, resume_epoch, model_results_path)

        # Update learning rate if a new one is specified
        if new_lr != None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        time_list = lists['time_list']
        train_total_loss_list = lists['train_total_loss_list']
        test_total_loss_list = lists['test_total_loss_list']

    # Handle case where training starts from scratch
    else:
        print("Starting training from scratch.")

        time_list = []
        train_total_loss_list = []
        test_total_loss_list = []

    # Get the number of samples in training and test datasets
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    # Start training loop for specified number of epochs
    for epoch_i in range(start_epoch, n_epochs):

        # Time instant when starts the training of the model for one epoch
        time_init = time.time()

        # Process training data in batches and prepare them in the needed format
        for batch_start in range(0, N_train, batch_size):
            X_batch = X_train[batch_start:(batch_start+batch_size)].to(device)
            y_batch = TensOps(y_train.values[batch_start:(batch_start+batch_size)].to(device), space_dimension=y_train.space_dim, 
                              contravariance=y_train.order[0], covariance=y_train.order[1])
            
            # Train a batch
            loss_train = train_autoencoder_epoch(model, optimizer, X_batch, y_batch).item()
        
        # Time instant when ends  the training of the model for one epoch and obtaining the total time
        time_end = time.time()
        train_time_epoch = time_end - time_init
        
        loss_test = test_autoencoder_epoch(model, X_test, y_test).item()

        # Record average training  and test losses after processing all batches
        time_list.append(train_time_epoch)
        train_total_loss_list.append(loss_train/batch_size)
        test_total_loss_list.append(loss_test/N_test)

        # Print training and testing metrics periodically
        if epoch_i % (1 if n_epochs < 100 else (10 if n_epochs <= 1000 else 100)) == 0:
            print(f'Epoch {epoch_i}, Train loss: {loss_train/batch_size:.3e}, Test loss: {loss_test/N_test:.3e}')

        # Save a checkpoint at regular intervals
        if epoch_i % (int(n_epochs/n_checkpoints)) == 0:
            save_checkpoint(model, optimizer, epoch_i, model_results_path, train_total_loss_list=train_total_loss_list, 
                            test_total_loss_list=test_total_loss_list, time_list=time_list)

    save_checkpoint(model, optimizer, epoch_i, model_results_path, end_flag=True, train_total_loss_list=train_total_loss_list, 
                    test_total_loss_list=test_total_loss_list, time_list=time_list)   