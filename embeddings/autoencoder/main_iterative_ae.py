import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))

import torch
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Own library imports
from vecopsciml.utils import TensOps
from vecopsciml.operators.zero_order import Mx, My
from vecopsciml.kernels.derivative import DerivativeKernels

# Function from this project
from utils.folders import create_folder
from utils.load_data import load_data
from trainers.train import train_loop, train_autoencoder_loop

# Import model
from architectures.autoencoder import Autoencoder
from architectures.pgnniv_decoder import PGNNIVAutoencoder

# Parameters of the data
# N_DATA = [10, 100, 1000] 
# SIGMA = [0, 1, 5] # The noise added in '%'
# N_MODES = [5, 10, 50]

N_DATA = [20, 50, 5000] 
SIGMA = [0, 1, 5] # The noise added in '%'
N_MODES = [1, 2, 3, 20, 100]

combinations = list(itertools.product(N_DATA, SIGMA, N_MODES))

for combination_i in combinations:
    
    print("COMBINATION: ", combination_i)

    N_data_i = combination_i[0]
    sigma_i = combination_i[1]
    n_modes_i = combination_i[2]

    dataset = 'non_linear'
    data_name = dataset + '_' + str(N_data_i) + '_' + str(sigma_i)

    model = 'autoencoder'
    model_name = model + '_model_' + str(n_modes_i)

    ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    DATA_PATH = os.path.join(ROOT_PATH, r'data/', data_name, data_name) + '.pkl'
    RESULTS_FOLDER_PATH = os.path.join(ROOT_PATH, r'results/', data_name)

    MODEL_RESULTS_AE_PATH = os.path.join(ROOT_PATH, r'results/', data_name, model_name) + '_AE'
    MODEL_RESULTS_PGNNIV_PATH = os.path.join(ROOT_PATH, r'results/', data_name, model_name) + '_NN'

    # Creamos las carpetas que sean necesarias (si ya están creadas se avisará de ello)
    create_folder(RESULTS_FOLDER_PATH)
    create_folder(MODEL_RESULTS_AE_PATH)
    create_folder(MODEL_RESULTS_PGNNIV_PATH)

    # Load dataset
    dataset = load_data(DATA_PATH)

    # Convolutional filters to derivate
    dx = dataset['x_step_size']
    dy = dataset['y_step_size']
    D = DerivativeKernels(dx, dy, 0).grad_kernels_two_dimensions()

    DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    X_train = torch.Tensor(dataset['X_train']).unsqueeze(1)
    y_train = torch.Tensor(dataset['y_train']).unsqueeze(1)
    K_train = torch.tensor(dataset['k_train']).unsqueeze(1)
    f_train = torch.tensor(dataset['f_train']).unsqueeze(1).to(torch.float32)

    X_val = torch.Tensor(dataset['X_val']).unsqueeze(1)
    y_val = TensOps(torch.Tensor(dataset['y_val']).unsqueeze(1).requires_grad_(True), space_dimension=2, contravariance=0, covariance=0)
    K_val = TensOps(torch.tensor(dataset['k_val']).unsqueeze(1).requires_grad_(True), space_dimension=2, contravariance=0, covariance=0)
    f_val = TensOps(torch.tensor(dataset['f_val']).to(torch.float32).unsqueeze(1).requires_grad_(True), space_dimension=2, contravariance=0, covariance=0)

    print("Train dataset length:", len(X_train))
    print("Validation dataset length:", len(X_val))

    N_data_AE = len(X_train)//2
    N_data_NN = len(X_train) - len(X_train)//2
    prop_data_NN = 1 - N_data_AE/(N_data_NN + N_data_AE)

    print("Dataset length for the autoencoder:", N_data_AE)
    print("Dataset length for the PGNNIV:", N_data_NN)

    X_AE, X_NN, y_AE, y_NN, K_AE, K_NN, f_AE, f_NN = train_test_split(X_train, y_train, K_train, f_train, test_size=prop_data_NN, random_state=42)

    y_train_AE, y_test_AE = train_test_split(y_AE, test_size=0.2, random_state=42)

    y_train_AE = TensOps(y_train_AE.requires_grad_(True).to(DEVICE), space_dimension=2, contravariance=0, covariance=0)
    y_test_AE = TensOps(y_test_AE.requires_grad_(True).to(DEVICE), space_dimension=2, contravariance=0, covariance=0)

    X_train_NN, X_test_NN, y_train_NN, y_test_NN, K_train_NN, K_test_NN, f_train_NN, f_test_NN = train_test_split(X_NN, y_NN, K_NN, f_NN, test_size=0.2, random_state=42)

    X_train_NN = X_train_NN.to(DEVICE)
    X_test_NN = X_test_NN.to(DEVICE)

    y_train_NN = TensOps(y_train_NN.requires_grad_(True).to(DEVICE), space_dimension=2, contravariance=0, covariance=0)
    y_test_NN = TensOps(y_test_NN.requires_grad_(True).to(DEVICE), space_dimension=2, contravariance=0, covariance=0)

    K_train_NN = TensOps(K_train_NN.to(DEVICE), space_dimension=2, contravariance=0, covariance=0)
    K_test_NN = TensOps(K_test_NN.to(DEVICE), space_dimension=2, contravariance=0, covariance=0)

    f_train_NN = TensOps(f_train_NN.to(DEVICE), space_dimension=2, contravariance=0, covariance=0)
    f_test_NN = TensOps(f_test_NN.to(DEVICE), space_dimension=2, contravariance=0, covariance=0)


    # Autoencoder
    autoencoder_input_shape = y_train_AE.values[0].shape
    latent_space_dim = [20, 10, n_modes_i, 10, 20]
    autoencoder_output_shape = y_train_AE.values[0].shape

    X_train = y_train_AE.values
    y_train = y_train_AE

    X_test = y_test_AE.values
    y_test = y_test_AE

    autoencoder = Autoencoder(autoencoder_input_shape, latent_space_dim, autoencoder_output_shape).to(DEVICE)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=3e-3)

    start_epoch = 0
    n_epochs = 25000
    batch_size = 64
    n_checkpoint = 3
    new_lr = None

    train_autoencoder_loop(autoencoder, optimizer, X_train, y_train, X_test, y_test,  
                        n_checkpoint, start_epoch, n_epochs, batch_size, MODEL_RESULTS_AE_PATH, DEVICE, new_lr)
    
    start_epoch = n_epochs-1
    n_epochs = 30000
    batch_size = 64
    n_checkpoint = 3
    new_lr = 3e-4

    train_autoencoder_loop(autoencoder, optimizer, X_train, y_train, X_test, y_test,  
                        n_checkpoint, start_epoch, n_epochs, batch_size, MODEL_RESULTS_AE_PATH, DEVICE, new_lr)
    


    # PGNNIV

    # Predictive network architecture
    input_shape = X_train_NN[0].shape
    predictive_layers = [20, 10, n_modes_i]
    predictive_output = y_train_NN.values[0].shape

    # Explanatory network architecture
    explanatory_input = Mx(My(y_train_NN)).values[0].shape
    explanatory_layers = [10]
    explanatory_output = Mx(My(f_train_NN)).values[0].shape

    # Other parameters
    n_filters_explanatory = 5

    pretrained_decoder = autoencoder.decoder

    for param in pretrained_decoder.parameters():
        param.requires_grad = False

    # for name, param in pretrained_decoder.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    model = PGNNIVAutoencoder(input_shape, predictive_layers, pretrained_decoder, predictive_output, explanatory_input,
                                    explanatory_layers, explanatory_output, n_filters_explanatory).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Parametros de entrenamiento (entrenamiento 1)
    start_epoch = 0
    n_epochs = 100000

    batch_size = 64
    n_checkpoints = 3

    train_loop(model, optimizer, X_train_NN, y_train_NN, f_train_NN, X_test_NN, y_test_NN, f_test_NN,
            D, n_checkpoints, start_epoch=start_epoch, n_epochs=n_epochs, batch_size=batch_size, 
            model_results_path=MODEL_RESULTS_PGNNIV_PATH, device=DEVICE)

    # Parametros de entrenamiento (entrenamiento 2)
    start_epoch = n_epochs-1
    n_epochs = 150000

    batch_size = 64 
    n_checkpoints = 3

    second_lr = 3e-4

    train_loop(model, optimizer, X_train_NN, y_train_NN, f_train_NN, X_test_NN, y_test_NN, f_test_NN,
            D, n_checkpoints, start_epoch=start_epoch, n_epochs=n_epochs, batch_size=batch_size, 
            model_results_path=MODEL_RESULTS_PGNNIV_PATH, device=DEVICE, new_lr=second_lr)