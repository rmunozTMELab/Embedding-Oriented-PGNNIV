import os
import torch
import GPUtil
import itertools
from trainers.train import train_loop
from sklearn.model_selection import train_test_split

# Imports de la libreria propia
from vecopsciml.kernels.derivative import DerivativeKernels
from vecopsciml.utils import TensOps
from vecopsciml.operators.zero_order import Mx, My

# Imports de las funciones creadas para este programa
from model.baseline_model import BaselineNonlinearModel
from utils.folders import create_folder
from utils.load_data import load_data

# Parameters of the data
N_DATA = [10, 20, 50, 100, 1000, 5000] 
SIGMA = [0, 1, 5, 10] # The noise added in '%'
N_MODES = [10, 20, 50, 100]

combinations = list(itertools.product(N_DATA, SIGMA, N_MODES))

for combination_i in combinations:
    
    print("COMBINATION: ", combination_i)

    N_data_i = combination_i[0]
    sigma_i = combination_i[1]
    n_modes_i = combination_i[2]

    data_name = 'non_linear_' + str(N_data_i) + '_' + str(sigma_i)
    n_modes = n_modes_i

    # Creamos los paths para las distintas carpetas
    ROOT_PATH = r'/home/rmunoz/Escritorio/rmunozTMELab/Physically-Guided-Machine-Learning'
    DATA_PATH = os.path.join(ROOT_PATH, r'data/', data_name, data_name) + '.pkl'
    RESULTS_FOLDER_PATH = os.path.join(ROOT_PATH, r'results/', data_name)
    MODEL_RESULTS_PATH = os.path.join(ROOT_PATH, r'results/', data_name, 'baseline_model_') + str(n_modes)

    # Creamos las carpetas que sean necesarias (si ya están creadas se avisará de ello)
    create_folder(RESULTS_FOLDER_PATH)
    create_folder(MODEL_RESULTS_PATH)

    # Load dataset
    dataset = load_data(DATA_PATH)

    # Convolutional filters to derivate
    dx = dataset['x_step_size']
    dy = dataset['y_step_size']
    D = DerivativeKernels(dx, dy, 0).grad_kernels_two_dimensions()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    # Train data splitting in train/test
    X = torch.tensor(dataset['X_train'], dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(dataset['y_train'], dtype=torch.float32).unsqueeze(1)
    K = torch.tensor(dataset['k_train'], dtype=torch.float32).unsqueeze(1)
    f = torch.tensor(dataset['f_train'], dtype=torch.float32).unsqueeze(1)

    X_train, X_test, y_train, y_test, K_train, K_test, f_train, f_test = train_test_split(X, y, K, f, test_size=0.3, random_state=42)

    # Data processing and adequacy with our TensOps library
    X_train = X_train.to(DEVICE)
    X_test = X_test.to(DEVICE)

    y_train = TensOps(y_train.to(DEVICE).requires_grad_(True), space_dimension=2, contravariance=0, covariance=0)
    y_test = TensOps(y_test.to(DEVICE).requires_grad_(True), space_dimension=2, contravariance=0, covariance=0)

    K_train = TensOps(K_train.to(DEVICE).requires_grad_(True), space_dimension=2, contravariance=0, covariance=0)
    K_test = TensOps(K_test.to(DEVICE).requires_grad_(True), space_dimension=2, contravariance=0, covariance=0)

    f_train = TensOps(f_train.to(DEVICE).requires_grad_(True), space_dimension=2, contravariance=0, covariance=0)
    f_test = TensOps(f_test.to(DEVICE).requires_grad_(True), space_dimension=2, contravariance=0, covariance=0)

    # Loading and processing validation data
    X_val = torch.tensor(dataset['X_val'], dtype=torch.float32).unsqueeze(1)
    y_val = TensOps(torch.tensor(dataset['y_val'], dtype=torch.float32, requires_grad=True).unsqueeze(1), space_dimension=2, contravariance=0, covariance=0)
    K_val = TensOps(torch.tensor(dataset['k_val'], dtype=torch.float32, requires_grad=True).unsqueeze(1), space_dimension=2, contravariance=0, covariance=0)
    f_val = TensOps(torch.tensor(dataset['f_val'], dtype=torch.float32, requires_grad=True).unsqueeze(1), space_dimension=2, contravariance=0, covariance=0)

    # Predictive network architecture
    input_shape = X_train[0].shape
    predictive_layers = [20, 10, n_modes, 10, 20]
    predictive_output = y_train.values[0].shape

    # Explanatory network architecture
    explanatory_input = Mx(My(y_train)).values[0].shape
    explanatory_layers = [10]
    explanatory_output = Mx(My(f_train)).values[0].shape

    # Other parameters
    n_filters_explanatory = 5

    # Load model and the optimizer
    model = BaselineNonlinearModel(input_shape, predictive_layers, predictive_output, explanatory_input, explanatory_layers, explanatory_output, n_filters_explanatory).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Parametros de entrenamiento (entrenamiento 1)
    start_epoch = 0
    n_epochs = 110000

    batch_size = 64
    n_checkpoints = 11

    train_loop(model, optimizer, X_train, y_train, f_train, X_test, y_test, f_test,
            D,  n_checkpoints, start_epoch=start_epoch, n_epochs=n_epochs, batch_size=batch_size, 
            model_results_path=MODEL_RESULTS_PATH, device=DEVICE)
    
    # Parametros de entrenamiento (entrenamiento 2)
    start_epoch = 100000
    n_epochs = 150000

    batch_size = 64
    n_checkpoints = 5

    second_lr = 3e-4

    train_loop(model, optimizer, X_train, y_train, f_train, X_test, y_test, f_test,
            D,  n_checkpoints, start_epoch=start_epoch, n_epochs=n_epochs, batch_size=batch_size, 
            model_results_path=MODEL_RESULTS_PATH, device=DEVICE, new_lr=second_lr) 