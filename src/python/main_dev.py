import os
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from vecopsciml.algebra import zero_order as azo
from vecopsciml.operators import zero_order as zo
from vecopsciml.kernels.derivative import DerivativeKernels
from vecopsciml.utils import Tensor

from models.constant_diffusivity import ConstantDiffusivityNeuralNetwork
from utils.folders import create_folder
from utils.load_data import load_data
from utils.checkpoints import load_checkpoint, save_checkpoint
from trainers.train import train_epoch


ROOT_PATH = r'C:\Users\usuario\Desktop\rmunozTMELab\Physically-Guided-Machine-Learning'
DATA_PATH = os.path.join(ROOT_PATH, r'data\linear_homogeneous\linear_homogeneous.pkl')
RESULTS_FOLDER_PATH = os.path.join(ROOT_PATH, r'results\linear_homogeneous')
MODEL_RESULTS_PATH = os.path.join(ROOT_PATH, r'results\linear_homogeneous\lineal_homogeneous_prueba')

create_folder(RESULTS_FOLDER_PATH)
create_folder(MODEL_RESULTS_PATH)

dataset = load_data(DATA_PATH)



############ ------------------------------------
dx = dataset['x_step_size']
dy = dataset['y_step_size']
D = DerivativeKernels(dx, dy, 0).grad_kernels_two_dimensions()

X_train = Tensor(torch.tensor(dataset['X_train'], dtype=torch.float32, requires_grad=True).unsqueeze(1), space_dimension=2, order=0)
y_train = Tensor(torch.tensor(dataset['y_train'], dtype=torch.float32, requires_grad=True).unsqueeze(1), space_dimension=2, order=0)
K_train = Tensor(torch.tensor(dataset['k_train'], dtype=torch.float32, requires_grad=True).unsqueeze(1), space_dimension=2, order=0)

X_val = Tensor(torch.tensor(dataset['X_val'], dtype=torch.float32, requires_grad=True).unsqueeze(1), space_dimension=2, order=0)
y_val = Tensor(torch.tensor(dataset['y_val'], dtype=torch.float32, requires_grad=True).unsqueeze(1), space_dimension=2, order=0)
K_val = Tensor(torch.tensor(dataset['k_val'], dtype=torch.float32, requires_grad=True).unsqueeze(1), space_dimension=2, order=0)

X_np = X_train.values
y_np = y_train.values
K_np = K_train.values


X_train_np, X_test_np, y_train_np, y_test_np, K_train_np, K_test_np = train_test_split(X_np, y_np, K_np, test_size=0.2, random_state=42)

X_train = Tensor(X_train_np, space_dimension=X_train.space_dim, order=X_train.order)
X_test = Tensor(X_test_np, space_dimension=X_train.space_dim, order=X_train.order)

y_train = Tensor(y_train_np, space_dimension=y_train.space_dim, order=y_train.order)
y_test = Tensor(y_test_np, space_dimension=y_train.space_dim, order=y_train.order)

K_train = Tensor(K_train_np, space_dimension=K_train.space_dim, order=K_train.order)
K_test = Tensor(K_test_np, space_dimension=K_train.space_dim, order=K_train.order)

# Cargar modelo con las formas de entrada y salida
input_shape = X_train.values[0].shape  # [1, 10, 8]
output_shape = y_train.values[0].shape  # [1, 10, 10]

# Dimensiones ocultas
hidden1_dim = 100
hidden2_dim = 100

# Crear el modelo
model = ConstantDiffusivityNeuralNetwork(input_shape, hidden1_dim, hidden2_dim, output_shape)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10

loss_list = []
e_list = []
pi1_list = []
pi2_list = []
pi3_list = []

for epoch_i in range(epochs):

    # for X_batch, y_batch in train_loader:

    loss, e_loss, pi1_loss, pi2_loss, pi3_loss = train_epoch(model, optimizer, X_train, y_train, D)

    loss_list.append(loss)
    e_list.append(e_loss)
    pi1_list.append(pi1_loss)
    pi2_list.append(pi2_loss)
    pi3_list.append(pi3_loss)

    if epoch_i % 5 == 0:
        save_checkpoint(model, optimizer, epoch_i, FOLDER, MODEL_NAME, 
                        loss_list=loss_list, e_list=e_list, pi1_list=pi1_list, pi2_list=pi2_list, pi3_list=pi3_list,)


