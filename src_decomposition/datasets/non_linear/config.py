import os

BASE_DIR = os.path.abspath(__file__)
DATA_BASE_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'data'))

# Configure the absolute path where the data will be saved
MODEL_NAME = 'non_linear'
DATA_PATH = os.path.join(DATA_BASE_PATH, MODEL_NAME)

# Save configuration in a dictionary
DATASET_CONFIG = {
    'MODEL_NAME': MODEL_NAME,
    'N_DATA': 1000,             
    'N_DISCRETIZATION': 50,
    'x0': 0, 
    'xN': 1,  
    'y0': 0,  
    'yN': 1,    
    'PATH': DATA_PATH,         
}