import os
import itertools
from non_linear.data_generator import DataGenerator

if __name__ == "__main__":

    BASE_DIR = os.path.abspath(__file__)
    DATA_BASE_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data')) 

    # Parameters of the data
    N_data = [10, 20, 50, 100, 1000, 5000, 10000] 
    sigma = [0, 1, 2, 5, 10] # The noise added in '%'
    combinations = list(itertools.product(N_data, sigma))


    for i, combination_i in enumerate(combinations):

        N_data_i = combination_i[0]
        sigma_i = combination_i[1]

        # Configure the absolute path where the data will be saved
        MODEL_NAME = 'non_linear_' + str(N_data_i) + "_" + str(sigma_i)
        DATA_PATH = os.path.join(DATA_BASE_PATH, MODEL_NAME)

        # Save configuration in a dictionary
        DATASET_CONFIG = {
            'MODEL_NAME': MODEL_NAME,
            'N_DATA': N_data_i,           
            'noise_sigma': sigma_i,    
            'N_DISCRETIZATION': 10,
            'x0': 0, 
            'xN': 1,  
            'y0': 0,  
            'yN': 1,    
            'PATH': DATA_PATH,         
        }

        # Load dataset config
        generator = DataGenerator(DATASET_CONFIG)
        
        # Generate and save dataset
        dataset = generator.generate_dataset()
        generator.save_data(dataset)