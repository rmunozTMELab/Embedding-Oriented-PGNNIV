from non_linear_sigmoid.config import DATASET_CONFIG
from non_linear_sigmoid.data_generator import DataGenerator

if __name__ == "__main__":
    
    # Load dataset config
    config = DATASET_CONFIG
    generator = DataGenerator(config)
    
    # Generate and save dataset
    dataset = generator.generate_dataset()
    generator.save_data(dataset)