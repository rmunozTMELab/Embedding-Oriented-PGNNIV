from linear_homogeneous.config import DATASET_CONFIG
from linear_homogeneous.data_generator import DataGenerator




if __name__ == "__main__":
    # Cargar configuración
    config = DATASET_CONFIG
    generator = DataGenerator(config)
    
    # Generar y guardar datos
    dataset = generator.generate_dataset()
    # generator.save_data(dataset)




    
