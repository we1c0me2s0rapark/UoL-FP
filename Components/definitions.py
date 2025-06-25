import os

class DataPaths:
    """
    Defines all relevant file and directory paths for the CBIS-DDSM dataset.
    Paths are organized by data type and usage.
    """
    def __init__(self, root):
        self.root = root

        # Metadata
        self.metadata_csv = os.path.join(self.root, 'metadata.csv')

        # Original case descriptions
        self.mass_train_csv = os.path.join(self.root, 'mass_case_description_train_set.csv')
        self.mass_test_csv = os.path.join(self.root, 'mass_case_description_test_set.csv')
        self.calc_train_csv = os.path.join(self.root, 'calc_case_description_train_set.csv')
        self.calc_test_csv = os.path.join(self.root, 'calc_case_description_test_set.csv')

        # Updated case descriptions
        self.new_mass_train_csv = os.path.join(self.root, 'new_mass_case_description_train_set.csv')
        self.new_mass_test_csv = os.path.join(self.root, 'new_mass_case_description_test_set.csv')
        self.new_calc_train_csv = os.path.join(self.root, 'new_calc_case_description_train_set.csv')
        self.new_calc_test_csv = os.path.join(self.root, 'new_calc_case_description_test_set.csv')

        # Pixel info files
        self.mass_train_pixels_csv = os.path.join(self.root, 'mass_train_pixels.csv')
        self.mass_test_pixels_csv = os.path.join(self.root, 'mass_test_pixels.csv')
        self.calc_train_pixels_csv = os.path.join(self.root, 'calc_train_pixels.csv')
        self.calc_test_pixels_csv = os.path.join(self.root, 'calc_test_pixels.csv')

        # Tensor directories
        self.mass_train_tensor_dir = os.path.join(self.root, 'mass_tensor', 'train')
        self.mass_test_tensor_dir = os.path.join(self.root, 'mass_tensor', 'test')
        self.calc_train_tensor_dir = os.path.join(self.root, 'calc_tensor', 'train')
        self.calc_test_tensor_dir = os.path.join(self.root, 'calc_tensor', 'test')

        # Model directories
        self.mass_model_dir = os.path.join(self.root, 'mass_model')
        self.calc_model_dir = os.path.join(self.root, 'calc_model')


# Instantiate with the dataset root directory
DATA_ROOT = os.path.join('/mnt', 'c', 'Users', 'lejam', 'Desktop', 'CBIS-DDSM', 'manifest-1748122768688')

Paths = DataPaths(DATA_ROOT)

# Constants for model / preprocessing
TARGET_SIZE = (128, 128, 3)
KERNEL_SIZE = (3, 3)




if False:
    data_root = os.path.join('/mnt', 'c', 'Users', 'lejam', 'Desktop', 'CBIS-DDSM', 'manifest-1748122768688')
    
    meta_path = os.path.join(data_root, 'metadata.csv')
    
    # Original files
    mass_train_path = os.path.join(data_root, 'mass_case_description_train_set.csv')
    mass_test_path = os.path.join(data_root, 'mass_case_description_test_set.csv')
    calc_train_path = os.path.join(data_root, 'calc_case_description_train_set.csv')
    calc_test_path = os.path.join(data_root, 'calc_case_description_test_set.csv')
    
    # Updated files
    new_mass_train_path = os.path.join(data_root, 'new_mass_case_description_train_set.csv')
    new_mass_test_path = os.path.join(data_root, 'new_mass_case_description_test_set.csv')
    new_calc_train_path = os.path.join(data_root, 'new_calc_case_description_train_set.csv')
    new_calc_test_path = os.path.join(data_root, 'new_calc_case_description_test_set.csv')
    
    # Pixel info files
    mass_train_pixels_path = os.path.join(data_root, 'mass_train_pixels.csv')
    mass_test_pixels_path = os.path.join(data_root, 'mass_test_pixels.csv')
    calc_train_pixels_path = os.path.join(data_root, 'calc_train_pixels.csv')
    calc_test_pixels_path = os.path.join(data_root, 'calc_test_pixels.csv')
    
    # Tensor data
    mass_train_tensor_path = os.path.join(data_root, 'mass_tensor', 'train')
    mass_test_tensor_path = os.path.join(data_root, 'mass_tensor', 'test')
    calc_train_tensor_path = os.path.join(data_root, 'calc_tensor', 'train')
    calc_test_tensor_path = os.path.join(data_root, 'calc_tensor', 'test')
    
    # Model data
    mass_model_path = os.path.join(data_root, 'mass_model')
    calc_model_path = os.path.join(data_root, 'calc_model')
    
    # Define the target size
    target_size = (128, 128, 3)
    kernel_size = (3, 3)
