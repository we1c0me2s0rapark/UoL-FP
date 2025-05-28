import os

data_root = os.path.join(os.path.normpath(os.path.expanduser("~/Desktop")), 'CBIS-DDSM', 'manifest-1748122768688')

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
mass_pixels_path = os.path.join(data_root, 'mass_pixels.csv')
calc_pixels_path = os.path.join(data_root, 'calc_pixels.csv')

# Define the target size
target_size = (224, 224, 3)
kernel_size = (3, 3)