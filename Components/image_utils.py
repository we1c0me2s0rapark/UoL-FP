#
# @brief Import all necessary libraries for the project.
#
# This set of imports is used for data handling, medical image processing,
# and deep learning model development.
#
import os
import re
import glob
import numpy as np
import cv2
import tensorflow as tf
import pydicom as dicom
import matplotlib.pyplot as plt

class ImageProcessor:
    """
    A class to preprocess DICOM images and their corresponding masks
    for medical image segmentation tasks.
    """

    def __init__(self, image_size):
        """
        Initialise the ImageProcessor.

        Args:
            image_size (tuple): Target image size (height, width).
        """
        self.image_size = image_size

    def load_image(self, image_path):
        """
        Load and preprocess a DICOM image.

        Steps:
            - Read and normalize DICOM pixel data.
            - Convert grayscale to RGB.
            - Resize to the target image size.
            - Normalize to [0, 1].
            - Optionally apply Laplacian enhancement.

        Args:
            image_path (str): File path to a DICOM image.

        Returns:
            tf.Tensor: A float32 RGB image tensor of shape (H, W, 3).
        """
        try:
            # Read the DICOM file and get the pixel data.
            ds = dicom.dcmread(image_path)
            image = ds.pixel_array.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to load DICOM file: {image_path}") from e

        # Normalize the pixel data and convert it to a uint8 format.
        image = (np.maximum(image, 0) / (image.max() + 1e-7)) * 255.0
        image = np.uint8(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize the image to the predefined target size.
        image_resized = tf.image.resize(
            image_rgb,
            self.image_size,
            method=tf.image.ResizeMethod.BILINEAR
        )

        # Normalize the image to a [0, 1] range.
        image_normalized = tf.cast(image_resized, tf.float32) / 255.0
        return self._apply_laplacian_filter(image_normalized)

    def load_mask(self, image_path):
        """
        Load and preprocess a DICOM mask image.

        Steps:
            - Read pixel data.
            - Binarise the mask.
            - Resize using nearest neighbor.
            - Add channel dimension.

        Args:
            image_path (str): File path to a DICOM mask.

        Returns:
            tf.Tensor: A float32 mask tensor of shape (H, W, 1).
        """
        try:
            # Read the DICOM file and get the pixel data.
            ds = dicom.dcmread(image_path)
            mask = ds.pixel_array
        except Exception as e:
            raise ValueError(f"Failed to load DICOM mask: {image_path}") from e

        # Binarise the mask by converting positive values to 1.0.
        mask = (mask > 0).astype(np.float32)

        # Add a channel dimension if it's a 2D image.
        if mask.ndim == 2:
            mask = mask[..., np.newaxis]

        # Resize the mask using nearest neighbor interpolation to preserve boundaries.
        mask_resized = tf.image.resize(
            mask,
            self.image_size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        return tf.cast(mask_resized, tf.float32)

    def _apply_laplacian_filter(self, img, apply_gaussian=True):
        """
        Apply optional Gaussian blur and Laplacian filtering
        to enhance edges in an image.

        Args:
            img (tf.Tensor): Float image tensor (H, W, C) in [0, 1].
            apply_gaussian (bool): Whether to apply Gaussian blur before Laplacian.

        Returns:
            tf.Tensor: Edge-enhanced float image tensor in [0, 1].
        """
        if not tf.executing_eagerly():
            raise RuntimeError("apply_laplacian requires eager execution for NumPy operations.")

        laplacian_factor = -0.3
        gaussian_factor = -0.15

        img_np = img.numpy()

        if apply_gaussian:
            gaussian_kernel = np.ones((5, 5), np.float32) / 25.0
            blurred = np.stack([
                cv2.filter2D(img_np[..., c], -1, gaussian_kernel)
                for c in range(img_np.shape[-1])
            ], axis=-1)
            img_np += gaussian_factor * blurred

        laplacian_kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)

        laplacian = np.stack([
            cv2.filter2D(img_np[..., c], -1, laplacian_kernel)
            for c in range(img_np.shape[-1])
        ], axis=-1)

        sharpened = img_np + laplacian_factor * laplacian
        sharpened = np.clip(sharpened, 0, 1).astype(np.float32)

        return tf.convert_to_tensor(sharpened, dtype=tf.float32)

class ImagePathUpdater:
    """
    A class to update and correct file paths in a DataFrame by
    recursively searching for image files and applying filters.
    """
    def __init__(self, meta_data, target_data, column, root_path, number=None):
        """
        Initialise the path updater with data and configuration.

        Args:
            meta_data (pd.DataFrame): The DataFrame containing metadata, including original file locations.
            target_data (pd.DataFrame): The DataFrame to be updated with correct file paths.
            column (str): The name of the column in `target_data` to be updated.
            root_path (str): The base directory for all image files.
            number (int, optional): The number of rows to process. If None, all rows are processed.
        """
        self.meta_data = meta_data
        self.target_data = target_data
        self.column = column
        self.root_path = root_path
        self.number = number

    def worker(self, index, current_path, subject_id):
        """
        Recursively searches for and validates DICOM image files.

        Args:
            index (int): The DataFrame index of the row being updated.
            current_path (str): The current directory path to search.
            subject_id (str): The subject ID to identify the correct path.
        """
        try:
            # Use glob to find all files and folders in the current path.
            for p in glob.glob(current_path):
                # If the item is a directory, recurse into it.
                if os.path.isdir(p):
                    new_path = os.path.join(p, '*')
                    # Fix: Pass all necessary arguments in the recursive call.
                    self.worker(index, new_path, subject_id)
                else:
                    # If the item is a file, parse it as a DICOM file.
                    ds = dicom.dcmread(p)
        
                    # Categorise the file as a cropped image or an ROI mask based on pixel values.
                    # ROI masks contain only two pixel values: 0 (background) and 255 (region of interest).
                    diff = np.setdiff1d(np.unique(ds.pixel_array), np.array([0, 255]))
                    
                    # Skip the file if it does not match the expected type based on the column name.
                    if self.column == 'cropped image file path' and diff.size == 0:
                        continue
                    if self.column == 'ROI mask file path' and diff.size > 0:
                        continue
        
                    # If the file is valid, update the DataFrame entry with its new relative path.
                    image_path_without_absolute_root = p[p.find(subject_id):]
                    self.target_data.at[index, self.column] = image_path_without_absolute_root
        except Exception as ex:
            print(ex)

    def update(self):
        """
        Orchestrates the path update process by iterating through the data
        and calling the worker function for each entry.
        """
        try:
            data_num = self.number
            if self.number is None:
                data_num = self.target_data.shape[0]
            
            # Loop through a specified number of images or all images in the DataFrame.
            for index, row in self.target_data.head(data_num).iterrows():
                # Validate the file path.
                splitted_path = row[self.column].split('/')
                if len(splitted_path) == 0: continue
        
                # Skip if no metadata is found for the subject ID.
                subject_id = splitted_path[0]
                meta_data = self.meta_data[self.meta_data['Subject ID']==subject_id]
                if meta_data.shape[0] == 0: continue
        
                # Iterate through available file locations for this subject.
                for d in meta_data['File Location']:
                    # Split the path on either forward slash or backslash for cross-platform compatibility.
                    parts = re.split(r"[\\/]+", d)
                    
                    # Join the parts with the root path and append the DICOM file extension.
                    path = os.path.join(self.root_path, *parts, "*")
        
                    # Start the recursive search to collect all image files.
                    self.worker(index, path, subject_id)
        except Exception as ex:
            print(ex)
        
class Augment(tf.keras.layers.Layer):
    """
    A class to apply data augmentation to images and masks.
    """
    def __init__(self, seed=42):
        super().__init__()
        # Both layers use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        
    def call(self, inputs, labels):
        # Apply horizontal flip augmentation to both image and mask.
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels

#
# @brief Renders raw DICOM images without filters.
#
# This function displays a specified number of DICOM images from the
# training dataset. It retrieves the image data from a given path,
# plots it on a subplot, and labels it with its pathology.
#
# @param root_path (str): The base directory for the image files.
# @param main_df (pd.DataFrame): The DataFrame containing metadata for image locations.
# @param target_df (pd.DataFrame): The DataFrame with image paths and labels to be displayed.
# @param column (str): The column name in `target_df` containing the file path.
# @param number (int): The number of images to display.
# @param update (bool): A flag to enable intelligent filtering for different image types.
#
def render_images(root_path, main_df, target_df, column, number, update=False):
    """
    Renders raw DICOM images from a training DataFrame.
    """
    # Set up figure
    fig, axes = plt.subplots(1, number, figsize=(10, 5))
    
    # Flatten axes if there is only one subplot, to ensure axes[index] can be used.
    if number == 1:
        axes = [axes]

    # Loop through images for display.
    for index, row in target_df.head(number).iterrows():
        # Validate the file path
        splitted_path = row[column].split('/')
        if len(splitted_path) == 0: continue

        # Check if the DICOM data exists
        subject_id = splitted_path[0]
        meta_data = main_df[main_df['Subject ID']==subject_id]
        if meta_data.shape[0] == 0: continue

        # Display DICOM (.dcm) files
        raw_path = meta_data.iloc[0]['File Location']

        # Split the path on either forward slash or backslash for cross-platform compatibility.
        parts = re.split(r"[\\/]+", raw_path)
        
        # Join the parts with the root path and append the DICOM file extension.
        path = os.path.join(root_path, *parts, "*.dcm")
        
        # This loop is fixed to only display one image.
        for image_path in glob.glob(path):
            # Parse each DICOM file
            ds = dicom.dcmread(image_path)

            # Categorise the file as either a cropped image or an ROI mask based on pixel values.
            # ROI masks contain only two pixel values: 0 (background) and 255 (region of interest).
            diff = np.setdiff1d(np.unique(ds.pixel_array), np.array([0, 255]))
            
            # Use the 'update' flag to filter out images that do not match the expected type.
            if update and column == 'cropped image file path' and diff.size == 0: continue
            if update and column == 'ROI mask file path' and diff.size > 0: continue
            
            # Render the image on the current axis
            ax = axes[index]
            ax.imshow(ds.pixel_array, cmap='gray')
            ax.set_title(f"{row['pathology']}")
            ax.axis('off')
            break # Exit the inner loop after one image is found and rendered

    # Display the subplot.
    plt.tight_layout()
    plt.show()