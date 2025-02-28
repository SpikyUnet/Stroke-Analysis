import os
import numpy as np
import pydicom
import cv2
from skimage import morphology
from scipy import ndimage
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, input_folder, output_folder, preprocess=False):
        """
        :param preprocess: If True, process images with CLAHE & morphological operations
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.preprocess = preprocess  # If False, returns original X

    @staticmethod
    def transform_to_hu(medical_image, image):
        """Convert DICOM pixel values to Hounsfield Units (HU)."""
        intercept = medical_image.RescaleIntercept
        slope = medical_image.RescaleSlope
        return image * slope + intercept

    @staticmethod
    def window_image(image, window_center, window_width):
        """Apply windowing to enhance the brain region."""
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        image[image < img_min] = img_min
        image[image > img_max] = img_max
        return image

    @staticmethod
    def to_grayscale(image):
        """Normalize image and convert to grayscale."""
        return np.uint8((np.maximum(image, 0) / image.max()) * 255.0)

    def process_dicom_file(self, filepath):
        """Convert DICOM to grayscale & return original and processed versions."""
        ds = pydicom.dcmread(filepath)
        image = ds.pixel_array
        hu_image = self.transform_to_hu(ds, image)
        brain_image = self.window_image(hu_image, ds.WindowCenter, ds.WindowWidth)
        brain_image = self.to_grayscale(brain_image)

        # Store original X before applying CLAHE & morphological operations
        original_x = brain_image.copy()

        # If preprocessing is required, apply morphological and CLAHE transformations
        if self.preprocess:
            segmentation = morphology.dilation(brain_image, np.ones((1, 1)))
            labels, _ = ndimage.label(segmentation)
            mask = labels == np.bincount(labels.ravel().astype(np.int64)).argmax()
            mask = ndimage.binary_fill_holes(morphology.dilation(mask, np.ones((3, 3))))
            processed_x = mask * brain_image

            clahe = cv2.createCLAHE(clipLimit=5)
            processed_x = clahe.apply(processed_x.astype(np.uint8))
        else:
            processed_x = original_x

        # Normalize both versions of X to [0,1]
        original_x = original_x.astype(np.float32) / 255.0
        processed_x = processed_x.astype(np.float32) / 255.0

        return original_x, processed_x

    def load_dicom_images(self, folder_path):
        """
        Load a set of DICOM images, returning original and processed versions.
        """
        original_images, processed_images = [], []
        dicom_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".dcm")])

        for filename in dicom_files:
            original_x, processed_x = self.process_dicom_file(os.path.join(folder_path, filename))
            original_images.append(original_x)
            processed_images.append(processed_x)

        return np.array(original_images), np.array(processed_images)

    def load_patient_data(self):
        """
        Load input (X) and output (Y) data from DICOM images.
        """
        original_x_data, processed_x_data, y_data = [], [], []
        perfusion_type = "NLR_CBV"

        for patient_folder in os.listdir(self.input_folder):
            print(f"Processing: {patient_folder}")

            original_x, processed_x = self.load_dicom_images(os.path.join(self.input_folder, patient_folder))
            y = self.load_dicom_images(os.path.join(self.output_folder, f"{patient_folder}_Filtered_3mm_20HU_Maps", perfusion_type))[1]

            original_x_data.append(original_x)
            processed_x_data.append(processed_x)
            y_data.append(y)

        # Normalize Y data to [0,1] for consistency
        y_data = np.array(y_data, dtype=np.float32) / 255.0

        return np.array(original_x_data), np.array(processed_x_data), y_data
    

    def run(self):
        """
        Process data and return original or processed X along with Y.
        """
        original_x, processed_x, y_data = self.load_patient_data()

        # Select whether to return original or preprocessed X
        X_data = processed_x if self.preprocess else original_x

        # Save the processed data
        np.save("X_data_original.npy", original_x)
        np.save("X_data_preprocessed.npy", processed_x)
        np.save("Y_data_preprocessed.npy", y_data)

        print("Data saved successfully.")

        return X_data, y_data

