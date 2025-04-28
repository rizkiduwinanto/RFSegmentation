import os
import skimage as ski
from skimage.transform import resize
import matplotlib.pyplot as plt
import albumentations as A
import random
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import img_as_ubyte
import keras
from PIL import Image, ImageOps

class Dataset():
    def __init__(self, input_dir, target_dir, img_size=(333, 333), batch_size=16):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_img_paths = sorted(
            [
                os.path.join(input_dir, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(".tif")
            ]
        )
        self.target_img_paths = sorted(
            [
                os.path.join(target_dir, fname)
                for fname in os.listdir(target_dir)
                if fname.endswith(".tif") and not fname.startswith(".")
            ]
        )
        self.paths = list(zip(self.input_img_paths, self.target_img_paths))
        self.img_size = img_size
        self.batch_size = batch_size

    def __getitem__(self, index):
        input_img = ski.io.imread(self.input_img_paths[index])
        target_img = ski.io.imread(self.target_img_paths[index])
        return input_img, target_img

    def train_test_val_split(self, val_ratio=0.2):
        random.Random(1337).shuffle(self.paths)

        n = len(self.paths)
        val_size = int(n * val_ratio)

        val = self.paths[:val_size]
        train = self.paths[val_size:]

        return train, val

    def display(self, input_img, target_img, index, bands=1, is_resize=True):
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        if is_resize:
            resize_img = self.resize(input_img[bands-1], (128, 128))
            axes[0].imshow(resize_img)
        else:
            axes[0].imshow(input_img[bands-1])
        
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        if is_resize:
            target_resize_img = self.resize(target_img, (128, 128))
        axes[1].imshow(target_img)
        axes[1].set_title("Mask")
        axes[1].axis('off')
        plt.show()

    def display_img(self, image, original_mask, predicted_mask, index, bands=1, is_resize=True):
        #Intersection over union (IoU)
        iou = self.calculate_iou(original_mask, predicted_mask)
        print(f"IoU for image {index}: {iou*100:.4f}")
        # binary predicted mask array
        #print("Predicted mask array (binary):", predicted_mask)
        
        # plot masks (true and predicted) plus original image
        plt.figure(figsize=(15, 5))
        plt.title(f"Image {index} - IoU: {iou*100:.4f}")
        
        # Original mask
        plt.subplot(1, 3, 1)
        plt.title("Original Mask")
        plt.imshow(original_mask, cmap="gray")
        plt.axis("off")
        
        # Predicted mask
        plt.subplot(1, 3, 2)
        plt.title("Predicted Mask")
        plt.imshow(predicted_mask, cmap="gray")
        plt.axis("off")
        
        # original image
        plt.subplot(1, 3, 3)
        plt.title("Original Image (green band)")
        plt.imshow(image[1])
        plt.axis("off")

        plt.show()

    def __len__(self):
        return len(self.input_img_paths)

    def __iter__(self):
        return self

    def crop_dataset_img(self, image_path):
        img = ski.io.imread(image_path)
        img_array = np.array(img, dtype=np.uint8)

        h, w, _ = img_array.shape

        start_y = (h - 333) // 2
        start_x = (w - 333) // 2

        cropped_input = img_array[start_y:start_y + 333, start_x:start_x + 333]

        return cropped_input

    def crop_dataset_mask(self, mask_path):
        img = ski.io.imread(mask_path)
        mask_array = np.array(img, dtype=np.uint8)

        h, w = mask_array.shape

        start_y = (h - 333) // 2
        start_x = (w - 333) // 2

        cropped_mask = mask_array[start_y:start_y + 333, start_x:start_x + 333]

        return cropped_mask

    def crop_dataset(self, index):
        cropped_input = self.crop_dataset_img(self.input_img_paths[i])
        cropped_mask = self.crop_dataset_mask(self.target_img_paths[i])

    def crop_dataset_img_and_mask(self):
        test_img_arrays = np.zeros((len(self.input_img_paths), 333, 333, 7),  dtype=np.uint8)
        test_msk_arrays = np.zeros((len(self.target_img_paths), 333, 333, 1), dtype=np.uint8)

        # Loop through input image paths and populate img_arrays
        for idx, image_path in enumerate(self.input_img_paths):
            test_img_arrays[idx] = self.crop_dataset_img(image_path)

        for idx, mask_path in enumerate(self.target_img_paths):
            test_msk_arrays[idx] = self.crop_dataset_mask(mask_path)

        return test_img_arrays, test_msk_arrays

    def augment_dataset(self, index):
        input_img = ski.io.imread(self.input_img_paths[index])
        target_img = ski.io.imread(self.target_img_paths[index])

        h, w = target_img.shape

        transpose_input = input_img.transpose(1, 2, 0)

        transform = A.Compose(
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(p=0.5),
                A.Compose([
                    A.RandomCrop(256, 256, p=0.5),
                    A.Resize(333, 333, p=1),
                ], p=1),
            ], p=1),
        )

        horizontal_transform =  A.Compose([
            A.RandomCrop(128, 128, p=0.5),
            A.Resize(333, 333, p=1),
        ])

        augmented = horizontal_transform(image=transpose_input, mask=target_img)
        augmented_input = augmented['image']
        augmented_target = augmented['mask']   

        return augmented_input, augmented_target
        
    @staticmethod
    def augment(image):
        transform = A.Compose(
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.SafeRotate(limit=45, p=1.0),
                A.Compose([
                    A.RandomCrop(256, 256, p=0.5),
                    A.Resize(333, 333, p=1),
                ], p=1),
            ], p=1),
        )
        augmented = transform(image=image)
        return augmented['image']

    def calculate_feature_vector(self, paths):
        feature_vectors = np.array([self.get_feature_vector(i) for i in range(len(paths))])
        return feature_vectors

    def get_feature_vector(self, index, is_resize=True, is_crop=False):
        if is_crop:
            input_img = self.crop_dataset_img(self.input_img_paths[index])
            input_img = np.array(input_img, dtype=np.uint8)
        else:
            input_img = ski.io.imread(self.input_img_paths[index])

        red_band = input_img[0, :, :]
        blue_band = input_img[1, :, :]
        red_edge_band = input_img[2, :, :]
        green_band = input_img[3, :, :]
        near_ir1_band = input_img[4, :, :]
        yellow_band = input_img[5, :, :]
        near_ir2_band = input_img[6, :, :]

        if is_resize:
            red_band = self.resize(red_band, (128, 128))
            blue_band = self.resize(blue_band, (128, 128))
            red_edge_band = self.resize(red_edge_band, (128, 128))
            green_band = self.resize(green_band, (128, 128))
            near_ir1_band = self.resize(near_ir1_band, (128, 128))
            yellow_band = self.resize(yellow_band, (128, 128))
            near_ir2_band = self.resize(near_ir2_band, (128, 128))

        savi_band = ((near_ir1_band - red_band) / (near_ir1_band + red_band + 0.5)) * (1 + 0.5)
        plants_band = ((red_edge_band - red_band) / (red_edge_band + red_band))

        # Normalizing the bands


        feature_vector = np.stack(
            [
                red_band,
                blue_band,
                red_edge_band,
                green_band,
                near_ir1_band,
                yellow_band,
                near_ir2_band,
                savi_band.astype(np.uint16),
                plants_band.astype(np.uint16),
            ]
        )
        return feature_vector

    def get_labels(self, paths):
        labels = np.array([self.get_label(i) for i in range(len(paths))])
        return labels

    def get_label(self, index, is_resize=True, is_crop=False):

        if is_crop:
            target_img = self.crop_dataset_mask(self.target_img_paths[index])
            target_img = np.array(target_img, dtype=np.uint8)
        else:
            target_img = ski.io.imread(self.target_img_paths[index])

        if is_resize:
            target_img = self.resize(target_img, (128, 128))

        return target_img

    def get_feature_vector_and_labels(self, is_resize=True, split=True, is_crop=False):
        if split:
            train, val = self.train_test_val_split()

            X_train = np.array([self.get_feature_vector(i, is_resize) for i in range(len(train))])
            y_train = np.array([self.get_label(i, is_resize) for i in range(len(train))])
            X_val = np.array([self.get_feature_vector(i, is_resize) for i in range(len(val))])
            y_val = np.array([self.get_label(i, is_resize) for i in range(len(val))])

            X_train = X_train.transpose(0, 2, 3, 1).reshape(-1, 9)
            y_train = y_train.transpose(0, 2, 1).reshape(-1, 1)
            X_val = X_val.transpose(0, 2, 3, 1).reshape(-1, 9)
            y_val = y_val.transpose(0, 2, 1).reshape(-1, 1)

            return X_train, y_train, X_val, y_val
        else:
            X = np.array([self.get_feature_vector(i, is_resize, is_crop) for i in range(len(self.input_img_paths))])
            y = np.array([self.get_label(i, is_resize, is_crop) for i in range(len(self.target_img_paths))])

            X = X.transpose(0, 2, 3, 1).reshape(-1, 9)
            y = y.transpose(0, 2, 1).reshape(-1, 1)

            return X, y

    @staticmethod
    def train_val_split_fv_l(X, y, val_ratio=0.1):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=42, shuffle=True, stratify=y
        )
        return X_train, X_val, y_train, y_val

    @staticmethod
    def savi_band(nir):
        savi = ((nir - red) / (nir + red + 0.5)) * (1 + 0.5)
        return savi

    @staticmethod
    def plants_band(red_edge, red):
        plants_ = ((red_edge - red) / (red_edge + red))
        return plants_

    @staticmethod
    def resize_image(image, size):
        resized_image = ski.transform.resize(image, size)
        return img_as_ubyte(resized_image)

    @staticmethod
    def resize(img, size):
        input_image = ski.transform.resize(img, size)
        input_image = img_as_ubyte(input_image)
        return input_image

    @staticmethod
    def calculate_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        # Intersection over Union (IoU)
        iou = intersection / union
        return iou

            


