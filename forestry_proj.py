import os
import matplotlib.pyplot as plt
import tensorflow as tf
#from keras.utils import load_img
from PIL import Image, ImageOps
import skimage as ski
import numpy as np
#from keras import preprocessing

input_dir = "treecover_segmentation_satellite_bengaluru/tiles/"
target_dir = "treecover_segmentation_satellite_bengaluru/masks/"
img_size = (333, 333)
num_classes = 2
batch_size = 32

test_img = "treecover_segmentation_satellite_bengaluru/tiles/tile_1.tif"
test_msk = "treecover_segmentation_satellite_bengaluru/masks/mask_1.tif"

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".tif")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".tif") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

# Loop through input and target paths
for input_path, target_path in zip(input_img_paths, target_img_paths):
    print(input_path, "|", target_path)

# Display input image and mask
fig, axes = plt.subplots(1, 2, figsize=(10, 10))

# Load and display the input image
input_img = ski.io.imread(test_img)
axes[0].imshow(input_img[3])
axes[0].set_title("Input Image")
axes[0].axis('off')

# Load and display the mask
target_img = ski.io.imread(test_msk)
axes[1].imshow(target_img)
axes[1].set_title("Mask")
axes[1].axis('off')

plt.show()

# keras pipeline
# preprocessing
img_arrays = np.zeros((len(input_img_paths), 7, img_size[0], img_size[1]), dtype=np.uint8)
mask_arrays = np.zeros((len(target_img_paths), 7, img_size[0], img_size[1]), dtype=np.uint8)

# Loop through input image paths and populate img_arrays
for idx, image_path in enumerate(input_img_paths):
    # load the image and convert it to a numpy array
    img = ski.io.imread(image_path)
    img_array = np.array(img, dtype=np.uint8)
    
    # assign the image array to the correct index in img_arrays
    img_arrays[idx] = img_array

print(img_arrays.dtype)
print(img_arrays.shape)

#savi_band 
def savi_band(nir, red):
    # element-wise operations
    savi = ((nir - red) / (nir + red + 0.5)) * (1 + 0.5)
    return savi

def plants_band(red_edge, red):
    # element-wise operations
    plants_ = ((red_edge - red) / (red_edge + red))
    return plants_

red_band = img_arrays[:, 0, :, :]
blue_band = img_arrays[:, 1, :, :]
red_edge_band = img_arrays[:, 2, :, :]
nir1_band = img_arrays[:, 4, :, :]
nir2_band = img_arrays[:, 6, :, :]

savi_ = savi_band(nir1_band, red_band)
#print(savi_)
print(savi_.shape)
plants_ = plants_band(rededge_band, red_band)
savi_ = np.expand_dims(savi_, axis=1)
plants_ = np.expand_dims(plants_, axis=1)
print(savi_.shape)
print(plants_.shape)

# add new bandS to the data matrix
img_arrays = np.append(img_arrays, savi_, axis=1)
img_arrays = np.append(img_arrays, plants_, axis=1)
print(img_arrays.shape)

np.save('new_data.npy', img_arrays)

# normalization
img_arrays = img_arrays / 255.0
