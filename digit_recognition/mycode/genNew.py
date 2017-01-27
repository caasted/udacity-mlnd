import numpy as np
import os
from scipy import ndimage
from scipy.misc import imresize

def generateData(folder):
    image_size = 54
    pixel_channels = 1
    max_sequence_length = 5
    num_labels = 11

    image_files = os.listdir(folder)

    dataset = np.ndarray(shape=(len(image_files), pixel_channels, image_size, image_size), dtype=np.int32)
    labels = np.ndarray(shape=(len(image_files), num_labels, max_sequence_length), dtype=np.int32)
    sequences = np.ndarray(shape=(len(image_files), max_sequence_length), dtype=np.int32)
    
    dataset = dataset[:, :, :, :]

    num_images = 0
    skipped_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        image_data = ndimage.imread(image_file, flatten=True)

        if min(image_data.shape) > 0:
            resized_image = imresize(image_data, (image_size, image_size), interp='bilinear')
            dataset[num_images, 0, :, :] = resized_image
            num_images += 1
        else:
            print("Image not loaded!")
    
    labels[:, :, :] = 0
    
    # 10
    labels[0, 1, 0] = 1
    labels[0, 10, 1] = 1
    labels[0, 0, 2:] = 1
    # 14
    labels[1, 1, 0] = 1
    labels[1, 4, 1] = 1
    labels[1, 0, 2:] = 1
    # 17
    labels[2, 1, 0] = 1
    labels[2, 7, 1] = 1
    labels[2, 0, 2:] = 1
    # 2000
    labels[3, 2, 0] = 1
    labels[3, 10, 1] = 1
    labels[3, 10, 2] = 1
    labels[3, 10, 3] = 1
    labels[3, 0, 4] = 1
    # 24
    labels[4, 2, 0] = 1
    labels[4, 4, 1] = 1
    labels[4, 0, 2:] = 1
    # 25
    labels[5, 2, 0] = 1
    labels[5, 5, 1] = 1
    labels[5, 0, 2:] = 1
    # 3
    labels[6, 3, 0] = 1
    labels[6, 0, 1:] = 1
    # 34
    labels[7, 3, 0] = 1
    labels[7, 4, 1] = 1
    labels[7, 0, 2:] = 1
    
    sequences[:, :] = 0

    #10
    sequences[0, 1] = 1
    #117
    sequences[1, 2] = 1
    #14
    sequences[2, 1] = 1
    #2000
    sequences[3, 3] = 1
    #24
    sequences[4, 1] = 1
    #25
    sequences[5, 1] = 1
    #3
    sequences[6, 0] = 1
    #34
    sequences[7, 1] = 1
    
    print "Images loaded:", num_images

    return dataset, labels, sequences


