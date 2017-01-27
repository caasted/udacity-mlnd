import os
import numpy as np
from scipy import ndimage
from scipy.misc import imresize

def generateData(folder, dictionary, crop=False):

    image_size = 64
    pixel_channels = 1
    max_sequence_length = 5
    num_labels = 11
    pixel_depth = 255.

    image_files = os.listdir(folder)

    dataset = np.ndarray(shape=(len(image_files), pixel_channels, image_size, image_size), dtype=np.int32)
    labels = np.ndarray(shape=(len(image_files), num_labels, max_sequence_length), dtype=np.int32)
    sequences = np.ndarray(shape=(len(image_files), max_sequence_length), dtype=np.int32)
    
    num_images = 0
    skipped_images = 0
    for image in image_files:
        if image in dictionary:
            label_sequence = dictionary[image]['label']
        else:
            label_sequence = None
            skipped_images += 1
            
        if label_sequence != None and len(label_sequence) > max_sequence_length:
            label_sequence = None
            skipped_images += 1
            print "Image", image, "has too many digits!"
        
        if label_sequence != None and len(label_sequence) > 0:
            image_file = os.path.join(folder, image)
            image_raw = (ndimage.imread(image_file, flatten=True).astype(float) - pixel_depth / 2) / pixel_depth
            
            if crop:
                lefts = dictionary[image]['left']
                tops = dictionary[image]['top']
                widths = dictionary[image]['width']
                heights = dictionary[image]['height']
                rights = []
                bottoms = []
                for left, width in zip(lefts, widths):
                    rights.append(left + width)
                for top, height in zip(tops, heights):
                    bottoms.append(top + height)
                if min(tops) < max(bottoms) and min(lefts) < max(rights):
                    image_data = image_raw[min(tops):max(bottoms), min(lefts):max(rights)]
                else:
                    print("Bounding box error!")
            else:
                image_data = image_raw
            
            #print(image_data.shape)
            if min(image_data.shape) > 0:
                resized_image = imresize(image_data, (image_size, image_size), interp='bilinear')
                dataset[num_images, 0, :, :] = resized_image

                labels[num_images, :, :] = 0
                if len(label_sequence) < max_sequence_length:
                    labels[num_images, 0, len(label_sequence):] = 1 # Blank class labels
                for index in range(len(label_sequence)):
                    labels[num_images, label_sequence[index], index] = 1

                sequences[num_images, :] = 0
                sequences[num_images, len(label_sequence) - 1] = 1

                num_images += 1
            else:
                skipped_images += 1

                print "Skipped zero-size bbox image:", image
        
    print '\nSkipped images:', skipped_images
    print 'Full dataset tensor:', dataset.shape
    print 'Mean:', np.mean(dataset)
    print 'Standard deviation:', np.std(dataset), "\n"

    dataset = dataset[0:num_images, :, :, :]
    labels = labels[0:num_images, :, :]
    sequences = sequences[0:num_images, :]

    return dataset, labels, sequences

