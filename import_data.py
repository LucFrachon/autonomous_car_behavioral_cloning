import os
import csv
import cv2
import random
import numpy as np
import matplotlib.image as img
import sklearn
from sklearn.utils import shuffle


def random_adjust_brightness(image, factor = .3):
    '''
    Randomly lower or increase brightness of a single image by 1 +/- 'factor'.

    Returns the modified image.
    '''
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Random adjustment requires converting array to float:
    image_hsv = image_hsv.astype(np.float32)

    random_factor = random.uniform((-1) * factor, factor)

    # Create array of maximum possible pixel values:
    max_values = np.empty_like(image_hsv[:, :, 2])
    max_values.fill(255.)

    # Cap pixel values to 255 to avoid artefacts:
    image_hsv[:, :, 2] = np.minimum(image_hsv[:, :, 2] * (1. + random_factor), 
        max_values)

    # Convert back to unsigned integer before passing to cv2 function:
    image_hsv = np.round(image_hsv).astype(np.uint8)
    image_out = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    return image_out


def random_translate(image, angle, x_pixels, y_pixels, angle_corr):
    '''
    Apply random translations on the x and y axes, up to x_pixels and y_pixels.
    When translating along x, also corrects steering angle by a constant factor 
    (an reasonable approximation of the actual correction function).

    Returns the single modified image and the corrected angle.
    '''
    # Random number of pixels to translate horizontally by:
    x_t = int(np.round(np.random.uniform(-x_pixels, x_pixels)))
    # Correction on steering angle, proportional to x_t:
    angle_out = angle + x_t * 5 * angle_corr / image.shape[1]
    # Random number of pixels to translate vertically by:
    y_t = np.random.uniform(-y_pixels, y_pixels)

    # Define translation matrix:
    M = np.array([[1, 0, x_t], [0, 1, y_t]], dtype = np.float32)
    # Apply translation:
    image_out = cv2.warpAffine(image, M, 
        dsize = (image.shape[1], image.shape[0]))

    return image_out, angle_out


CROP = (55, 25)
SIZE = (80, 40)

def crop_resize(image, crop_pixels, new_size):
    '''
    Crops crop_pixels[0] from top of the image, crop_pixels[1] from bottom,
    then resizes to shape new_size.

    Returns the cropped and resized single image.
    '''
    height = image.shape[0]
    image_out = image[crop_pixels[0] : height - crop_pixels[1], :]
    image_out = cv2.resize(image_out, new_size)
    return image_out


def append_images_and_angles_train(batch_sample, images, angles, 
            log_path, angle_corr = 0.23, camera = 0, channels = 3):
    '''
    Updates the passed images list by appending:
        - Original images, preprocessed
        - And reversed images, pre-processed
    And the passed angles by appending:
        - Angles for the current camera
        - And pposite angles for the current camera (correspond to flipped
        images)
    Used during the training phase therefore pre-processing includes data
    augmentation (random translations and brightness adjustments).

    batch_sample: A batch of rows for the 'driving_log.csv' picked from the
        training set, containing paths to the center, left and right camera 
        images as well as the corresponding steering angles.
    images: A list of images (ie. 3-dimensional np.arrays)
    angles: A list of steering angles (scalars)
    log_path: path to the 'driving_log.csv' file
    angle_corr: fixed additive constant, added to the steering angle when
        using a left-camera image, or substracted when using a right-camera
        images. This simulates an off-center position on the road.
    camera: 0 for center, 1 for left, 2 for right
    channels: number of channels in the image (tested with 0 and 3)

    Returns: An expanded images list, an expanded angles list.
	'''

    # Correction on steering angle depending on camera used:
    if camera == 0:
        corr_sign = 0.
    elif camera == 1:
       corr_sign = 1.
    elif  camera == 2:
        corr_sign = -1.
    else:
        raise ValueError("Wrong camera value - can only be 0, 1 or 2")

    # Get image file:
    name = log_path + 'IMG/' + batch_sample[camera].split('/')[-1]

    if channels > 1:  # Load as greyscale if channel == 1, BGR otherwise
        image = cv2.imread(name, 1)
    else:
        image = cv2.imread(name, 0)
    
    # Get corresponding steering angle and adjust for camera position:
    angle = float(batch_sample[3])
    angle_trans = angle + corr_sign * angle_corr
    # Note that the exact correction would not be a constant but we can show 
    # that it is a reasonable (and cheap) approximation.
    
    # Apply data augmentation and crop/resize:
    image_trans, angle_trans = random_translate(image, angle_trans, 30., 30., 
        angle_corr)
    image_trans = crop_resize(image_trans, CROP, SIZE)
    image_trans = random_adjust_brightness(image_trans, .3)

    # Histogram equalization (also required in the drive.py file):
    if channels == 1:
        image_trans = cv2.equalizeHist(image_trans)

    else: # if BGR, convert to YUV, equalize histogram and convert back to BGR
        image_yuv = cv2.cvtColor(image_trans, cv2.COLOR_BGR2YCrCb)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        image_trans = cv2.cvtColor(image_yuv, cv2.COLOR_YCrCb2BGR)

    images.append(image_trans)  # Original image and angle
    angles.append(angle_trans)
    images.append(cv2.flip(image_trans, flipCode = 1))  # Flipped image and angle
    angles.append(angle_trans * (-1))
    
    return images, angles


def append_images_and_angles_valid(batch_sample, images, angles, 
            log_path, channels = 3):
    '''
    Updates the passed images list by appending original images, preprocessed,
    and the passed angles by appending steering angles.
    Used during the validation phase therefore pre-processing DOES NOT include 
    data augmentation and only the center camera is used.

    batch_sample: A batch of rows for the 'driving_log.csv' picked from the 
        validation set, containing paths to the center camera images as well 
        as the corresponding steering angles.
    images: A list of images (ie. 3-dimensional np.arrays)
    angles: A list of steering angles (scalars)
    log_path: path to the 'driving_log.csv' file
    channels: number of channels in the image (tested with 0 and 3)

    Returns: An expanded images list, an expanded angles list.
    '''
   
    name = log_path + 'IMG/' + batch_sample[0].split('/')[-1]

    if channels > 1:  # Load as greyscale if channel == 1, BGR otherwise
        image = cv2.imread(name, 1)
    else:
        image = cv2.imread(name, 0)

    # Get corresponding steering angle 
    angle = float(batch_sample[3])

    # Crop / resize image to fit model
    image_cropped = crop_resize(image, CROP, SIZE)

    # Histogram equalization (also required in the drive.py file)
    if channels == 1:
        image_cropped = cv2.equalizeHist(image_cropped)
    else: # if BGR, convert to YUV, equalize histogram and convert back to BGR
        image_yuv = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2YCrCb)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        image_cropped = cv2.cvtColor(image_yuv, cv2.COLOR_YCrCb2BGR)

    images.append(image_cropped)
    angles.append(angle)

    return images, angles


def train_generator(samples, log_path, batch_size = 32, angle_corr = 0.25, 
    channels = 3):
    '''
    Generate batches of images and steering angles, applying data augmentation
    and pre-processing routines. Used for training.

    samples: A set of rows from the 'driving_log.csv' file used as the training
    set.
    log_path: Path to the 'driving_log.csv' file. Passed to 
        'append_images_and_angles_train()'.
    angle_corr: Constant additive correction to steering angle when using 
        left/right camera images (simulates off-center position on road).
    channels: Image depth. Tested with values 1 and 3.

    Yields: A shuffled batch of images with preprocessing and random 
        augmentation applied; a corresponding batch of steering angles 
        corrected accordingly.
    '''

    num_samples = len(samples)

    while True: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []

            for batch_sample in batch_samples:
                camera = np.random.randint(0, 3) # Pick a random camera (0, 1 or 2)
                images, angles = append_images_and_angles_train(batch_sample,
                    images, angles, log_path, angle_corr, camera = camera,
                    channels = channels)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)
 
def valid_generator(samples, log_path, batch_size = 32, channels = 3):
    '''
    Generate batches of images and steering angles, applying data augmentation
    and pre-processing routines. Used for validation.

    samples: A set of rows from the 'driving_log.csv' file used as the training
    set.
    log_path: Path to the 'driving_log.csv' file. Passed to 
        'append_images_and_angles_train()'.
    channels: Image depth. Tested with values 1 and 3.

    Yields: A shuffled batch of images with preprocessing applied; a 
        corresponding batch of steering angles.
    '''

    num_samples = len(samples)

    while True: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []

            for batch_sample in batch_samples:
                images, angles = append_images_and_angles_valid(batch_sample,
                    images, angles, log_path, channels = channels)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)