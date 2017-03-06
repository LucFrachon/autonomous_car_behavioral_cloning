import os
import csv
import cv2
import random
import numpy as np
import matplotlib.image as img
import sklearn
from sklearn.utils import shuffle


def import_data(ex_from = None, ex_to = None, log_dir = '/home/lucfrachon/udacity_sim/data/',
        log_filename = 'driving_log.csv', images_dir = '/home/lucfrachon/udacity_sim/data/IMG/'):
    '''
    Looks for the log file output from the training mode of the simulator and creates numpy 
    arrays containing each image and its corresponding steering input, straight and reversed.

    ex_from: If sampling from the images, starting point of the sample. None <--> 0
    ex_to:  If sampling from the images, end point of the sample. None <--> last image
    log_dir: 

    '''
    lines = []
    with open(log_dir + log_filename) as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []

    if ex_from == None:
        ex_from = 0
    if ex_to == None:
        ex_to = len(lines)

    for line in lines[ex_from:ex_to]:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = images_dir + filename
        image = img.imread(current_path)
        images.append(image)
        images.append(cv2.flip(image, flipCode = 1))
        measurement = float(line[3])
        measurements.append(measurement)
        measurements.append(-measurement)

    x_out = np.array(images)
    y_out = np.array(measurements)

    return x_out, y_out


def random_adjust_brightness(image, min_factor = .4):
    '''
    Randomly lower brightness of a single image
    '''
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image_hsv[:, :, 2] = random.uniform(min_factor, 1.) * image_hsv[:, :, 2]
    image_out = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    return image_out


def random_translate(image, angle, x_pixels, y_pixels, angle_corr):
    '''
    Apply random translations on the x and y axes, up to x_pixels and y_pixels.
    When translating along x, also corrects steering angle by a constant factor 
    (an approximation of the actual correction function)
    '''
    x_t = np.random.uniform(-x_pixels, x_pixels) 
    angle_out = x_t * angle_corr / image.shape[1]
    y_t = np.random.uniform(-y_pixels, y_pixels)
    M = np.array([[1, 0, x_t], [0, 1, y_t]], dtype = np.float32)
    image_out = cv2.warpAffine(image, M, dsize = image.shape[:2])

    return image_out, angle_out

CROP = (60, 20)
SIZE = (100, 100)

def crop_resize(image, crop_pixels, new_size):
    '''
    Crops crop_pixels[0] from top of the image, crop_pixels[1] from bottom,
    then resizes to shape new_size.
    Returns the cropped and resized image.
    '''
    height = image.shape[0]
    image_out = image[crop_pixels[0] : height - crop_pixels[1], :]
    image_out = cv2.resize(image_out, new_size)
    return image_out


def append_images_and_angles_train(batch_sample, images, angles, 
            log_path, angle_corr = 0.23, camera = 0, channels = 3):
    '''
    Creates an images list containing:
        - Original images, preprocessed
        - Or Reversed images, pre-processed (either of the two, at random)
    and an angles list containing:
        - Angles for the current camera
        - Opposite angles for the current camera (correspond to flipped
        images)

    camera: 0 for center, 1 for left, 2 for right
	'''
    if camera == 0:
        corr_sign = 0.
    elif camera == 1:
       corr_sign = 1.
    else:
        corr_sign = -1.

    flip_yes_no = random.randint(0, 1)  # Randomly decide to flip image or not

    name = log_path + 'IMG/' + batch_sample[camera].split('/')[-1]

    if channels == 3:
        image = img.imread(name)
    else:
        image = img.imread(name, 0)

    angle = float(batch_sample[3]) + corr_sign * angle_corr

    image_trans, angle_trans = random_translate(image, angle, 30., 0., angle_corr)
    image_trans = crop_resize(image_trans, CROP, SIZE)

    # Histogram equalization (also required in the drive.py file)
    if channels == 1:
        image_trans = cv2.equalizeHist(image_trans)
    else: # if BGR, convert to YUV, equalize histogram and convert back to BGR
        image_yuv = cv2.cvtColor(image_trans, cv2.COLOR_RGB2YCrCb)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        image_trans = cv2.cvtColor(image_yuv, cv2.COLOR_YCrCb2RGB)

    if flip_yes_no == 0:
        images.append(random_adjust_brightness(image_trans, .4))
        angles.append(angle_trans)
    else:
        images.append(cv2.flip(random_adjust_brightness(image_trans, .4), flipCode = 1))
        angles.append(angle_trans * (-1))

    return images, angles


def append_images_and_angles_valid(batch_sample, images, angles, 
            log_path, channels = 3):
    '''
    Creates an images list containing:
        - Original images, preprocessed
        - Or Reversed images, pre-processed (either of the two, at random)
    and an angles list containing:
        - Angles for the current camera
        - Opposite angles for the current camera (correspond to flipped
        images)

    camera: 0 for center, 1 for left, 2 for right
    '''
    
    name = log_path + 'IMG/' + batch_sample[0].split('/')[-1]

    if channels == 3:
        image = img.imread(name)
    else:
        image = img.imread(name, 0)

    angle = float(batch_sample[3])

    image_cropped = crop_resize(image, CROP, SIZE)

    # Histogram equalization (also required in the drive.py file)
    if channels == 1:
        image_cropped = cv2.equalizeHist(image_cropped)
    else: # if BGR, convert to YUV, equalize histogram and convert back to BGR
        image_yuv = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2YCrCb)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        image_cropped = cv2.cvtColor(image_yuv, cv2.COLOR_YCrCb2RGB)

    images.append(image_cropped)
    angles.append(angle)

    return images, angles


def train_generator(samples, log_path, batch_size = 32, angle_corr = 0.25, 
    channels = 3):

    num_samples = len(samples)

    while True: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []

            for batch_sample in batch_samples:
                camera = np.random.randint(0,2) # Pick a random camera
                images, angles = append_images_and_angles_train(batch_sample,
                    images, angles, log_path, angle_corr, camera = camera,
                    channels = channels)

            X_train = np.array(images)
            y_train = np.array(angles)
            #print(X_train.shape)
            yield shuffle(X_train, y_train)
 
def valid_generator(samples, log_path, batch_size = 32, channels = 3):

    num_samples = len(samples)

    while True: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []

            for batch_sample in batch_samples:
                images, angles = append_images_and_angles_valid(batch_sample,
                    images, angles, log_path, channels = channels)

            X_train = np.array(images)
            y_train = np.array(angles)
            #print(X_train.shape)
            yield shuffle(X_train, y_train)