from import_data import *
import matplotlib 
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.image as im
import cv2
from copy import copy
# from drive import crop_resize_equaliz

matplotlib.get_backend()

log_dir = '/home/lucfrachon/udacity_sim/ud_data/'
images_dir = '/home/lucfrachon/udacity_sim/ud_data/IMG/'
samples = []

with open(log_dir + 'driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        samples.append(line)

# im = random.randint(0, len(samples))
im = 7333
camera = 1


X, y = import_data(im, im + 1, log_dir = log_dir, log_filename = "driving_log.csv", 
    images_dir = images_dir, camera = camera, angle_corr = 0.28)

# images = []
# angles = []



font = cv2.FONT_HERSHEY_SIMPLEX
agl_txt_1 = str(round(y[0], 3))
X0_t = copy(X[0])
cv2.putText(X0_t, agl_txt_1,(80,150), font, 2,(255,255,255), 2)

X0_brightness = random_adjust_brightness(X[0])
X0_brightness_t = copy(X0_brightness)
cv2.putText(X0_brightness_t, agl_txt_1,(80,150), font, 2,(255,255,255), 2)

X0_translate, y_translate = random_translate(X0_brightness, y[0], 30, 30, 0.2)
agl_txt_2 = str(round(y_translate, 3))
X0_translate_t = copy(X0_translate)
cv2.putText(X0_translate_t, agl_txt_2,(80,150), font, 2,(255,255,255), 2)

X0_crop = crop_resize(X0_translate, CROP, SIZE)
X0_crop_t = copy(X0_crop)
cv2.putText(X0_crop_t, agl_txt_2,(20,38), font, 0.5,(255,255,255), 1)

X0_crop_flip = cv2.flip(X0_crop, flipCode = 1)
agl_txt_3 = str(round(-y_translate, 3))
cv2.putText(X0_crop_flip, agl_txt_3, (20, 38), font, 0.5, (255, 255, 255), 1)

# _,_,X_final = append_images_and_angles_train(samples[500],images, angles, log_dir, 
#     camera = 0, channels = 3) 

cv2.imshow("original", cv2.cvtColor(X0_t, cv2.COLOR_RGB2BGR))
cv2.imshow("brightness", cv2.cvtColor(X0_brightness_t, cv2.COLOR_RGB2BGR))
cv2.imshow("translate", cv2.cvtColor(X0_translate_t, cv2.COLOR_RGB2BGR))
cv2.imshow("crop", cv2.cvtColor(X0_crop_t, cv2.COLOR_RGB2BGR))
cv2.imshow("crop, flip", cv2.cvtColor(X0_crop_flip, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)

cv2.imwrite("original_" + str(camera) + ".png", 
    cv2.cvtColor(X0_t, cv2.COLOR_RGB2BGR))
cv2.imwrite("brightness_" + str(camera) + ".png", 
    cv2.cvtColor(X0_brightness_t, cv2.COLOR_RGB2BGR))
cv2.imwrite("translate_" + str(camera) + ".png", 
    cv2.cvtColor(X0_translate_t, cv2.COLOR_RGB2BGR))
cv2.imwrite("crop_" + str(camera) + ".png", 
    cv2.cvtColor(X0_crop_t, cv2.COLOR_RGB2BGR))
cv2.imwrite("crop_flip_" + str(camera) + ".png", 
    cv2.cvtColor(X0_crop_flip, cv2.COLOR_RGB2BGR))

# im = crop_resize_equalize(X[0], CROP, SIZE)
# cv2.imshow("original", cv2.cvtColor(X[0], cv2.COLOR_RGB2BGR))
# cv2.imshow("result", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)

# images, angles, image = append_images_and_angles_train(samples[500], images, angles,
#     log_dir, camera = 2)
# print(len(images))
# print(angles[0])
# cv2.imshow("original", image)
# cv2.imshow("result", images[0])
# cv2.waitKey(0)


# _, _, image, image_trans, angle, angle_trans = append_images_and_angles_train(samples[500],
#      images, angles, log_dir, camera = 2)


# print(angle, angle_trans)


# cv2.imshow("camera", image)
# cv2.imshow("result", image_trans)
# cv2.waitKey(0)

# gen = train_generator(samples, log_dir, batch_size = 32)
# i = 0
# batch_x, batch_y = next(gen)
# print(len(batch_x))
# for im in batch_x:
#     print(np.squeeze(im).shape)
#     cv2.imshow(str(i), np.squeeze(im))
#     cv2.waitKey()