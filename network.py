from import_data import import_data
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from optparse import OptionParser
from zipfile import ZipFile
import gc

parser = OptionParser()
parser.add_option("-l", "--local", action = "store_true", dest = "local", default = False,
	help = "Tells the code to run locally. Defaults to False (then path names correspond to a remote instance")
(options, args) = parser.parse_args()

if options.local == True:
	log_dir = '/home/lucfrachon/udacity_sim/data/'
	im_dir = '/home/lucfrachon/udacity_sim/data/IMG/'
else:
	with ZipFile("./data/data.zip", 'r') as zipped:
		zipped.extractall("./data/")
		log_dir = "./data/"
		im_dir = "./data/IMG/"

# Load data into np.arrays:
X_train, y_train = import_data(None, None, log_dir = log_dir, log_filename = 'driving_log.csv',
	images_dir = im_dir)
print("Features Dimensions = {}".format(X_train.shape))
print("Measurements Dimensions = {}".format(y_train.shape))

# Build model:
model = Sequential()
model.add(Cropping2D(cropping = ((60, 20),(0,0)), input_shape = (160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.) - 0.5))
model.add(Convolution2D(24, 5, 5, init = 'glorot_normal', border_mode = 'valid', 
	subsample = (2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, init = 'glorot_normal', border_mode = 'valid', 
	subsample = (2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, init = 'glorot_normal', border_mode = 'valid', 
	subsample = (2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, init = 'glorot_normal', border_mode = 'valid', 
	subsample = (1, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, init = 'glorot_normal', border_mode = 'valid', 
	subsample = (1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Configure learning process and train model:
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = .2, shuffle = True, nb_epoch = 5,
	verbose = 2)

model.save('model.h5')
gc.collect()