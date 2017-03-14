#**Behavioral Cloning** 

##Project Writeup

_Author: Luc Frachon - March 13, 2017_


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around Track One without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./im/angle_correction.png "Camera angle correction"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

With this submission, I included the following files:

* `network_large.py`: My final ConvNet architecture, derived from the nVidia architecture [(Ref.1)](#ref1).
* `import_data.py`: The code used to import data, augment it, pre-process it and build generators to feed batches into the model
* `drive.py`: The code used to drive the car. It is based on the provided code but I added the possibility to pass different a target speed as well as multipliers to the steering and throttle inputs. I also included data pre-processing into it in line with what is applied to validation sets during the training phase.
* `model_final.h5`: The trained model, passed to `drive.py` to drive the car in autonomous mode.
* `video.mp4`: A video recording of the model completing XX laps of track 1.
* `self_driving_car_mountain.mp4`: A video recording of the model completing track 2 of the 50MHz simulator (mountain road), with added music. Note that the steering and throttle multipliers used during recording were 1.5 and 0.3, but the model works with 1.4 and throttle multiplier = 1. I added some music to make it nicer (The Smiths' *Oscillate Wildly* -- a fitting name considering the behaviour the car occasionally exhibits!).

####2. Submission includes functional code

* **To run the model on track one, follow the steps below:**

	1. Start the 50MHz simulator (the 10MHz sim does not work as well and more importantly, does not contain the mountain track which I also tested the model on) and select the faster graphics quality (this is what I used to train the model). Other options will works too but with a slightly lower accuracy.
	2. Open another terminal window and activate the `carnd-term1` environment
	3. Type `python drive.py model_final.h5`. This runs the model with a target speed of 25, a steering multiplier of 1. and a throttle multiplier of 1., which work well on this track. You can also increase the target speed to 30 if you like, adding the optional flag `-s 30`. Note that the driving parameters used for the mountain track also work (see below) but generate a slightly wobblier drive, with the car meandering from left to right on the straights.

* **To run the model on track two (mountains), follow the steps below:**

	1. Start the 50MHz simulator. Screen resolution does not seem to matter much, however you will need to select the fastest graphics quality in order to prevent the model from getting confused with shadows (see discussion later).
	2. Open another terminal window and activate the `carnd-term1` environment
	3. Type `python drive.py model_final.h5 -m 1.4 -t 1`. This runs the model with a target speed of 25, a steering multiplier of 1.4 and a throttle multiplier of 1., which work better on this track. You can also increase target speed to 30 without any noticeable loss in accuracy.

I also made an attempt at the "jungle track", provided with the 10MHz simulator. To achieve this I drove on that track as well using the same strategy as described later, but due to the fact that this simulator records five times less data, it constitutes only a small portion of the overall data that the model was trained on. It performs reasonably well on most of the track except on two occasions where it seems to get confused by the high, red-and-white roadside delimiters and goes off track. I can manually reverse back into the track and then the car goes on. The model also sometimes switches lanes, which I believe could be corrected with more training on a two-lane road (the jungle track is the only track to display this feature).

* **To run the model on the jungle track, follow the steps below:**

	1. Start the 10MHz simulator, as the other application does not include this track. Select the fastest graphics quality.
	2. Open another terminal window and activate the `carnd-term1` environment.
	3. Type `python drive.py model_final.h5 -s 15 -m 2 -t 0.2`. This runs the model with a target speed of 15, a steering multiplier of 2.0 and a throttle multiplier of 0.2, which work better on this track.
	4. Be prepared to manually reverse out of the ditch in a couple of places! If that happens, simply bring the back onto the correct lane using manual controls.

Ideally, I would have liked to train a model that uses the same driving parameters for all three tracks (speed, steering and throttle multipliers), but unfortunately I ran out of time. As it is, the model works on both track 1 and 'mountain' with identical parameters (steering multiplier = 1.4) but on track 1, the driving is smoother with steering multiplier = 1.


####3. Submission code is usable and readable

The `model_final.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture

####1. General Design Strategy

I started by building the helper code to import images in an early version that did not use a generator. I then coded a rudimentary neural network using dense layers only to test the basic concepts with a very small sample of data.

I then wrote the code to generate training batches during the training phase, using a Python generator as suggested in the project videos. This allowed me to use the full data set provided by Udacity.

The next step was to improve the model architecture. To save time, I decided to start from the nVidia model [(Ref.1)](#ref1) and added dropout layers after each Dense layer, to prevent overfitting. At that point, my desktop computer was no longer up to the task as I have an AMD GPU, which I couldn't use with the CUDA libraries. I therefore set up a "GPU Compute" AWS EC2 instance and all the subsequent training was performed on it.

After checking that the whole pipeline worked as intended (which naturally involved hours of debugging), I decided that it was time to record more data. I describe the data collection process into more detail [below](#data-collection).

Then began an iterative process of tuning metaparameters, training, testing, until no further progress seemed possible with the available dataset. I then decided to implement data augmentation and pre-processing. I developped several functions to add random translations, random brightness adjustments and histogram adjustments. For translations, I add a steering correction depending on the direction of the translation. Finally, I add horizontally flipped copies of each image (with the opposite steering angle as the outcome to predict).

In addition to the above, I also pick randomly from the left, center and right camera view for each timestamp of the training data and apply a correction to the steering angle. I describe these corrections in more detail below.

At that point I had a decent model for track one (although it displayed an unhealthy affinity for yellow lines and red and white kerbs), but I couldn't get the car go drive around the mountain track without hitting a wall or concrete barrier.

In an attempt to increse the model's accuracy, I did three things:

 - Increased the depth of the network,
 - To prevent overfitting, I added extra dropout units and L2 regularization everywhere,
 - Recorded my own dataset, with data from track one as well as the mountain and jungle tracks.

Then as I was running out of time came a major breakthrough: I realized that in addition to target speed, I could also add a multiplicative factor in the `drive.py` file that is used to execute the model in autonomous mode. Setting it to 1.4 almost immediately allowed me to complete the mountain track.

Given more time, I could have tried to streamline the model now that I know the steering multiplier trick. Maybe a simpler model would work equally well or even better, but that is something I was not able to work before the submission deadline.

####2. An appropriate model architecture has been employed

The model I used in this project is derived from the aforementioned nVidia architecture [Ref. 1](#ref1). The differences are:

 - Input shape is different (I used 40 x 80 vs. the original 66 x 220)
 - A 1x1x6 convolution layer after normalization, increasing depth to 6
 - Consequently, depths of the next two 5x5 conv layers increased to 48 and 72 (vs. 24 and 36 respectively in the original nVidia paper)
 - Depth of the next 3x3 conv layer increased to 96 (vs. 64)
 - Replace next 3x3x64 layer with a 1x1x128 layer
 - Depth of next 3 dense layers increased to 128, 64 and 16 (from 100, 50, 10)
 - Dropout layers inserted after each activation
 - L2 regularization used on every convolutional or dense layer

####3. Attempts to reduce overfitting in the model

To prevent overfitting, the following measures were taken:

 - Record data on all three tracks in both directions
 - Augment data by generating randomly shifted and brightness-adjusted copies, selected at random in each batch for every epoch of the training process
 - Insert dropout layers after each ReLU activation unit, using a constant keep probability of 50%
 - Use L2 regularisation 
 - Split the dataset into train (80%) and validation (20%) sets. Please note, the data on which the model is actually trained is not the raw images from the train set, but random batches with random transformations applied to them. Each batch contains both the unflipped and the flipped images.

> *Note:* Detecting overfitting is far from obvious. The only measure of accuracy that we have is subjective (beyond the vehicle staying on track). Of course one could say that if the vehicle manages track 1 but not the other two, it is overfitting, but I believe this is too restrictive a definition because the other two tracks have far more challenging features; thus this could also be a case of high bias rather than high variance.
It seems to me that overfitting could manifest itself by sudden unexplainable behavior of the car: A sudden turn  where there is no curve, a sudden attraction to a wall...
The behaviour of my final model is not entirely free of such behavior: for instance, on track one, it sometimes skims the red and white kerbs; on the mountain track it sometimes get very close to the side lines; on the jungle track is goes off track on a couple of occasions. Whether these are instances of overfitting or not is hard to tell.

####4. Model parameter tuning

Besides model architecture, the training parameters available for tuning are: Batch size, initial learning rate, number of epochs, corrective angle for left/right cameras, color depth of images, dropout probability.

> *Note:* At test time, there is mainly target speed and steering multiplier, which I discussed earlier.

Parameter tuning was another challenge. The correlation between parameter adjustments and driving quality is sometimes hard to understand and so this phase can be very time-consumming. I quickly discarded color depth to stick to RGB (grayscale just wasn't working). With regards to learning rate, I set the initial rate at 0.0001 and did not touch it afterwards, as I am using the Adam optimizer which is able to adjust its learning rate automatically.
To speed up tuning time, I started using subsamples of data and only trained full models once I thought I had a good set of parameters.

Of course this was an iterative process and each change to the model required a new search through the parameter space to find the right combination.

 
####5. Final architecture

|Layer (type)                     |Output Shape          |Param #     |Connected to         |
|---------------------------------|----------------------|:----------:|---------------------|
|Lambda - Normalization           |(None, 40, 80, 3)     |0           |lambda_input_1[0][0] |
|Convolution2D - 1x1, stride 1    |(None, 40, 80, 6)     |24          |lambda_1[0][0]       |
|Activation - ReLU                |(None, 40, 80, 6)     |0           |convolution2d_1[0][0]|
|Convolution2D - 5x5, stride 2    |(None, 18, 38, 48)    |7248        |activation_1[0][0]   |
|Activation - ReLU                |(None, 18, 38, 48)    |0           |convolution2d_2[0][0]|
|Dropout - Keep 50%               |(None, 18, 38, 48)    |0           |activation_2[0][0]   |
|Convolution2D - 5x5, stride 2    |(None, 7, 17, 72)     |86472       |dropout_1[0][0]      |
|Activation - ReLU                |(None, 7, 17, 72)     |0           |convolution2d_3[0][0]|
|Dropout - Keep 50%               |(None, 7, 17, 72)     |0           |activation_3[0][0]   |
|Convolution2D - 3x3, stride 2    |(None, 5, 15, 96)     |62304       |dropout_2[0][0]      |
|Activation - ReLU                |(None, 5, 15, 96)     |0           |convolution2d_4[0][0]|
|Dropout - Keep 50%               |(None, 5, 15, 96)     |0           |activation_4[0][0]   |
|Convolution2D - 1x1, stride 1    |(None, 5, 15, 128)    |12416       |dropout_3[0][0]      |
|Activation - ReLU                |(None, 5, 15, 128)    |0           |convolution2d_5[0][0]|
|Dropout - Keep 50%               |(None, 5, 15, 128)    |0           |activation_5[0][0]   |
|Flatten                          |(None, 9600)          |0           |dropout_4[0][0]      |
|Dense                            |(None, 128)           |1228928     |flatten_1[0][0]      |
|Activation - ReLU                |(None, 128)           |0           |dense_1[0][0]        |
|Dropout - Keep 50%               |(None, 128)           |0           |activation_6[0][0]   |
|Dense                            |(None, 64)            |8256        |dropout_5[0][0]      |
|Activation - ReLU                |(None, 64)            |0           |dense_2[0][0]        |
|Dropout - Keep 50%               |(None, 64)            |0           |activation_7[0][0]   |
|Dense                            |(None, 16)            |1040        |dropout_6[0][0]      |
|Activation - ReLU                |(None, 16)            |0           |dense_3[0][0]        |
|Dropout - Keep 50%               |(None, 16)            |0           |activation_8[0][0]   |
|Dense                            |(None, 1)             |17          |dropout_7[0][0]      |

Total number of parameters: 1,406,705

The model took about 90 minutes to train on an AWS EC2 GPU Compute instance. The most cumbersome part was transfering all the training data from my local computer to the server, as I had over 3GB of data and hit Linux's limit on file counts. I therefore had to split the data in three groups, zip them individually, tranfer the zip files and unzip on the server, which took well over an hour.


###Training strategy

####1. Data collection
<a name="data-collection"></a>

I devoted a lot of time to data collection. For smoother steering inputs, I used a Logitech G25 steering wheel and was very careful about the car's position on the road. For each of the three tracks, I recorded the following:
 - 2 laps one way
 - 2 laps in the reverse direction
 - 1 "recovery" lap one way, where I would bring the car to one side of the road, hit "record" and drive back to the center of the road. I did this on the straights and on the inside and the outside of curves to provide the model with examples of every conceivable scenario
 - 1 "recovery" lap in the reverse direction

Track 1 and the mountain track were both recorded at 50MHz whereas the jungle track was recorded at 10MHz. Thus, I had roughly 5 times less data for the latter. Moreover, when driving the jungle track, I kept the car on the right-hand lane, which is a unique feature of this track.

The collected data amounts to a total of approximately 3GB and 160,000 timestamps, which seems like a very large dataset compared to some discussions I saw on Slack or the forums. This was then used as the base dataset on which to apply various pre-preocessing and data augmentation routines, which I detail below.

Out of the 160,000 timestamps, 20% were set aside randomly as a validation set.

####2. Data augmentation and pre-processing

The purpose of data augmentation here is not to increase the amount of data available, as I already have a very large dataset, but rather to help the model generalize better, for instance to situations where the car is far from the ideal central position.

To achieve this, each **training batch** is built by:

 - Randomly selecting an image from left, center and right camera
 - Randomly adding horizontal and vertical translations, from a uniform distribution within the range [-30, +30] pixels,
 - Cropping the top 55 and bottom 25 pixels to retain only the parts than contain the road
 - Resizing it to 40x80
 - Randomly selecting brightness adjustments, from a random distribution within the range [0.70, 1.30], capped at a maximum value of 255 to avoid errors
  - Applying histogram equalization to improve contrast
 - Appending this image and its horizontally flipped counterpart to the training batch
 - During training, the image is normalized to values within the range [-1., +1.] by a Keras Lambda layer

Whenever applying horizontal translation, image flipping or simply selecting a left- or right-camera image, the corresponding steering input needs to be corrected. Detail of this below.

**Validation batches** are built in the same way, except that no data augmentation (random translations and brightness adjustments) is used and only center-camera images are considered.

####3. Steering angle adjustments

There are three occasions during training batch generation where steering angles need to be modified:

 - Image flipped horizontally: The steering angle is simply reversed ($\sigma$ := $-\sigma$)
 - Left- or Right-hand camera image selected: See discussion below.
 - Image translated horizontally: A fixed steering correction is applied for every pixel of translation. For consistency, this correction is proportional to the angle correction for left and right images, with a fixed coefficient that I determined empirically. A detailed calculation would have been possible, but after I made the decision discussed below, there was no point in refining this second-order term.
 
**Selecting the right steering angle correction for left- and right-hand camera images:**

When steering into a left-hand corner, the car is in a situation similar to below:

[image1]


 


References:

<a name="ref1"></a>Ref.1: https://arxiv.org/pdf/1604.07316.pdf 
*"End-to-End Learning for Self-Driving Cars"* - Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba


With this project, I quickly realized that one could easily get lost in fiddling with the model, metaparameters, recording more and more data etc. I personally struggled to see logical connections between some of the changes I made and the outcome in terms of driving accuracy. I was trying to get the model to drive around the mountain track and it seemed that no matter how much data I was feeding the model with, or how many changes I made to the neural network's architecture, I could never achieve that. 

Next steps:
 - Drive like a pilot
 - Reduce model complexity