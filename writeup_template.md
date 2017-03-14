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

[image1]: ./examples/placeholder.png "Model Visualization"
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

* `network_large.py`: My final ConvNet architecture, derived from the [nVidia architecture][1]
* `import_data.py`: The code used to import data, augment it, pre-process it and build generators to feed batches into the model
* `drive.py`: The code used to drive the car. It is based on the provided code but I added the possibility to pass different a target speed as well as multipliers to the steering and throttle inputs. I also included data pre-processing into it in line with what is applied to validation sets during the training phase.
* `model_final.h5`: The trained model, passed to `drive.py` to drive the car in autonomous mode.
* `video.mp4`: A video recording of the model completing XX laps of track 1.
* `self_driving_car_mountain.mp4`: A video recording of the model completing track 2 of the 50MHz simulator (mountain road), with added music. Note that the steering and throttle multipliers used during recording were 1.5 and 0.3, but the model works with 1.4 and throttle multiplier = 1. I added some music to make it nicer (The Smiths' *Oscillate Wildly* -- a fitting name considering the behaviour the car occasionally exhibits!).

####2. Submission includes functional code

To run the model on track one, follow the steps below:

1. Start the 50MHz simulator (the 10MHz sim does not work as well and more importantly, does not contain the mountain track which I also tested the model on) and select the faster graphics quality (this is what I used to train the model). Other options will works too but with a slightly lower accuracy.
2. Open another terminal window and activate the `carnd-term1` environment
3. Type `python drive.py model_final.h5`. This runs the model with a target speed of 25, a steering multiplier of 1. and a throttle multiplier of 1., which work well on this track. You can also increase the target speed to 30 if you like, adding the optional flag `-s 30`. Note that the driving parameters used for the mountain track also work but generate quite wobbly drive, with the car meandering from left to right on the straights.

To run the model on track two (mountains), follow the steps below:

1. Start the 50MHz simulator. Screen resolution does not seem to matter much, however you will need to select the fastest graphics quality in order to prevent the model from getting confused with shadows (see discussion later).
2. Open another terminal window and activate the `carnd-term1` environment
3. Type `python drive.py model_final.h5 -m 1.4 -t 1`. This runs the model with a target speed of 25, a steering multiplier of 1.4 and a throttle multiplier of 1., which work better on this track. You can also increase target speed to 30 without any noticeable loss in accuracy.

I also made an attempt at the "jungle track", provided with the 10MHz simulator. To achieve this I drove on that track as well using the same strategy as described later, but due to the fact that this simulator records five times less data, it constitutes only a small portion of the overall data that the model was trained on. It performs reasonably well on most of the track except on two occasions where it seems to get confused by the high, red-and-white roadside delimiters and goes off track. I can manually reverse back into the track and then the car goes on. The model also sometimes switches lanes, which I believe could be corrected with more training on a two-lane road (the jungle track is the only track to display this feature).

To run the model on the jungle track, follow the steps below:

1. Start the 10MHz simulator, as the other application does not include this track. Select the fastest graphics quality.
2. Open another terminal window and activate the `carnd-term1` environment.
3. Type `python drive.py model_final.h5 -s 15 -m 2 -t 0.2`. This runs the model with a target speed of 15, a steering multiplier of 2.0 and a throttle multiplier of 0.2, which work better on this track.

Ideally, I would have liked to train a model that uses the same driving parameters for all three tracks (speed, steering and throttle multipliers), but unfortunately I ran out of time.


####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

With this project, I realized that one could easily get lost in fiddling with the model, metaparameters, recording more and more data etc. I personally struggled to see logical connections between some of the changes I made and the outcome in terms of driving accuracy. I was trying to get the model to drive around the mountain track and it seemed that no matter how much data I was feeding the model with, or how many changes I made to the neural network's architecture, I could never achieve that. One of the major breakthroughs for me was when I realized that I could tune driving parameters to amplify the steering angles computed by the neural network.

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

References:

[1]: https://arxiv.org/pdf/1604.07316.pdf *"End-to-End Learning for Self-Driving Cars"* - Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba

