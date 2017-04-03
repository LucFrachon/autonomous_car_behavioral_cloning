
# Project 3: Use Deep Learning to Clone Driving Behavior

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains my implementation of behavioral cloning applied to a self-driving car simulator.

I collected driving data form three different tracks then used a convolutional neural network to teach the model to drive.

## Files

`network_large.py`: The code implementing the model architecture and training parameters

`import_data.py`: Functions and generators producting image batches for training and validation and applying pre-processing and data augmentation;

`model.h5`: The final, trained model

`drive.py`: Code used to drive the car in autonomous mode. Uses some flags to define target speed, steering multiplicative factor and throttle input multiplicative factor.

`video.py`: Code to generate videos from recorded screen shots

`video.mp4`, `video_track2.mp4`, `video_track3_edit.mp4`: Videos showcasing the result of the trained model on three different tracks of increasing difficulty. Notice how the model performs best on track 2.

`writeup_report.md`: The project writeup that explains design choices and reflections.

`im` folder: Contains images to render `writeup_report.md` correctly.

`LICENSE`: MIT license.

`README`: This file.
