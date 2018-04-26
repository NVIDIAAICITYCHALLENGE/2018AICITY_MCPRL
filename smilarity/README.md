# Train Model for Classifying Vehicle and Non-vehicle

## How it works

### 0.Dependencies
Our code has been tested on Ubuntu 14.04. You will need the following dependencies to run our code:

*Pytorch
*Numpy
*Pillow

### 1.Data
We train our model with the VeRi dataset, which is a large-scale benchmark dateset for vehicle Re-Id in the real-world urban surveillance scenario.

## 2.Code Structure
* Scripts `train.py` and `test.py` are for training or testing models seprarately
* Scripts `model.py` and `mylayer1.py` are the network architecture and loss function (triplet loss) respectively.
