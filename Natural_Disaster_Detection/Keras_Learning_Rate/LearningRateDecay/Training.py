
import matplotlib
matplotlib.use("Agg")


from LearningRateScheduler import StepDecay
from LearningRateScheduler import PolynomialDecay
from ResNet.resnet import ResNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--schedule", type=str, default="", help="learning rate schedule method")
ap.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs to train for")
ap.add_argument("-l", "--lr-plot", type=str, default="lr.png", help="path to output learning rate plot")
ap.add_argument("-t", "--train-plot", type=str, default="train.png", help="path to output training plot")
args = vars(ap.parse_args())