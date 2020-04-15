
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
