import matplotlib
matplotlib.use("Agg")

from LibraryFile.minigooglenet import MiniGoogLeNet
from LibraryFile.clr_callback import CyclicLR
from LibraryFile import config
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np












