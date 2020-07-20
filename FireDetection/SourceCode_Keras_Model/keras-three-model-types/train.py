# USAGE
# python train.py --model sequential --plot output/sequential.png
# python train.py --model functional --plot output/functional.png
# python train.py --model class --plot output/class.png

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# there seems to be an issue with TensorFlow 2.0 throwing non-critical
# warnings regarding gradients when using the model sub-classing
# feature -- I found that by setting the logging level I can suppress
# the warnings from showing up (likely won't be required in future
# releases of TensorFlow)
import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

# import the necessary packages
from pyimagesearch.models import MiniVGGNetModel
from pyimagesearch.models import minigooglenet_functional
from pyimagesearch.models import shallownet_sequential
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD # 随机梯度下降算法
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="sequential",
	choices=["sequential", "functional", "class"],
	help="type of model architecture")
ap.add_argument("-p", "--plot", type=str, required=True,
	help="path to output plot file")
args = vars(ap.parse_args())

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-2
BATCH_SIZE = 128
NUM_EPOCHS = 60

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog",
	"frog", "horse", "ship", "truck"]

# load the CIFAR-10 dataset
print("[INFO] loading CIFAR-10 dataset...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# scale the data to the range [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	 horizontal_flip=True, fill_mode="nearest")

# check to see if we are using a Keras Sequential model
if args["model"] == "sequential":
	# instantiate a Keras Sequential model
	print("[INFO] using sequential model...")
	model = shallownet_sequential(32, 32, 3, len(labelNames))

# check to see if we are using a Keras Functional model
elif args["model"] == "functional":
	# instantiate a Keras Functional model
	print("[INFO] using functional model...")
	model = minigooglenet_functional(32, 32, 3, len(labelNames))

# check to see if we are using a Keras Model class
elif args["model"] == "class":
	# instantiate a Keras Model sub-class model
	print("[INFO] using model sub-classing...")
	model = MiniVGGNetModel(len(labelNames))

# initialize the optimizer compile the model and
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
print("[INFO] training network...")
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // BATCH_SIZE,
	epochs=NUM_EPOCHS,
	verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

# determine the number of epochs and then construct the plot title
N = np.arange(0, NUM_EPOCHS)
title = "Training Loss and Accuracy on CIFAR-10 ({})".format(
	args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title(title)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])