import matplotlib
matplotlib.use("Agg")

from LearningRateDecay.learningRateScheduler import StepDecay
from LearningRateDecay.learningRateScheduler import PolynomialDecay
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

epochs = args["epoch"]
callbacks = []
schedule = None

if args["schedule"] == "step":
    print("[INFO] using 'step-based' learning rate decay...")
    schedule = StepDecay(initAlpha=1e-1, factor=0.25, dropEvery=15)
elif args["schedule"] == "linear":
    print("[INFO] using 'linear' learning rate decay...")
    schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=1)
elif args["schedule"] == "poly":
    print("[INFO] using 'ploynomial' learning rate decay...")
    schedule = DeploynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=5)

#如果learing rate schedule不为空，将其放入回调函数中
if schedule is not None:
    callbacks = [LearningRateScheduler(schedule)]

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#在CIFAR-10数据集中初始化标签名
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#初始化优化器和模型
opt = SGD(lr=1e-1, momentum=0.9, decay=decay)
modele = ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#训练网络
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=epochs, callbacks=callbacks, verbose=1)

