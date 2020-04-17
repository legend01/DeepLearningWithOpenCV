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

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testY = testX.astype("float")

mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
opt = SGD(lr=config.MIN_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=32, classes=10)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] using '{}' method".format(config.CLR_METHOD))
clr = CyclicLR(
    mode = config.CLR_METHOD,
    base_lr = config.MIN_LR,
    max_lr = config.MAX_LR,
    step_size = config.STEP_SIZE * (trainX.shape[0]//conifg.BATCH_SIZE)
)

print("[INFO] traininig network....")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
    validation_data = (testX, testY), 
    steps_per_epoch = trainX.shape[0],
    epochs = config.NUM_EPOSHS, 
    callbacks = [clr], 
    verbose = 1
)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=config.BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predictions.argumax(axis=1), target_names=config.CLASSES))

N = np.arange(0, conifg.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower_left")
plt.savefig(config.TRAINING_PLOT_PATH)

N = np.range(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH)





