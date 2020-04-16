import os

#初始化标签
CLASSES=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

MIN_LR = 1e-7
MAX_LR = 1e-2
BATCH_SIZE = 64
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 96

TRAINING_PLOT_PATH = os.path.sep.join(["E:/PythonWorkSpace/DeepLearningWithOpenCV/Natural_Disaster_Detection/Keras_CyclicalLearningRate/keras-cyclical-learning-rates/MyCode/Output", "training_plot.png"])
CLR_PLOT_PATH = os.path.join(["E:/PythonWorkSpace/DeepLearningWithOpenCV/Natural_Disaster_Detection/Keras_CyclicalLearningRate/keras-cyclical-learning-rates/MyCode/Output"], "clr_plot.png")




