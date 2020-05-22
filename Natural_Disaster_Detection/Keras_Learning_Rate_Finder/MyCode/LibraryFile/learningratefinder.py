from keras.callbacks import LambdaCallback
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile

class LearningRateFinder:
    def __init__(self, model, stopFactor=4, beta=0.98):
        #存储模型，停止因素，beta值
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta

        #初始化学习率和损失值
        self.lrs = []
        self.losses = []

        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None
    
    def reset(self):
        #在构造器中重新初始化所有变量
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def is_data_iter(self, data):
        iterClasses = ["NumpyArrayIterator", "DirectoryIterator", "DAtaFrameIterator", "Iterator", "Sequence"]
        return data.__class__.__name__ in iterClasses

    def on_batch_end(self, batch, logs):
        #获取当前学习率并且增加log至存放学习率元组中
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        #在每个batch分组结束时获取损失值，增加批处理的批次总数，计算平均损失，平滑后更新损失列表
        l= logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * 1)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)

        #计算最大的损失停止值
        stopLoss = self.stopFactor * self.bestLoss

        #检查损失值是否增长太快
        if self.batchNum >1 and smooth >stopLoss:
            self.model.stop_training = True
            return

        #检查最优损失值是否需要跟新
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth

        #增加学习率
        lr * = self.lrMult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, trainDAta, startLR, endLR, epochs=None, stepsPerEpoch=None, batchSize=32, sampleSize=2048, verbose=1):
        

