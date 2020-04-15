'''
@Description: 学习率衰减表 
@version: 
@Author: HLLI8
@Date: 2020-04-15 10:36:18
@LastEditors: HLLI8
@LastEditTime: 2020-04-15 15:18:09
'''
import matplotlib.pyplot as plt
import numpy as np

class LearningRateDecay:
    def plot(self, epochs, title='Learning Rate Schedule'):
        lrs = [self(i) for i in epochs]

        plt.style.use("ggplot")
        plt.figure()    
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")

class StepDecay(LearningRateDecay):
    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
        self.initAlpha = initAlpha
        self.factor = factor 
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        #计算当前迭代的学习率
        exp np.floor((1+epoch)/self.dropEvery)
        alpha = self.initAlpha * (self.factor**exp)
        #返回学习率
        return float(alpha)