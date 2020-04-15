'''
@Description: 学习率衰减表 
@version: 
@Author: HLLI8
@Date: 2020-04-15 10:36:18
@LastEditors: HLLI8
@LastEditTime: 2020-04-15 11:03:56
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