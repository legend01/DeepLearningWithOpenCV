# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K

class MiniGoogLeNet:
	@staticmethod
	def conv_module(x, K, kX, kY, stride, chanDim, padding="same"): # x:网络层的输入  K:CONV层的filter个数   kX和kY:filter的大小    stride:步长   padding:填充模式,"same",保持输入跟输出大小一致
		# define a CONV => BN => RELU pattern
		x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation("relu")(x)

		# return the block
		return x

	@staticmethod
	def inception_module(x, numK1x1, numK3x3, chanDim): # x:输入层   numK1*1:1*1的filter个数   numK3*3:3*3的filter个数  chanDim:通道维度
		# define two CONV modules, then concatenate across the
		# channel dimension
		conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1,
			(1, 1), chanDim)
		conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3,
			(1, 1), chanDim)
		x = concatenate([conv_1x1, conv_3x3], axis=chanDim)

		# return the block
		return x

	@staticmethod
	def downsample_module(x, K, chanDim): # x:输入层   K:filter的个数     chanDim:特征维度
		# define the CONV module and POOL, then concatenate
		# across the channel dimensions
		conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 3, (2, 2),
			chanDim, padding="valid")
		pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
		x = concatenate([conv_3x3, pool], axis=chanDim)

		# return the block
		return x

	@staticmethod
	def build(width, height, depth, classes): # width:特征图像的宽度    height:特征图像的高度    depth:通道数    classes:类别个数
		# initialize the input shape to be "channels last" and the
		# channels dimension itself
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# define the model input and first CONV module
		inputs = Input(shape=inputShape)
		x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1),   #96个3*3的filter
			chanDim)

		# two Inception modules followed by a downsample module
		x = MiniGoogLeNet.inception_module(x, 32, 32, chanDim) #32个1*1的filter和32个1*1的filters   K = 32+32= 64个filters
		x = MiniGoogLeNet.inception_module(x, 32, 48, chanDim) #32个1*1的filters和48个3*3的filters  K = 32+48= 80个filters
		x = MiniGoogLeNet.downsample_module(x, 80, chanDim)

		# four Inception modules followed by a downsample module
		x = MiniGoogLeNet.inception_module(x, 112, 48, chanDim)
		x = MiniGoogLeNet.inception_module(x, 96, 64, chanDim)
		x = MiniGoogLeNet.inception_module(x, 80, 80, chanDim)
		x = MiniGoogLeNet.inception_module(x, 48, 96, chanDim)
		x = MiniGoogLeNet.downsample_module(x, 96, chanDim)

		# two Inception modules followed by global POOL and dropout
		x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
		x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim) # 7*7*336输出特征图像大小
		x = AveragePooling2D((7, 7))(x) # 1*1*336输出特征图像大小
		x = Dropout(0.5)(x)

		# softmax classifier
		x = Flatten()(x)
		x = Dense(classes)(x)
		x = Activation("softmax")(x)

		# create the model
		model = Model(inputs, x, name="googlenet")

		# return the constructed network architecture
		return model