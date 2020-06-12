# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate

def shallownet_sequential(width, height, depth, classes):
	# initialize the model along with the input shape to be
	# "channels last" ordering 初始化模型以及输入形状为通道最后次序 栈模型
	model = Sequential()
	inputShape = (height, width, depth)

	# define the first (and only) CONV => RELU layer
	model.add(Conv2D(32, (3, 3), padding="same", #卷积层
		input_shape=inputShape))
	model.add(Activation("relu")) #使用ReLU激活函数

	# softmax classifier
	model.add(Flatten()) #降维，矩阵转化为向量+全连接层
	model.add(Dense(classes))
	model.add(Activation("softmax")) #使用softmax激活函数

	# return the constructed network architecture
	return model

def minigooglenet_functional(width, height, depth, classes):
	def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
		# define a CONV => BN => RELU pattern
		x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x) #卷积层
		x = BatchNormalization(axis=chanDim)(x) #单个batch归一化
		x = Activation("relu")(x) #relu激活层

		# return the block
		return x

	def inception_module(x, numK1x1, numK3x3, chanDim):
		# define two CONV modules, then concatenate across the
		# channel dimension
		conv_1x1 = conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
		conv_3x3 = conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
		x = concatenate([conv_1x1, conv_3x3], axis=chanDim) #将模块分支聚集在通道维度上,两个分支填充,输出的尺寸相等,可以沿信道维度连接

		# return the block
		return x

	def downsample_module(x, K, chanDim):
		# define the CONV module and POOL, then concatenate
		# across the channel dimensions
		conv_3x3 = conv_module(x, K, 3, 3, (2, 2), chanDim, #卷积层
			padding="valid")
		pool = MaxPooling2D((3, 3), strides=(2, 2))(x) #池化层
		x = concatenate([conv_3x3, pool], axis=chanDim)

		# return the block
		return x

	# initialize the input shape to be "channels last" and the
	# channels dimension itself
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input and first CONV module
	inputs = Input(shape=inputShape)
	x = conv_module(inputs, 96, 3, 3, (1, 1), chanDim) #卷积层

	# two Inception modules followed by a downsample module
	x = inception_module(x, 32, 32, chanDim)
	x = inception_module(x, 32, 48, chanDim)
	x = downsample_module(x, 80, chanDim)

	# four Inception modules followed by a downsample module
	x = inception_module(x, 112, 48, chanDim)
	x = inception_module(x, 96, 64, chanDim)
	x = inception_module(x, 80, 80, chanDim)
	x = inception_module(x, 48, 96, chanDim)
	x = downsample_module(x, 96, chanDim)

	# two Inception modules followed by global POOL and dropout
	x = inception_module(x, 176, 160, chanDim)
	x = inception_module(x, 176, 160, chanDim)
	x = AveragePooling2D((7, 7))(x) #平均池化层
	x = Dropout(0.5)(x) # dropout正则化 随机失活0.5的概率 防止过拟合

	# softmax classifier
	x = Flatten()(x) #平滑层
	x = Dense(classes)(x) #全连接层
	x = Activation("softmax")(x) #激活层

	# create the model
	model = Model(inputs, x, name="minigooglenet")

	# return the constructed network architecture
	return model

class MiniVGGNetModel(Model):
	def __init__(self, classes, chanDim=-1):
		# call the parent constructor
		super(MiniVGGNetModel, self).__init__()

		# initialize the layers in the first (CONV => RELU) * 2 => POOL
		# layer set
		self.conv1A = Conv2D(32, (3, 3), padding="same") #卷积层
		self.act1A = Activation("relu") #relu激活层
		self.bn1A = BatchNormalization(axis=chanDim) #batch归一化
		self.conv1B = Conv2D(32, (3, 3), padding="same") #卷积层
		self.act1B = Activation("relu") #relu激活层
		self.bn1B = BatchNormalization(axis=chanDim) #batch归一化
		self.pool1 = MaxPooling2D(pool_size=(2, 2)) #最大池化层

		# initialize the layers in the second (CONV => RELU) * 2 => POOL
		# layer set
		self.conv2A = Conv2D(32, (3, 3), padding="same") #卷积层
		self.act2A = Activation("relu") #relu激活层
		self.bn2A = BatchNormalization(axis=chanDim) #batch归一化
		self.conv2B = Conv2D(32, (3, 3), padding="same") #卷积层
		self.act2B = Activation("relu") #relu激活层
		self.bn2B = BatchNormalization(axis=chanDim) #batch归一化
		self.pool2 = MaxPooling2D(pool_size=(2, 2)) #最大池化层

		# initialize the layers in our fully-connected layer set
		self.flatten = Flatten() #平滑层
		self.dense3 = Dense(512) #全连接层
		self.act3 = Activation("relu") #relu激活层
		self.bn3 = BatchNormalization() #batch归一化
		self.do3 = Dropout(0.5) #随机失活层 dropout归一化

		# initialize the layers in the softmax classifier layer set
		self.dense4 = Dense(classes) #全连接层
		self.softmax = Activation("softmax") #softmax分类

	def call(self, inputs):
		# build the first (CONV => RELU) * 2 => POOL layer set
		x = self.conv1A(inputs)
		x = self.act1A(x)
		x = self.bn1A(x)
		x = self.conv1B(x)
		x = self.act1B(x)
		x = self.bn1B(x)
		x = self.pool1(x)

		# build the second (CONV => RELU) * 2 => POOL layer set
		x = self.conv2A(x)
		x = self.act2A(x)
		x = self.bn2A(x)
		x = self.conv2B(x)
		x = self.act2B(x)
		x = self.bn2B(x)
		x = self.pool2(x)

		# build our FC layer set
		x = self.flatten(x)
		x = self.dense3(x)
		x = self.act3(x)
		x = self.bn3(x)
		x = self.do3(x)

		# build the softmax classifier
		x = self.dense4(x)
		x = self.softmax(x)

		# return the constructed model
		return x