import numpy as np
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.io as io

train_data = '' # Training data
test_data = '' # Testing data

tf.random.set_random_seed(1234)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

size = 224
batch_size = 5
X = tf.placeholder(tf.float32 , [None,size,size,1])
Y = tf.placeholder(tf.float32 , [None , 10])
droupout_prob = tf.placeholder(tf.float32)
X_pred = tf.placeholder(tf.float32 , [None,size,size,3])
Y_true = tf.placeholder(tf.int32 , [None,size,size])
Y_x = tf.placeholder(tf.float32, [None, size, size, 3])


def one_hot_label(img):
	## Contains all the categories and returns their one-hot encodings.
    continue


def train_data_with_label():
    train_images=[]
    for i in tqdm(os.listdir(train_data)):
        path=os.path.join(train_data,i)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(size,size))
        train_images.append([np.array(img),one_hot_label(i)])
    shuffle(train_images)
    return train_images


def test_data_with_label():
    test_images=[]
    for i in tqdm(os.listdir(test_data)):
        path=os.path.join(test_data,i)
        #print(path)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(size,size))
        test_images.append([np.array(img),one_hot_label(i)])
        shuffle(test_images)
    return test_images

def important_function(c):
	# This function will give the modified images.
	y = np.empty(c.shape[:-1])
	print(c.shape[2])
	sing = np.zeros(c.shape[:-1])
	for i in range(c.shape[0]):
		for j in range(c.shape[1]):
			flat = c[i, j, :].flatten()
			k = np.max(flat)
			for g in range(c.shape[2]):
				if k == c[i,j,g]:
					y[i,j] = g
	return y


def segmented_train_data_with_label():
	segment_images = []
	for i in tqdm(os.listdir(segment_label)):
		path = os.path.join(segment_label, i)
		img_ = io.imread(path)
		img_ = cv2.resize(img_, (size, size))
		img_classes = important_function(img_)
		#i[-4:] = '.jpg'
		j = i[:-4] + '.jpg'
		path_ = os.path.join(train_data+"/", j)
		print(path_)
		img = cv2.imread(path_)
		if img is None:
			continue
		else:
			img = cv2.resize(img, (size, size))
			segment_images.append([np.array(img), np.array(img_classes), np.array(img_)])
	shuffle(segment_images)
	return segment_images


training_images = train_data_with_label()
testing_images = test_data_with_label()
segment_images = segmented_train_data_with_label()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,size,size,1)
tr_lbl_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,size,size,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

tr_segment_data = np.array([i[0] for i in segment_images]).reshape(-1,size,size,3)
tr_segment_lbl = np.array([i[1] for i in segment_images]).reshape(-1,size,size)
tr_segment = np.array([i[2] for i in segment_images]).reshape(-1,size,size,3)

def vgg(X,droupout_prob, train, circ = False, circ1 = False, segmentation = False):

	kernel = [3,3,1,64]
	kernel_ = [3,3,64,64]
	kernel2 = [3,3,64,128]
	kernel2_ = [3,3,128,128]
	kernel3 = [3,3,128,256]
	kernel3_ = [3,3,256,256]
	kernel4 = [3,3,256,512]
	kernel5 = [3,3,512,512]
	kernel6 = [7 * 7 * 512,4096]
	kernel7 = [4096,4096]
	kernel8 = [4096 , 10]

	ksize = [1,2,2,1]
	strides = [1,1,1,1]
	strides2 = [1,2,2,1]
	X /= 255
	initial = tf.contrib.layers.xavier_initializer()


	#CONVOLUTION BLOCK - 1
	#filter1 = tf.Variable(tf.random_normal(kernel , stddev = 0.05))

	if segmentation:
		filter1 = tf.Variable(initial([3,3,3,64]))
		layer1_conv_1 = tf.layers.batch_normalization(tf.nn.relu(tf.nn.conv2d(X ,filter1 , strides = strides , padding = 'SAME')), training = train)

	else:
		filter1 = tf.Variable(initial(kernel))
		layer1_conv_1 = tf.layers.batch_normalization(tf.nn.relu(tf.nn.conv2d(X ,filter1 , strides = strides , padding = 'SAME')), training = train)

	#bias1 = tf.Variable(initial([64]))

	#filter2 = tf.Variable(tf.random_normal(kernel_ , stddev = 0.05))
	filter2 = tf.Variable(initial(kernel_))
	#bias2 = tf.Variable(initial([64]))
	layer1_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer1_conv_1 ,filter2 , strides = strides , padding = 'SAME')), droupout_prob), training = train)

	layer1_maxpool = tf.nn.relu(tf.nn.max_pool(layer1_conv_2 , ksize = ksize , strides = [1,2,2,1] , padding = 'SAME'))
	print("Layer-1",layer1_maxpool.get_shape().as_list())

	#CONVOLUTION BLOCK - 2
	#filter3 = tf.Variable(tf.random_normal(kernel2 , stddev = 0.05))
	filter3 = tf.Variable(initial(kernel2))
	#bias3 = tf.Variable(initial([128]))
	layer2_conv_1 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer1_maxpool ,filter3 ,strides = strides , padding = 'SAME')), droupout_prob), training = train)

	#filter4 = tf.Variable(tf.random_normal(kernel2_ , stddev = 0.05))
	filter4 = tf.Variable(initial(kernel2_))
	#bias4 = tf.Variable(initial([128]))
	layer2_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer2_conv_1 ,filter4 ,strides = strides , padding = 'SAME')), droupout_prob), training = train)

	layer2_maxpool = tf.nn.relu(tf.nn.max_pool(layer2_conv_2 , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer2 = tf.nn.dropout(layer2_maxpool , droupout_prob)
	print("Layer-2",layer2.get_shape().as_list())

	#CONVOLUTION BLOCK - 3
	#filter5 = tf.Variable(tf.random_normal(kernel3 , stddev = 0.05))
	filter5 = tf.Variable(initial(kernel3))
	#bias5 = tf.Variable(initial([256]))
	layer3_conv_1 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer2 , filter5 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	#filter6 = tf.Variable(tf.random_normal(kernel3_ , stddev = 0.05))
	filter6 = tf.Variable(initial(kernel3_))
	#bias6 = tf.Variable(initial([256]))
	layer3_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer3_conv_1 , filter6 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	#filter7 = tf.Variable(tf.random_normal(kernel3_ , stddev = 0.05))
	filter7 = tf.Variable(initial(kernel3_))
	#bias7 = tf.Variable(initial([256]))
	layer3_conv_3 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer3_conv_2 , filter7 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	layer3_maxpool = tf.nn.relu(tf.nn.max_pool(layer3_conv_3 , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer3 = tf.nn.dropout(layer3_maxpool , droupout_prob)
	print("Layer-3",layer3.get_shape().as_list())
	shape = layer3.get_shape().as_list()

	if circ1:
		return layer3

	#CONVOLUTION BLOCK - 4
	#filter8 = tf.Variable(tf.random_normal(kernel4 , stddev = 0.05))
	filter8 = tf.Variable(initial(kernel4))
	#bias8 = tf.Variable(initial([512]))
	layer4_conv_1 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer3 , filter8 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	#filter9 = tf.Variable(tf.random_normal(kernel5 , stddev = 0.05))
	filter9 = tf.Variable(initial(kernel5))
	#bias9 = tf.Variable(initial([512]))
	layer4_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer4_conv_1 , filter9 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	#filter10 = tf.Variable(tf.random_normal(kernel5 , stddev = 0.05))
	filter10 = tf.Variable(initial(kernel5))
	#bias10 = tf.Variable(initial([512]))
	layer4_conv_3 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer4_conv_2 , filter10 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	layer4_maxpool = tf.nn.relu(tf.nn.max_pool(layer4_conv_3 , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer4 = tf.nn.dropout(layer4_maxpool , droupout_prob)
	print("Layer-4",layer4.get_shape().as_list())
	shape = layer4.get_shape().as_list()

	if circ:
		return layer4

	#CONVOLUTION BLOCK - 5
	#filter11 = tf.Variable(tf.random_normal(kernel5 , stddev = 0.05))
	filter11 = tf.Variable(initial(kernel5))
	#bias11 = tf.Variable(initial([kernel5[-1]]))
	layer5_conv_1 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer4 , filter11 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	#filter12 = tf.Variable(tf.random_normal(kernel5 , stddev = 0.05))
	filter12 = tf.Variable(initial(kernel5))
	#bias12 = tf.Variable(initial([kernel5[-1]]))
	layer5_conv_2 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer5_conv_1 , filter12 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	#filter13 = tf.Variable(tf.random_normal(kernel5 , stddev = 0.05))
	filter13 = tf.Variable(initial(kernel5))
	#bias13 = tf.Variable(initial([kernel5[-1]]))
	layer5_conv_3 = tf.layers.batch_normalization(tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(layer5_conv_2 , filter13 ,strides = strides , padding ='SAME')), droupout_prob), training = train)

	layer5_maxpool = tf.nn.relu(tf.nn.max_pool(layer5_conv_3 , ksize = ksize , strides = strides2 , padding = 'SAME'))
	layer5 = tf.layers.batch_normalization(tf.nn.dropout(layer5_maxpool , droupout_prob), training = train)
	print("Layer-5",layer5.get_shape().as_list())
	shape = layer5.get_shape().as_list()

	#FIRST DENSE LAYER
	filter14 = tf.Variable(initial(kernel6))
	#bias14 = tf.Variable(initial([kernel6[-1]]))
	shapestraight = [-1 , shape[1] * shape[2] * shape[3]]
	layer6_shaping = tf.reshape(layer5 , shapestraight)
	layer6drop = tf.nn.dropout(layer6_shaping , droupout_prob)
	layer6 = tf.layers.batch_normalization(tf.nn.relu(tf.matmul(layer6drop , filter14)), training = train)
	print("Layer-6",layer6.get_shape().as_list())

	#SECOND DENSE LAYER
	filter15 = tf.Variable(initial(kernel7))
	#bias15 = tf.Variable(initial([kernel7[-1]]))
	layer7 = tf.nn.relu(tf.matmul(layer6, filter15))
	layer7 = tf.layers.batch_normalization(tf.nn.dropout(layer7 , droupout_prob), training = train)
	print("Layer-7",layer7.get_shape().as_list())

	#OUTPUT LAYER
	filter16 = tf.Variable(initial(kernel8))
	#bias16 = tf.Variable(initial([kernel8[-1]]))
	#layer8drop = tf.nn.dropout(layer4 , droupout_prob)
	output = tf.matmul(layer7 , filter16)
	print("Output",output.get_shape().as_list())
	return output

def DeepLab(X, rates = [1, 2, 4], mrates = [], mgrid = False, apply_batchnorm = False, depth = 256):

	# X is the input batch of tensors or images.
	# rates are the dilation rates for the Atrous Spatial Pyramid Pooling(ASPP)
	# mrates are the dilation rates for the multi-grid system just before the ASPP module.
	# mgrid is a boolean variable to perform multi-grid convolution before ASPP or not.
	# apply_batchnorm is a boolean variable to do Batch Normalization or not.
	initial = tf.contrib.layers.xavier_initializer()
	# Applying Global Average pooling.
	res = tf.reduce_mean(X, [1,2], name = 'global_pool', keepdims = True)
	k = tf.Variable(tf.random_normal([1, 1, 512, depth], stddev = 0.05))
	image_level_features = tf.nn.conv2d(res, k, strides = [1, 1, 1, 1], padding = 'SAME', name = 'resize')
	res_ = tf.image.resize_bilinear(image_level_features, (tf.shape(X)[1], tf.shape(X)[2]))
	filter_1 = tf.Variable(initial([1, 1, 512, depth]))
	res = tf.nn.conv2d(X, filter_1, strides = [1, 1, 1, 1], padding = 'SAME', name = 'conv_1_1')

	filter_2_1 = tf.Variable(initial([3, 3, 512, depth]))
	filter_2_2 = tf.Variable(initial([3, 3, 512, depth]))
	filter_2_3 = tf.Variable(initial([3, 3, 512, depth]))
	
	res1 = tf.nn.conv2d(X, filter_2_1, strides = [1, 1, 1, 1], dilations = [1, rates[0], rates[0], 1], padding = 'SAME', name = 'conv_3_3_1')
	res2 = tf.nn.conv2d(X, filter_2_2, strides = [1, 1, 1, 1], dilations = [1, rates[1], rates[1], 1], padding = 'SAME', name = 'conv_3_3_2')
	res3 = tf.nn.conv2d(X, filter_2_3, strides = [1, 1, 1, 1], dilations = [1, rates[2], rates[2], 1], padding = 'SAME', name = 'conv_3_3_3')

	final = tf.concat((res_, res, res1, res2, res3), axis = 3, name = 'concat')
	final_w = tf.Variable(initial([1, 1, 5 * depth, depth]))
	final_ = tf.nn.conv2d(final, final_w, strides = [1, 1, 1, 1], padding = 'SAME', name = 'final_conv')

	return final_


def complete_DeepLab(input_img, dropout, N = 3, bn = False):

	initial = tf.contrib.layers.xavier_initializer()

	X = vgg(input_img, droupout_prob = dropout, train = bn, circ = True, circ1 = False, segmentation = True) # change circ to False and circ1 to true for the output stride to be 8 rather than 16.

	result = DeepLab(X)
	filter_final= tf.Variable(initial([1, 1, 256, N]))
	res_ = tf.nn.conv2d(result, filter_final, strides = [1, 1, 1, 1], padding = 'SAME', name = 'conv_final')

	return tf.image.resize_bilinear(res_, input_img.shape[1:-1])

def compute_iou(original, prediction):

	H, W, N = original.get_shape().as_list()[1:]
	pred = tf.reshape(prediction, [-1, H * W, N])
	orig = tf.reshape(original, [-1, H * W, N])
	intersection = tf.reduce_sum(pred * orig, axis = 2) + 1e-7
	denominator = tf.reduce_sum(pred, axis = 2) + tf.reduce_sum(orig, axis = 2) + 1e-7
	iou = tf.reduce_mean(intersection / denominator)

	return iou

is_train = tf.placeholder(tf.bool)
train_layer = vgg(X, droupout_prob, is_train)
Y_ = tf.nn.softmax(train_layer)
correct = tf.equal(tf.argmax(Y,1) , tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))

#cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = train_layer)
feature_map_16 = complete_DeepLab(X_pred, dropout = droupout_prob, bn = is_train)
iou = compute_iou(feature_map_16, Y_x)
loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = Y_true, logits = feature_map_16)
with tf.control_dependencies(update_ops):
    optimize = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
    optimize_segment =  tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss_cls)
optimize_segment =  tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss_cls)

init = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(init)

	# TRAINING PHASE
	for ep in range(1):
		batch_size = 1
		print("Epoch number %d" %(ep))
		x = 0
		ls = []
		count = 0
		for i in range(int(len(tr_img_data)/batch_size)):
			testing = []
			trainimg = tr_img_data[batch_size * i : batch_size * (i+1)]
			trainlbl = tr_lbl_data[batch_size * i : batch_size * (i+1)]
			trainimg = tr_segment_data[batch_size * i : batch_size * (i+1)]
			trainlbl = tr_segment_lbl[batch_size * i : batch_size * (i+1)]
			train_ = tr_segment[batch_size * i : batch_size * (i+1)]
			for i in trainlbl:
				testing.append(np.reshape(i, (-1, 10)))
			testing = np.reshape(testing, (batch_size, 10))
			#print(testing.shape)
			#print(trainimg.shape)
			data = {X : trainimg , Y: testing , X_pred : np.zeros((batch_size, size, size, 3)), Y_x : np.zeros((batch_size, size, size, 3)), Y_true : np.zeros((batch_size, size, size)), is_train : True, droupout_prob: 1.0}
			session.run(optimize , feed_dict = data)
			ls.append(session.run(accuracy , feed_dict = data))
			#print("Loss", session.run(cost, feed_dict = data))
		print("Highest : ", max(ls))
		print("Minimum : ",min(ls))

	# TESTING PHASE
	print("TESTING PHASE!!")
	testing = []
	batch_size = 5
	for i in range(int(len(tst_img_data)/batch_size)):
		testing = []
		testimg = tst_img_data[batch_size * i : batch_size * (i+1)]
		testlbl = tst_lbl_data[batch_size * i : batch_size * (i+1)]
		for i in testlbl:
			testing.append(np.reshape(i, (-1, 10)))
		testing = np.reshape(testing, (batch_size, 10))
		#print(testing.shape)
		#print(trainimg.shape)
		data = {X : testimg , Y: testing , X_pred : np.zeros((batch_size, size, size, 3)), Y_x : np.zeros((batch_size, size, size, 3)), Y_true : np.zeros((batch_size, size, size)), is_train : False, droupout_prob: 1.0}
		print(session.run(accuracy , feed_dict = data))
	## Uncomment the below lines if the machine can handle all the testing data at once. (VRAM should be enough)
	'''for i in tst_lbl_data:
		testing.append(np.reshape(i, (-1, 10)))
	testing = np.reshape(testing, (len(tst_lbl_data), 10))
	data = {X : tst_img_data , Y : testing , droupout_prob : 1}

	print("after training : ")
	print(session.run(accuracy , feed_dict = data))'''
	

	# SEGMENTATION CODE
	batch_size = 5
	print("SEGMENTATION TRAINING PHASE")
	for ep in range(5):
		print("Epoch number %d" %(ep))
		ls = []
		for i in range(int(len(tr_segment_data)/batch_size)):
			testing = []
			trainimg = tr_segment_data[batch_size * i : batch_size * (i+1)]
			trainlbl = tr_segment_lbl[batch_size * i : batch_size * (i+1)]
			train_ = tr_segment[batch_size * i : batch_size * (i+1)]
			data = {X : np.zeros((batch_size, size, size, 1)), Y : np.zeros((batch_size, 10)) ,X_pred : trainimg , Y_true : trainlbl , Y_x : train_ ,  is_train : False, droupout_prob: 1.0}
			session.run(optimize_segment , feed_dict = data)
			ls.append(session.run(iou , feed_dict = data))
			#print("Loss", session.run(cost, feed_dict = data))
		print("Highest : ", max(ls))
		print("Minimum : ",min(ls))

	print("SEGMENTATION TESTING PHASE")
	for _ in range(1):
		img = ip.imread('000045.jpg')
		print(img.shape)
		img = cv2.resize(img,(size,size))
		img = np.reshape(img, (1, size, size, 3))
		img_true = ip.imread('2.png')
		img_true = cv2.resize(img_true,(size,size))
		img_true = np.reshape(img_true[:,:,:-1], (1, size, size, 3))
		data = {X : np.zeros((batch_size, size, size, 1)), Y : np.zeros((batch_size, 10)), X_pred : img, Y_true : img_true[:,:,:,:], droupout_prob : 1.0, is_train : False}
		print(session.run(iou, feed_dict = data))
		o = session.run(feature_map_16, feed_dict = {X_pred : img, droupout_prob : 1.0, is_train : False})
		o = np.reshape(o, (o.shape[1], o.shape[2], o.shape[3]))
		print(o.shape)
		ip.imshow(o)
		plt.show()
