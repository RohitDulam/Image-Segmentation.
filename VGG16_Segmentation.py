import numpy as np
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
#import IP as ip
import skimage.io as io

#train_data= 'C:/Users/Sneha/Desktop/abc/Output'
#train_data = 'C:/Users/dulam/Desktop/Advanced_CV/Sneha Project/abc/Output'
train_data = 'C:/Users/dulam/Desktop/Advanced_CV/Sneha_Project/testing'
#train_data = 'C:/Users/dulam/Desktop/Advanced_CV/Sneha_Project/alltrain'
#test_data = 'C:/Users/dulam/Desktop/Advanced_CV/Sneha Project/testing'
test_data = 'C:/Users/dulam/Desktop/Advanced_CV/Sneha_Project/Final_test'
#test_data=  'C:/Users/Sneha/Desktop/samples/testing'
#train_data = "../input/testing_bfr/testing/"
#test_data = "../input/sampletest/sampletest/"
#segment_data = "C:/Users/dulam/Desktop/Advanced_CV/Sneha_Project/real/"
segment_label = "C:/Users/dulam/Desktop/Advanced_CV/Sneha_Project/segimaages/"

tf.random.set_random_seed(1234)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

size = 224
batch_size = 5
X = tf.placeholder(tf.float32 , [None,size,size,1])
#X_reshaped = tf.reshape(X , [-1,size,size,1])
Y = tf.placeholder(tf.float32 , [None , 10])
droupout_prob = tf.placeholder(tf.float32)
X_pred = tf.placeholder(tf.float32 , [None,size,size,3])
Y_true = tf.placeholder(tf.int32 , [None,size,size])
Y_x = tf.placeholder(tf.float32, [None, size, size, 3])


def one_hot_label(img):
    label=str(img.split('.')[0])
    if 'gossiping' in label:
        ohl=[1,0,0,0,0,0,0,0,0,0]
        return ohl
    elif 'isolation' in label:
        ohl=[0,1,0,0,0,0,0,0,0,0]
        return ohl
    elif 'laughing' in label:
        ohl=[0,0,1,0,0,0,0,0,0,0]
        return ohl
    elif 'lp' in label or 'pullinghair' in label:
        ohl=[0,0,0,1,0,0,0,0,0,0]
        return ohl
    elif 'punching' in label:
        ohl=[0,0,0,0,1,0,0,0,0,0]
        return ohl
    elif 'slapping' in label:
        ohl=[0,0,0,0,0,1,0,0,0,0]
        return ohl
    elif 'stabbing' in label:
        ohl=[0,0,0,0,0,0,1,0,0,0]
        return ohl
    elif 'strangle' in label:
        ohl=[0,0,0,0,0,0,0,1,0,0]
        return ohl
    elif '00' in label:
        ohl=[0,0,0,0,0,0,0,0,1,0]
        #print(label)
        return ohl
    else:
        ohl=[0,0,0,0,0,0,0,0,0,1]
        return ohl
    	#print("IMG", img)
    	#print("LABEL", label)


def train_data_with_label():
    train_images=[]
    #print("hi")
    for i in tqdm(os.listdir(train_data)):
        path=os.path.join(train_data,i)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(size,size))
        train_images.append([np.array(img),one_hot_label(i)])
    shuffle(train_images)
    #print("hi")
    #print("\nTraining images:",len(train_images))
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


'''def segmented_train_data_with_label():
    segment_images=[]
    #print("hi")
    for i in tqdm(os.listdir(segment_label)):
        path=os.path.join(segment_label,i)
        path_label = os.path.join(segment_label,i)
        #print(path)
        img=io.imread(path)
        img = cv2.resize(img, (size,size))
        #print(type(img))
        img_label = cv2.imread(path_label)
        img_label = cv2.resize(img_label,(size,size))
        segment_images.append([np.array(img),np.array(img_label)])
    shuffle(segment_images)
    #print("hi")
    #print("\nTraining images:",len(train_images))
    return segment_images'''

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
					'''if g == 2:
						y[i,j] = -1
					else:
						y[i,j] = g'''
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


#training_images = train_data_with_label()
#testing_images = test_data_with_label()
segment_images = segmented_train_data_with_label()

'''tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,size,size,1)
tr_lbl_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,size,size,1)
tst_lbl_data = np.array([i[1] for i in testing_images])'''

tr_segment_data = np.array([i[0] for i in segment_images]).reshape(-1,size,size,3)
tr_segment_lbl = np.array([i[1] for i in segment_images]).reshape(-1,size,size)
tr_segment = np.array([i[2] for i in segment_images]).reshape(-1,size,size,3)

'''def take_only(test_img_batch):

	return test_img_batch[:,:,:,:2]'''

'''def for_display(img):
	print(img.shape)
	z = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
	img_final = np.stack([img[:,:,:,0],img[:,:,:,1],z], axis = -1)
	print("VAMMO", img_final.shape)
	img_final = np.reshape(img_final, (img_final.shape[1], img_final.shape[2], img_final.shape[3]))
	return img_final'''

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
	#kernel4 = [shape[1] * shape[2] * shape[3],500]
	#filter14 = tf.Variable(tf.random_normal(kernel6 , stddev = 0.05))
	filter14 = tf.Variable(initial(kernel6))
	#bias14 = tf.Variable(initial([kernel6[-1]]))
	shapestraight = [-1 , shape[1] * shape[2] * shape[3]]
	layer6_shaping = tf.reshape(layer5 , shapestraight)
	layer6drop = tf.nn.dropout(layer6_shaping , droupout_prob)
	layer6 = tf.layers.batch_normalization(tf.nn.relu(tf.matmul(layer6drop , filter14)), training = train)
	print("Layer-6",layer6.get_shape().as_list())

	#SECOND DENSE LAYER
	#filter15 = tf.Variable(tf.random_normal(kernel7, stddev = 0.05))
	filter15 = tf.Variable(initial(kernel7))
	#bias15 = tf.Variable(initial([kernel7[-1]]))
	layer7 = tf.nn.relu(tf.matmul(layer6, filter15))
	layer7 = tf.layers.batch_normalization(tf.nn.dropout(layer7 , droupout_prob), training = train)
	print("Layer-7",layer7.get_shape().as_list())

	#OUTPUT LAYER
	#filter16 = tf.Variable(tf.random_normal(kernel8 ,stddev = 0.05))
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

	# P.S - I don't think adding the Mutli-grid method before the ASPP layer will improve the accuracy
	# given this is not a ResNet. But, it has been left in the implementation if I change my mind in 
	# the future.

	# Get the input from vgg which is a feature map of 14 by 14 with a depth of 512.
	# How do we use a multi grid system here? With that small of a feature map.
	# What would be the rates that'd be used? 

	initial = tf.contrib.layers.xavier_initializer()
	# Applying Global Average pooling.
	res = tf.reduce_mean(X, [1,2], name = 'global_pool', keepdims = True)
	print("RES", res.get_shape().as_list())
	k = tf.Variable(tf.random_normal([1, 1, 512, depth], stddev = 0.05))
	image_level_features = tf.nn.conv2d(res, k, strides = [1, 1, 1, 1], padding = 'SAME', name = 'resize')
	#image_level_features = tf.contrib.slim.conv2d(res, depth, [1, 1], scope="image_level_conv_1x1", activation_fn=None, weights_initializer = tf.initializers.random_normal(stddev = 0.05))
	res_ = tf.image.resize_bilinear(image_level_features, (tf.shape(X)[1], tf.shape(X)[2]))
	#filter_1 = tf.Variable(tf.random_normal([1, 1, 512, depth], stddev = 0.05))
	filter_1 = tf.Variable(initial([1, 1, 512, depth]))
	res = tf.nn.conv2d(X, filter_1, strides = [1, 1, 1, 1], padding = 'SAME', name = 'conv_1_1')

	filter_2_1 = tf.Variable(initial([3, 3, 512, depth]))
	filter_2_2 = tf.Variable(initial([3, 3, 512, depth]))
	filter_2_3 = tf.Variable(initial([3, 3, 512, depth]))
	#filter_2_1 = tf.Variable(tf.random_normal([3, 3, 512, depth], stddev = 0.05))
	#filter_2_2 = tf.Variable(tf.random_normal([3, 3, 512, depth], stddev = 0.05))
	#filter_2_3 = tf.Variable(tf.random_normal([3, 3, 512, depth], stddev = 0.05))
	
	res1 = tf.nn.conv2d(X, filter_2_1, strides = [1, 1, 1, 1], dilations = [1, rates[0], rates[0], 1], padding = 'SAME', name = 'conv_3_3_1')
	res2 = tf.nn.conv2d(X, filter_2_2, strides = [1, 1, 1, 1], dilations = [1, rates[1], rates[1], 1], padding = 'SAME', name = 'conv_3_3_2')
	res3 = tf.nn.conv2d(X, filter_2_3, strides = [1, 1, 1, 1], dilations = [1, rates[2], rates[2], 1], padding = 'SAME', name = 'conv_3_3_3')

	final = tf.concat((res_, res, res1, res2, res3), axis = 3, name = 'concat')
	#final_w = tf.Variable(tf.random_normal([1, 1, 5 * depth, depth], stddev = 0.05))
	final_w = tf.Variable(initial([1, 1, 5 * depth, depth]))
	final_ = tf.nn.conv2d(final, final_w, strides = [1, 1, 1, 1], padding = 'SAME', name = 'final_conv')

	return final_


def complete_DeepLab(input_img, dropout, N = 3, bn = False):

	# N is the number of output classes and it is 3 in our case. Background being the third class.

	# Get the feature maps from our own VGG16 network and then send them to DeepLab for the ASPP thing.
	# The returned results will be used to create the final predictions.

	initial = tf.contrib.layers.xavier_initializer()

	X = vgg(input_img, droupout_prob = dropout, train = bn, circ = True, circ1 = False, segmentation = True) # change circ to False and circ1 to true for the output stride to be 8 rather than 16.

	result = DeepLab(X)
	#filter_final = tf.Variable(tf.random_normal([1, 1, 256, N], stddev = 0.05))
	filter_final= tf.Variable(initial([1, 1, 256, N]))
	res_ = tf.nn.conv2d(result, filter_final, strides = [1, 1, 1, 1], padding = 'SAME', name = 'conv_final')

	return tf.image.resize_bilinear(res_, input_img.shape[1:-1])

def compute_iou(original, prediction):

	# Write formula for calculating Intersection over Union.

	H, W, N = original.get_shape().as_list()[1:]
	pred = tf.reshape(prediction, [-1, H * W, N])
	orig = tf.reshape(original, [-1, H * W, N])
	intersection = tf.reduce_sum(pred * orig, axis = 2) + 1e-7
	denominator = tf.reduce_sum(pred, axis = 2) + tf.reduce_sum(orig, axis = 2) + 1e-7
	#iou = tf.metrics.mean_iou(tf.argmax(orig, 2), tf.argmax(pred, 2), N)
	iou = tf.reduce_mean(intersection / denominator)

	return iou

'''kernel = [3,3,1,64]
kernel_ = [3,3,64,64]
kernel2 = [3,3,64,128]
kernel2_ = [3,3,128,128]
kernel3 = [3,3,128,256]
kernel3_ = [3,3,256,256]
kernel4 = [3,3,256,512]
kernel5 = [3,3,512,512]
kernel6 = [7 * 7 * 512,4096]
kernel7 = [4096,4096]
kernel8 = [4096 , 10]'''

#droupout_prob = tf.placeholder(tf.float32)
is_train = tf.placeholder(tf.bool)
#train_layer = vgg(X, kernel, kernel_, kernel2, kernel2_, kernel3, kernel3_, kernel4, kernel5, kernel6, kernel7, kernel8, droupout_prob, is_train)
#train_layer = vgg(X, droupout_prob, is_train)

#Y_ = tf.nn.softmax(train_layer)
#correct = tf.equal(tf.argmax(Y,1) , tf.argmax(Y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))

#cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = train_layer)
#optimize = tf.train.MomentumOptimizer(learning_rate = 1e-4, momentum = 0.9).minimize(cost)
#optimize = tf.train.GradientDescentOptimizer(0.000001).minimize(cost)
feature_map_16 = complete_DeepLab(X_pred, dropout = droupout_prob, bn = is_train)
iou = compute_iou(feature_map_16, Y_x)
#shape = tf.shape(iou)
#print("IOU SHAPE", type(iou))
#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#recon_loss = tf.losses.mean_squared_error(labels = Y_true , predictions = feature_map_16)
loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = Y_true, logits = feature_map_16)
#with tf.control_dependencies(update_ops):
    #optimize = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
    #optimize_segment =  tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss_cls)
optimize_segment =  tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss_cls)
#feature_map_16 = complete_DeepLab(X_pred)
#iou = compute_iou(feature_map_16, Y_true)
#shape = tf.shape(iou)
#print("IOU SHAPE", type(iou))

init = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(init)

	# TRAINING PHASE
	'''for ep in range(1):
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
		print("Minimum : ",min(ls))'''

	# TESTING PHASE
	'''print("TESTING PHASE!!")
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
		print(session.run(accuracy , feed_dict = data))'''
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
			'''for i in trainlbl:
				testing.append(np.reshape(i, (-1, 10)))
			testing = np.reshape(testing, (batch_size, 10))'''
			#print(testing.shape)
			#print(trainimg.shape)
			data = {X : np.zeros((batch_size, size, size, 1)), Y : np.zeros((batch_size, 10)) ,X_pred : trainimg , Y_true : trainlbl , Y_x : train_ ,  is_train : False, droupout_prob: 1.0}
			session.run(optimize_segment , feed_dict = data)
			ls.append(session.run(iou , feed_dict = data))
			#print("Loss", session.run(cost, feed_dict = data))
		print("Highest : ", max(ls))
		print("Minimum : ",min(ls))

	'''print("SEGMENTATION TESTING PHASE")
	for _ in range(1):
		img = ip.imread('000045.jpg')
		print(img.shape)
		img = cv2.resize(img,(size,size))
		img = np.reshape(img, (1, size, size, 3))
		img_true = ip.imread('2.png')
		img_true = cv2.resize(img_true,(size,size))
		img_true = np.reshape(img_true[:,:,:-1], (1, size, size, 3))
		data = {X : np.zeros((batch_size, size, size, 1)), Y : np.zeros((batch_size, 10)), X_pred : img, Y_true : img_true[:,:,:,:], droupout_prob : 1.0, is_train : False}
		#session.run(optimize_, feed_dict = data)
		print(session.run(iou, feed_dict = data))
		o = session.run(feature_map_16, feed_dict = {X_pred : img, droupout_prob : 1.0, is_train : False})
		o = np.reshape(o, (o.shape[1], o.shape[2], o.shape[3]))
		print(o.shape)
		ip.imshow(o)
		plt.show()'''
