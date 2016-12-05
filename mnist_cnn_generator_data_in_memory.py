'''
Train a simple convnet on the MNIST dataset using a generator

from the command line run ipython:
>>>ipython --pylab

in ipython:
>>>run mnist_cnn_generator_data_in_memory.py
>>>pred, Y_test = fit()

GTX 760: 1 epoch takes 4 seconds
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.optimizers import Adam



#list(myGenerator(X_train, y_train, batch_size, fnames_train))[0]
def myGenerator(X_train, y, batch_size):

	order = np.arange(X_train.shape[0])

	while True:
		
		if not y is None:
			np.random.shuffle(order)	
			X_train = X_train[order]	
			y = y[order]	
		
		for i in xrange(np.ceil(1.0*X_train.shape[0]/batch_size).astype(int)):
			
			#training set
			if not y is None:	
				yield X_train[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]
				
			#test set
			else:
				yield X_train[i*batch_size:(i+1)*batch_size]
					
					



#pred, Y_test = fit()
def fit():
	batch_size = 128
	nb_epoch = 15

	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train.reshape((X_train.shape[0],) + (1,) + X_train.shape[1:]).astype(np.float)
	X_test = X_test.reshape((X_test.shape[0],) + (1,) + X_test.shape[1:]).astype(np.float)
	
	#normalize
	X_train /= 255.0
	X_test /= 255.0
	
	# input image dimensions
	img_rows, img_cols = 28, 28
	# number of convolutional filters to use
	nb_filters = 32
	# size of pooling area for max pooling
	nb_pool = 2
	# convolution kernel size
	nb_conv = 3
	

	#load all the labels for the train and test sets
	#y_train = np.loadtxt('labels_train.csv')
	#y_test = np.loadtxt('labels_test.csv')
	
	#fnames_train = np.array(['train/train'+str(i)+'.png' for i in xrange(len(y_train))])
	#fnames_test = np.array(['test/test'+str(i)+'.png' for i in xrange(len(y_test))])
	
	nb_classes = len(np.unique(y_train))

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train.astype(int), nb_classes)
	Y_test = np_utils.to_categorical(y_test.astype(int), nb_classes)

	model = Sequential()

	
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
							border_mode='valid', init='he_normal',
							input_shape=(1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.1))	
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid',init='he_normal'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.1))
	

	model.add(Flatten())
	model.add(Dense(32, init='he_normal'))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))	
	model.add(Dense(nb_classes, init='he_normal'))
	model.add(Activation('softmax'))

	optimizer = Adam(lr=1e-3)
	model.compile(loss='categorical_crossentropy',
				  optimizer=optimizer,
				  metrics=['accuracy'])

	model.fit_generator(myGenerator(X_train, Y_train, batch_size), samples_per_epoch = Y_train.shape[0], nb_epoch = nb_epoch, verbose=1,callbacks=[], validation_data=None) # show_accuracy=True, nb_worker=1 
		  

	pred = model.predict_generator(myGenerator(X_test, None, batch_size), X_test.shape[0]) # show_accuracy=True, nb_worker=1 

	#score = model.evaluate(X_test, Y_test, verbose=0)
	#print('Test score:', score[0])
	#print('Test accuracy:', score[1])	
	print( 'Test accuracy:', np.mean(np.argmax(pred, axis=1) == np.argmax(Y_test, axis=1)) )
	
	return pred, Y_test	
		  

