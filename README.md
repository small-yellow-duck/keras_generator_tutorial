# keras_generator_tutorial
tutorial on using fit_generator and predict_generator in keras with data either

1) in CPU memory 
2) on the HD

First, you need to download the mnist data and split it into directories of test and
train images

from the command line run ipython:
>>>ipython --pylab

compile this script
>>>run mnist_split_to_test_and_train.py

run the do split function
>>>do_split()


To see how long an epoch takes when the data needs to be read from files on the hard drive:

from the command line run ipython:
>>>ipython --pylab

in ipython:
>>>run mnist_cnn_generator_data_on_hd.py
>>>pred, Y_test = fit()

GTX 760: 1 epoch takes 12 seconds



To see how long an epoch takes when the data fits in CPU memory:

from the command line run ipython:
>>>ipython --pylab

in ipython:
>>>run mnist_cnn_generator_data_in_memory.py
>>>pred, Y_test = fit()

GTX 760: 1 epoch takes 4 seconds
'''