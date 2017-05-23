# this file is the utility python files which defines all the ops used in the MATA(Mobile Attention-based Template Adaptation) network
import os
import tensorflow as tf
import numpy as np

def distorted_inputs(list_path, image_dir, re_size, crop_size, batch_size, is_color):    

    channel = 3 if is_color else 1

    with open(list_path,'r') as f:
        lines = f.readlines()
    print 'content has %d lines' % len(lines)

    content = [ image_dir + line.strip() for line in lines ]   
    value_queue = tf.train.string_input_producer(content,shuffle=True)  
    value = value_queue.dequeue()
 
    dir, label = tf.decode_csv(records=value, 
                               record_defaults=[tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.string)],
                               field_delim = ',')  
    label = tf.string_to_number(label, tf.int32)

    imagecontent = tf.read_file(dir)  
    image = tf.image.decode_jpeg(imagecontent)
    image = tf.image.resize_images(image, [re_size, re_size])
    image = tf.random_crop(image, [crop_size, crop_size, channel])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_standardization(image)
    
    image.set_shape([224, 224, channel])
    images, labels = tf.train.shuffle_batch(
                                    [image, label], batch_size = batch_size, num_threads = 6,  
                                    capacity = 10 * batch_size , 
                                    min_after_dequeue = 2 * batch_size)
    return images, labels

def inputs(list_path, image_dir, re_size, crop_size, batch_size, is_color=False):

    channel = 3 if is_color else 1

    with open(list_path,'r') as f:
        lines = f.readlines()
    print 'content has %d lines' % len(lines)

    content = [ image_dir + line.strip() for line in lines ]   
    value_queue = tf.train.string_input_producer(content,shuffle=True)  
    value = value_queue.dequeue()
 
    dir, label = tf.decode_csv(records=value, 
                               record_defaults=[tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.string)],
                               field_delim = ',')  
    label = tf.string_to_number(label, tf.int32)

    imagecontent = tf.read_file(dir)  
    image = tf.image.decode_jpeg(imagecontent)
    image = tf.image.resize_images(image, [re_size, re_size])
    image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
    image = tf.image.per_image_standardization(image)

    image.set_shape([crop_size, crop_size, channel])

    images, labels = tf.train.batch([image, label],batch_size=batch_size)

    return images, labels


def batch_norm(x, phase_train, scope='bn', affine=True):
    """
    Batch normalization on convolutional maps.
    from: https://stackoverflow.com/questions/33949786/how-could-i-
    use-batch-normalization-in-tensorflow
    Only modified to infer shape from input tensor x.
    Parameters
    ----------
    x
        Tensor, 4D BHWD input maps
    phase_train
        boolean tf.Variable, true indicates training phase
    scope
        string, variable scope
    affine
        whether to affine-transform outputs
    Return
    ------
    normed
        batch-normalized maps
    """

    with tf.variable_scope(scope):
        og_shape = x.get_shape().as_list()
        if len(og_shape) == 2:
            x = tf.reshape(x, [-1, 1, 1, og_shape[1]])
        shape = x.get_shape().as_list()
        beta = tf.Variable(tf.constant(0.0, shape=[shape[-1]]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[shape[-1]]),
                            name='gamma', trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            """Summary
            Returns
            -------
            name : TYPE
                Description
            """
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train,
                                          mean_var_with_update,
                                          lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(
            x, mean, var, beta, gamma, 1e-3, affine)
        if len(og_shape) == 2:
            normed = tf.reshape(normed, [-1, og_shape[-1]])
    return normed

## define the custom convolutional layer,including conv,bias,relu,and regular term
def cusConv(input_imgs, weight_shape, name, is_train, 
	        stddev=None, need_dropout = False,
            W_regular=0, need_relu = True, need_bn = False,
            downsample = False, reuse=None):
    _stride = 2 if downsample else 1
    if stddev is None:
        print name, weight_shape
        fan_in = np.prod(weight_shape[:3])
        stddev = np.sqrt(2. / fan_in) #xavier initialization
    with tf.name_scope(name) as scope: # using this tf.name_scope, can draw nice tensorboard graph,make sure it is using the same graph

        weights = tf.Variable(
            tf.truncated_normal(weight_shape, dtype=tf.float32, stddev=stddev),
            name='weights')
        conv = tf.nn.conv2d(input_imgs, # with NHWC order
                            weights,  # with HWIO oreder
                            strides = [1, _stride, _stride, 1], # strides[0] must equal strides[3]
                            padding='SAME') # out_height = ceil(float(in_height)/stride[1])
        biases = tf.Variable(tf.constant(0.0, shape=[weight_shape[3]], # shape = out_channels
                                         dtype=tf.float32),
                             name='biases')
        tmp_imgs = tf.nn.bias_add(conv, biases)

        if W_regular > 0:
            weight_decay = tf.nn.l2_loss(weights) * W_regular # add normalization term
            tf.add_to_collection("losses", weight_decay)

        if need_bn:
            tmp_imgs = batch_norm(tmp_imgs, is_train)
        if need_relu:
            tmp_imgs = tf.nn.relu(tmp_imgs, name=scope)
        if need_dropout:
            tmp_imgs = tf.nn.dropout(tmp_imgs, keep_prob=0.618)
        param = [weights, biases]
    return tmp_imgs, param

# define depthwise separable convolution
# pay attention that now we add batch norm layer regardless of speed concern
def depth_sep_conv( input_imgs, weight_shape, name, is_train, 
	                downsample = False, W_regular=0, need_relu = True, 
	                width_multiplier = 1, reuse=None):

    _stride = 2 if downsample else 1

    # this weight shape is the overall standard weight shape, [Dk, Dk, M, N]
    print name, weight_shape 
    
    # multiply width multiplier
    weight_shape = [weight_shape[0], weight_shape[1], weight_shape[2] * width_multiplier, weight_shape[3] * width_multiplier]

    # initialize the depthwise-convolution
    dwc_filter_shape = [weight_shape[0], weight_shape[1], weight_shape[2], 1]
    print name + '-depthwise-conv',
    print dwc_filter_shape
    dwc_stddev = np.sqrt(2. / np.prod(dwc_filter_shape[:3]))
    
    # initialize the pointwise-convolution
    pwc_filter_shape = [1, 1, weight_shape[2], weight_shape[3]]
    print name + '-pointwise-conv',
    print pwc_filter_shape
    pwc_stddev = np.sqrt(2. / np.prod(pwc_filter_shape[:3]))

    with tf.name_scope(name) as scope: # using this tf.name_scope, can draw nice tensorboard graph,make sure it is using the same graph

        # first do depthwise convolution
        dwc_filter = tf.Variable(
            tf.truncated_normal(dwc_filter_shape, dtype=tf.float32, stddev=dwc_stddev),
            name='dwc_filter')
        dwc = tf.nn.depthwise_conv2d(input_imgs, dwc_filter, strides=[1, _stride, _stride, 1], padding="SAME") # depth_wise convolution

        dwc_biases = tf.Variable(tf.constant(0.0, shape=[dwc_filter_shape[2]], # shape = in_channels 
                                         dtype=tf.float32),
                             name='dwc_biases')
        tmp_imgs = tf.nn.bias_add(dwc, dwc_biases)
        # batch norm it
        tmp_imgs = batch_norm(tmp_imgs, is_train)

        if W_regular > 0:
            dwc_weight_decay = tf.nn.l2_loss(dwc_filter) * W_regular # add normalization term
            tf.add_to_collection("losses", dwc_weight_decay)

        if need_relu:
            tmp_imgs = tf.nn.relu(tmp_imgs, name='relu_dwc')

        
        # then do pointwise convolution
        pwc_filter = tf.Variable(
            tf.truncated_normal(pwc_filter_shape, dtype=tf.float32, stddev=pwc_stddev),
            name='pwc_filter')
        pwc = tf.nn.conv2d(tmp_imgs, # with NHWC order
                           pwc_filter,  # with HWIO oreder
                           strides = [1, 1, 1, 1], # strides[0] must equal strides[3]
                           padding='SAME') # out_height = ceil(float(in_height)/stride[1])
        pwc_biases = tf.Variable(tf.constant(0.0, shape=[weight_shape[3]], # shape = out_channels
                                         dtype=tf.float32),
                             name='pwc_biases')
        tmp_imgs = tf.nn.bias_add(pwc, pwc_biases)
        # batch norm it
        tmp_imgs = batch_norm(tmp_imgs, is_train)

        if W_regular > 0:
            pwc_weight_decay = tf.nn.l2_loss(pwc_filter) * W_regular # add normalization term
            tf.add_to_collection("losses", pwc_weight_decay)

        if need_relu:
            tmp_imgs = tf.nn.relu(tmp_imgs, name='relu_pwc')
        param = [dwc_filter, dwc_biases, pwc_filter, pwc_biases]

    return tmp_imgs, param

## define the custom fully connected layer
def cusFc(input_imgs, weight_shape, name, is_train, need_bn = False,
          need_relu=True, W_regular=0.0005, stddev=None, reuse=None):
    if stddev is None:
        stddev = np.sqrt(2. / weight_shape[0]) # xarvier initialization
    with tf.name_scope(name) as scope:
        fc_weights = tf.Variable(tf.truncated_normal(
            weight_shape, stddev=stddev, dtype=tf.float32), name="weights")
        fc_biases = tf.Variable(tf.constant(0.0, shape=[weight_shape[-1]],
                                            dtype=tf.float32), name="biases")
        if W_regular > 0:
            weight_decay = tf.nn.l2_loss(fc_weights) * W_regular
            tf.add_to_collection("losses", weight_decay)

        fc = tf.add(tf.matmul(input_imgs, fc_weights), fc_biases, name="add")

        if need_bn:
        	fc = batch_norm(fc, is_train, name = "bn")
        if need_relu:
            fc = tf.nn.relu(fc,name = "relu")
        if need_dropout:
        	fc = tf.layers.dropout(fc, training = is_train, name = "dropout")
        param = [fc_weights, fc_biases]
    return fc, param

def get_deconv_filter(f_shape):
    width = f_shape[0]
    heigh = f_shape[0]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="up_filter", initializer=init,
                          shape=weights.shape)
    return var

def _upscore_layer(bottom, shape,
                   num_classes, name,
                   ksize = 4, stride = 2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value ## what is get_shape().val? get the out_channels of bottom

        if shape is None:
            # Compute shape of the Bottom
            in_shape = tf.shape(bottom) # get the shape of this layer

            h = ((in_shape[1] - 1) * stride) + 1 # why not h = (h_in - 1) * stride + k OR h = H_in * stride??????
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.stack(new_shape) # combined as a 1-D tensor

        f_shape = [ksize, ksize, num_classes, in_features] # filter shape, i.e. W shape used in deconv

        # create
        weights = get_deconv_filter(f_shape) # why not using random initialization???
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')
                                        # value: A 4-D Tensor of type float and shape [batch, height, width, in_channels]
                                        # filter: A 4-D Tensor with the same type as value and shape [height, width, output_channels, in_channels]
                                        # filter's in_channels dimension must match that of value
    # print deconv.get_shape()
    return deconv, (weights,)



