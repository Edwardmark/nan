# define the mobile neural network for neural aggregation network
import tensorflow as tf 
from utils import *

def inference(images, is_train, num_classes):
    input_shape = tf.shape(images)
    input_imgs = tf.cast(images, tf.float32)
    parameters = []

    name = "conv1_s2"
    weight_shape = [3, 3, 3, 32]
    conv1, param = cusConv(
                    input_imgs, weight_shape = weight_shape,
                    name = name, is_train = is_train, need_relu = True, need_bn = True, downsample = True)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, input_imgs.get_shape().value, weight_shape, conv1.get_shape().value)
    print 'downsample at %s' % name

    name = "conv2_dsc_s1"
    weight_shape = [3, 3, 32, 64]
    conv2, param = depth_sep_conv(
                    conv1, weight_shape,
                    name, is_train = is_train, need_relu = True, downsample = False)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv1.get_shape().value, weight_shape, conv2.get_shape().value)
    
    name = "conv3_dsc_s2"
    weight_shape = [3, 3, 64, 128]
    conv3, param = depth_sep_conv(
                    conv2, weight_shape,
                    name, is_train = is_train, need_relu = True, downsample = True)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv2.get_shape().value, weight_shape, conv3.get_shape().value)
    print 'downsample at %s' % name

    name = "conv4_dsc_s1"
    weight_shape = [3, 3, 128, 128]
    conv4, param = depth_sep_conv(
                    conv3, weight_shape,
                    name, is_train = is_train, need_relu = True, downsample = False)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv3.get_shape().value, weight_shape, conv4.get_shape().value)

    name = "conv5_dsc_s2"
    weight_shape = [3, 3, 128, 256]
    conv5, param = depth_sep_conv(
                    conv4, weight_shape,
                    name, is_train = is_train, need_relu = True, downsample = True)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv4.get_shape().value, weight_shape, conv5.get_shape().value)
    print 'downsample at %s' % name

    name = "conv6_dsc_s1"
    weight_shape = [3, 3, 256, 256]
    conv6, param = depth_sep_conv(
                    conv5, weight_shape,
                    name, is_train = is_train, need_relu = True, downsample = False)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv5.get_shape().value, weight_shape, conv6.get_shape().value)

    name = "conv7_dsc_s2"
    weight_shape = [3, 3, 256, 512]
    conv7, param = depth_sep_conv(
                    conv6, weight_shape,
                    name, is_train = is_train, need_relu = True, downsample = True)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv6.get_shape().value, weight_shape, conv7.get_shape().value)
    print 'downsample at %s' % name

    name = "conv8_1_dsc_s1"
    weight_shape = [3, 3, 512, 512]
    conv8_1, param = depth_sep_conv(
                    conv7, weight_shape,
                    name, is_train = is_train, need_relu = True, downsample = False)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv7.get_shape().value, weight_shape, conv8.get_shape().value)

    name = "conv8_2_dsc_s1"
    weight_shape = [3, 3, 512, 512]
    conv8_2, param = depth_sep_conv(
                    conv8_1, weight_shape,
                    name, is_train = is_train, need_relu = True, downsample = False)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv8_1.get_shape().value, weight_shape, conv8_2.get_shape().value)

    name = "conv8_3_dsc_s1"
    weight_shape = [3, 3, 512, 512]
    conv8_3, param = depth_sep_conv(
                    conv8_2, weight_shape,
                    name, is_train = is_train, need_relu = True, downsample = False)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv8_2.get_shape().value, weight_shape, conv8_3.get_shape().value)

    name = "conv8_4_dsc_s1"
    weight_shape = [3, 3, 512, 512]
    conv8_4, param = depth_sep_conv(
                    conv8_3, weight_shape,
                    name, is_train = is_train, need_relu = True, downsample = False)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv8_3.get_shape().value, weight_shape, conv8_4.get_shape().value)
    
    name = "conv8_5_dsc_s1"
    weight_shape = [3, 3, 512, 512]
    conv8_5, param = depth_sep_conv(
                    conv8_4, weight_shape,
                    name, is_train = is_train, need_relu = True, downsample = False)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv8_4.get_shape().value, weight_shape, conv8_5.get_shape().value)

    name = "conv9_dsc_s2"
    weight_shape = [3, 3, 512, 1024]
    conv9, param = depth_sep_conv(
                    conv8_5, weight_shape,
                    name, is_train = is_train, need_relu = True, downsample = False)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv8_5.get_shape().value, weight_shape, conv9.get_shape().value)
    print 'downsample at %s' % name

    name = "conv10_dsc_s1"
    weight_shape = [3, 3, 1024, 1024]
    conv10, param = depth_sep_conv(
                    conv9, weight_shape,
                    name, is_train = is_train, need_relu = True, downsample = False)
    parameters.append(param)
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv9.get_shape().value, weight_shape, conv10.get_shape().value)
    
    name = "avg_pool_11"
    weight_shape = [1, conv10.get_shape()[1].value, conv10.get_shape()[2].value, 1]
    pool11 = tf.nn.avg_pool(conv10, ksize = weight_shape, strides = [1, 1, 1, 1], padding= "SAME")
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, conv10.get_shape().value, weight_shape, pool11.get_shape().value)

    name = "feature"
    feature = tf.contrib.layers.flatten(pool11, scope = 'pool11_flat')
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, pool11.get_shape().value, weight_shape, feature.get_shape().value)

    name = "logits"
    weight_shape = [1024, num_classes]
    logits = fully_connected(feature, num_classes, is_train = is_train, activation=lambda x: x, scope='fc')
    print "layer:%s, input_shape:%s, weight_shape:%s, output_shape:%s" % \
            (name, feature.get_shape().value, weight_shape, logits.get_shape().value)
    
    return feature, logits, parameters

if __name__ == '__main__':
    import cv2
    images = tf.convert_to_tensor(cv2.imread('test_image'),dtype = tf.float32)
    images = tf.expand_dims(images,axis = 0)
    is_train = tf.placeholder(tf.bool)
    feature, logits, _ = inference(images, is_train, num_classes = 10)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feature, logits = sess.run([feature, logits], feed_dict = {images: images, is_train: True})
        print feature.shape
        print feature
        print logits.shape
        print logits
