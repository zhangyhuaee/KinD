import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers

def lrelu(x, trainbable=None):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable= True)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1], name=scope_name)

        deconv_output =  tf.concat([deconv, x2],3)
        deconv_output.set_shape([None, None, None, output_channels*2])

        return deconv_output

def DecomNet_simple(input):
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        pool1=slim.max_pool2d(conv1, [2, 2], stride = 2, padding='SAME' )
        conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        pool2=slim.max_pool2d(conv2, [2, 2], stride = 2, padding='SAME' )
        conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
        up8 =  upsample_and_concat( conv3, conv2, 64, 128 , 'g_up_1')
        conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
        up9 =  upsample_and_concat( conv8, conv1, 32, 64 , 'g_up_2')
        conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
        # Here, we use 1*1 kernel to replace the 3*3 ones in the paper to get better results.
        conv10=slim.conv2d(conv9,3,[1,1], rate=1, activation_fn=None, scope='g_conv10')
        R_out = tf.sigmoid(conv10)

        l_conv2=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='l_conv1_2')
        l_conv3=tf.concat([l_conv2, conv9],3)
        # Here, we use 1*1 kernel to replace the 3*3 ones in the paper to get better results.
        l_conv4=slim.conv2d(l_conv3,1,[1,1], rate=1, activation_fn=None,scope='l_conv1_4')
        L_out = tf.sigmoid(l_conv4)

    return R_out, L_out

def Restoration_net(input_r, input_i):
    with tf.variable_scope('Restoration_net', reuse=tf.AUTO_REUSE):
        input_all = tf.concat([input_r,input_i], 3)
        
        conv1=slim.conv2d(input_all,32,[3,3], rate=1, activation_fn=lrelu,scope='de_conv1_1')
        conv1=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='de_conv1_2')
        pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

        conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='de_conv2_1')
        conv2=slim.conv2d(conv2,64,[3,3], rate=1, activation_fn=lrelu,scope='de_conv2_2')
        pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

        conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='de_conv3_1')
        conv3=slim.conv2d(conv3,128,[3,3], rate=1, activation_fn=lrelu,scope='de_conv3_2')
        pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

        conv4=slim.conv2d(pool3,256,[3,3], rate=1, activation_fn=lrelu,scope='de_conv4_1')
        conv4=slim.conv2d(conv4,256,[3,3], rate=1, activation_fn=lrelu,scope='de_conv4_2')
        pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

        conv5=slim.conv2d(pool4,512,[3,3], rate=1, activation_fn=lrelu,scope='de_conv5_1')
        conv5=slim.conv2d(conv5,512,[3,3], rate=1, activation_fn=lrelu,scope='de_conv5_2')

        up6 =  upsample_and_concat( conv5, conv4, 256, 512, 'up_6')

        conv6=slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=lrelu,scope='de_conv6_1')
        conv6=slim.conv2d(conv6,256,[3,3], rate=1, activation_fn=lrelu,scope='de_conv6_2')

        up7 =  upsample_and_concat( conv6, conv3, 128, 256, 'up_7'  )
        conv7=slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=lrelu,scope='de_conv7_1')
        conv7=slim.conv2d(conv7,128,[3,3], rate=1, activation_fn=lrelu,scope='de_conv7_2')

        up8 =  upsample_and_concat( conv7, conv2, 64, 128, 'up_8' )
        conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='de_conv8_1')
        conv8=slim.conv2d(conv8,64,[3,3], rate=1, activation_fn=lrelu,scope='de_conv8_2')

        up9 =  upsample_and_concat( conv8, conv1, 32, 64, 'up_9' )
        conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='de_conv9_1')
        conv9=slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=lrelu,scope='de_conv9_2')

        conv10=slim.conv2d(conv9,3,[3,3], rate=1, activation_fn=None, scope='de_conv10')
    
        out = tf.sigmoid(conv10)
        return out

def Illumination_adjust_net(input_i, input_ratio):
    with tf.variable_scope('Illumination_adjust_net', reuse=tf.AUTO_REUSE):
        input_all = tf.concat([input_i, input_ratio], 3)
        
        conv1=slim.conv2d(input_all,32,[3,3], rate=1, activation_fn=lrelu,scope='en_conv_1')
        conv2=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='en_conv_2')
        conv3=slim.conv2d(conv2,32,[3,3], rate=1, activation_fn=lrelu,scope='en_conv_3')
        conv4=slim.conv2d(conv3,1,[3,3], rate=1, activation_fn=lrelu,scope='en_conv_4')

        L_enhance = tf.sigmoid(conv4)
    return L_enhance