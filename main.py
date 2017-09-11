import os.path
import tensorflow as tf
import time
import numpy as np
import scipy.misc

import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

print("Initializing...")

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'


    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    fcn_enc_input = graph.get_tensor_by_name(vgg_input_tensor_name)

    fcn_enc_keep_prob = graph.get_tensor_by_name (vgg_keep_prob_tensor_name)

    fcn_enc_layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)

    fcn_enc_layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)

    fcn_enc_layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return fcn_enc_input, fcn_enc_keep_prob, fcn_enc_layer3, fcn_enc_layer4, fcn_enc_layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    hyper_l2_reg = 1e-3
    hyper_init = 0.01

    fcn_enc_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes,
                                      kernel_size=1,
                                      strides=(1, 1),
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=hyper_init),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(hyper_l2_reg))

    fcn_enc_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes,
                                      kernel_size=1,
                                      strides=(1, 1),
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=hyper_init),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(hyper_l2_reg))

    fcn_enc_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes,
                                      kernel_size=1,
                                      strides=(1, 1),
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=hyper_init),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(hyper_l2_reg))

    fcn_dec_layer7_upscale = tf.layers.conv2d_transpose(fcn_enc_layer7,
                                      num_classes,
                                      kernel_size=(4, 4),
                                      strides=(2, 2),
                                      padding='SAME',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=hyper_init),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(hyper_l2_reg))

    fcn_dec_layer4_skip = tf.add(fcn_dec_layer7_upscale, fcn_enc_layer4)

    fcn_dec_layer4_skip = tf.layers.dropout(fcn_dec_layer4_skip, )

    fcn_dec_layer4_upscale = tf.layers.conv2d_transpose(fcn_dec_layer4_skip,
                                      num_classes,
                                      kernel_size=(4, 4),
                                      strides=(2, 2),
                                      padding='SAME',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=hyper_init),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(hyper_l2_reg))

    fcn_dec_layer3_skip = tf.add(fcn_dec_layer4_upscale, fcn_enc_layer3)

    output = tf.layers.conv2d_transpose(fcn_dec_layer3_skip, num_classes,
                                      kernel_size=(16, 16),
                                      strides=(8, 8),
                                      padding='SAME',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=hyper_init),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(hyper_l2_reg))

    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, [-1, num_classes])
    labels = tf.reshape(correct_label, [-1, num_classes])

    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_operation = optimizer.minimize(loss)

    return logits, train_operation, loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, models_dir, training_timestamp, unit_test = False):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    print("Training...")

    model_saved = False


    for i in range(epochs):

        batch = 0

        for image, gt_image, image_val, gt_image_val in get_batches_fn(batch_size):
            batch += 1
            start_time = time.time()

            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                input_image: image,
                correct_label: gt_image,
                keep_prob: 0.5,
                learning_rate: 0.0001 #/(1+epoch/2)
            })

            stop_time = time.time()

            print("Epoch {}/{} batch {} {}s - loss: {}".format(i+1,
                                                               epochs,
                                                               batch,
                                                               int((stop_time - start_time)),
                                                               loss))

        if not unit_test:
            if (i+1) % 3 == 0:
                helper.save_trained_model(models_dir, training_timestamp, sess, epoch=i+1)
                model_saved = True

    if not unit_test:
        if not model_saved:
            helper.save_trained_model(models_dir, training_timestamp, sess, epoch=i+1)

    pass

tests.test_train_nn(train_nn)


GLOBAL = 1
IMAGE_SHAPE = None
KEEP_PROB = None
LOGITS = None
SESSION = None
VGG_INPUT = None

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    models_dir = './models'
    tests.test_for_kitti_dataset(data_dir)

    epochs = 12 # 5
    batch_size = 15

    training_timestamp = str(time.time())

    # Download pretrained vgg model

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)

        vgg_input, vgg_keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)
        fcn_output = layers(vgg_layer3,
                            vgg_layer4,
                            vgg_layer7,
                            num_classes)

        logits, train_op, cross_entropy_loss = optimize(fcn_output, label, learning_rate, num_classes)

        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())

        if 0:
            train_nn(sess,
                 epochs,
                 batch_size,
                 get_batches_fn,
                 train_op,
                 cross_entropy_loss,
                 vgg_input,
                 label,
                 vgg_keep_prob,
                 learning_rate,
                 models_dir,
                 training_timestamp,
                 )

            # Save inference data using helper.save_inference_samples
            if 0:
                helper.save_inference_samples(runs_dir,
                                      training_timestamp,
                                      data_dir,
                                      sess,
                                      image_shape,
                                      logits,
                                      vgg_keep_prob,
                                      vgg_input,
                                      )

        # Apply the trained model to a video
        if 1:

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()

                model_file = "./models/1505134276.3176534/20-epoch"
                
                print("Restoring model...")
                saver.restore(sess, model_file)
                print("Model restored.")

                from moviepy.video.io.VideoFileClip import VideoFileClip

                global IMAGE_SHAPE
                IMAGE_SHAPE = image_shape

                global KEEP_PROB
                KEEP_PROB = vgg_keep_prob

                global LOGITS
                LOGITS = logits

                global SESSION
                SESSION = sess

                global VGG_INPUT
                VGG_INPUT = vgg_input

                clip = VideoFileClip("video/drive_3.mp4")
                white_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
                white_clip.write_videofile("video/drive_3_out.mp4", audio=False)


def process_image(image):

    sess = SESSION
    logits = LOGITS
    image_shape = IMAGE_SHAPE

    height = image.shape[0]

    crop_top = np.math.floor(height*0.25)
    crop_bottom = height - np.math.floor(height*0.25)

    image = image[crop_top:crop_bottom, :]
    image = scipy.misc.imresize(image, IMAGE_SHAPE)

    keep_prob = KEEP_PROB
    vgg_input = VGG_INPUT

    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, vgg_input : [image]})

    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    return np.array(street_im)

    return img


if __name__ == '__main__':
    run()
