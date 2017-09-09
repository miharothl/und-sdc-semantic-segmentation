import os.path
import tensorflow as tf
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

    fcn_env_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes,
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

    fcn_dev_layer3_skip = tf.add(fcn_dec_layer4_upscale, fcn_env_layer3)

    output = tf.layers.conv2d_transpose(fcn_dev_layer3_skip, num_classes,
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
             correct_label, keep_prob, learning_rate):
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

    for epoch in range(epochs):
        for image, image_gt in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                input_image: image,
                correct_label: image_gt,
                keep_prob: 0.5,
                learning_rate: 0.0001 #/(1+epoch/2)
            })

            print("EPOCH: {}/{} Loss: {}".format(epoch, epochs, loss))

    pass
tests.test_train_nn(train_nn)





def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    epochs = 3 # 5
    batch_size = 4

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

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

        # TODO: Build NN using load_vgg, layers, and optimize function
        label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)

        vgg_input, vgg_keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)
        fcn_output = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)
        logits, train_op, cross_entropy_loss = optimize(fcn_output, label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, vgg_input, label,
                 vgg_keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
