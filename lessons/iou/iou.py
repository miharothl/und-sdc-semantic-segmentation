import tensorflow as tf


def mean_iou(ground_truth, prediction, num_classes):
    # TODO: Use `tf.metrics.mean_iou` to compute the mean IoU.
    iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
    return iou, iou_op


ground_truth = tf.constant([
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]], dtype=tf.float32)
prediction = tf.constant([
    [0, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 2, 2, 1],
    [3, 3, 0, 3]], dtype=tf.float32)

gt = tf.placeholder(tf.float32, (4,4))
pred = tf.placeholder(tf.float32, (4,4))

# TODO: use `mean_iou` to compute the mean IoU
iou, iou_op = mean_iou(ground_truth, prediction, 4)

iou2, iou_op2 = mean_iou(gt, pred, 4)



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    # need to initialize local variables for this to run `tf.metrics.mean_iou`
    sess.run(tf.local_variables_initializer())

    pred_val = sess.run(prediction)

    a = [[0, 0, 0, 0],
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]]

    b = [
    [0, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 2, 2, 1],
    [3, 3, 0, 3]]
    pred_val, gt_val, iou_op2_val, iou2_val = sess.run([prediction, gt, iou_op2, iou2], feed_dict={
        gt:a,
        pred:b,
    })


    bbb = sess.run(iou2)




    iou_op_val = sess.run(iou_op)
    # should be 0.53869
    print("Mean IoU =", sess.run(iou))