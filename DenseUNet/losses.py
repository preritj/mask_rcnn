import tensorflow as tf


def reg_loss(gt_regs, pred_regs, labels):
    gt_regs = tf.reshape(gt_regs, [-1, 4])
    pred_regs = tf.reshape(pred_regs, [-1, 4])
    weights = tf.cast(labels, tf.float32)
    weights = tf.maximum(weights, 0.)
    weights = tf.reshape(weights, [-1])
    d = tf.losses.huber_loss(labels=gt_regs, predictions=pred_regs,
                             delta=1., reduction=tf.losses.Reduction.NONE)
    # d = gt_regs - pred_regs
    # d = d * d
    loss = tf.reduce_sum(d, axis=1)
    loss = tf.reduce_sum(loss * weights)
    w_sum = tf.reduce_sum(weights)
    loss = tf.cond(tf.less(0.0, w_sum), lambda: loss / w_sum, lambda: 0.)
    return loss


def cross_entropy_loss(logits, labels, weights):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=labels)
    w_sum = tf.reduce_sum(weights)
    loss = tf.reduce_sum(loss * weights)
    loss = tf.cond(tf.less(0.0, w_sum), lambda: loss / w_sum, lambda: 0.)
    return loss


def mask_loss(mask_logits, mask_labels):
    mask_labels = tf.cast(mask_labels, tf.int32)
    mask_labels = tf.reshape(mask_labels, [-1])
    sem_labels = tf.minimum(mask_labels, 1)
    dir_labels = tf.maximum(mask_labels - 1, 0)
    n_labels = mask_logits.get_shape().as_list()[-1]
    mask_logits = tf.reshape(mask_logits, [-1, n_labels])
    bkg_logits, obj_logits = tf.split(mask_logits, [1, n_labels - 1], 1)
    dir_seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=obj_logits,
                                                                  labels=dir_labels)
    dir_seg_loss = tf.cast(sem_labels, tf.float32) * dir_seg_loss
    bkg_logits = tf.squeeze(bkg_logits)
    obj_logits = tf.reduce_logsumexp(obj_logits, 1)
    sem_logits = tf.stack([bkg_logits, obj_logits], axis=1)
    sem_seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sem_logits,
                                                                  labels=sem_labels)
    return sem_seg_loss, dir_seg_loss
