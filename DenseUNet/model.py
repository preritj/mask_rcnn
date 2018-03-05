import os
import numpy as np
import tensorflow as tf
import losses
from bbox import generate_anchors, label_anchors
from bbox import param_bboxes, unparam_bbox
from data import ImageLoader
import sys


class RCNN(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.input_shape = cfg.input_shape
        self.anchor_scales = cfg.anchor_scales
        self.anchor_ratios = cfg.anchor_ratios
        self.block_layers = [i + 1 for i in range(cfg.n_block_layers)]
        self.feat_shape = {}
        self.build_feat_pyramid()
        self.anchors, self.anchor_is_untruncated = {}, {}
        for i in self.block_layers:
            self.anchors[i], self.anchor_is_untruncated[i] = \
                self.build_anchors(self.feat_shape[i], self.anchor_scales[str(i)])
        self.tf_placeholders = {}
        self.create_tf_placeholders()
        self.global_step = tf.Variable(0, name='global_step',
                                       trainable=False)
        self.train_op, self.loss_op = None, None
        self.eval_op = None

    def build_feat_pyramid(self):
        h, w, _ = self.input_shape
        for i in self.block_layers:
            fh = h / 2**i
            fw = w / 2**i
            feat_shape = (fh, fw)
            self.feat_shape[i] = feat_shape

    def create_tf_placeholders(self):
        h, w, c = self.input_shape
        images = tf.placeholder(tf.float32, [None, h, w, c])
        dir_masks = tf.placeholder(tf.float32, [None, h, w])
        n_ratios = len(self.cfg.anchor_ratios)
        bbox_regs, labels, weights = {}, {}, {}
        for i in self.block_layers:
            n_scales = len(self.anchor_scales[str(i)])
            fh, fw = self.feat_shape[i]
            n_anchors = fh * fw * n_scales * n_ratios
            bbox_regs[i] = tf.placeholder(tf.float32, [None, n_anchors, 4])
            labels[i] = tf.placeholder(tf.int32, [None, n_anchors])
            weights[i] = tf.placeholder(tf.float32, [None, n_anchors])
        learning_rate = tf.placeholder(tf.float32)
        self.tf_placeholders = {'images': images,
                                'dir_masks': dir_masks,
                                'bbox_regs': bbox_regs,
                                'labels': labels,
                                'weights': weights,
                                'learning_rate': learning_rate}

    def build_anchors(self, feat_shape, anchor_scales):
        img_shape = self.input_shape[:2]
        anchors = []
        anchor_is_untruncated = []
        for scale in anchor_scales:
            for ratio in self.anchor_ratios:
                _anchors, _anchor_is_untruncated = \
                    generate_anchors(img_shape, feat_shape, scale, ratio)
                anchors.append(_anchors)
                anchor_is_untruncated.append(_anchor_is_untruncated)
        anchors = np.vstack(anchors)
        anchor_is_untruncated = np.hstack(anchor_is_untruncated)
        return anchors, anchor_is_untruncated

    def generate_bbox_regs(self, i_layer, batch_bboxes):
        anchors = self.anchors[i_layer]
        batch_regs, batch_labels = [], []
        batch_is_object, batch_is_background = [], []
        n_object, n_background = 0., 0.
        scale_min, scale_max = self.cfg.scale_range[str(i_layer)]
        for gt_bboxes in batch_bboxes:
            gt_bboxes = np.array(gt_bboxes)
            if len(gt_bboxes) > 0:
                cond_min = np.sqrt(np.product(gt_bboxes[:, 2:], axis=1)) >= scale_min
                cond_max = np.sqrt(np.product(gt_bboxes[:, 2:], axis=1)) <= scale_max
                gt_bboxes = gt_bboxes[cond_min & cond_max]
            labels, bboxes, classes = \
                label_anchors(anchors,
                              self.anchor_is_untruncated[i_layer],
                              gt_bboxes,
                              iou_low_threshold=self.cfg.iou_low,
                              iou_high_threshold=self.cfg.iou_high)
            if bboxes is None:
                regs = np.zeros_like(anchors)
            else:
                regs = param_bboxes(bboxes, anchors)
            batch_regs.append(regs)
            is_object = np.uint8(labels == 1)
            is_background = np.uint8(labels == 0)
            n_object += np.sum(is_object)
            n_background += np.sum(is_background)
            batch_is_object.append(is_object)
            batch_is_background.append(is_background)
            labels[labels < 0] = 0  # set ambiguous labels to 0, so tf doesn't complain
            batch_labels.append(labels)

        p = ((1. + n_background) / (1. + n_object)) ** self.cfg.class_balance
        batch_weights = np.float32(batch_is_background)
        batch_weights += np.float32(batch_is_object) * p
        return np.array(batch_regs), np.array(batch_labels), batch_weights

    def build_rpn_net(self, input_, training=False):
        raise NotImplementedError("Not yet implemented")

    def build_rcn_refine_net(self, input_, training=False):
        raise NotImplementedError("Not yet implemented")

    def make_rpn_op(self, training=False):
        images = self.tf_placeholders['images']
        logits, pred_regs, mask_logits = \
            self.build_rpn_net(images, training=training)
        return logits, pred_regs, mask_logits

    def make_rcn_op(self, training=False):
        _, rpn_regs, _ = self.make_rpn_op(training=training)
        rpn_bboxes = pred_regs
        clf_logits, pred_regs = self.build_mask_net(rpn_feats, rpn_bboxes,
                                                    training=False)

    def make_train_op(self):
        masks = self.tf_placeholders['dir_masks']
        logits, pred_regs, mask_logits = self.make_rpn_op(training=True)
        labels = self.tf_placeholders['labels']
        gt_regs = self.tf_placeholders['bbox_regs']
        weights = self.tf_placeholders['weights']
        learning_rate = self.tf_placeholders['learning_rate']
        batch_size = self.cfg.batch_size

        clf_losses, reg_losses = [], []
        for i in self.block_layers:
            # classification loss (object vs background)
            logits_i = tf.reshape(logits[i], [batch_size, -1, 2])
            labels_i = tf.maximum(labels[i], 0)
            cross_entropy_loss = losses.cross_entropy_loss(logits=logits_i,
                                                           labels=labels_i,
                                                           weights=weights[i])

            # regression loss
            reg_loss = losses.reg_loss(gt_regs[i], pred_regs[i], labels[i])
            reg_loss = self.cfg.reg_weight * reg_loss
            clf_losses.append(cross_entropy_loss)
            reg_losses.append(reg_loss)

        sem_seg_loss, dir_seg_loss = losses.mask_loss(mask_logits, masks)
        mask_loss = self.cfg.sem_seg_weight * tf.reduce_mean(sem_seg_loss) \
            + self.cfg.dir_seg_weight * tf.reduce_mean(dir_seg_loss)

        solver = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)

        # regular_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # loss = cross_entropy_loss + reg_loss + tf.add_n(regular_losses)
        clf_loss = tf.add_n(clf_losses)
        reg_loss = tf.add_n(reg_losses)
        loss = clf_loss + reg_loss + mask_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = solver.minimize(loss, global_step=self.global_step)
            self.loss_op = [clf_loss, reg_loss, mask_loss]

    def make_eval_op(self):
        logits, pred_regs, mask_logits, _ = self.make_rpn_op(training=True)
        probs, regs = {}, {}
        for i in self.block_layers:
            # classification loss (object vs background)
            logits_i = tf.reshape(logits[i], [-1, 2])
            regs[i] = tf.reshape(pred_regs[i], [-1, 4])
            probs[i] = tf.nn.softmax(logits_i)
        mask_probs = tf.nn.softmax(mask_logits)
        self.eval_op = [regs, probs, mask_probs]

    def train(self):
        """ Train the model. """
        self.make_train_op()
        epochs = self.cfg.epochs
        batch_size = self.cfg.batch_size
        learning_rate = self.cfg.learning_rate
        display_period = self.cfg.display_period
        save_dir = os.path.join(self.cfg.model_save_dir, 'model')
        save_period = self.cfg.save_period
        image_loader = ImageLoader(self.cfg)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if self.cfg.load_model:
                self.load(sess, saver)
            sum_clf_loss, sum_reg_loss, sum_mask_loss = 0., 0., 0.
            for epoch in range(epochs):
                batch_idx = 0
                batch_gen = image_loader.batch_generator()
                for batch_images, batch_bboxes, _, batch_dir_masks in batch_gen:
                    feed_dict = {self.tf_placeholders['images']: batch_images,
                                 self.tf_placeholders['dir_masks']: batch_dir_masks,
                                 self.tf_placeholders['learning_rate']: learning_rate}
                    for i in self.block_layers:
                        batch_regs, batch_labels, batch_weights = \
                            self.generate_bbox_regs(i, batch_bboxes)
                        feed_dict[self.tf_placeholders['bbox_regs'][i]] = batch_regs
                        feed_dict[self.tf_placeholders['labels'][i]] = batch_labels
                        feed_dict[self.tf_placeholders['weights'][i]] = batch_weights
                    _, global_step, clf_loss, reg_loss, mask_loss = \
                        sess.run([self.train_op, self.global_step]
                                 + self.loss_op, feed_dict=feed_dict)
                    batch_idx += 1
                    sum_clf_loss += clf_loss
                    sum_reg_loss += reg_loss
                    sum_mask_loss += mask_loss
                    if batch_idx % display_period == 0:
                        print("Classification loss after {} batches : {:3.5f}"
                              .format(batch_idx, sum_clf_loss / display_period))
                        print("Regression loss after {} batches : {:3.5f}"
                              .format(batch_idx, sum_reg_loss / display_period))
                        print("Segmentation loss after {} batches : {:3.5f}"
                              .format(batch_idx, sum_mask_loss / display_period))
                        sum_clf_loss, sum_reg_loss, sum_mask_loss = 0., 0., 0.
                    if (global_step + 1) % save_period == 0:
                        print("Saving model in {}".format(save_dir))
                        saver.save(sess, save_dir, global_step)
                print("{} epochs finished.".format(epoch + 1))

    def test(self, test_image):
        """ Test the model. """
        self.make_eval_op()
        image_loader = ImageLoader(self.cfg)
        img = image_loader.preprocess_image(test_image)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            self.load(sess, saver)
            feed_dict = {self.tf_placeholders['images']: [img]}
            regs, probs, mask_probs = sess.run(self.eval_op,
                                               feed_dict=feed_dict)
            bboxes = {i: unparam_bbox(regs[i], self.anchors[i])
                      for i in regs.keys()}
            # bboxes = self.anchors
            return bboxes, probs, mask_probs

    def load(self, sess, saver):
        """ Load the trained model. """
        save_dir = os.path.join(self.cfg.model_save_dir, 'model')
        print("Loading model...")
        checkpoint = tf.train.get_checkpoint_state(os.path.dirname(save_dir))
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        saver.restore(sess, checkpoint.model_checkpoint_path)
