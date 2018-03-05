import tensorflow as tf
from layers import concat, conv2d, fc_layer
from layers import dense_block, transition_down, transition_up
from model import RCNN


class DenseUNet(RCNN):
    def __init__(self, cfg):
        RCNN.__init__(self, cfg)
        self.n_block_layers = cfg.n_block_layers
        self.rpn_outputs = None

    def build_rpn_net(self, input_, training=False):
        anchor_ratios = self.cfg.anchor_ratios
        anchor_scales = self.cfg.anchor_scales
        mask_logits, rpn_inputs = self.build_unet(input_, training)
        n_anchor_ratios = len(anchor_ratios)
        rpn_outputs, clf_logits, regs = {}, {}, {}
        for i in self.block_layers:
            n_anchors_per_cell = len(anchor_scales[str(i)])
            n_anchors_per_cell *= n_anchor_ratios
            rpn_outputs[i], clf_logits[i], regs[i] = \
                self.build_rpn(rpn_inputs[i],
                               n_anchors_per_cell)
        self.rpn_outputs = rpn_outputs
        return clf_logits, regs, mask_logits

    def build_rcn_refine_net(self, rpn_bboxes, training=False):
        assert self.cfg.batch_size == 1, \
            "RCN training supported for batch_size = 1 only."
        assert self.rpn_outputs is not None, \
            "Run RCN first."
        crop_size = self.cfg.feat_crop_size
        all_feats, all_boxes = [], []
        box_ind = []
        for i, i_layer in enumerate(self.block_layers):
            all_feats.append(self.rpn_outputs[i_layer])
            boxes = rpn_bboxes[i_layer]
            n_boxes = boxes.get_shape().as_list()[0]
            box_ind += [i] * n_boxes
            all_boxes.append(boxes)
        box_ind = tf.cast(box_ind, tf.int32)
        all_boxes = tf.reshape(all_boxes, [-1, 4])
        all_feats = tf.reshape(all_feats, [-1])
        crops = tf.image.crop_and_resize(all_feats, all_boxes,
                                         box_ind=box_ind,
                                         crop_size=crop_size)
        crops = tf.stop_gradient(crops)
        clf_logits, regs = self.build_refine_bbox(crops, training)
        return clf_logits, regs

    def build_unet(self, x, training=False):
        dir_bins = self.cfg.dir_bins
        drop_rate = self.cfg.drop_rate
        batch_norm = self.cfg.batch_norm
        layers_per_block = self.cfg.layers_per_block
        # first conv layer:
        x = conv2d(x, 48)
        stack_skip = []
        # downsample (encoder):
        for i in range(self.n_block_layers):
            n = layers_per_block[str(i)]
            name = 'down_block_' + str(i)
            x_new = dense_block(x, name, n_layers=n, training=training)
            x = concat(x, x_new)
            stack_skip.append(x)
            x = transition_down(x, training)
        # bottleneck:
        n = layers_per_block[str(self.n_block_layers)]
        x_new = dense_block(x, name='bottleneck_block', n_layers=n,
                            training=training)
        rpn_inputs = {self.n_block_layers: concat(x, x_new)}
        # upsample (decoder):
        for i in reversed(range(self.n_block_layers)):
            n = layers_per_block[str(i)]
            name = 'up_block_' + str(i)
            x_skip = stack_skip[i]
            x = transition_up(x_new)
            x = concat(x, x_skip)
            x_new = dense_block(x, name, n_layers=n, training=training,
                                batch_norm=batch_norm, drop_rate=drop_rate)
            rpn_inputs[i] = concat(x, x_new)
        # last conv layer:
        mask_logits = conv2d(x_new, 1 + dir_bins, k_size=1)
        return mask_logits, rpn_inputs

    def build_rpn(self, x, anchors_per_cell):
        _, h, w, _ = x.get_shape().as_list()
        n = anchors_per_cell
        x = conv2d(x, 512)
        rpn_outputs = x
        x = tf.nn.relu(x)
        x = conv2d(x, n_filters=6 * n, k_size=1)
        x = tf.transpose(x, (0, 3, 1, 2))
        x = tf.reshape(x, [-1, 6, n * h * w])
        x = tf.transpose(x, (0, 2, 1))
        clf_logits, regs = tf.split(x, [2, 4], axis=2)
        return rpn_outputs, clf_logits, regs

    def build_refine_bbox(self, feat_crops, training=False):
        x = conv2d(feat_crops, n_filters=256, k_size=1)
        x = fc_layer(x, 1024, training=training)
        x = fc_layer(x, 1024, training=training)
        x = tf.layers.dense(x, 6)
        clf_logits, regs = tf.split(x, [2, 4], axis=1)
        return clf_logits, regs



