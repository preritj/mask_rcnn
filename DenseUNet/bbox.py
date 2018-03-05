import numpy as np
import tensorflow as tf


def iou_bbox(bboxes1, bboxes2):
    """ Compute the IoUs between bounding boxes. """
    bboxes1 = np.array(bboxes1, np.float32)
    bboxes2 = np.array(bboxes2, np.float32)

    intersection_min_y = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
    intersection_max_y = np.minimum(bboxes1[:, 0] + bboxes1[:, 2] - 1,
                                    bboxes2[:, 0] + bboxes2[:, 2] - 1)
    intersection_height = np.maximum(intersection_max_y - intersection_min_y + 1,
                                     np.zeros_like(bboxes1[:, 0]))

    intersection_min_x = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
    intersection_max_x = np.minimum(bboxes1[:, 1] + bboxes1[:, 3] - 1,
                                    bboxes2[:, 1] + bboxes2[:, 3] - 1)
    intersection_width = np.maximum(intersection_max_x - intersection_min_x + 1,
                                    np.zeros_like(bboxes1[:, 1]))

    area_intersection = intersection_height * intersection_width
    area_first = bboxes1[:, 2] * bboxes1[:, 3]
    area_second = bboxes2[:, 2] * bboxes2[:, 3]
    area_union = area_first + area_second - area_intersection

    iou = area_intersection * 1.0 / area_union
    return iou


def param_bboxes(bboxes, anchors):
    """ Parameterize bounding boxes with respect to anchors.
    Namely, (y,x,h,w)->(ty,tx,th,tw). """
    bboxes = np.array(bboxes, np.float32)
    anchors = np.array(anchors, np.float32)
    tyx = 10. * (bboxes[:, :2] - anchors[:, :2]) / anchors[:, 2:]
    thw = 5. * np.log(bboxes[:, 2:] / anchors[:, 2:])
    return np.concatenate((tyx, thw), axis=1)


def unparam_bbox(t, anchors):
    """ Unparameterize bounding boxes with respect to anchors.
    Namely, (ty,tx,th,tw)->(y,x,h,w). """
    t = np.array(t, np.float32)
    anchors = np.array(anchors, np.float32)
    yx = .1 * t[:, :2] * anchors[:, 2:] + anchors[:, :2]
    hw = np.exp(.2 * t[:, 2:]) * anchors[:, 2:]
    return np.concatenate((yx, hw), axis=1)


def tf_unparam_bbox(t, anchors):
    """ tensorflow implementation of unparam_bbox.
    Namely, (ty,tx,th,tw)->(y,x,h,w). """
    yx = .1 * t[:, :2] * anchors[:, 2:] + anchors[:, :2]
    hw = tf.exp(.2 * t[:, 2:]) * anchors[:, 2:]
    return tf.concat([yx, hw], axis=1)


def generate_anchors(img_shape, feat_shape, scale, ratio):
    """ Generate the anchors. """
    ih, iw = img_shape
    fh, fw = feat_shape
    n = fh * fw

    # Compute the coordinates of the anchors
    j = np.arange(fh)
    j = np.expand_dims(j, 1)
    j = np.tile(j, (1, fw))
    j = j.reshape((-1))

    i = np.arange(fw)
    i = np.expand_dims(i, 0)
    i = np.tile(i, (fh, 1))
    i = i.reshape((-1))

    s = np.ones(n) * scale
    r0 = np.ones(n) * min(ratio[0], .8 * ih / scale)
    r1 = np.ones(n) * min(ratio[1], .8 * iw / scale)

    h = s * r0
    w = s * r1
    y = (j + 0.5) * ih / fh - h * 0.5
    x = (i + 0.5) * iw / fw - w * 0.5

    # Determine if the anchors cross the boundary
    anchor_is_untruncated = np.ones(n, np.int32)
    anchor_is_untruncated[np.where(y < 0)[0]] = 0
    anchor_is_untruncated[np.where(x < 0)[0]] = 0
    anchor_is_untruncated[np.where(h + y > ih)[0]] = 0
    anchor_is_untruncated[np.where(w + x > iw)[0]] = 0

    y = np.expand_dims(y, 1)
    x = np.expand_dims(x, 1)
    h = np.expand_dims(h, 1)
    w = np.expand_dims(w, 1)
    anchors = np.concatenate((y, x, h, w), axis=1)
    anchors = np.array(anchors, np.int32)
    return anchors, anchor_is_untruncated


def label_anchors(anchors, anchor_is_untruncated, gt_bboxes,
                  iou_low_threshold=0.3, iou_high_threshold=0.7):
    """ Get the labels of the anchors.
    Each anchor can be labeled as positive (1), negative (0) or ambiguous (-1)
    Truncated anchors are always labeled as ambiguous. """
    n = anchors.shape[0]
    k = gt_bboxes.shape[0]

    # if no GT bboxes:
    if k == 0:
        labels = np.zeros(n, np.int32)
        # Truncated anchors are always ambiguous
        ignore_idx = np.where(anchor_is_untruncated == 0)[0]
        labels[ignore_idx] = -1
        return labels, None, None

    # Compute the IoUs of the anchors and ground truth boxes
    tiled_anchors = np.tile(np.expand_dims(anchors, 1), (1, k, 1))
    tiled_gt_bboxes = np.tile(np.expand_dims(gt_bboxes, 0), (n, 1, 1))

    tiled_anchors = tiled_anchors.reshape((-1, 4))
    tiled_gt_bboxes = tiled_gt_bboxes.reshape((-1, 4))

    ious = iou_bbox(tiled_anchors, tiled_gt_bboxes)
    ious = ious.reshape(n, k)

    # Label each anchor based on its max IoU
    max_ious = np.max(ious, axis=1)
    best_gt_bbox_ids = np.argmax(ious, axis=1)

    positive_idx = np.where(max_ious >= iou_high_threshold)[0]
    negative_idx = np.where(max_ious < iou_low_threshold)[0]
    labels = -np.ones(n, np.int32)
    labels[positive_idx] = 1
    labels[negative_idx] = 0

    # Label at least one anchor as positive for each GT bbox
    anchor_idx = np.argmax(ious, axis=0)
    labels[anchor_idx] = 1

    # Truncated anchors are always ambiguous
    ignore_idx = np.where(anchor_is_untruncated == 0)[0]
    labels[ignore_idx] = -1

    bboxes = gt_bboxes[best_gt_bbox_ids]
    classes = np.ones(n, np.int32)
    classes[np.where(labels < 1)[0]] = 0

    max_ious[np.where(anchor_is_untruncated == 0)[0]] = -1
    return labels, bboxes, classes
