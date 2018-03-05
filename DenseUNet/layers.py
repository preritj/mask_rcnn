import tensorflow as tf


def concat(x, y):
    return tf.concat([x, y], axis=-1)


def conv2d(x, n_filters=None, k_size=3, stride=1, padding='SAME'):
    if not n_filters:
        n_filters = x.get_shape().as_list()[-1]
    return tf.layers.conv2d(x, n_filters, k_size, strides=stride,
                            padding=padding)


def transpose_conv2d(x, n_filters=None, k_size=3, stride=2, padding='SAME'):
    if not n_filters:
        n_filters = x.get_shape().as_list()[-1]
    return tf.layers.conv2d_transpose(x, n_filters, k_size, strides=stride,
                                      padding=padding)


def layer(x, n_filters, training=False, name='layer', batch_norm=True,
          drop_rate=0.2):
    """BN + Relu + 3x3 conv + dropout"""
    with tf.variable_scope(name):
        if batch_norm:
            x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = conv2d(x, n_filters)
        x = tf.layers.dropout(x, rate=drop_rate, training=training)
        return x


def transition_down(x, training=False, batch_norm=True, drop_rate=0.2):
    """BN + Relu + 1x1 conv + dropout + max-pool"""
    if batch_norm:
        x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = conv2d(x, stride=1)
    x = tf.layers.dropout(x, rate=drop_rate, training=training)
    x = tf.layers.max_pooling2d(x, 2, 2)
    return x


def transition_up(x):
    """3x3 transpose conv with stride 2"""
    return transpose_conv2d(x)


def dense_block(x, name, n_layers=4, training=False, growth_rate=16,
                batch_norm=True, drop_rate=0.2):
    with tf.variable_scope(name):
        x_stack = []
        for i in range(n_layers):
            x_in = x
            layer_name = 'layer_' + str(i+1)
            x = layer(x, n_filters=growth_rate, training=training,
                      batch_norm=batch_norm, drop_rate=drop_rate,
                      name=layer_name)
            x_stack.append(x)
            x = tf.concat([x, x_in], axis=-1)
        x = tf.concat(x_stack, axis=-1)
        return x


def fc_layer(x, n_units, training=False, drop_rate=0.2):
    x = tf.layers.dense(x, n_units, activation=tf.nn.relu)
    x = tf.layers.dropout(x, rate=drop_rate, training=training)
    return x
