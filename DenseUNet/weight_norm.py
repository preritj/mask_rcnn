import tensorflow as tf


def get_var_maybe_avg(var_name, ema, **kwargs):
    """ utility for retrieving polyak averaged params """
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v


def mean_only_batch_norm(input_, training, bias=None, init=False):
    n_filters = input_.get_shape().as_list()[-1]
    var_init = tf.constant_initializer(0.)

    with tf.variable_scope('batch_norm', reuse=(not init)):
        x = input_
        moving_mean = tf.get_variable('mean', [n_filters], tf.float32, var_init,
                                      trainable=False)

        def _mean_with_update():
            decay = 0.999
            x_mean = tf.reduce_mean(x, [0, 1, 2])
            difference = (1 - decay) * (moving_mean - x_mean)
            with tf.control_dependencies([moving_mean.assign_sub(difference)]):
                return x_mean

        if init:
            mean = tf.reduce_mean(x, [0, 1, 2])
            with tf.control_dependencies([moving_mean.assign(mean)]):
                x = tf.identity(x)
        else:
            # apply mean only batch norm
            mean = tf.cond(training, _mean_with_update, lambda: moving_mean)
            x = x - mean

        if bias:
            x = tf.nn.bias_add(x, bias)
        return x


def conv_weight_norm(input_, n_filters=None, k_size=3, stride=1, init=False,
                     padding='SAME', ema=None, init_scale=1.):
    """ Convolutional layer with weight-norm"""
    kernel_stride = [1] + 2 * [stride] + [1]
    in_shape = input_.get_shape().as_list()
    n_channels = in_shape[-1]
    if not n_filters:
        n_filters = n_channels
    shape_V = [k_size, k_size, n_channels, n_filters]

    with tf.variable_scope('conv2d', reuse=(not init)):
        var_init = tf.random_normal_initializer(0, .1)
        V = get_var_maybe_avg('V', ema, shape=shape_V, dtype=tf.float32,
                              initializer=var_init, trainable=True)
        var_init = tf.constant_initializer(1.)
        g = get_var_maybe_avg('g', ema, shape=[n_filters], dtype=tf.float32,
                              initializer=var_init, trainable=True)
        var_init = tf.constant_initializer(0.)
        b = get_var_maybe_avg('b', ema, shape=[n_filters], dtype=tf.float32,
                              initializer=var_init, trainable=True)

        def _weight_norm():
            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, 1, n_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])
            return tf.nn.conv2d(input_, W, kernel_stride, padding=padding)

        x = _weight_norm()

        if init:
            m_init, v_init = tf.nn.moments(x, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init),
                                          b.assign_add(-m_init * scale_init)]):
                x = _weight_norm()

        return x, b
