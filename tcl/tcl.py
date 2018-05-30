"""TCL"""





import os
import sys
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('FILTER_COLLECTION', 'filter',
                           """filter collection.""")

# =============================================================
# =============================================================
def _variable_init(name, shape, wd, initializer = tf.contrib.layers.variance_scaling_initializer(), trainable = True, collections=None):
    """Helper to create an initialized Variable with weight decay.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """

    if collections is None:
        collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    else:
        collections = [tf.GraphKeys.GLOBAL_VARIABLES]+collections

    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable, collections=collections)

    # Weight decay
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


# =============================================================
# =============================================================
def inference(x, list_hidden_nodes, num_class, wd = 1e-4, maxout_k = 2, MLP_trainable = True, feature_nonlinearity='abs'):
    """Build the model.
        MLP with maxout activation units
    Args:
        x: data holder.
        list_hidden_nodes: number of nodes for each layer. 1D array [num_layer]
        num_class: number of classes of MLR
        wd: (option) parameter of weight decay (not for bias)
        maxout_k: (option) number of affine feature maps
        MLP_trainable: (option) If false, fix MLP layers
        feature_nonlinearity: (option) Nonlinearity of the last hidden layer (feature value)
    Returns:
        logits: logits tensor:
        feat: feature tensor
    """
    print("Building model...")

    # Maxout --------------------------------------------------
    def maxout(y, k):
        #   y: data tensor
        #   k: number of affine feature maps
        input_shape = y.get_shape().as_list()
        ndim = len(input_shape)
        ch = input_shape[-1]
        assert ndim == 4 or ndim == 2
        assert ch is not None and ch % k == 0
        if ndim == 4:
            y = tf.reshape(y, [-1, input_shape[1], input_shape[2], ch / k, k])
        else:
            y = tf.reshape(y, [-1, int(ch / k), k])
        y = tf.reduce_max(y, ndim)
        return y

    num_layer = len(list_hidden_nodes)

    # Hidden layers -------------------------------------------
    for ln in range(num_layer):
        with tf.variable_scope('layer'+str(ln+1)) as scope:
            in_dim = list_hidden_nodes[ln-1] if ln > 0 else x.get_shape().as_list()[1]
            out_dim = list_hidden_nodes[ln]

            if ln < num_layer - 1: # Increase number of nodes for maxout
                out_dim = maxout_k * out_dim

            # Inner product
            W = _variable_init('W', [in_dim, out_dim], wd, trainable=MLP_trainable, collections=[FLAGS.FILTER_COLLECTION])
            b = _variable_init('b', [out_dim], 0, tf.constant_initializer(0.0), trainable=MLP_trainable, collections=[FLAGS.FILTER_COLLECTION])
            x = tf.nn.xw_plus_b(x, W, b)

            # Nonlinearity
            if ln < num_layer-1:
                x = maxout(x, maxout_k)
            else: # The last layer (feature value)
                if feature_nonlinearity == 'abs':
                    x = tf.abs(x)
                else:
                    raise ValueError

            # Add summary
            tf.summary.histogram('layer'+str(ln+1)+'/activations', x)

    feats = x

    # MLR -----------------------------------------------------
    with tf.variable_scope('MLR') as scope:
        in_dim = list_hidden_nodes[-1]
        out_dim = num_class

        # Inner product
        W = _variable_init('W', [in_dim, out_dim], wd, collections=[FLAGS.FILTER_COLLECTION])
        b = _variable_init('b', [out_dim], 0, tf.constant_initializer(0.0), collections=[FLAGS.FILTER_COLLECTION])
        logits = tf.nn.xw_plus_b(x, W, b)

    return logits, feats


# =============================================================
# =============================================================
def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
    Args:
        logits: logits from inference().
        labels: labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acurracy')

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss'), accuracy


# =============================================================
# =============================================================
def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.

    Args:
        total_loss: total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


# =============================================================
# =============================================================
def train(total_loss,
          accuracy,
          global_step,
          initial_learning_rate,
          momentum,
          decay_steps,
          decay_factor,
          moving_average_decay = 0.9999,
          moving_average_collections = tf.trainable_variables()):
    """Train model.
        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.
    Args:
        total_loss: total loss from loss().
        accuracy: accuracy tensor
        global_step: integer variable counting the number of training steps processed.
        initial_learning_rate: initial learning rate
        momentum: momentum parameter (tf.train.MomentumOptimizer)
        decay_steps: decay steps (tf.train.exponential_decay)
        decay_factor: decay factor (tf.train.exponential_decay)
        moving_average_decay: (option) moving average decay of variables to be saved
        moving_average_collections: (option) variables to be saved with those moving averages
    Returns:
        train_op: op for training.
    """

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                  global_step,
                                  decay_steps,
                                  decay_factor,
                                  staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Generate moving averages of accuracy and associated summaries.
    accu_averages = tf.train.ExponentialMovingAverage(0.9, name='avg_accu')
    accu_averages_op = accu_averages.apply([accuracy])
    tf.summary.scalar(accuracy.op.name + ' (raw)', accuracy)
    tf.summary.scalar(accuracy.op.name, accu_averages.average(accuracy))

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op, accu_averages_op]):
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.MomentumOptimizer(lr, momentum)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op, lr


