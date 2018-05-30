""" Evaluation
    Main script for evaluating the model trained by tcl_training.py
"""





import os
import numpy as np
import pickle
import tensorflow as tf

from subfunc.generate_artificial_data import generate_artificial_data
from subfunc.preprocessing import pca
from subfunc.showdata import *
from tcl import tcl, tcl_eval
from sklearn.decomposition import FastICA

FLAGS = tf.app.flags.FLAGS

# parameters ==================================================
# =============================================================

eval_dir = './storage/temp'
parmpath = os.path.join(eval_dir, 'parm.pkl')

apply_fastICA = True
nonlinearity_to_source = 'abs' # Assume that sources are generated from laplacian distribution



# =============================================================
# =============================================================

# Load trained file -------------------------------------------
ckpt = tf.train.get_checkpoint_state(eval_dir)
modelpath = ckpt.model_checkpoint_path

# Load parameter file
with open(parmpath, 'rb') as f:
    model_parm = pickle.load(f)

num_comp = model_parm['num_comp']
num_segment = model_parm['num_segment']
num_segmentdata = model_parm['num_segmentdata']
num_layer = model_parm['num_layer']
list_hidden_nodes = model_parm['list_hidden_nodes']
moving_average_decay = model_parm['moving_average_decay']
random_seed = model_parm['random_seed']
pca_parm = model_parm['pca_parm']


# Generate sensor signal --------------------------------------
sensor, source, label = generate_artificial_data(num_comp=num_comp,
                                                 num_segment=num_segment,
                                                 num_segmentdata=num_segmentdata,
                                                 num_layer=num_layer,
                                                 random_seed=random_seed)

# Preprocessing -----------------------------------------------
sensor, pca_parm = pca(sensor, num_comp, params = pca_parm)


# Evaluate model ----------------------------------------------
with tf.Graph().as_default() as g:

    data_holder = tf.placeholder(tf.float32, shape=[None, sensor.shape[0]], name='data')
    label_holder = tf.placeholder(tf.int32, shape=[None], name='label')

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, feats = tcl.inference(data_holder, list_hidden_nodes, num_class=num_segment)

    # Calculate predictions.
    top_value, preds = tf.nn.top_k(logits, k=1, name='preds')

    # Restore the moving averaged version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)

        tensor_val = tcl_eval.get_tensor(sensor, [preds, feats], sess, data_holder, batch=256)
        pred_val = tensor_val[0].reshape(-1)
        feat_val = tensor_val[1]


# Calculate accuracy ------------------------------------------
accuracy, confmat = tcl_eval.calc_accuracy(pred_val, label)


# Apply fastICA -----------------------------------------------
if apply_fastICA:
    ica = FastICA(random_state=random_seed)
    feat_val = ica.fit_transform(feat_val)


# Evaluate ----------------------------------------------------
if nonlinearity_to_source == 'abs':
    xseval = np.abs(source) # Original source
else:
    raise ValueError
feateval = feat_val.T # Estimated feature
#
corrmat, sort_idx, _ = tcl_eval.correlation(feateval, xseval, 'Pearson')
meanabscorr = np.mean(np.abs(np.diag(corrmat)))


# Display results ---------------------------------------------
print("Result...")
print("    accuracy(train) : {0:7.4f} [%]".format(accuracy*100))
print("    correlation     : {0:7.4f}".format(meanabscorr))

print("done.")


