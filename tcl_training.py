""" Classification
    Main script for the simulation described in Hyvarinen and Morioka, NIPS 2016.

    Perform time-contrastive learning from artificial data.
    Source signals are generated based on segment-wise-modulated Laplace distribution (q = |.|).
"""





import os
import pickle
import shutil

from subfunc.generate_artificial_data import generate_artificial_data
from subfunc.preprocessing import pca
from tcl.tcl_train import train


# Parameters ==================================================
# =============================================================

# Data generation ---------------------------------------------
random_seed = 0 # random seed
num_comp = 20 # number of components (dimension)
num_segment = 256 # number of segments
num_segmentdata = 512 # number of data-points in each segment
num_layer = 5 # number of layers of mixing-MLP

# MLP ---------------------------------------------------------
list_hidden_nodes = [40, 40, 40, 40, 20]
# list of the number of nodes of each hidden layer of feature-MLP
# [layer1, layer2, ..., layer(num_layer)]

# Training ----------------------------------------------------
initial_learning_rate = 0.01 # initial learning rate
momentum = 0.9 # momentum parameter of SGD
max_steps = int(7e5) # number of iterations (mini-batches)
decay_steps = int(5e5) # decay steps (tf.train.exponential_decay)
decay_factor = 0.1 # decay factor (tf.train.exponential_decay)
batch_size = 512 # mini-batch size
moving_average_decay = 0.999 # moving average decay of variables to be saved
checkpoint_steps = 1e5 # interval to save checkpoint

# for MLR initialization
max_steps_init = int(7e4) # number of iterations (mini-batches) for initializing only MLR
decay_steps_init = int(5e4) # decay steps for initializing only MLR

# Other -------------------------------------------------------
# # Note: save folder must be under ./storage
train_dir = './storage/temp' # save directory (Caution!! this folder will be removed at first)
saveparmpath = os.path.join(train_dir, 'parm.pkl') # file name to save parameters



# =============================================================
# =============================================================

# Prepare save folder -----------------------------------------
if train_dir.find("./storage/") > -1:
    if os.path.exists(train_dir):
        print("delete savefolder: {0:s}...".format(train_dir))
        shutil.rmtree(train_dir)  # Remove folder
    print("make savefolder: {0:s}...".format(train_dir))
    os.makedirs(train_dir)  # Make folder
else:
    assert False, "savefolder looks wrong"


# Generate sensor signal --------------------------------------
sensor, source, label = generate_artificial_data(num_comp=num_comp,
                                                 num_segment=num_segment,
                                                 num_segmentdata=num_segmentdata,
                                                 num_layer=num_layer,
                                                 random_seed=random_seed)


# Preprocessing -----------------------------------------------
sensor, pca_parm = pca(sensor, num_comp=num_comp)


# Train model (only MLR) --------------------------------------
train(sensor,
      label,
      num_class = num_segment,
      list_hidden_nodes = list_hidden_nodes,
      initial_learning_rate = initial_learning_rate,
      momentum = momentum,
      max_steps = max_steps_init, # For init
      decay_steps = decay_steps_init, # For init
      decay_factor = decay_factor,
      batch_size = batch_size,
      train_dir = train_dir,
      checkpoint_steps = checkpoint_steps,
      moving_average_decay = moving_average_decay,
      MLP_trainable = False, # For init
      save_file='model_init.ckpt', # For init
      random_seed = random_seed)

init_model_path = os.path.join(train_dir, 'model_init.ckpt')


# Train model -------------------------------------------------
train(sensor,
      label,
      num_class = num_segment,
      list_hidden_nodes = list_hidden_nodes,
      initial_learning_rate = initial_learning_rate,
      momentum = momentum,
      max_steps = max_steps,
      decay_steps = decay_steps,
      decay_factor = decay_factor,
      batch_size = batch_size,
      train_dir = train_dir,
      checkpoint_steps = checkpoint_steps,
      moving_average_decay = moving_average_decay,
      load_file=init_model_path,
      random_seed = random_seed)


# Save parameters necessary for evaluation --------------------
model_parm = {'random_seed':random_seed,
              'num_comp':num_comp,
              'num_segment':num_segment,
              'num_segmentdata':num_segmentdata,
              'num_layer':num_layer,
              'list_hidden_nodes':list_hidden_nodes,
              'moving_average_decay':moving_average_decay,
              'pca_parm':pca_parm}

print("Save parameters...")
with open(saveparmpath, 'wb') as f:
    pickle.dump(model_parm, f, pickle.HIGHEST_PROTOCOL)

print("done.")

