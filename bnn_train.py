# This is code from https://github.com/dtanoglidis/BNN_LSBGs_ICML/blob/main/BNN_Training_Baseline.ipynb, made into one script

# usage: run BNN on data presented in data_stash/

# to run: python bnn_train.py [dataset_motherpath] [config_path] [output_path]
# e.g:    python bnn_traom.py generated_dataset/original/{\'PA\': [0.0, 180.0], \'I_sky\': 22.23, \'ell\': [0.05, 0.7], \'n\': [0.5, 1.5], \'I_e\': [24.3, 25.5], \'r_e\': [2.5, 6.0]}_BNN_weight.h5 param_config/original bnn_train_out/

#==============================================================
# TODO:
# [ ] ckean up code
# [ ] last run to check
#==============================================================

# Import basic packages
import numpy as np
import scipy as sp
import pandas as pd
import pylab as plt
import scipy.stats
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from keras.layers import InputLayer
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
#%matplotlib inline
from PIL import Image
import os
import ast
import sys
from matplotlib import rcParams, use

use('Agg')
rcParams['font.family'] = 'serif'

# Define basic numbers
IMAGE_SHAPE = [64,64,1]
NUM_TRAIN_EXAMPLES = 150000
NUM_VAL_EXAMPLES = 10000
NUM_TEST_EXAMPLES = 10000
NUM_CLASSES = 5
EPOCHS = 150

def plot_pretty(dpi=200, fontsize=9):
    plt.rc("savefig", dpi=dpi)       # dpi resolution of saved image files
    plt.rc('text', usetex=False)      # use LaTeX to process labels
    plt.rc('font', size=fontsize)    # fontsize
    plt.rc('xtick', direction='in')  # make axes ticks point inward
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=10)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=10)
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [0.5, 1.1]) # fix dotted lines
    return


def negloglik(y_true, y_pred):
    return -tf.reduce_mean(y_pred.log_prob(y_true))

def compile_model():
    # define KL divergence function
    tfd = tfp.distributions

    # KL divergence weighted by the number of training samples, using
    # lambda function to pass as input to the kernel_divergence_fn on
    # flipout layers.
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                               tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))


    # Bayesian DeepBFit in functional form
    model_input = Input(shape=(64,64,1))
    # Convolutional part =================
    # 1st convolutional chunk
    x = tfp.layers.Convolution2DFlipout(
              filters = 4,
              kernel_size=(3,3),
              padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu)(model_input)
    x = keras.layers.MaxPool2D(pool_size=(2, 2),
                                   strides=None,
                                   padding='valid')(x)
    # 2nd convolutional chunk
    x = tfp.layers.Convolution2DFlipout(
              filters = 8,
              kernel_size=(3,3),
              padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2),
                                   strides=None,
                                   padding='valid')(x)
    # 3rd convolutional chunk
    x = tfp.layers.Convolution2DFlipout(
              filters = 16,
              kernel_size=(3,3),
              padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2),
                                   strides=None,
                                   padding='valid')(x)
    # 4th convolutional chunk
    x = tfp.layers.Convolution2DFlipout(
              filters = 32,
              kernel_size=(3,3),
              padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2),
                                   strides=None,
                                   padding='valid')(x)
    # 5th convolutional chunk
    x = tfp.layers.Convolution2DFlipout(
              filters = 64,
              kernel_size=(3,3),
              padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2),
                                   strides=None,
                                   padding='valid')(x)
    x = keras.layers.Flatten()(x)
    x = tfp.layers.DenseFlipout(
              units = 1024,
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu)(x)
    distribution_params = keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(5))(x)
    model_output = tfp.layers.MultivariateNormalTriL(event_size=5)(distribution_params)
    model = Model(model_input, model_output)

    # Define the optimizer
    optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.2, rho=0.98)
    model.compile(optimizer,
                  loss=negloglik,
                  metrics=['mae'],experimental_run_tf_function=False)
    return model


def train(model, x_train, x_val, y_train, y_val, path):
    # load data
    # Images
    X_train = np.float64(np.load(x_train))
    X_val = np.float64(np.load(x_val))

    # Labels
    y_train = np.float64(np.load(y_train))
    y_val = np.float64(np.load(y_val))

    scaler = StandardScaler()

    # Rescale the labels
    scaler.fit(y_train)
    y_train_sc = scaler.transform(y_train)
    y_val_sc = scaler.transform(y_val)

    model.fit(x=X_train, y=y_train_sc,
              epochs=EPOCHS, batch_size=64,
              shuffle=True,
              validation_data=(X_val,y_val_sc))

    model.save_weights(f'{path}_BNN_weight.h5',overwrite=True)
    print(f"model weights saved as {path}_BNN_weight.h5*.pdf")

    history_dict = model.history.history
    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    train_mae = history_dict['mae']
    val_mae = history_dict['val_mae']

    Epochs_sp = np.arange(1,EPOCHS+1,1)

    # =====================================================================
    f = plt.figure(figsize=(7.2, 6.0))

    # Plot Loss
    plt.plot(Epochs_sp, train_loss, c = 'mediumblue',linewidth=2.5,label='Training loss')
    plt.plot(Epochs_sp, val_loss , c = 'dodgerblue',linewidth=2.5,label='Validation loss')

    # ==========================================
    # ==========================================
    plt.grid(ls='--',alpha=0.6)
    plt.xlabel('Epoch', fontsize=17);plt.ylabel('Loss',fontsize=17)
    plt.legend(frameon=True, loc='upper right', fontsize=17)
    plt.tick_params(axis='both', labelsize=14.5)

    #plt.xlim(20,)
    #plt.ylim(0,0.4)
    plt.savefig(f"{path}_Loss.pdf")

    f.clear()
    plt.close(f)
    f = plt.figure(figsize=(7.2, 6.0))

    # Plot MAE
    plt.plot(Epochs_sp, train_mae, c = 'mediumblue',linewidth=2.5,label='Training MAE')
    plt.plot(Epochs_sp, val_mae, c = 'dodgerblue',linewidth=2.5,label='Validation MAE')

    plt.grid(ls='--',alpha=0.6)
    plt.xlabel('Epoch', fontsize=17);plt.ylabel('MAE',fontsize=17)
    plt.legend(frameon=True, loc='upper right', fontsize=17)
    plt.tick_params(axis='both', labelsize=14.5)

    #plt.xlim(20,)
    #plt.ylim(0.1,0.4)
    plt.savefig(f"{path}_MAE.pdf")

    print(f"plots saved as {path}_*.pdf")
    f.clear()
    plt.close(f)



if __name__ == "__main__":
    plot_pretty()
    dataset_path = sys.argv[1] # ../out_data/
    out_dir = sys.argv[3]
    try:
        os.mkdir(os.path.join('./', str(out_dir)))
    except FileExistsError:

    config_file = open(sys.argv[2], 'r').readlines()
    param_list = [ast.literal_eval(param) for param in config_file]
    model = compile_model()
    for params in param_list:
        print("Doing param:", params)
        param = f"Params_{list(params.values())}"
        x_train = f'{dataset_path}/X_train.npy'
        x_val =  f'{dataset_path}/X_val.npy'
        y_train =  f'{dataset_path}/y_train.npy'
        y_val =  f'{dataset_path}/y_val.npy'
        path = f'{out_dir}/'
        train(model, x_train, x_val, y_train, y_val, path)
    print("done BNN train.......................................................")
