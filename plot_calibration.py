# This is https://github.com/dtanoglidis/BNN_LSBGs_ICML/blob/main/Calibration_Plots.ipynb made into a python script

# usage: create calibration plots using bnn trained on h5_weight_file_path to test on data_conparison_source

# use: python plot_calibration.py [h5_weight_file_path] [data_comparison_source] [out_directory_name]
# e.g: python plot_calibration.py bnn_train_out/original/{\'PA\': [0.0, 180.0], \'I_sky\': 22.23, \'ell\': [0.05, 0.7], \'n\': [0.5, 1.5], \'I_e\': [24.3, 25.5], \'r_e\': [2.5, 6.0]}_BNN_weight.h5 generated_data/original" "out_directory_name"

# data output is at ./calibration_plots/out_directory_name

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
import time
import sys
import os
import joblib as jl
# Import resampling, we will need it for bootstap resampling
from sklearn.utils import resample #Resampling

# Colab in order to download files
#from google.colab import files

# Scikit-learn for scaling and preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# PyImfit, ChainConsumer and Emcee
import pyimfit
from chainconsumer import ChainConsumer

# Matplotlib, seaborn and plot pretty
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
#%matplotlib inline
from matplotlib import rcParams, use
rcParams['font.family'] = 'serif'

# Pillow
from PIL import Image

# Tensorflow and Keras
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
# Keras Layers
from keras.layers import InputLayer
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# Import some layers etc that are useful in the Functional approach
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

use('Agg')
# global params
IMAGE_SHAPE = [64, 64, 1]
NUM_TRAIN_EXAMPLES = 150000
NUM_VAL_EXAMPLES = 10000
NUM_TEST_EXAMPLES = 10000
NUM_CLASSES = 5

# Define percents and percentiles
percentiles = np.linspace(0,1,40)
n_test = 1000

#---------------------------------------------------------------------
# Adjust rc parameters to make plots pretty
def plot_pretty(dpi=200, fontsize=9):

    plt.rc("savefig", dpi=dpi)       # dpi resolution of saved image files
    plt.rc('text', usetex=True)      # use LaTeX to process labels
    plt.rc('font', size=fontsize)    # fontsize
    plt.rc('xtick', direction='in')  # make axes ticks point inward
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=10)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=10)
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [0.5, 1.1]) # fix dotted lines
    return


# conversion functions
# Converts from counts/pixel to mag/arcsec
def counts_to_SB(Counts):
    Area = (0.263)**2.0
    SB = -2.5*np.log10(Counts/Area)+30.
    return SB

# Converts from mag/arcsec to counts/pixel
def SB_to_counts(SB):
    Area = (0.263)**2.0
    exponent = (30.0-SB)/2.5
    Counts = Area*(10.0**exponent)
    return Counts

# Convert pixels to arcsec
def pix_to_asec(pix):
    asec = pix*0.263
    return asec

# Convert arcsec to pixels
def asec_to_pix(asec):
    pix = asec/0.263
    return pix

#function that calculates the fraction of galaxies within the percentiles.
def fraction_param(lims_par,par_true):
    nums = np.zeros(20)

    for i in range(20):
        nums_i = 0
        for j in range(n_test):
            u_lim = lims_par[39-i,j]
            l_lim = lims_par[i,j]
            true_par_j = par_true[j]

            if ((true_par_j>=l_lim)&(true_par_j<=u_lim)):
                nums_i +=1

            nums[i] = nums_i
            fraction = nums/1000
    return fraction

# Write function that returns limits of percentiles
# for specific parameter
def lims_perc(sample_param):
    sample_shift = sample_param
    lims = np.quantile(sample_shift,percentiles, axis=1)
    return lims


# calculate confidence intervals using bootstrap resampling
def bootstrap_error(sample_par,par_true):
  # Computes upper and lower bound using bootstrap resampling

    n_bootstrap = 100
    indices = np.arange(1000)
    nums_boot = np.zeros((100,20))

    for i in range(n_bootstrap):
        if (i%10==0):
            print(i)

        # Resample
        ind_loc = resample(indices, n_samples=len(indices), replace=True)
        # Create new sample
        param_true_local = par_true[ind_loc]
        sample_shift_loc = sample_par[ind_loc]
        lims_loc = np.quantile(sample_shift_loc,percentiles, axis=1)

        for j in range(20):
            nums_j = 0
            for k in range(n_test):
                u_lim = lims_loc[39-j,k]
                l_lim = lims_loc[j,k]
                true_par_k = param_true_local[k]

                if ((true_par_k>=l_lim)&(true_par_k<=u_lim)):
                    nums_j +=1

                nums_boot[i,j] = nums_j/1000.

    # Compute the lower and upper bound of the 95% confidence interval
    lower = np.quantile(nums_boot,0.025, axis=0)
    upper = np.quantile(nums_boot,0.975, axis=0)

    return lower, upper


def get_model():
    # Define KL function

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
    # =========================================
    # =========================================
    x = keras.layers.Flatten()(x)
    # =========================================
    # =========================================
    x = tfp.layers.DenseFlipout(
              units = 1024,
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu)(x)
    distribution_params = keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(5))(x)
    model_output = tfp.layers.MultivariateNormalTriL(event_size=5)(distribution_params)
    model = Model(model_input, model_output)
    return model



def main(model, weight_source: str, motherpath: str):
    # Images
    # motherpath = "../../data_stash/out_data/original"
    X_test = np.float64(np.load(f"{motherpath}/X_test.npy"))
    # Labels
    y_train = np.float64(np.load(f"{motherpath}/y_train.npy"))
    y_test = np.float64(np.load(f"{motherpath}/y_test.npy"))
    #Scaling
    scale_test = np.float64(np.load(f"{motherpath}/scale_test.npy"))

    scaler = StandardScaler()
    # Rescale the labels
    scaler.fit(y_train)
    y_train_sc = scaler.transform(y_train)
    y_test_sc = scaler.transform(y_test)


    # '../../bnn_codes/train_out/original/{\'PA\': [0.0, 180.0], \'I_sky\': 22.23, \'ell\': [0.05, 0.7], \'n\': [0.5, 1.5], \'I_e\': [24.3, 25.5], \'r_e\': [2.5, 6.0]}_BNN_weight.h5'
    model.load_weights(weight_source)

    # make prediction

    X_keep = X_test[:n_test]
    y_keep = y_test[:n_test]
    y_keep_sc = y_test_sc[:n_test]
    scale_keep = scale_test[:n_test]

    #print(np.shape(X_keep))

    pred_dist = model(X_keep)
    sample = np.asarray(pred_dist.sample(300))

    #print(np.shape(sample))

    n_rands = 400

    for i in range(n_rands):
        pred_dist = model(X_keep)
        sample_loc = np.asarray(pred_dist.sample(300))
        sample = np.concatenate((sample,sample_loc))

    inv_sample = []
    for i in range(n_test):
        inv_sample_loc = scaler.inverse_transform(sample[:,i,:])
        inv_sample.append(inv_sample_loc)

    # Get means
    #mean_preds = np.mean(inv_sample,axis=1)
    # Get medians
    #median_preds = np.median(inv_sample,axis=1)
    # Get standard deviations
    #std_preds = np.std(inv_sample,axis=1)
    return np.asarray(inv_sample), y_keep, sample


def make_calibration(sample_eff, eff_true, title: str, out_dir: str):
    # Percents
    percents = np.zeros(20)
    for i in range(20):
        percents[i] = percentiles[39-i]-percentiles[i]

    low_reff, upp_reff = bootstrap_error(sample_eff,eff_true)
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    lims = lims_perc(sample_eff)
    fraction = fraction_param(lims, eff_true)

    x = np.linspace(0,1,100)
    f = plt.figure(figsize=(8., 8.))

    # Plot title
    plt.title(title, fontsize=23)

    # TODO: here: percents/fraction plot
    # should:

    plt.plot(x,x,ls='--',c='k',label='Perfect Calibration')
    plt.plot(percents,fraction,c='mediumblue',label='BNN')
    plt.fill_between(percents,low_reff,upp_reff,color='mediumblue',
                     rasterized=True,alpha=0.3,label='95$\%$ C.L.')


    plt.grid(alpha=0.6)

    plt.xlabel('Posterior Percentile', fontsize=25)
    plt.ylabel('Fraction of LSBGs within Percentile', fontsize=25)

    # Text
    plt.text(0.7,0.05,'Overconfident', bbox=props, fontsize=20)
    plt.text(0.02,0.95,'Underconfident', bbox=props,fontsize=20)

    #Configure plot
    plt.tick_params(axis='both', labelsize=19)
    plt.legend(frameon=True, loc=(0.02,0.6),ncol=1, fontsize=22)
    #Save
    plt.savefig(f"{out_dir}/{title}_Calibration_Plot_reff.pdf",bbox_inches='tight')
    plt.show()
    f.clear()
    plt.close(f)



if __name__ == "__main__":
    plot_pretty()
    model = get_model()
    inv_sampl, y_keep, sample = main(model, sys.argv[1], sys.argv[2])
    out_dir = str(sys.argv[3])
    # Effective radius
    sample_r_eff = inv_sampl[:,:,-1]
    r_eff_true = y_keep[:,-1]
    # Mean Surface brightness
    sample_I_eff = inv_sampl[:,:,-2]
    I_eff_true = y_keep[:,-2]
    # Sersic Index
    sample_n = inv_sampl[:,:,-3]
    n_true = y_keep[:,-3]
    # Ellipticity
    sample_ell = inv_sampl[:,:,-4]
    ell_true = y_keep[:,-4]
    # Position Angle
    sample_PA = inv_sampl[:,:,-5]
    PA_true = y_keep[:,-5]

    # outdict: "name": (sample, true)
    dictionary = {"radius $r_e$": (sample_r_eff, r_eff_true),
                "surface brightness $I_e$": (sample_I_eff, I_eff_true),
                "sersic index $n$": (sample_n, n_true),
                "ellipticity $\epsilon$": (sample_ell, ell_true),
                "position angle PA": (sample_PA, PA_true)}

    try:
        os.mkdir(os.path.join('./', 'calibration_plots'))
        os.mkdir(os.path.join('./calibration_plots/',out_dir))
    except FileExistsError:
        print()

    print("done prepping. Now onto making calibration plots")
    calibration_dict = {}
    for param in dictionary.keys():
        sample, true = dictionary[param]
        make_calibration(sample, true, param, f'./calibration_plots/{out_dir}')

    print("savings samples as samples.jl")
    jl.dump(samples, './calibration_plots/{out_dir}')
