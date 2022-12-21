# This is https://github.com/dtanoglidis/BNN_LSBGs_ICML/blob/main/Dataset_Generation_Sersic_Model.ipynb made into a script

# usage: generate synthetic data

# to run: python data_gen.py [param_file_name]
# e.g:    python data_gen.py param_config/original_param

#==============================================================
# TODO:
# [x] ckean up code
# [x] last run to check
#==============================================================

import numpy as np
import scipy as sp
import pandas as pd
import pylab as plt
import scipy.stats
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib import rcParams, use
from PIL import Image
import pyimfit
import sys
import ast
import os
import joblib as jl

use('Agg') # helps matplotlib memory

rcParams['font.family'] = 'serif'
nsims = 170000                 # number of simulation
SIZE = 64
DTYPE = [
    ('id',int),                # object id
    ('x',float),               # x-centroid
    ('y',float),               # y-centroid
    ('I_sky',float),           # background sky intensity (counts/pixel)
    ('PA',float),              # mean number of background photons
    ('ell',float),             # radius (pixels)
    ('n',float),               # ellipticity
    ('I_e',float),             # surface brightness at half-light radius
    ('r_e',float),             # half-light radius (pixels)
]


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


def SB_conversion(SB_phys):
    """
    Convert surface brightness from
    mag/arcsec^2 to counts/pixel

    Input:
    SB_brigth: surface brightness in mag/arcsec
    """
    plate_scale = 0.263 # Plate scale of DES
    zpm = 30.0 # DES zero-point magnitude

    A = plate_scale**2.0 #Area
    exponent = (zpm - SB_phys)/2.5
    counts = A*(10.**exponent)
    return counts


def create_population(nsim=100, x=SIZE//2, y=SIZE//2, I_sky = 23,
                      PA=90., ell=0.5, n=1.0, I_e = 25., r_e = 3.):
    """ Create the population parameters
    Parameters
    ----------
    nsim : number of simulations
    x,y  : centroid location [pix]
    I_sky: background intensity [mag/arcsec^2]
    PA: position angle [deg]
    ell: ellipticity
    n: Sérsic index
    I_e: surface brightness at effective radius [mag/arcsec^2]
    r_e: effective radius [arcsec]
    """
    ones = np.array([1,1])
    x *= ones
    y *= ones
    I_sky *= ones
    PA *= ones
    ell *= ones
    n *= ones
    I_e *= ones
    r_e *= ones

    params = np.recarray(nsim,dtype=DTYPE)
    params['id']   = np.arange(nsim)
    params['x']    = np.random.uniform(x[0],x[1],nsim)
    params['y']    = np.random.uniform(y[0],y[1],nsim)
    params['I_sky'] = np.random.uniform(I_sky[0],I_sky[1],nsim)
    params['PA'] = np.random.uniform(PA[0],PA[1],nsim)
    params['ell'] = np.random.uniform(ell[0],ell[1],nsim)
    params['n']  = np.random.uniform(n[0],n[1],nsim)
    params['I_e']  = np.random.uniform(I_e[0],I_e[1],nsim)
    params['r_e']  = np.random.uniform(r_e[0],r_e[1],nsim)

    return params


def create_galaxy(params):
    """
    Function that creates halaxy images given a set of parameters
    (coordinates, background photons, PA, ell, n, I_e, r_e)
    """
    #below 2 lines are to supress redundant pyimfit outputs
    old_stdout = sys.stdout         # backup current stdout
    sys.stdout = open(os.devnull, "w")

    # Get relevant parameters
    x_0 = params['x']
    y_0 = params['y']
    I_sky = params['I_sky']
    PA = params['PA']
    ell = params['ell']
    n = params['n']
    I_e = params['I_e']
    r_e = params['r_e']


    # ==========================================
    # Convert physical units to pixel and counts/pixel

    # Plate scale (arcsec/pixel)
    p_scale = 0.263


    # Effective radius in pixels
    r_e_pix  = r_e/p_scale

    # Background sky surface brightess in counts/pixel
    I_sky_counts = SB_conversion(I_sky)
    # Galaxy surface brightness at effective radius in counts/pixel
    I_e_counts = SB_conversion(I_e)

    # ==========================================
    # ==========================================

    # Create model
    model = pyimfit.SimpleModelDescription()
    # define the X0,Y0
    model.x0.setValue(x_0)
    model.y0.setValue(y_0)
    # create a FlatSky uniform background
    FlatSky_function = pyimfit.make_imfit_function("FlatSky")
    FlatSky_function.I_sky.setValue(I_sky_counts)
    # create a Sersic profile
    Sersic_function = pyimfit.make_imfit_function("Sersic")
    Sersic_function.PA.setValue(PA)
    Sersic_function.ell.setValue(ell)
    Sersic_function.n.setValue(n)
    Sersic_function.I_e.setValue(I_e_counts)
    Sersic_function.r_e.setValue(r_e_pix)

    # now add the Flatsky and Sersic profiles to the model function
    model.addFunction(FlatSky_function)
    model.addFunction(Sersic_function)

    imfitter = pyimfit.Imfit(model)
    img = imfitter.getModelImage(shape=(64,64))

    sys.stdout = old_stdout # reset old stdout
    return sp.stats.poisson.rvs(img)


def main(create_population_param: dict):
    """
    Now we create a catalog of

    Paramater values and ranges:

    Position angle, PA ∈[0,180]
    Background surface brightness, Isky= 22.23 mag/arcsec
    Ellipticity in the range e∈[0.05,0.7].
    Sersic index in the range n∈[0.5,1.5].
    Effective surface brightness Ie∈[24.3−25.5]mag/arcsec2
    Effective radius in the range re∈[2.5−6.0] arcsec
    These numbers have been selected to approximately reproduce the parameter range of the bulk of DES Y3 LSBGs (not outliers).

    example create_population_param:
    {'PA': [0.0, 180.0], 'I_sky': 22.23, 'ell': [0.05, 0.7], 'n': [0.5, 1.5], 'I_e': [24.3, 25.5], 'r_e': [2.5, 6.0]}
    """
    catalog = create_population(nsim=nsims, **create_population_param)

    # Get labels - we will use them in the regression task
    #I_sky_true = catalog['I_sky']
    PA_true = catalog['PA']                   # Position angle
    ell_true = catalog['ell']                 # ellipticity
    n_true = catalog['n']                     # Sersic index
    I_e_true = catalog['I_e']                 # surface brightness at effective radius [count/]
    r_e_true = catalog['r_e']                 # effective radius [pix]

    # Concatenate the above parameters in a common y vector
    y_tot = np.column_stack((PA_true,ell_true,n_true,I_e_true,r_e_true))

    scaler = StandardScaler()
    # Rescale the feature space
    scaler.fit(y_tot)
    y_scaled = scaler.transform(y_tot)

    images_training = []
    for i,params in enumerate(catalog):
        img = create_galaxy(params)
        images_training.append(img)


    # Plot them
    n_rows = 4
    n_cols = 5

    f = plt.figure(figsize=(4*n_cols*0.7, 4*n_rows*0.7))

    for i in range(n_rows*n_cols):
        if (i==3):
            plt.title(create_population_param, fontsize=25)
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(images_training[i]/430.)
        plt.axis('off')


    plt.subplots_adjust(wspace=0.01, hspace=0.03)
    plt.savefig(f"generated_datasets/{param}/plot.pdf")

    f.clear()
    plt.close(f)


    X_gals_scaled = np.zeros([nsims,64,64,1])
    scaling = np.zeros(nsims)

    for i in range(nsims):
        scaling[i] = np.max(images_training[i])
        inp = images_training[i]/scaling[i]

        X_gals_scaled[i,:,:,0] = inp

    # Images
    X_train = X_gals_scaled[0:150000]
    X_val = X_gals_scaled[150000:160000]
    X_test = X_gals_scaled[160000:]

    # Labels
    y_train = y_tot[0:150000]
    y_val = y_tot[150000:160000]
    y_test = y_tot[160000:]

    #scaling
    scale_train = scaling[0:150000]
    scale_val = scaling[150000:160000]
    scale_test = scaling[160000:]

    # save training, validation, test
    # Images
    print(f"saving data to ./generated_datasets/{create_population_param}/")

    np.save(f"generated_datasets/{create_population_param}/X_train.npy",X_train)
    np.save(f"generated_datasets/{create_population_param}/X_val.npy",X_val)
    np.save(f"generated_datasets/{create_population_param}/X_test.npy",X_test)

    # Labels
    np.save(f"generated_datasets/{create_population_param}/y_train.npy",y_train)
    np.save(f"generated_datasets/{create_population_param}/y_val.npy",y_val)
    np.save(f"generated_datasets/{create_population_param}/y_test.npy",y_test)

    # Scaling
    np.save(f"generated_datasets/{create_population_param}/scale_train.npy",scale_train)
    np.save(f"generated_datasets/{create_population_param}/scale_val.npy",scale_val)
    np.save(f"generated_datasets/{create_population_param}/scale_test.npy",scale_test)


if __name__ == "__main__":
    plot_pretty()
    config_file = open(sys.argv[1], 'r').readlines()
    param_list = [ast.literal_eval(param) for param in config_file]
    try:
        os.mkdir(os.path.join('./', 'generated_datasets'))
    except FileExistsError:
        print()
    for param in param_list:
        print("Doing param:", param)
        try:
            os.mkdir(os.path.join('generated_datasets/', str(param)))
        except FileExistsError:
            print()
        main(param)
    print("data (train sets, test sets, etc...) created.............................................")
