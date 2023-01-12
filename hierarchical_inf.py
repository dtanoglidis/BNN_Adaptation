"""
-------------------------------------------
: Purpose                                 :
-------------------------------------------
File for conducting hierarchical inference, following Wagner-Carena's Ovejero.
So far, the function `run_sampler` takes care of everything.

to use: python3 hierarchical_inf.py [weight_source] [motherpath]


-------------------------------------------
: Overview                                :
-------------------------------------------
We will be following Wagner-Carena's ovejero, in particular, hierarchical_inference.py

However, whereas ovejero concerns itself with three types of BNN parameterization, we only concern
ourselves with one: we will work with the BNN parameterization we (tanoglidis) already have, which
is a 5D gaussian. (This case is similar to ovejero's parameterization choice "Full Gaussian"?)


-------------------------------------------
: Proceeding                              :
-------------------------------------------
Overview: Wagner-Carena et al's hierarchical inference proceed as follow:
    1/ gen_samples: sampling xi from trained BNN
    2/ initialize_sampler: an emcee sampler, initialized as follow:
            emcee.EnsembleSampler(n_walker, ndim, log_post_omega)
    3/ run_sampler(num_sample)
   (4/ plot_chain)

The main thing we need to be concern with is how to obtain log_post_omega, or equation 10 in the paper.
According to Wagner-Carena:
    log_post_omega returns lprior + np.sum(like_ratio)
    - lprior = log_p_omega
    - like_ratio = a function dependent on log_p_xi_omega

"""

from scipy import special
import numpy as np
import scipy.stats as stats
import emcee, copy, sys

from plot_calibration import main, get_model, make_calibration


###########################################
# Not from ovejero                        #
###########################################

def tbi():
    raise ValueError("need implementation")



def make_target_eval_dict():
    """
    ovejero, target_eval_dict.keys() give
    dict_keys(['hyp_len', 'hyp_init', 'hyp_sigma', 'hyp_prior', 'hyp_names', 'external_shear_gamma_ext', 'external_shear_psi_ext', 'lens_mass_center_x', 'lens_mass_center_y', 'lens_mass_e1', 'lens_mass_e2', 'lens_mass_gamma', 'lens_mass_theta_E'])
    """
    tbi()



def make_interim_eval_dict():
    """
    maybe
    dict_keys(['hyp_len', 'hyp_values', 'hyp_names', 'external_shear_gamma_ext', 'external_shear_psi_ext', 'lens_mass_center_x', 'lens_mass_center_y', 'lens_mass_e1', 'lens_mass_e2', 'lens_mass_gamma', 'lens_mass_theta_E'])
    """
    tbi()



#def make_hyp():
#    """
#    hyp: np array with dimensions (n_hyperparameters).
#         These are the values of omega's parameters
#    """
#    tbi()



def make_lens_params_train():
    """
    cfg['dataset_params']['lens_params'].copy()
    """
    tbi()




def gen_samples(weight_source: str, motherpath: str, k: int):
    """ [should b done]
    generating xi ~ BNN

    parameters:
        weight_source: path to bnn weight source
        motherpath   : path to data to compare current model against
        k            : number of sample

    note: Tanoglidis' BNN is a 5D Gaussian with full covariance
    """
    model = get_model()
    sample, y_keep, _ = main(model, weight_source, motherpath, k)

    xi = sample

    return xi, y_keep



def plot(inv_sampl, true):
    """
    plotting code.
    Have the choice of plotting non-inference and inference. Should we?
    """

    # plotting
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


    print("done prepping. Now onto making calibration plots")
    calibration_dict = {}
    for param in dictionary.keys():
        sample, true = dictionary[param]
        make_calibration(sample, true, param, f'./hierarchical_inf_out')


###########################################
# From ovejero (but slightly modified)    #
###########################################
def convert_tril_to_cov(tril):
	""" [ copied. Kept as is ]
	Convert a list of lower triangular matrix entries to the corresponding
	covariance matrix.

	Parameters:
		tril (np.array): A 1D array with the lower triangular values

	Returns:
		(np.array): The covariance matrix
	"""
	# Format the lower triangular matrix and return the dot product.
	n_params = int(0.5*(np.sqrt(1+8*len(tril))-1))
	tril_mask = np.tri(n_params,dtype=bool, k=0)
	tril_mat = np.zeros((n_params,n_params))
	tril_mat[tril_mask] = tril
	return np.dot(tril_mat,tril_mat.T)



def log_p_omega(hyp):
    """ [done]
    calculate p(omega)
    """
	# We iterate through each hyperparamter and evaluate it on its prior.
	logpdf = 0
	for hpi, hyper_param in enumerate(hyp):
		logpdf += target_eval_dict['hyp_prior'][hpi](hyper_param)

	# Give -np.inf in the case of a nan.
	if np.sum(np.isnan(logpdf))>0:
		logpdf = -np.inf

	return logpdf



def log_p_xi_omega(hyp: np.array, eval_dict: dict, lens_params: list):
	""" [done]
	Calculate log p(xi|omega) - the probability of the lens parameters in
	the data given the proposed lens parameter distribution.

	hyp (np.array): A numpy array with dimensions (n_hyperparameters).
			These are the hyperparameters that will be used for evaluation.
	eval_dict (dict): A dictionary from build_evaluation_dictionary to
			query for the evaluation functions.
	lens_params ([str,...]): A list of strings of the lens parameters
			generated by baobab.

	Returns:
		np.array: A numpy array of the shape (n_samps,batch_size) containing
		the log p(xi|omega) for each sample.
	"""

    #lens_samps = make_lens_samps()
    samples = xi

    # We iterate through each lens parameter and carry out the evaluation
	logpdf = np.zeros((samples.shape[1],samples.shape[2]))

	for li, lens_param in enumerate(lens_params):
		# Skip covariance parameters.
		if ('cov_params_list' in eval_dict and
			lens_param in eval_dict['cov_params_list']):
			continue
		logpdf += eval_dict[lens_param]['eval_fn'](samples[li],
			*hyp[eval_dict[lens_param]['hyp_ind']],
			**eval_dict[lens_param]['eval_fn_kwargs'])

	# Calculate covariate parameters
	if 'cov_params_list' in eval_dict:
		# Identify the samples associated with the covariance parameters
		cov_samples_index = []
		for cov_lens_param in eval_dict['cov_params_list']:
			cov_samples_index.append(lens_params.index(cov_lens_param))
		cov_samples = samples[cov_samples_index]
		for ili, is_log in enumerate(eval_dict['cov_params_is_log']):
			if is_log:
				cov_samples[ili,cov_samples[ili]<=0] = 1e-22
				cov_samples[ili] = np.log(cov_samples[ili])

		# Get the mean and covariance we want to use
		mu = hyp[eval_dict['cov_mu_hyp_ind']]
		tril = hyp[eval_dict['cov_tril_hyp_ind']]
		cov = convert_tril_to_cov(tril)

		# Reshape the covariance samples to feed into the logpdf function
		orig_shape = cov_samples.T.shape
		cov_samples = cov_samples.T.reshape(-1,len(mu))

		logpdf_cov = stats.multivariate_normal(mean=mu,cov=cov).logpdf(
			cov_samples)

		# This is a hardcode, but for axis ratio we want to renormalize by
		# the area cut by q<=1.
		if 'lens_mass_q' in eval_dict['cov_params_list']:
			qi = eval_dict['cov_params_list'].index('lens_mass_q')
			logpdf_cov -= np.log(stats.norm(mu[qi],np.sqrt(cov[qi,qi])).cdf(1))

		logpdf += logpdf_cov.reshape(orig_shape[:-1]).T

	# Clean up any lingering nans.
	logpdf[np.isnan(logpdf)] = -np.inf

	return logpdf



def log_post_omega(hyp):
    """ [done]
    hyp: np array with dimensions (n_hyperparameters).
         These are the values of omega's parameters

    return: log posterior of omega given generated data
    """
    lprior = log_p_omega(hyp)
    if lprior == -np.inf:
        return lprior

    lens_params_train = make_lens_params_train()
    lens_params_test = copy.deepcopy(lens_params_train)
    interim_eval_dict = make_interim_eval_dict()

    pt_omegai = log_p_xi_omega(interim_eval_dict['hyp_values'], interim_eval_dict, lens_params_train)
    pt_omega = log_p_xi_omega(hyp, target_eval_dict, lens_params_test)

    like_ratio = pt_omega - pt_omegai
    like_ratio[np.isinf(pt_omegai)] = -np.inf
	like_ratio = special.logsumexp(like_ratio,axis=0)
	like_ratio[np.isnan(like_ratio)] = -np.inf

    return lprior + np.sum(like_ratio)



def initialize_sampler(n_walkers: int):
    """ [done]
    n_walkers : number of walkers used by the sampler
                must be at least twice the number of hyperparameters

    Return Ensemble sampler and the current state
    """
    #hyp = make_hyp()
    #log_post_omega_ = log_post_omega(hyp)

    save_path = './hierarhical_inf_out'

    ndim = target_eval_dict['hyp_len']

    # start samples at initial value randomly distr around +- sigma
    cur_state = ((np.random.rand(n_walkers, ndim)*2-1)*target_eval_dict['hyp_sigma']
                 + target_eval_dict['hyp_init'])

    # ensure no walkers start at point with log prob -np.inf
    while all_finite is False:
        all_finite = True
        f_counter = 0.0
        for w_i in range(n_walkers):
            if log_post_omega(cur_state[w_i]) == -np.inf:
                all_finite = False
                f_counter +=1
                cur_state[w_i] = cur_state[np.random.randint(n_walkers)]
        if f_counter > n_walkers*0.7:
            raise RuntimeError('Too few (%.3f) of the initial'%(1-f_counter/n_walkers)+
                               'walkers have finite probability!')

    # Initialize backend hdf5 file that will store samples as we go
    self.backend = emcee.backends.HDFBackend(save_path)

    # Very important I pass in prob_class.log_post_omega here to allow
    # pickling.
    sampler = emcee.EnsembleSampler(n_walkers, ndim,
        log_post_omega, backend=self.backend, pool=None)

    return sampler, cur_state



def run_sampler(n_samps: int, n_walkers: int):
    """ [done]
    run an emcee sampler to get a posterior on the hyperparameters

    n_samps   : number of samples to take
    n_walkers : number of walkers used by the sampler
                must be at least twice the number of hyperparameters

    note: we break with tradition and instead of initializing and running the sampler
          in two steps, this function handles both initializing and sampling.
    """
    sampler, cur_state = initialize_sampler(n_walkers)

    sampler.run_mcmc(cur_state, n_samps, progress=True)
    return sampler



if __name__=="__main__":
    try:
        os.mkdir(os.path.join('./', 'hierarchical_inf_out'))
    except FileExistsError:
        print()

    global target_eval_dict
    target_eval_dict = make_target_eval_dict()

    global xi
    xi, y_keep = gen_samples(str(sys.argv[1]), str(sys.argv[2]), 1000)

    sampler_ran = run_sampler(100, 50)

    inv_sampl = sampler_ran.get_chain() #TODO
    plot(inv_sampl, y_keep)
