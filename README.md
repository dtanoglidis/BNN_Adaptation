# BNN_Adaptation
Bayesian Neural Networks with Domain Adaptation

Repo that contains notes and code on Bayesian Neural Networks and Domain Adaptation

Codes are built upon `https://github.com/dtanoglidis/BNN_LSBGs_ICML`

# Modules and usages
`data_gen.py`: generates synthetic data (test sets, training sets,...etc) for training and testing

`bnn_train.py`: trains bnn (i.e creating h5 files) on generated synthetic data

`hierarchical_inf.py`: doing hierarchical inference on trained bnn

`plot_calibration.py`: plot calibration plots (see notebook for a demonstration)

`param_config/`: directory of parameter config

`notebooks/`: directory of jupyter notebooks/reports

# The pipeline and how it's going:
[x] synthetic data
[x] train bnn
[] hierarchical inference (not yet started)
[x] calibration plotting

# some current concerns to be addressed next meeting:
[] data generation phase: what does it mean to sample from data of different distributions? Is my current approach (data sampled from different distribution BEFORE being put into pyimfit for processing) correct?
