# Overview
This directory hosts all parameter configs. These specify which parameter we would like to run our code on.

## original_param
The original config is based on the one appeared in [the original code](https://github.com/dtanoglidis/BNN_LSBGs_ICML), which specify:

{"PA":[0., 180.], "I_sky":22.23, "ell":[0.05, 0.7], "n":[0.5, 1.5], "I_e":[24.3, 25.5], "r_e":[2.5, 6.0]}

## To specify a new param

To specify new parameters, simply substitute the values of the original param config:

{"PA":[from this, to this], "I_sky":value, "ell":[from this, to this], "n":[from this, to this], "I_e":[from this, to this], "r_e":[from this, to this]}

Note that a param config file can have multiple param configs. For example:
```
{"PA":[0., 180.], "I_sky":22.23, "ell":[0.05, 0.7], "n":[0.5, 1.5], "I_e":[24.3, 25.5], "r_e":[2.5, 6.0]}
{"PA":[0., 90.], "I_sky":2.0, "ell":[0.1, 1.0], "n":[0.5, 5.0], "I_e":[20., 25.], "r_e":[2.5, 3.0]}
{"PA":[0., 180.], "I_sky":20., "ell":[0., 100.], "n":[10., 11.5], "I_e":[14.3, 15.5], "r_e":[1.5, 2.0]}
```

All codes would run like normal.
