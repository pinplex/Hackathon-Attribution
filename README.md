
# Attribution hackathon

This repository contains code and data for the attribution hackathon.

# Getting started

1. [Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create the conda environent: `conda env create -f attr_hack_env.yml`
3. Run data processing / simulation (only needed to re-run data processing, you may skip this step).
    * GPP model
        * `python Simple_GPP_model/process_predictor-variables.py`
        * `python Simple_GPP_model/gpp_model.py`
    * SM model
        * `python simple_sm_model/era_sm_sim.py`
