
# Attribution hackathon

This repository contains code and data for the attribution hackathon.

[This pad](https://pad.gwdg.de/YoDSoLPUQmaAgNcuw_vVAA#) contains more discussion and ideas.

# Getting started

1. [Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
1. Create the conda environent: `conda env create -f attr_hack_env.yml`
1. Activate conda environment: `conda activate attr_hack`
1. Run data processing / simulation (only needed to re-run data processing, you may skip this step).
    * GPP model
        * `python Simple_GPP_model/process_predictor-variables.py`
        * `python Simple_GPP_model/gpp_model.py`
    * SM model
        * `python simple_sm_model/era_sm_sim.py`

# Hackathon guide

## Relevant files

| file | description |
| ---  | ---         |
| `hackathon/models/linear.py` | A dummy model. |
| `hackathon/base_model.py` | The base model class. |
| `hackathon/base_runner.py` | The base runner class. |
| `hackathon/data_pipeline.py` | The dataloader. |
| `benchmark.py` | Evaluate model. |

## Getting started

1. Make a copy of `hackathon/models/linear.py` within the same directory and rename it to a meaningful name, e.g., `hackathon/models/rnn.py`.
1. Replace the class `Linear` with your model, give it a meaningful name, e.g., `class RNN(BaseModel)`. The model **must** subclass `BaseModel`!
1. Create your own runner:
    * Rename the existing class `LinearRunner` to something meaningful (e.g., `LinearRunner`), and define how the `datamodule` and the `model` are created in the methods `data_setup` and `model_setup`, respectively. Hard-code all parameters. The parameters of `DataModule` may be changed, they current values are probably not very smart (e.g., temporal splitting).
    * Override the method `train`: this is the training routine. It is important that you return a single model from the `train` method; you can do a cross validation within this method and combine multiple trained models with the class `hackathon.base_runner.Ensemble`.
1. Add your model to the `benchmark.py` file (import and add it `models = [MyModel]`).
1. To run your model with `python benchmark.py`.

__Craete your own model and don't push it to the common repository until the hackathon meeting (to make sure that we come up with independent solutions).__

__Don't just take the parameters from the dumy model, they are intentionally set to bad values (e.g., temporal splitting does not even use all data)__

## Logging

* The `log_dir` argument sets the base directory of the experiment.
* The `version` argument in `MyRunner.trainer_setup(version=...)` sets the logging to `log_dir/version`, where checkpoints and logs are saved to. A version could be a cross validation fold, for example. Use tensorboard to check the logs. You most likely call `MyRunner.trainer_setup` from within `MyRunner.train`.
* If you need predictions for every version run (e.g., cross validation fold), call `self.predict(trainer=trainer, datamodule=datamodule, version=version)` as in the `linear.py` example. The final predictions on the test set are saved to `log_dir/version/predictions.nc`
* The `benchmark.py` script evaluates the model and saves whatever model is return from `MyRunner.run(...)` along with final predictions, both with version `final`.
