
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
| `hackathon/model_runner.py` | The model runner class. |
| `hackathon/data_pipeline.py` | The dataloader. |
| `benchmark.py` | Evaluate model. |

## Getting started

1. Make a copy of `hackathon/models/linear.py` within the same directory and rename it to a meaningful name, e.g., `hackathon/models/rnn.py`.
1. Replace the class `Linear` with your model, give it a meaningful name, e.g., `class RNN(BaseModel)`. The model **must** subclass `BaseModel`!
1. Define a function `model_setup` within the same file that returns an initialized model. The function does not take any arguments (hint: number of features=8). The `model_setup` function must follow this pattern:
    ```
    def model_setup(norm_stats: dict[str, Tensor]) -> BaseModel:
        """Create a model as subclass of hackathon.base_model.BaseModel.

        Parameters
        ----------
        norm_stats: Feature normalization stats with signature {'mean': Tensor, 'std': Tensor},
            both tensors with shape (num_features,).

        Returns
        -------
        A model.
        """
        model = MyCustomModel(num_layers=1000000)  # <- Is a subclass of BaseModel.

        return model
    ```
1. Add your model to the `benchmark.py` file (import and add it `models = [MyModel]`).
1. To run your model with `python benchmark.py` (`--quickrun` for developer run over one epoch / CV fold with less training and validation data).

## Logging

* The `log_dir` argument sets the base directory of the experiment.
* Cross validation runs are each put into a subdirectory (`log_dir/fold_00` etc.)
* The final model (ensemble of all CV models) and its predictions are saved in `log_dir/final`
