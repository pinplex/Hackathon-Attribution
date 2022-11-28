
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
1. The model `__init__` method must take `**kwargs` and pass them to the parent class (`BaseModel`) like so:
    ```
    class MyModel(BaseModel):
        def __init__(self, num_features: int, num_targets: int, **kwargs) -> None:
            super(MyModel, self).__init__(**kwargs)

            self.linear = torch.nn.Linear(num_features, num_targets)
            self.softplus = torch.nn.Softplus()

        def forward(self, x: Tensor) -> Tensor:
            out = self.softplus(self.linear(x))
            return out
    ```


1. Define a function `model_setup` within the same file that returns an initialized model (e.g., `hackathon/models/rnn.py`). The function must take the argument `norm_stats`. The `model_setup` function must follow this pattern:
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
        model = MyCustomModel(  # <- Is a subclass of BaseModel
            num_features=8,  # <- A model HP
            num_targets=1,  # <- A model HP
            num_layers=1000000,  # <- A model HP
            ... # more model HPs
            learning_rate=0.01,  # <- BaseModel kwarg
            weight_decay=0.0,  # <- BaseModel kwarg
            norm_stats=norm_stats)  # <- BaseModel kwarg

        return model
    ```
1. Add your model to the `benchmark.py` file by importing and adding the `model_setup` function:
    ```
    from hackathon.models.linear import model_setup as linear_model

    model_funs = [linear_model]
    ```
1. Run your model with `python benchmark.py` (`--quickrun` for developer run over one epoch / CV fold with less training and validation data).

## Logging

* The `log_dir` argument sets the base directory of the experiment.
* Cross validation runs are each put into a subdirectory (`log_dir/fold_00` etc.)
* The final model (ensemble of all CV models) and its predictions are saved in `log_dir/final`

0  0 1  1 2
1  1 0  2 1

