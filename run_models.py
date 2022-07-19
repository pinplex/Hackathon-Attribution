
import os

from hackathon.models.linear import run as linear

for runner in [linear]:
    model_name = runner.__module__.split('.')[-1]
    log_dir = f'./hackathon/eval/{model_name}'

    # Evaluate.
    model, eval_loader = runner(root_dir=log_dir, version='final')
    eval_loader.dataset.ds.to_netcdf(os.path.join(log_dir, 'predictions.nc'), seed=910)

    # Can apply XAI here.
    model, eval_loader = runner(rerun=False, root_dir=log_dir, version='final', seed=910)
