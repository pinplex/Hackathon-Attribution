
import os
import shutil

from hackathon.models.simplemlp import SimpleMLPRunner
from hackathon.models.second import RandomBaselineRunner
from hackathon.models.linear import LinearRunner

models = [LinearRunner, RandomBaselineRunner]

for Runner in models:
    model_name = Runner.__module__.split('.')[-1]
    log_dir = f'./hackathon/logs/{model_name}'
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    # Training.
    runner = Runner(log_dir=log_dir, seed=910)
    trainer, datamodule, model = runner.train()
    runner.predict(trainer=trainer, datamodule=datamodule, version='final')
    runner.save_model(model=model, version='final')
