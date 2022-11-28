
import os
import shutil
from argparse import ArgumentParser, Namespace

from hackathon.model_runner import ModelRunner

from hackathon.models.attn import model_setup as attn_model
from hackathon.models.Conv1D import model_setup as conv1d_model
from hackathon.models.linear import model_setup as linear_model
from hackathon.models.LSTM import model_setup as lstm_model
from hackathon.models.multimodel import model_setup as efficiency_model
from hackathon.models.resnet import model_setup as resnet_model
from hackathon.models.simplemlp import model_setup as simplemlp_model

model_funs = [
    attn_model,
    conv1d_model,
    linear_model,
    lstm_model,
    efficiency_model,
    resnet_model,
    simplemlp_model
]

def main(args: Namespace):
    for model_fn in model_funs:
        
        model_name = model_fn.__module__.split('.')[-1]
        log_dir = f'./hackathon/logs/{model_name}/xval'
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)

        # Training.
        runner = ModelRunner(log_dir=log_dir, quickrun=args.quickrun, seed=910)
        trainer, model, _ = runner.train(
            model_fn=model_fn,
            patience=15,
            max_epochs=1 if args.quickrun else -1,
            accelerator=None if args.gpu == -1 else 'gpu',
            devices=None if args.gpu == -1 else f'{args.gpu},')

        # Evaluating.
        datamodule = runner.data_setup(fold=-1)
        runner.predict(
            trainer=trainer,
            model=model,
            datamodule=datamodule,
            version='final')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--quickrun',
        action='store_true',
        help='Quick development run with only 1 epoch and 2 CV folds, less data.')
    parser.add_argument(
        '--gpu',
        type=int,
        default=-1,
        help='GPU ID to use, -1 (default) deactivates GPU.')

    args = parser.parse_args()

    main(args)
