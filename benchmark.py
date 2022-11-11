
import os
import shutil
from argparse import ArgumentParser, Namespace

from hackathon.model_runner import ModelRunner

from hackathon.models.transformer import model_setup as attn_model
from hackathon.models.simplemlp import model_setup as simplemlp
from hackathon.models.linear import model_setup as linear_model
from hackathon.models.multimodel import model_setup as efficiency_model

model_funs = [efficiency_model, simplemlp]

def main(args: Namespace):
    for model_fn in model_funs:
        
        model_name = model_fn.__module__.split('.')[-1]
        log_dir = f'./hackathon/logs/{model_name}'
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)

        # Training.
        runner = ModelRunner(log_dir=log_dir, quickrun=args.quickrun, seed=910)
        trainer, model = runner.train(
            model_fn=model_fn,
            max_epochs=1 if args.quickrun else -1,
            accelerator=None if args.gpu == -1 else 'gpu',
            devices=None if args.gpu == -1 else f'{args.gpu},')

        # Evaluating.
        datamodule = runner.data_setup(fold=-1)
        #from IPython import embed; embed()
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
