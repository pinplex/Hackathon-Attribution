
import os
import shutil
from argparse import ArgumentParser, Namespace
from itertools import product
from functools import partial
import numpy as np
import json

from hackathon.model_runner import ModelRunner

from hackathon.models.attn_nores import model_setup as attn_model

model_funs = [attn_model]


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
            patience=5,
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


def tune(args: Namespace):
    search_space = {
        'd_model': [8, 16, 32],
        'num_head': [1, 2, 4],
        'num_hidden': [16, 32, 64],
        'num_layers': [1, 2, 3],
        'dropout': [0.0, 0.15, 0.3],
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'weight_decay': [1e-4, 1e-3, 1e-2],
    }
    search_params = list()
    for params in product(*search_space.values()):
        params_kw = {k: v for k, v in zip(search_space.keys(), params)}
        search_params.append(params_kw)
    num_trials = min(args.num_trials, len(search_params))
    print(f'\nTuning model with {num_trials} trials\n{"=" * 40}\n')
    rand_search_params = np.random.choice(search_params, size=num_trials)

    model_name = attn_model.__module__.split('.')[-1]
    base_dir = f'./hackathon/logs/{model_name}/tune/'
    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)

    summary = {}
    for i, params in enumerate(rand_search_params):
        print(f'\n >> Starting trial {i + 1:3d} of {num_trials}\n')
        model_fun = partial(attn_model, **params)
        log_dir = os.path.join(base_dir, f'trial_{i:03d}')

        # Training.
        runner = ModelRunner(log_dir=log_dir, quickrun=args.quickrun, seed=910)
        _, _, scores = runner.train(
            model_fn=model_fun,
            fold=0,
            patience=5,
            max_epochs=1 if args.quickrun else -1,
            accelerator=None if args.gpu == -1 else 'gpu',
            devices=None if args.gpu == -1 else f'{args.gpu},')

        params.update({'test_loss': scores[0][0]['test_loss']})
        summary.update({f'trial_{i:03d}': params})

        with open(os.path.join(base_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)


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
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform model tuning.')
    parser.add_argument(
        '--num_trials',
        type=int,
        default=100,
        help='Number.')

    args = parser.parse_args()

    if args.tune:
        tune(args)
    else:
        main(args)
