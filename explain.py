
from argparse import ArgumentParser, Namespace
import torch

from hackathon.model_runner import ModelRunner
from hackathon.explainers.test_explainer import TestExplainer
from hackathon.base_model import BaseModel
# from hackathon.models.attn import MultiheadAttn as attn_model
# from hackathon.models.Conv1D import Conv1D as conv1d_model
from hackathon.models.linear import Linear as linear_model
# from hackathon.models.LSTM import LSTM as lstm_model
# from hackathon.models.multimodel import EfficiencyModel as efficiency_model
# from hackathon.models.resnet import ResNetModule as resnet_model
# from hackathon.models.simplemlp import SimpleMLP as simplemlp_model

model_funs = [
    # attn_model,
    # conv1d_model,
    linear_model,
    # lstm_model,
    # efficiency_model,
    # resnet_model,
    # simplemlp_model
]

def main(args: Namespace):
    explainer = TestExplainer()
    model: BaseModel
    for model in model_funs:
        
        model_name = model.__module__.split('.')[-1]
        log_dir = f'./hackathon/logs/{model_name}/expl'
        checkpoint_path = f'./hackathon/logs/{model_name}/xval/final/final.ckpt'

        runner = ModelRunner(log_dir=log_dir, quickrun=args.quickrun, seed=910)
        #model.load_from_checkpoint(checkpoint_path)
        model = torch.load(checkpoint_path)
        test_dataloader = runner.data_setup(fold=-1).test_dataloader()

        explanations = explainer.get_explanations(model, test_dataloader)

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
