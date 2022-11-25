
import xarray as xr
from torch.utils.data import DataLoader

from hackathon.base_model import BaseModel
from hackathon.base_explanation import BaseExplainer

class TestExplainer(BaseExplainer):
    def __init__(self):
        pass

    def _custom_explanations(
            self,
            model: BaseModel,
            test_dataloader: DataLoader) -> tuple[tuple[float], float, xr.Dataset]:

        for batch in test_dataloader:
            y_hat = model(batch)
            print(y_hat.shape)
            # test_dataloader.dataset.assign_sensitivities(
            #     sensitivities=...,
            #     data_sel=batch['data_sel']
            # )


