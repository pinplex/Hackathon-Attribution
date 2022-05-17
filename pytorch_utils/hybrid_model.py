import torch
from torch import nn

class HybridModel(nn.Module):
    
    def __init__(self, replace_vpd = True):
        super(HybridModel, self).__init__()
        self.replace_vpd = replace_vpd
        #if replace_vpd:
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self,x):

        swrad = x[:,0:1]
        tmin = x[:, 1:2]
        vpd = x[:, 2:3]
        fPAR = x[:, 3:4]
       

        if self.replace_vpd:
            vpd_scalar_0 = self.fc1(vpd)
            vpd_scalar_0 = self.relu(vpd_scalar_0)
            vpd_scalar = self.fc2(vpd_scalar_0)
        else:
            vpd_scalar = 1 - (torch.clamp(vpd, 650, 2400) - 650) / 1750

        tmin_scalar = (torch.clamp(tmin, -7, 9.5) + 7) / 16.5
        
        apar = (swrad * 0.45) * fPAR


        return 1.051 * tmin_scalar * vpd_scalar * apar 
