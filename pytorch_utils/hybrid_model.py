import torch
from torch import nn

class HybridModel(nn.Module):
    
    def __init__(
            self, 
            lue = 1.051,
            replace_vpd = True,
            replace_tmin = False,
            replace_apar = False,
            replace_swc = False,
            replace_combine = False
            mean_vpd = 0,
            mean_tmin = 0,
            mean_swrad = 0,
            mean_fpar = 0,
            mean_swc = 0,
            std_vpd = 1,
            std_tmin = 1,
            std_swrad = 1,
            std_fpar = 1,
            std_swc = 1,
            data_frame = None,
            ):

        super(HybridModel, self).__init__()
        self.replace_vpd = replace_vpd
        self.replace_tmin = replace_tmin
        self.lue = lue
        #if replace_vpd:
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(1, 128)
        self.fc4 = nn.Linear(128, 1)
        self.fc5 = nn.Linear(4, 128)
        self.fc6 = nn.Linear(128, 1)
        self.fc7 = nn.Linear(2, 128)
        self.fc8 = nn.Linear(128, 1)
        self.fc9 = nn.Linear(1, 128)
        self.fca = nn.linear(128,1)
        self.relu = nn.ReLU()

        if not data_frame is None:
            self.get_mean_std()
        else: 
            self.mean_vpd = mean_vpd
            self.mean_tmin = mean_tmin
            self.mean_swrad = mean_swrad
            self.mean_fpar = mean_fpar
            self.mean_swc = mean_swc
            self.std_vpd = std_vpd
            self.std_tmin = std_tmin
            self.std_swrad = std_swrad
            self.std_fpar = std_fpar
            self.std_swc = std_swc




    def forward(self,x):

        swrad = x[:,0:1]
        tmin = x[:, 1:2]
        vpd = x[:, 2:3]
        fpar = x[:, 3:4]
        swc = x[:, 4:5]
       
        if self.replace_vpd:
            vpd = (vpd - self.mean_vpd) / self.std_vpd
            vpd_scalar_0 = self.fc1(vpd)
            vpd_scalar_0 = self.relu(vpd_scalar_0)
            vpd_scalar = self.fc2(vpd_scalar_0)
        else:
            vpd_scalar = 1 - (torch.clamp(vpd, 650, 2400) - 650) / 1750

        if self.replace_swc:
            swc = (swc - self.mean_swc) / self.std_swc
            swc_scalar = self.fc9(swc)
            swc_scalar = self.relu(swc_scalar)
            swc_scalar = self.fca(swc)
        else:
            swc = ((swc - 0.1) / (0.25 - 0.1))**0.383
            swc = torch.clamp(swc, 0, 1)

        if self.replace_tmin:
            tmin = (tmin - self.mean_tmin) / self.std_tmin
            tmin_scalar = self.fc3(tmin)
            tmin_scalar = self.relu(tmin_scalar)
            tmin_scalar = self.fc4(tmin_scalar)
        else:
            tmin_scalar = (torch.clamp(tmin, -7, 9.5) + 7) / 16.5
        
        if self.replace_apar:
            swrad = (swrad - self.mean_swrad) / self.std_swrad
            fpar = (fpar - self.mean_fpar) / self.std_fpar
            apar = self.fc7(torch.concat([swrad, fpar], axis = -1))
            apar = self.relu(apar)
            apar = self.fc8(apar)
        else:
            apar = (swrad * 0.45) * fpar

        if self.replace_combine:
            out = self.fc5(torch.concat([tmin_scalar, vpd_scalar, apar, swc_scalar], axis = -1))
            out = self.relu(out)
            out = self.fc6(out)
        else:
            out =  tmin_scalar * vpd_scalar * apar 

        return self.lue * out
