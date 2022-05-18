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
            replace_combine = False):

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

    def forward(self,x):

        swrad = x[:,0:1]
        tmin = x[:, 1:2]
        vpd = x[:, 2:3]
        fPAR = x[:, 3:4]
        swc = x[:, 4:5]
       
        if self.replace_vpd:
            vpd_scalar_0 = self.fc1(vpd)
            vpd_scalar_0 = self.relu(vpd_scalar_0)
            vpd_scalar = self.fc2(vpd_scalar_0)
        else:
            vpd_scalar = 1 - (torch.clamp(vpd, 650, 2400) - 650) / 1750

        if self.replace_swc:
            swc_scalar = self.fc9(swc)
            swc_scalar = self.relu(swc_scalar)
            swc_scalar = self.fca(swc)
        else:
            swc = ((swc - 25) / (100 - 25))**0.383
            swc = torch.clamp(swc, 0, 1)

        if self.replace_tmin:
            tmin_scalar = self.fc3(tmin)
            tmin_scalar = self.relu(tmin_scalar)
            tmin_scalar = self.fc4(tmin_scalar)
        else:
            tmin_scalar = (torch.clamp(tmin, -7, 9.5) + 7) / 16.5
        
        if self.replace_apar:
            apar = self.fc7(torch.concat([swrad, fPAR], axis = -1))
            apar = self.relu(apar)
            apar = self.fc8(apar)
        else:
            apar = (swrad * 0.45) * fPAR

        if self.replace_combine:
            out = self.fc5(torch.concat([tmin_scalar, vpd_scalar, apar, swc_scalar], axis = -1))
            out = self.relu(out)
            out = self.fc6(out)
        else:
            out =  tmin_scalar * vpd_scalar * apar 

        return self.lue * out

    def get_mean_std(self,df):
        self.mean_vpd = df.vpd.mean()
        self.std_vpd = df.vpd.std()

        self.mean_tmin = df.t2mmin.mean()
        self.std_tmin = df.t2mmin.std()

        self.mean_swrad = df.ssrd.mean()
        self.std_swrad = df.ssrd.std()

        self.mean_fPAR = df.FPAR.mean()
        self.std_fPAR = df.FPAR.std()

        self.mean_swc = df.sSWC.mean()
        self.std_swc = df.sSWC.std()