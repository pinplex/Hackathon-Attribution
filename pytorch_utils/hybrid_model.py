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
            replace_combine = False, 
            train_lue = False,
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
        
        """ 
        The class to create various hybrid models.

        Params:
            lue (float): the maximum light-use efficiency. The default is 1.051. If train_lue is True, this parameter will be ignored.
            replace_vpd (Bool): Indicates whether the vpd function should be learned or used from the MODIS modell. default is True.
            replace_tmin (Bool): Indicates whether the tmin function should be learned or used from the MODIS modell. default is False.
            replace_apar (Bool): Indicates whether the swrs and fPAR function should be learned or used from the MODIS modell. default is False.
            replace_swc (Bool): Indicates whether the swc function should be learned or used from the MODIS modell. default is False.
            replace_combine (Bool): Indicates whether the combination of fuctions (the multiplication) should be learned or taken from the MODIS model. The default value is False
            train_lue (Bool): Indicates whether the provided LUE value should be used or a LUE value should be trained from the data.
        """

        super(HybridModel, self).__init__()
        self.replace_vpd = replace_vpd
        self.replace_tmin = replace_tmin
        self.replace_apar = replace_apar
        self.replace_swc = replace_swc
        self.replace_combine = replace_combine
        self.train_lue = train_lue

        if train_lue:
            self.register_parameter( name = 'lue', parameter = nn.Parameter(torch.random.rand(1)))
        else:
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
        self.fca = nn.Linear(128,1)
        self.relu = nn.ReLU()
        
        # mean and std 
        if not data_frame is None:
            self.get_mean_std(data_frame)
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

    def get_mean_std(self,data_frame):
        self.mean_vpd = data_frame.vpd.mean()
        self.std_vpd = data_frame.vpd.std()

        self.mean_tmin = data_frame.t2mmin.mean()
        self.std_tmin = data_frame.t2mmin.std()

        self.mean_swrad = data_frame.ssrd.mean()
        self.std_swrad = data_frame.ssrd.std()

        self.mean_fpar = data_frame.FPAR.mean()
        self.std_fpar = data_frame.FPAR.std()

        self.mean_swc = data_frame.sSWC.mean()
        self.std_swc = data_frame.sSWC.std()
