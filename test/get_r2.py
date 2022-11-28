import xarray as xr 
import glob
import numpy as np
import matplotlib.pyplot as plt

for fold_nr in range(10):
    path = f'../hackathon/logs/multimodel/fold_0{fold_nr}/predictions.nc'

    predictions = xr.open_dataset(path)
    gpp_hat = predictions.GPP_hat.data
    gpp = predictions.GPP.data

    gpp = gpp[:,365:]
    gpp_hat = gpp_hat[:,365:]

    a = gpp_hat.shape[1] % 365

    gpp_hat = gpp_hat[:, a:]
    gpp = gpp[:, a:]

    gpp_hat = gpp_hat.reshape(-1,365)
    T = gpp_hat.shape[0]
    time = np.arange(T)

    print(gpp.shape)
    print(gpp_hat.shape)
    print(gpp[0].shape, gpp_hat.shape)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(gpp[0], label = 'gpp')
    plt.plot(gpp_hat.reshape(-1), label = 'gpp_hat')
    plt.legend()
    plt.subplot(1,2,2)
    plt.scatter(gpp[0], gpp_hat.reshape(-1))
    plt.show()
    plt.close()

    print(np.corrcoef(gpp[0], gpp_hat.reshape(-1))[0,1]**2)

    gpp_hat = gpp_hat - np.mean(gpp_hat, axis = 0)
    gpp = gpp.reshape(-1,365) - np.mean(gpp.reshape(-1,365), axis = 0)

    gpp_hat = gpp_hat.reshape(-1)
    gpp = gpp.reshape(-1)


    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(gpp, label = 'gpp')
    plt.plot(gpp_hat, label = 'gpp_hat')
    plt.legend()
    plt.subplot(1,2,2)
    plt.scatter(gpp, gpp_hat)
    plt.show()
    plt.close()

    print(np.corrcoef(gpp, gpp_hat)[0,1]**2)

