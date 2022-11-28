import csv

import xarray as xr

data_path = '../simple_gpp_model/data/CMIP6/predictor-variables_historical+GPP.nc'
prediction_path = '../hackathon/logs/simplemlp/final/predictions.nc'

prediction = xr.open_dataset(prediction_path)
gpp_hat = prediction.GPP_hat.data.reshape(-1)

data_nc = xr.open_dataset(data_path)
var1 = data_nc.var1.data
var2 = data_nc.var2.data
var3 = data_nc.var3.data
var4 = data_nc.var4.data
var5 = data_nc.var5.data
var6 = data_nc.var6.data
var7 = data_nc.var7.data
co2 = data_nc.co2.data
GPP = data_nc.GPP.data

for data, name in [(var1.reshape(-1), 'var1'), (var2.reshape(-1), 'var2'), (var3.reshape(-1), 'var3'), (var4.reshape(-1), 'var4'), (var5.reshape(-1), 'var5'), (var6.reshape(-1), 'var6'), (var7.reshape(-1), 'var7'), (co2.reshape(-1), 'co2')]:
    with open('features/' + name + '.csv', 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', name])
        for i in range(data.shape[0]):
            writer.writerow([i, data[i]])

with open('labels/GPP.csv', 'w') as fp:
    writer = csv.writer(fp)
    writer.writerow(['id', 'GPP'])
    for i in range(GPP.reshape(-1).shape[0]):
        writer.writerow([i, GPP.reshape(-1)[i]])

with open('predictions/predictions.csv', 'w') as fp:
    writer = csv.writer(fp)
    writer.writerow(['id', 'GPP'])
    for i in range(gpp_hat.shape[0]):
        writer.writerow([i, gpp_hat[i]])

