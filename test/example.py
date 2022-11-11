import csv
import numpy as np

from test_dependence import Tester

tester = Tester('./results.json', subset_size = None)
tester.test_dependence(
        features = "features/*",
        labels = "labels/*",
        predictions="predictions/*",
        ind_tests = ['partial_correlation_cont'])

