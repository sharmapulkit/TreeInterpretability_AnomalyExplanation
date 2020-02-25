import numpy as np
import csv 
import pandas as pd
import os

PATH='~/Documents/PDF/umass/sem2/cs696ds/'

dataset = os.path.join(PATH, 'datasets/networking-results.csv')

data = pd.read_csv(dataset)

print(data.shape)
