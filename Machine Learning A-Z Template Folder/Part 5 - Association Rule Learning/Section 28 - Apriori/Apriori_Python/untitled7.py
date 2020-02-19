import numpy as np
import pandas as pd
import matplotlib as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv',header = None)

# since this accepts data in list of lists as transactions
transaction = []
for i in range(0,dataset.shape[0]):
    transaction.append([str(dataset.values[i,j]) for j in range(0,dataset.shape[1])])
    
from apyori import apriori
rules = apriori(transaction,min_support =0.0028,min_confidence=0.2,min_lift =3,min_length = 2 )
results = list(rules)