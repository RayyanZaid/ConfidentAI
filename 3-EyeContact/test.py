import pickle as pkl
import pandas as pd
with open("3-EyeContact/MIT_DATA/results2.pkl", "rb") as f:
    object = pkl.load(f, encoding='latin1')
    
df = pd.DataFrame(object)
df.to_csv(r'results2.csv')