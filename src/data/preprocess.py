import pandas as pd

def preprocess(path):
    df = pd.read_csv(path)
    df = df.dropna()
    return df
