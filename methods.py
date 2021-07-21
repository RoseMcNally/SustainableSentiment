import pandas as pd


def read_model_data(path):
    data = pd.read_csv(path)
    data = data.dropna()
    data["Relevance"] = data["Relevance"].apply(lambda x: int(x))
    return data
