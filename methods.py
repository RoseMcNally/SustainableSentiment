import pandas as pd
import re


def read_model_data(path):
    data = pd.read_csv(path)
    data = data.dropna()
    data["Relevance"] = data["Relevance"].apply(lambda x: int(x))
    data["Text"] = data["Text"].apply(lambda x: re.sub("&amp;", "and", x))
    return data
