import os
import re

import emoji
import pandas as pd
import wordninja
from pysentimiento import SentimentAnalyzer
from tqdm import tqdm

from bert_preprocessor import BertPreprocessor

input_path = "data/sentiment_input/filtered"
output_path = "data/sentiment_output"


def analyse_sentiment(text, model):
    text = BertPreprocessor.bert_text_preprocess(text, True)
    sentiment = model.predict(text)
    return sentiment.output, sentiment.probas["POS"], sentiment.probas["NEU"], sentiment.probas["NEG"]


if __name__ == '__main__':
    # Run through folder and read all csvs into one big pandas df
    print("Loading files...")
    data_list = list()
    for file in tqdm(os.listdir(input_path)):
        if file.endswith(".csv"):
            try:
                file_df = pd.read_csv(input_path + "/" + file)
                data_list.append(file_df)
            except pd.errors.EmptyDataError:
                pass
    data = pd.concat(data_list, ignore_index=True)
    # data = pd.read_csv("data/sentiment_input/filtered/sub_6.csv")

    print("Loading model...")
    analyser = SentimentAnalyzer(lang='en')

    print("Analysing sentiment...")
    tqdm.pandas()
    sent = data.progress_apply(lambda row: analyse_sentiment(row["Text"], analyser), axis=1)

    print("Saving results...")
    sent = pd.DataFrame([[a, b, c, d] for a, b, c, d in sent.values], columns=["Sentiment", "POS", "NEU", "NEG"])
    sent = pd.DataFrame(sent, columns=["Sentiment", "POS", "NEU", "NEG"])
    data = pd.concat([data, sent], axis=1)
    data.to_csv(output_path + "/" + "sentiment.csv", index=False)
    # Save df in csv file