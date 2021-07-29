import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel
from general_bert_optimizer import BertTokenizer

from general_bert_arch import GeneralBertArch

model_name = "vinai/bertweet-base"
max_len = 50
length_of_sub = 1000


def split_data_and_filter(input_file, output_folder):
    data = pd.read_csv(input_file, index_col=0)
    num_subs = data.shape[0] // length_of_sub + 1

    bert = AutoModel.from_pretrained(model_name)
    model = GeneralBertArch(bert)
    model.load_state_dict(torch.load("data/bertweet/other/best_weights.pt"))

    for i in tqdm(range(num_subs)):
        filter_data(data.iloc[i * length_of_sub: (i + 1) * length_of_sub, :], output_folder + f"/sub_{i}.csv", model)


def filter_data(data, output_file, model):
    print("Cleaning data...")

    text = data["Text"]
    text = text.apply(lambda x: re.sub("&amp;", "and", x))
    text = text.apply(lambda x: re.sub("@", "", x))

    print("Tokenizing data...")

    bert_tokenizer = BertTokenizer(model_name, max_len)
    seq, mask = bert_tokenizer.tokenize_text_only(text)

    print("Labelling data...")

    with torch.no_grad():
        preds = model(seq, mask)
        preds = preds.numpy()

    preds = np.argmax(preds, axis=1)

    print("Saving data...")

    preds = pd.DataFrame(preds, columns=["Label"])
    data.reset_index(drop=True, inplace=True)
    data_labelled = pd.concat([data, preds], axis=1)
    data_labelled = data_labelled[data_labelled["Label"] == 1]
    data_labelled = data_labelled.drop(columns=["Label"])
    data_labelled.to_csv(output_file, index=False)

    print("Done!")


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    split_data_and_filter("data/all/clean_filtered.csv", "data/sentiment_input/filtered")
