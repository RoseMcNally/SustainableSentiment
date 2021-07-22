import re
import methods
from bert_optimisation import tune_bert_hyperparameters

if __name__ == '__main__':

    model_data = methods.read_model_data("data/all/mum_filtered_with_duplicates.csv")
    model_data["Text"] = model_data["Text"].apply(lambda x: re.sub("@", "", x))

    learning_rates = [1e-5]
    batches = [16]
    weight_decays = [0.1]
    hyps = {"learning_rates": learning_rates, "batches": batches, "weight_decays": weight_decays}

    tune_bert_hyperparameters(model_data, "vinai/bertweet-base", "data/bertweet", 50, hyps)
