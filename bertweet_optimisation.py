import re
import numpy as np
import torch
import methods
from bert_optimisation import tune_bert_hyperparameters

if __name__ == '__main__':

    np.random.seed(42)
    torch.manual_seed(42)

    model_data = methods.read_model_data("data/all/mum_filtered_with_duplicates.csv")
    model_data["Text"] = model_data["Text"].apply(lambda x: re.sub("@", "", x))

    learning_rates = [5e-6]
    batches = [64]
    weight_decays = [0.2]
    hyps = {"learning_rates": learning_rates, "batches": batches, "weight_decays": weight_decays}

    tune_bert_hyperparameters(model_data, "vinai/bertweet-base", "data/bertweet/other", 50, hyps)

    # Best model result:
    # Batch size: 64 , Learning rate: 5e-06 , Weight decay: 0.2
    #
    #              precision    recall  f1-score   support
    #
    #            0       0.84      0.62      0.71        98
    #            1       0.85      0.95      0.90       224
    #
    #     accuracy                           0.85       322
    #    macro avg       0.84      0.78      0.80       322
    # weighted avg       0.85      0.85      0.84       322
    #
    # Confusion matrix:
    # [[ 61  37]
    #  [ 12 212]]
    #
    # Normalised:
    # [[0.62 0.38]
    #  [0.05 0.95]]
    #
    #     learning_rates = [5e-6, 1e-5, 2e-5, 3e-5]
    #     batches = [8, 16, 32, 64]
    #     weight_decays = [0, 0.1, 0.2]
