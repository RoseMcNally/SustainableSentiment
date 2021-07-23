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

    learning_rates = [1e-5, 2e-5, 3e-5, 4e-5]
    batches = [8, 16, 32, 64]
    weight_decays = [0, 0.1, 0.2]
    hyps = {"learning_rates": learning_rates, "batches": batches, "weight_decays": weight_decays}

    tune_bert_hyperparameters(model_data, "vinai/bertweet-base", "data/bertweet", 50, hyps)

    # Best model result:
    # Batch size: 64 , Learning rate: 1e-05 , Weight decay: 0.2
    #
    #              precision    recall  f1-score   support
    #
    #            0       0.81      0.61      0.70        98
    #            1       0.85      0.94      0.89       224
    #
    #     accuracy                           0.84       322
    #    macro avg       0.83      0.77      0.79       322
    # weighted avg       0.84      0.84      0.83       322
    #
    # Confusion matrix:
    # [[ 60  38]
    #  [ 14 210]]
    #
    # Normalised:
    # [[0.61 0.39]
    #  [0.06 0.94]]
