import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import methods
from general_bert_optimizer import BertModelOptimizer, BertTokenizer
from bert_preprocessor import BertPreprocessor


def tune_bert_hyperparameters(data, model_name, path, max_len, params):
    device = torch.device("cuda")

    train_text, temp_text, train_labels, temp_labels = train_test_split(data['Text'], data['Relevance'],
                                                                        shuffle=True, test_size=0.2, random_state=42)
    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, shuffle=True,
                                                                    test_size=0.5, random_state=42)

    bert_tokenizer = BertTokenizer(model_name, max_len)
    train_seq, train_mask, train_y = bert_tokenizer.tokenize(train_text, train_labels)
    val_seq, val_mask, val_y = bert_tokenizer.tokenize(val_text, val_labels)
    test_seq, test_mask, test_y = bert_tokenizer.tokenize(test_text, test_labels)

    bert_optimizer = BertModelOptimizer(model_name, device, path, 6)
    bert_optimizer.load_training_data(train_seq, train_mask, train_y)
    bert_optimizer.load_validation_data(val_seq, val_mask, val_y)

    for lr in params["learning_rates"]:
        for bs in params["batches"]:
            for wd in params["weight_decays"]:
                bert_optimizer.run(bs, lr, wd)

    bs, lr, wd = bert_optimizer.best_model_parameters
    print("Best model parameters - Batch size:", bs, ", Learning rate:", lr, ", Weight decay:", wd)

    best_model = bert_optimizer.best_model()

    with torch.no_grad():
        preds = best_model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()

    preds = np.argmax(preds, axis=1)
    print(classification_report(test_y, preds))

    cf_matrix = confusion_matrix(test_y, preds)
    print(cf_matrix)

    cf_matrix = confusion_matrix(test_y, preds, normalize='true')

    cf_plot = sns.heatmap(cf_matrix, annot=True,
                          fmt='.2%', cmap='Blues')
    cf_plot.get_figure().savefig(path + "/cf_plot.png")


if __name__ == '__main__':

    np.random.seed(42)
    torch.manual_seed(42)

    model_data = methods.read_model_data("data/all/mum_filtered_with_duplicates.csv")
    model_data = BertPreprocessor.preprocess(model_data)

    learning_rates = [5e-6, 1e-5, 2e-5, 3e-5]
    batches = [8, 16, 32, 64]
    weight_decays = [0, 0.1]
    hyps = {"learning_rates": learning_rates, "batches": batches, "weight_decays": weight_decays}

    tune_bert_hyperparameters(model_data, "bert-base-uncased", "data/bert", 128, hyps)

    # Best model result:
    # Batch size: 64 , Learning rate: 5e-06 , Weight decay: 0
    #
    #                precision    recall  f1-score   support
    #
    #            0       0.72      0.64      0.68        98
    #            1       0.85      0.89      0.87       224
    #
    #     accuracy                           0.82       322
    #    macro avg       0.79      0.77      0.78       322
    # weighted avg       0.81      0.82      0.81       322
    #
    # Confusion matrix:
    # [[ 63  35]
    #  [ 24 200]]
    #
    # Normalised:
    # [[0.64 0.36]
    #  [0.11 0.89]]