import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import methods
from general_bert_optimizer import BertModelOptimizer, BertTokenizer
from bert_preprocessor import BertPreprocessor

if __name__ == '__main__':
    device = torch.device("cuda")

    model_data = methods.read_model_data("data/all/labelled_filtered_3000.csv")
    model_data = BertPreprocessor.preprocess(model_data)

    train_text, temp_text, train_labels, temp_labels = train_test_split(model_data['Text'], model_data['Relevance'],
                                                                        shuffle=True, test_size=0.2, random_state=42)
    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, shuffle=True,
                                                                    test_size=0.5, random_state=42)

    bert_tokenizer = BertTokenizer("bert-base-uncased", 128)
    train_seq, train_mask, train_y = bert_tokenizer.tokenize(train_text, train_labels)
    val_seq, val_mask, val_y = bert_tokenizer.tokenize(val_text, val_labels)
    test_seq, test_mask, test_y = bert_tokenizer.tokenize(test_text, test_labels)

    bert_optimizer = BertModelOptimizer("bert-base-uncased", device, "data/bert", 6)
    bert_optimizer.load_training_data(train_seq, train_mask, train_y)
    bert_optimizer.load_validation_data(val_seq, val_mask, val_y)

    learning_rates = [1e-5, 5e-5]
    batches = [8, 16]
    weight_decays = [0.1]

    for lr in learning_rates:
        for bs in batches:
            for wd in weight_decays:
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

    cf_plot = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                          fmt='.2%', cmap='Blues')
    cf_plot.get_figure().savefig("data/bert/cf_plot.png")
