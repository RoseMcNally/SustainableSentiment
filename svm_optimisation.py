import joblib
import pandas as pd
import methods
import numpy as np
from svm_model import SvmModel, SvmTokenizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import seaborn as sns

# Constants

word_vector_size = 300

# Main

if __name__ == '__main__':

    tokenizer = SvmTokenizer()
    tokenizer.train_embedding("data/all/clean.csv", "data/svm/word_model_svm.kv", word_vector_size)

    model_data = methods.read_model_data("data/all/labelled_filtered_3000.csv")
    model_data = SvmModel.preprocess(model_data)

    word_train = tokenizer.apply(model_data)

    train_count = int(word_train.shape[0] * 0.8)

    train_text = word_train[:train_count, :word_vector_size]
    test_text = word_train[train_count:, :word_vector_size]
    train_labels = word_train[:train_count, word_vector_size:].ravel()
    test_labels = word_train[train_count:, word_vector_size:].ravel()

    scaler = StandardScaler().fit(train_text)
    train_text = scaler.transform(train_text)
    test_text = scaler.transform(test_text)

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]}
    grid = GridSearchCV(svm.LinearSVC(random_state=0, max_iter=10000), param_grid, refit=True, verbose=2)
    grid.fit(train_text, train_labels)
    print(grid.best_estimator_)
    grid_predictions = grid.predict(test_text)
    joblib.dump(grid.best_estimator_, "data/svm/best_model_3000.pkl")
    print(classification_report(test_labels, grid_predictions))
    cr = classification_report(test_labels, grid_predictions, output_dict=True)
    df = pd.DataFrame(cr).transpose()
    df.to_html("data/svm/report_3000.html")
    cf_matrix = confusion_matrix(test_labels, grid_predictions)
    cm_plot = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
    cm_plot.get_figure().savefig("data/svm/cm_svm_3000.png")

