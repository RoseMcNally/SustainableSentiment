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

word_vector_size = 300

if __name__ == '__main__':

    tokenizer = SvmTokenizer()
    tokenizer.train_embedding("data/all/clean.csv", "data/svm/word_model_svm.kv", word_vector_size)

    model_data = methods.read_model_data("data/all/mum_filtered_with_duplicates.csv")
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

    param_grid = {'C': [0.001, 0.01, 0.1, 1], 'tol': [1e-5, 1e-4, 1e-3, 1e-2]}
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

    # Best model result:
    # C=0.1, tol=1e-05
    #
    #                 precision    recall  f1-score   support
    #
    #          0.0       0.45      0.25      0.32       117
    #          1.0       0.85      0.93      0.89       527
    #
    #     accuracy                           0.81       644
    #    macro avg       0.65      0.59      0.60       644
    # weighted avg       0.78      0.81      0.79       644
    #
    # Confusion matrix:
    # [[ 29  88]
    #  [ 35 492]]
    #
    # Normalised:
    # [[0.25 0.75]
    #  [0.07 0.93]]
