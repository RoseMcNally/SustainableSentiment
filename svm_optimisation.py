import joblib
import pandas as pd
import methods
import numpy as np
from svm_model import SvmModel, SvmTokenizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import seaborn as sns

word_vector_size = 300

if __name__ == '__main__':
    np.random.seed(42)

    tokenizer = SvmTokenizer()
    tokenizer.train_embedding("data/all/clean_filtered.csv", "data/svm/word_embeddings.kv", word_vector_size)

    model_data = methods.read_model_data("data/all/mum_filtered_with_duplicates.csv")
    model_data = SvmModel.preprocess(model_data)

    train_text, temp_text, train_labels, temp_labels = train_test_split(model_data['Text'], model_data['Relevance'],
                                                                        shuffle=True, test_size=0.2, random_state=42)
    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, shuffle=True,
                                                                    test_size=0.5, random_state=42)

    train_text = train_text.append(val_text)
    train_labels = train_labels.append(val_labels)
    train_text = tokenizer.apply(train_text)
    test_text = tokenizer.apply(test_text)

    scaler = StandardScaler().fit(train_text)
    train_text = scaler.transform(train_text)
    test_text = scaler.transform(test_text)

    param_grid = {'C': [0.001, 0.01, 0.1, 1], 'tol': [1e-5, 1e-4, 1e-3, 1e-2]}
    grid = GridSearchCV(svm.LinearSVC(random_state=0, max_iter=10000), param_grid, refit=True, verbose=2)
    grid.fit(train_text, train_labels)
    print(grid.best_estimator_)
    grid_predictions = grid.predict(test_text)
    joblib.dump(grid.best_estimator_, "data/svm/best_model_3000_correct.pkl")
    print(classification_report(test_labels, grid_predictions))
    cr = classification_report(test_labels, grid_predictions, output_dict=True)
    df = pd.DataFrame(cr).transpose()
    df.to_html("data/svm/report_3000_correct.html")
    cf_matrix = confusion_matrix(test_labels, grid_predictions)
    cm_plot = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
    cm_plot.get_figure().savefig("data/svm/cm_svm_3000.png")

    cf_matrix = confusion_matrix(test_labels, grid_predictions)
    print(cf_matrix)

    cf_matrix = confusion_matrix(test_labels, grid_predictions, normalize='true')
    sns.set(font_scale=1.2)
    cf_plot = sns.heatmap(cf_matrix, annot=True,
                          fmt='.2%', cmap='Blues')
    cf_plot.get_figure().savefig("data/svm/cf_plot_correct.png", dpi=400)

    # Best model result:
    # C=0.001, tol=1e-05
    #
    #               precision    recall  f1-score   support
    #
    #            0       0.68      0.55      0.61        98
    #            1       0.82      0.89      0.85       224
    #
    #     accuracy                           0.79       322
    #    macro avg       0.75      0.72      0.73       322
    # weighted avg       0.78      0.79      0.78       322
    #
    # Confusion matrix:
    # [[ 54  44]
    #  [ 25 199]]
    #
    # Normalised:
    # [[0.55 0.45]
    #  [0.11 0.89]]
