import os

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import get_scorer, confusion_matrix
from sklearn.model_selection import cross_validate, KFold, train_test_split, cross_val_predict

from scripts.ex1.dataset_member import Member
from scripts.ex1.dataset_zoo import Zoo
from scripts.ex1.dataset_cancer import Cancer
from scripts.ex1.dataset_loan import Loan

import warnings

warnings.filterwarnings("ignore")

METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
CLASSIFIERS = [
    ('dt', {"criterion": "gini", "splitter": "best"}),  # default
    ('dt', {"criterion": "entropy", "splitter": "best"}),  # entropy
    ('dt', {"criterion": "gini", "splitter": "random"}),  # random
    ('nb', {"alpha": 1.0, "fit_prior": True}),  # default
    ('nb', {"alpha": 0.1, "fit_prior": True}),  # low alpha
    ('nb', {"alpha": 1.0, "fit_prior": False}),  # no prior
    ('mlp', {"hidden_layer_sizes": (100,), "solver": "adam", "learning_rate_init": 0.001}),  # default
    ('mlp', {"hidden_layer_sizes": (100,), "solver": "lbfgs", "learning_rate_init": 0.0001}),  # lbfgs, low learning rate
    ('mlp', {"hidden_layer_sizes": (200, 50), "solver": "adam", "learning_rate_init": 0.001}),  # more layers
]
DATASET_CLASSES = [
    Loan,
    Member,
    Zoo,
    Cancer
]
result_log = ""

for classifier, hyper_params in CLASSIFIERS:
    for dataset_cls in DATASET_CLASSES:
        print(f"Cross validating classifier: {classifier} on dataset: {dataset_cls.__name__}")

        dataset = dataset_cls(classifier, hyper_params, seed=42)
        splitter = KFold(n_splits=10, shuffle=True, random_state=42)
        cv = cross_validate(dataset.pipeline, dataset.X, dataset.y, cv=splitter, scoring=METRICS, error_score="raise",
                            verbose=2)
        result_log += f"\nCross validation results for classifier: {classifier} on dataset: {dataset_cls.__name__}\n"
        result_log += f"with hyper parameters: {hyper_params if hyper_params else 'default'}\n"
        result_log += f"fit_time: {cv['fit_time'].mean():.2f}s (mean) {cv['fit_time'].var():.2f} (var)\n"
        result_log += f"score_time: {cv['score_time'].var():.2f}s (mean) {cv['score_time'].var():.2f} (var)\n"
        result_log += f"Accuracy: {cv['test_accuracy'].mean():.2%} (mean) {cv['test_accuracy'].var():.2%} (var)\n"
        result_log += f"Precision: {cv['test_precision_macro'].mean():.2%} (mean) {cv['test_precision_macro'].var():.2%} (var)\n"
        result_log += f"Recall: {cv['test_recall_macro'].mean():.2%} (mean) {cv['test_recall_macro'].var():.2%} (var)\n"
        result_log += f"F1-score: {cv['test_f1_macro'].mean():.2%} (mean) {cv['test_f1_macro'].var():.2%} (var)\n"

        X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.2, random_state=42)
        pipe = dataset.pipeline
        pipe.fit(X_train, y_train)
        result_log += f"\nHoldout test results for classifier: {classifier} on dataset: {dataset_cls.__name__}\n"
        result_log += f"with hyper parameters: {hyper_params if hyper_params else 'default'}\n"
        for metric in METRICS:
            scorer = get_scorer(metric)
            y_pred = pipe.predict(X_test)
            result_log += f"Test {metric}: {scorer(X=X_test, estimator=pipe, y_true=y_test):.2%}\n"

        # Use cross_val_predict to obtain predicted labels during cross-validation
        y_pred_cv = cross_val_predict(dataset.pipeline, dataset.X, dataset.y, cv=splitter)

        # Calculate the confusion matrix
        cm = confusion_matrix(dataset.y, y_pred_cv)

        # Print or visualize the confusion matrix
        print("Confusion Matrix:")
        print(cm)

        # Plot the confusion matrix using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{classifier} on dataset: {dataset_cls.__name__}\nhyper parameters: {hyper_params if hyper_params else 'default'}\nConfusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Specify the folder path
        output_folder = os.path.abspath('../../plots/ex1')

        # Ensure the folder exists, create it if necessary
        os.makedirs(output_folder, exist_ok=True)

        # Save the DataFrame to a colon-separated CSV file with headers in the specified folder
        output_file_path = os.path.join(output_folder, f"{classifier}_on_dataset_{dataset_cls.__name__}_{hyper_params}.png")
        plt.savefig(output_file_path)
        plt.show()


print(result_log)
with open(os.path.abspath('../../results/ex1_results_other.txt'), 'w') as f:
    f.write(result_log)
