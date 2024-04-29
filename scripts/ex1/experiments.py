import os

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import get_scorer, confusion_matrix
from sklearn.model_selection import cross_validate, KFold, train_test_split, cross_val_predict

from scripts.ex1.dataset_member import Member
from scripts.ex1.dataset_zoo import Zoo
from scripts.ex1.dataset_cancer import Cancer
from scripts.ex1.dataset_loan import Loan
import json

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
result_data = []

for classifier, hyper_params in CLASSIFIERS:
    for dataset_cls in DATASET_CLASSES:
        print(f"Cross validating classifier: {classifier} on dataset: {dataset_cls.__name__}")

        dataset = dataset_cls(classifier, hyper_params, seed=42)
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        cv = cross_validate(dataset.pipeline, dataset.X, dataset.y, cv=splitter, scoring=METRICS, error_score="raise",
                            verbose=2)
        # Store results in a structured dictionary
        cv_results = {
            "classifier": classifier,
            "dataset": dataset_cls.__name__,
            "hyper_parameters": hyper_params,
            "cross_validation_results": {
                "fit_time": {"mean": cv['fit_time'].mean(), "variance": cv['fit_time'].var()},
                "score_time": {"mean": cv['score_time'].mean(), "variance": cv['score_time'].var()},
                "metrics": {}
            }
        }

        for metric in METRICS:
            cv_results["cross_validation_results"]["metrics"][metric] = {
                "mean": cv[f'test_{metric}'].mean(),
                "variance": cv[f'test_{metric}'].var()
            }

        X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.2, random_state=42)
        pipe = dataset.pipeline
        pipe.fit(X_train, y_train)
        holdout_results = {}
        for metric in METRICS:
            scorer = get_scorer(metric)
            score = scorer(pipe, X_test, y_test)
            holdout_results[metric] = score

        cv_results["holdout_test_results"] = holdout_results

        # Optionally, add confusion matrix and save plots as you did before
        result_data.append(cv_results)

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


# Save the result data as JSON
output_folder = os.path.abspath('../../results')
os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, 'experiment_results.json')

with open(output_file_path, 'w') as f:
    json.dump(result_data, f, indent=4)

