import os
import pandas as pd

from scripts.ex1.dataset_loan import Loan

train = Loan('dt', {"criterion": "gini", "splitter": "best"})
test = Loan('dt', {"criterion": "gini", "splitter": "best"}, test=True)

print(train.X.head())
print(train.X.shape)

print(test.X.head())
print(test.X.shape)

pipeline = train.pipeline
pipeline.fit(train.X, train.y)

test_idx = test.X['ID']
test.X.drop('ID', axis=1, inplace=True)

# Make predictions on the test set
y_pred = pipeline.predict(test.X)

# Assuming 'id' is also in the test_data DataFrame
results = pd.DataFrame({'ID': test_idx, 'grade': y_pred})

# Display the results
print(results)

# Specify the folder path
output_folder = os.path.abspath('../../results')

# Ensure the folder exists, create it if necessary
os.makedirs(output_folder, exist_ok=True)

# Save the DataFrame to a colon-separated CSV file with headers in the specified folder
output_file_path = os.path.join(output_folder, 'output_file_loan.csv')
results.to_csv(output_file_path, sep=',', index=False)
