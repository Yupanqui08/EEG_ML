import pandas as pd

#Path of datafile
eeg_file_path = '/Users/raquelyupanqui/Downloads/scottwellington-FEIS-7e726fd/experiments/01/thinking.csv'

#Read file into variable: home_data
eeg_data = pd.read_csv(eeg_file_path)
eeg_data = eeg_data.drop(columns=['Time:256Hz', 'Epoch', 'Stage', 'Flag'], axis=1)
eeg_data = eeg_data.sample(frac=1).reset_index(drop=True)

#Select the prediction target
y = eeg_data.Label

#Create the list of features below
feature_names = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']

#Select data corresponding to features in feature_names
X = eeg_data[feature_names]

#******************************

from sklearn.tree import DecisionTreeClassifier
#specify the model
#For model reproducibility, set a numeric value for random_state when specifying the model
eeg_model = DecisionTreeClassifier(random_state=1)

# Fit the model
eeg_model.fit(X, y)

#Make predictions
predictions = eeg_model.predict(X)

#Split data into training and validation sets
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0, shuffle=True)
eeg_model = DecisionTreeClassifier(random_state=0)
eeg_model.fit(train_X, train_y)
val_predictions = eeg_model.predict(val_X)
final_eeg_model = DecisionTreeClassifier(max_leaf_nodes=100, random_state=1)

#Fit the final model
final_eeg_model.fit(X, y)

from sklearn.metrics import accuracy_score

print("Decision Tree Accuracy Score:")
print(accuracy_score(val_y, val_predictions))

#**************************************************************************************
#Now try Random Forest instead of Decision Tree
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=1)
rf_model.fit(train_X, train_y)
rf_preds = rf_model.predict(val_X)

print("RF Accuracy Score:")
print(accuracy_score(val_y, rf_preds))