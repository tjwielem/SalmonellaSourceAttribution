# Import Necessary Libraries
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

seed = 34 # random_state value
input_file = './enterobase/enterobase_train_filtered.csv'

input_df=pd.read_csv(input_file)

labels = "Region" # Column in the dataframe containing labels
y = input_df[labels] # labels
input_df.drop([labels], axis=1, inplace=True)
X = input_df # features

encoder = OrdinalEncoder()
X = encoder.fit_transform(X)

categorical_features = np.ones(X.shape[1], dtype=bool) # All features are categorical

encode = True
if encode:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)


# Split the dataset in train and test part
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=seed)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision_macro = precision_score(y_test, rf_predictions, average='macro')
rf_recall_macro = recall_score(y_test, rf_predictions, average='macro')
rf_f1_macro = f1_score(y_test, rf_predictions, average='macro')
rf_precision_micro = precision_score(y_test, rf_predictions, average='micro')
rf_recall_micro = recall_score(y_test, rf_predictions, average='micro')
rf_f1_micro = f1_score(y_test, rf_predictions, average='micro')
rf_precision_weighted = precision_score(y_test, rf_predictions, average='weighted')
rf_recall_weighted = recall_score(y_test, rf_predictions, average='weighted')
rf_f1_weighted = f1_score(y_test, rf_predictions, average='weighted')
print("Random Forest")
print(f" 'Accuracy': {rf_accuracy}, 'MacroPrecision' : {rf_precision_macro}, 'MacroRecall' : {rf_recall_macro}, 'MacroF1' : {rf_f1_macro}, 'WeightedPrecision' : {rf_precision_weighted},  'WeightedRecall' : {rf_recall_weighted},  'WeightedF1' : {rf_f1_weighted},  'MicroPrecision' : {rf_precision_micro},  'MicroRecall' : {rf_recall_micro},  'MicroF1' : {rf_f1_micro}")
print()

# XGBoost
xgb_classifier = XGBClassifier(n_estimators=1000, learning_rate=0.1, random_state=seed)
xgb_classifier.fit(X_train, y_train)
xgb_predictions = xgb_classifier.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_precision_macro = precision_score(y_test, xgb_predictions, average='macro')
xgb_recall_macro = recall_score(y_test, xgb_predictions, average='macro')
xgb_f1_macro = f1_score(y_test, xgb_predictions, average='macro')
xgb_precision_micro = precision_score(y_test, xgb_predictions, average='micro')
xgb_recall_micro = recall_score(y_test, xgb_predictions, average='micro')
xgb_f1_micro = f1_score(y_test, xgb_predictions, average='micro')
xgb_precision_weighted = precision_score(y_test, xgb_predictions, average='weighted')
xgb_recall_weighted = recall_score(y_test, xgb_predictions, average='weighted')
xgb_f1_weighted = f1_score(y_test, xgb_predictions, average='weighted')
print("XGBoost")
print(f" 'Accuracy': {xgb_accuracy}, 'MacroPrecision' : {xgb_precision_macro}, 'MacroRecall' : {xgb_recall_macro}, 'MacroF1' : {xgb_f1_macro}, 'WeightedPrecision' : {xgb_precision_weighted},  'WeightedRecall' : {xgb_recall_weighted},  'WeightedF1' : {xgb_f1_weighted},  'MicroPrecision' : {xgb_precision_micro},  'MicroRecall' : {xgb_recall_micro},  'MicroF1' : {xgb_f1_micro}")
print()

# Gradient Boost
gb_classifier = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=None, random_state=seed)
gb_classifier.fit(X_train, y_train)
gb_predictions = gb_classifier.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_predictions)
gb_precision_macro = precision_score(y_test, gb_predictions, average='macro')
gb_recall_macro = recall_score(y_test, gb_predictions, average='macro')
gb_f1_macro = f1_score(y_test, gb_predictions, average='macro')
gb_precision_micro = precision_score(y_test, gb_predictions, average='micro')
gb_recall_micro = recall_score(y_test, gb_predictions, average='micro')
gb_f1_micro = f1_score(y_test, gb_predictions, average='micro')
gb_precision_weighted = precision_score(y_test, gb_predictions, average='weighted')
gb_recall_weighted = recall_score(y_test, gb_predictions, average='weighted')
gb_f1_weighted = f1_score(y_test, gb_predictions, average='weighted')
print("Gradient Boost")
print(f" 'Accuracy': {gb_accuracy}, 'MacroPrecision' : {gb_precision_macro}, 'MacroRecall' : {gb_recall_macro}, 'MacroF1' : {gb_f1_macro}, 'WeightedPrecision' : {gb_precision_weighted},  'WeightedRecall' : {gb_recall_weighted},  'WeightedF1' : {gb_f1_weighted},  'MicroPrecision' : {gb_precision_micro},  'MicroRecall' : {gb_recall_micro},  'MicroF1' : {gb_f1_micro}")
print()

# HistGradientBoost
hist_gb_classifier = HistGradientBoostingClassifier(categorical_features=categorical_features, random_state=seed)
hist_gb_classifier.fit(X_train, y_train)
hist_gb_predictions = hist_gb_classifier.predict(X_test)
hist_gb_accuracy = accuracy_score(y_test, hist_gb_predictions)
hist_gb_precision_macro = precision_score(y_test, hist_gb_predictions, average='macro')
hist_gb_recall_macro = recall_score(y_test, hist_gb_predictions, average='macro')
hist_gb_f1_macro = f1_score(y_test, hist_gb_predictions, average='macro')
hist_gb_precision_micro = precision_score(y_test, hist_gb_predictions, average='micro')
hist_gb_recall_micro = recall_score(y_test, hist_gb_predictions, average='micro')
hist_gb_f1_micro = f1_score(y_test, hist_gb_predictions, average='micro')
hist_gb_precision_weighted = precision_score(y_test, hist_gb_predictions, average='weighted')
hist_gb_recall_weighted = recall_score(y_test, hist_gb_predictions, average='weighted')
hist_gb_f1_weighted = f1_score(y_test, hist_gb_predictions, average='weighted')
print("HistGradient Boost")
print(f" 'Accuracy': {hist_gb_accuracy}, 'MacroPrecision' : {hist_gb_precision_macro}, 'MacroRecall' : {hist_gb_recall_macro}, 'MacroF1' : {hist_gb_f1_macro}, 'WeightedPrecision' : {hist_gb_precision_weighted},  'WeightedRecall' : {hist_gb_recall_weighted},  'WeightedF1' : {hist_gb_f1_weighted},  'MicroPrecision' : {hist_gb_precision_micro},  'MicroRecall' : {hist_gb_recall_micro},  'MicroF1' : {hist_gb_f1_micro}")
print()

# Support Vector Machine
svm_classifier = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=seed)
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision_macro = precision_score(y_test, svm_predictions, average='macro')
svm_recall_macro = recall_score(y_test, svm_predictions, average='macro')
svm_f1_macro = f1_score(y_test, svm_predictions, average='macro')
svm_precision_micro = precision_score(y_test, svm_predictions, average='micro')
svm_recall_micro = recall_score(y_test, svm_predictions, average='micro')
svm_f1_micro = f1_score(y_test, svm_predictions, average='micro')
svm_precision_weighted = precision_score(y_test, svm_predictions, average='weighted')
svm_recall_weighted = recall_score(y_test, svm_predictions, average='weighted')
svm_f1_weighted = f1_score(y_test, svm_predictions, average='weighted')
print("SVM")
print(f" 'Accuracy': {svm_accuracy}, 'MacroPrecision' : {svm_precision_macro}, 'MacroRecall' : {svm_recall_macro}, 'MacroF1' : {svm_f1_macro}, 'WeightedPrecision' : {svm_precision_weighted},  'WeightedRecall' : {svm_recall_weighted},  'WeightedF1' : {svm_f1_weighted},  'MicroPrecision' : {svm_precision_micro},  'MicroRecall' : {svm_recall_micro},  'MicroF1' : {svm_f1_micro}")
print()

# Gaussian Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision_macro = precision_score(y_test, nb_predictions, average='macro')
nb_recall_macro = recall_score(y_test, nb_predictions, average='macro')
nb_f1_macro = f1_score(y_test, nb_predictions, average='macro')
nb_precision_micro = precision_score(y_test, nb_predictions, average='micro')
nb_recall_micro = recall_score(y_test, nb_predictions, average='micro')
nb_f1_micro = f1_score(y_test, nb_predictions, average='micro')
nb_precision_weighted = precision_score(y_test, nb_predictions, average='weighted')
nb_recall_weighted = recall_score(y_test, nb_predictions, average='weighted')
nb_f1_weighted = f1_score(y_test, nb_predictions, average='weighted')
print("Gaussian Naive Bayes")
print(f" 'Accuracy': {nb_accuracy}, 'MacroPrecision' : {nb_precision_macro}, 'MacroRecall' : {nb_recall_macro}, 'MacroF1' : {nb_f1_macro}, 'WeightedPrecision' : {nb_precision_weighted},  'WeightedRecall' : {nb_recall_weighted},  'WeightedF1' : {nb_f1_weighted},  'MicroPrecision' : {nb_precision_micro},  'MicroRecall' : {nb_recall_micro},  'MicroF1' : {nb_f1_micro}")
print()

# Multinomial Naive Bayes
mnb_classifier = MultinomialNB()
mnb_classifier.fit(X_train, y_train)
mnb_predictions = mnb_classifier.predict(X_test)
mnb_accuracy = accuracy_score(y_test, mnb_predictions)
mnb_precision_macro = precision_score(y_test, mnb_predictions, average='macro')
mnb_recall_macro = recall_score(y_test, mnb_predictions, average='macro')
mnb_f1_macro = f1_score(y_test, mnb_predictions, average='macro')
mnb_precision_micro = precision_score(y_test, mnb_predictions, average='micro')
mnb_recall_micro = recall_score(y_test, mnb_predictions, average='micro')
mnb_f1_micro = f1_score(y_test, mnb_predictions, average='micro')
mnb_precision_weighted = precision_score(y_test, mnb_predictions, average='weighted')
mnb_recall_weighted = recall_score(y_test, mnb_predictions, average='weighted')
mnb_f1_weighted = f1_score(y_test, mnb_predictions, average='weighted')
print("Multinomial Naive Bayes")
print()