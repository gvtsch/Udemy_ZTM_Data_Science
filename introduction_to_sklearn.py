# %% [markdown]
# # A simple Scikit-Learn CLassification Workflow
# 
# This notebook shows a brief workflow you might use with `scikit-learn` to build a machine learning model to classify wheteth or not a patien has heart disease.
# 
# What we're going to cover:
# 
# 0. An end-to-end Scikit-Learn workflow
# 1. Getting the data ready
# 2. Choose the right estimator/algorithm for out problems
# 3. Fit the model/algorithm and use it to make predictions on our data
# 4. Evaluating a model
# 5. Improve a model
# 6. Save and load a trained model
# 7. Putting it all together!

# %% [markdown]
# ## An end to end Scikit-Learn workflow

# %%
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
# Import dataset
import os
if not os.path.exists("heart-disease.csv"):
    !python -w get https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv
heart_disease = pd.read_csv("heart-disease.csv")
heart_disease.head()

# %%
# Create X (feature matrix)
X = heart_disease.drop("target", axis=1)

# Create y (labels)
y = heart_disease["target"]

# %%
# 2. Choose the right model and hyperparameters
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

# We'll keep the default hyperparameters
clf.get_params()

# %%
# 3. Fit the model to the training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
clf.fit(X_train, y_train)

# %%
# make a prediction
y_preds = clf.predict(X_test)
y_preds, y_test

# %%
# 4 Evaluate the model
clf.score(X_train, y_train)

# %%
clf.score(X_test, y_test)

# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, y_preds))

# %%
confusion_matrix(y_test, y_preds)

# %%
accuracy_score(y_test, y_preds)

# %%
# 5. Improve a modeel
# Try different amount of n_estimators

np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with {i} estimators.")
    clf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    print(f"Model accuracy on test set: {clf.score(X_test, y_test)*100}")

# %%
# 6. Save a model and load it
import pickle
pickle.dump(clf, open("random_forest_model_1.pkl", "wb"))

# %%
loaded_model = pickle.load(open("random_forest_model_1.pkl", "rb"))
loaded_model.score(X_test, y_test)

# %% [markdown]
# ## 1. Getting our data ready to be used with machine learning
# 
# Three main thing we have to do:
# 1. Split the data into features and labels (usually `X` & `y`)
# 2. Filling (also called imputing) or disregarding missing values
# 3. Converting non-numerical values to numerical values (also called feature encoding)

# %%
heart_disease.head()

# %%
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]
X.head(), y.head()

# %%
# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# ## 1. Getting data ready to be used with machine learning
# 
# Three main things we have to do:
# 1. Split the data into features and labels (usually `X` & `y`)
# 2. Filling (also called imputing) or disregarding missing values
# 3. Converting non-numerical values to numerical values (also called feature encoding)

# %%
heart_disease.head()

# %%
X = heart_disease.drop("target", axis=1)
X.head()

# %%
y = heart_disease["target"]
y.head()

# %%
# Slit the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# ### 1.1 Make sure it is all numerical

# %%
import os
if not os.path.exists("car-sales-extended.csv"):
    !python -m wget "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/car-sales-extended.csv"

# %%
car_sales = pd.read_csv("car-sales-extended.csv")
car_sales.head()

# %%
len(car_sales)

# %%
car_sales.dtypes

# %%
# Split into X/y
X = car_sales.drop("Price", axis = 1) 
y = car_sales["Price"]

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categrical_features = ["Make", "Colour", "Doors"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([(
    "one_hot",
    one_hot,
    categrical_features)],
    remainder="passthrough")
transformed_X = transformer.fit_transform(X)
transformed_X

# %%
pd.DataFrame(transformed_X)

# %%
dummies = pd.get_dummies(car_sales[["Make", "Colour", "Doors"]])
dummies

# %%
# Build machine learning model
from sklearn.ensemble import RandomForestRegressor
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)

# %% [markdown]
# ### 1.2 What if there are missing values?
# 1. Fill them with some value (also known as imputation).
# 2. Remove the samples with missing data all together.

# %%
import os
if not os.path.exists("car-sales-extended-missing-data.csv"):
    !python -m wget "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/car-sales-extended-missing-data.csv"
car_sales_missing = pd.read_csv("car-sales-extended-missing-data.csv")
car_sales_missing.head()

# %%
car_sales_missing.isna().sum()

# %%
# Create X and y
X = car_sales_missing.drop("Price", axis=1)
y = car_sales_missing["Price"]

# %% [markdown]
# #### Option 1: Fill missing data with Pandas 

# %%
# Fill the "Make" coloumn
car_sales_missing["Make"].fillna("missing", inplace=True)

# Fill the "Colour" column
car_sales_missing["Colour"].fillna("missing", inplace=True)

# Fill the "Odometer (KM)" column
car_sales_missing["Odometer (KM)"].fillna(car_sales_missing["Odometer (KM)"].mean, inplace=True)

# Fill the "Doors" column
# Fill with most common number for doors
car_sales_missing["Doors"].fillna(4, inplace=True)

# %%
car_sales_missing.isna().sum()

# %%
# Remove doors with missing price value
car_sales_missing.dropna(inplace=True)  

# %%
car_sales_missing.isna().sum(), len(car_sales_missing)

# %%
X = car_sales_missing.drop("Price", axis=1)
y = car_sales_missing["Price"]

# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Make", "Colour", "Doors"] 
one_hot = OneHotEncoder()
transformer = ColumnTransformer([(
    "one_hot",
    one_hot,
    categorical_features)],
    remainder="passthrough"
    )
transformed_X = transformer.fit_transform(car_sales_missing)
transformed_X

# %% [markdown]
# #### Option 2: Fill missing values with Scikit-Learn

# %%
car_sales_missing = pd.read_csv("car-sales-extended-missing-data.csv")
car_sales_missing.head()

# %%
car_sales_missing.isna().sum()

# %%
# dropna values that are in the subset of the price-column
car_sales_missing.dropna(subset=["Price"], inplace=True)
car_sales_missing.isna().sum()

# %%
# Split into X & y
X = car_sales_missing.drop("Price", axis=1)
y = car_sales_missing["Price"]

# %%
# Fill missing values with Scikit-Learn
# Turn data to numbers
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Fill categorical values with 'missing' & numerical valuies with 'mean'
cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
door_imputer = SimpleImputer(strategy="constant", fill_value=4)
num_imputer = SimpleImputer(strategy="mean")

# Define columns
cat_features = ["Make", "Colour"]
door_features = ["Doors"]
num_features = ["Odometer (KM)"]

# Create an imputer (something that fills missing data)
imputer = ColumnTransformer([
    ("cat_imputer", cat_imputer, cat_features),
    ("door_imputer", door_imputer, door_features),
    ("num_imputer", num_imputer, num_features)
])

# Transform the data
filled_X = imputer.fit_transform(X)
filled_X

# %%
car_sales_filled = pd.DataFrame(
    filled_X,
    columns=["Make", "Colour", "Doors", "Odometer (KM)"]
)
car_sales_filled.head()

# %%
car_sales_filled.isna().sum()

# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Make", "Colour", "Doors"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([(
    "one_hot",
    one_hot,
    categorical_features)],
    remainder="passthrough"
)

transformed_X = transformer.fit_transform(car_sales_filled)
transformed_X

# %%
# Now we've got our data as numbers and filled (no missing values)
# Let's fit a model
np.random.seed(42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    transformed_X, 
    y,
    test_size = 0.2
)

model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)

# %%
len(car_sales_filled), len(car_sales)

# %% [markdown]
# **Note:** The 50 less values in the transformed data is becase we dropped the rows (50 total) with missing values in the Price column

# %% [markdown]
# ## 2. Choosing the right estimator/algorithm for your problem
# 
# Some things to note:
# 
# * Sklearn refers to machine learning models, algorithms as estimators.
# * Classification problem - predicting a category (heart disease or not)
#     * Sometimes you'll see `clf` (short for classifier) used as a classification estimator
# * Regression problem - predicting a number (selling price of a car)
# 
# If you're working in a machine learning problem and looking to use Sklearn and not sure what model you should use refer to the sklearn machine learning map: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# %% [markdown]
# ### 2.1 Picking a machine learning for a regression problem
# 
# Let's use the California Housing Dataset.

# %%
# Get California Housing Dataset
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
housing

# %%
housing_df = pd.DataFrame(housing["data"], columns=housing["feature_names"])
housing_df.head()

# %%
housing_df["target"] = housing["target"]
housing_df.head()

# %%
# Import algorithm
from sklearn.linear_model import Ridge

# Setup random seed
np.random.seed(42)

# Create the data
X = housing_df.drop("target", axis=1)
y = housing_df["target"] # median house price in 100.000$

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate and fit the model (on the training set)
model = Ridge()
model.fit(X_train, y_train)

# Check the score of the model (on the test set)
model.score(X_test, y_test)

# %% [markdown]
# Waht if `Ridge` did not work or the score did not fit our needs?
# We can always try a different model...
# 
# Let's try an ensemble model (an ensemble is a combination of smaller models to try and make better predictions than just a simble model)?
# 
# Sklearn ensemble models can be found [here](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees9).

# %%
# Import the RandomForestRegressor model calss from the ensemble module
from sklearn.ensemble import RandomForestRegressor

# Setup random seed
np.random.seed(42)

# Create the data
X = housing_df.drop("target", axis=1)
y = housing_df["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create random forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Check the score of the model 
model.score(X_test, y_test)

# %% [markdown]
# ## 2.2 Picking a machine learning model for a classification problem

# %%
heart_disease = pd.read_csv("heart-disease.csv")
heart_disease.head()

# %%
len(heart_disease)

# %% [markdown]
# Consulting the map and it says to try `LinearSVC`

# %%
# Import LinearSVC estimatro class
from sklearn.svm import LinearSVC

# Setup random seed
np.random.seed(42)

# Make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate LinearSVC
clf = LinearSVC(max_iter=10_000)
clf.fit(X_train, y_train)

# Score
clf.score(X_test, y_test)

# %%
# Import RandomForestClassifier estimatro class
from sklearn.ensemble import RandomForestClassifier

# Setup random seed
np.random.seed(42)

# Make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Score
clf.score(X_test, y_test)

# %% [markdown]
# Tidbit:
# 
# 1. If you have structured data, use ensemble methods
# 2. If you have unstructured data, use deep learning or transfer learning

# %% [markdown]
# ## 3. Fit the model/algorithm in our data and use it to make predictions
# 
# ### 3.1 Fitting the model to the data
# 
# Different names for:
# * `X` = features, features variables, data
# * `y` = labels, targets, target variables

# %%
# Import RandomForestClassifier estimatro class
from sklearn.ensemble import RandomForestClassifier

# Setup random seed
np.random.seed(42)

# Make the data
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Score
clf.score(X_test, y_test)

# %% [markdown]
# ### 3.2 Make predictions using a machine learning model
# 
# 2 ways to make predictions:
# 1. `predict()`
# 2. `predict_proba()`

# %%
# Use a trained model to make predictions
clf.predict(X_test)

# %%
np.array(y_test)

# %%
# Compare predictions to truth labels to evaluate the model
y_preds = clf.predict(X_test)
np.mean(y_preds == y_test)

# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_preds)

# %% [markdown]
# Make predictions with `predict_proba()`

# %%
# predict_proba() returns probabilities of a classification label
clf.predict_proba(X_test[:5])

# %%
# Let's predict on the same data
clf.predict(X_test[:5])

# %%
heart_disease["target"].value_counts()

# %% [markdown]
# `predict()` can also be used for regression models

# %%
from sklearn.ensemble import RandomForestRegressor
np.random.seed(42)
X = housing_df.drop("target", axis=1)
y = housing_df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model=RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
y_pred[:10], np.array(y_test[:10])

# %%
len(y_pred), len(y_test)

# %%
# Compare the predictions to the truth
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_pred, y_test)

# %%
housing_df["target"]

# %% [markdown]
# ## 4. Evaluating a machine learning model
# 
# Threee ways to evaluate Scikit-Learn models/estimators:
# 1. Estimator's built-in `score()` method
# 2. The `scoring` parameter
# 3. Problem-specific metric functions
# 
# You can read more about these here: https://scikit-learn.org/stable/modules/model_evaluation.html

# %% [markdown]
# ## 4.1 Evaluating a model with the `score` method

# %%
from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)

X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf=RandomForestClassifier()
clf.fit(X_train, y_train)

# %%
clf.score(X_train, y_train), clf.score(X_test, y_test)

# %% [markdown]
# Let's use the `score()` on our regression problem ...

# %%
from sklearn.ensemble import RandomForestRegressor
np.random.seed(42)
X = housing_df.drop("target", axis=1)
y = housing_df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model=RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
model.score(X_train, y_train), model.score(X_test, y_test)

# %% [markdown]
# ## 4.2 Evaluating a model using the `scoring` parameter

# %%
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf=RandomForestClassifier()
clf.fit(X_train, y_train)

# %%
clf.score(X_test, y_test)

# %%
cross_val_score(clf, X, y, cv=5)

# %%
np.random.seed(42)

# Single training and test split score
clf_single_score = clf.score(X_test, y_test)

# Take the mean of 5-fold cross-validation score
clf_cross_val_score = np.mean(cross_val_score(clf, X, y, cv=5))

# Compare the two
clf_single_score, clf_cross_val_score

# %%
# Scoring parameter set to None by default
cross_val_score(clf, X, y, cv=5, scoring=None)

# %% [markdown]
# ### 4.2.1 Classification model evaluation metrics
# 
# 1. Accuracy
# 2. Area under ROC curve
# 3. Confusion matrix
# 4. Classification report
# 
# **Accuracy**

# %%
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

clf = RandomForestClassifier()
cross_val_score = cross_val_score(clf, X, y, cv=5)

# %%
np.mean(cross_val_score)

# %%
print(f"Heart disease classifier accuracy Cross-validated accuracy: {np.mean(cross_val_score)*100:.2f}%")

# %% [markdown]
# **Area under the receiver operating characteristic curve (AUC/ROC)**
# 
# * Area under curve (AUC)
# * ROC curve
# 
# ROC curves are a comparison of a model's true positive rate (tpr) versus a models false positive false positive rate (fpr).
# 
# * True positive = model predicts 1, when truth is 1
# * False positive = model predicts 1, when truth is 0
# * True negative = model predicts 0, when truth is 0
# * False negative = model predicts 0, when truth is 1

# %%
# Create X_test... etc
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
from sklearn.metrics import roc_curve

clf.fit(X_train, y_train)

# Make predictions with probabilitis
y_probs = clf.predict_proba(X_test)

y_probs[:10]

# %%
y_probs_positive = y_probs[:, 1]
y_probs_positive[:10]

# %%
# Calculate fpr, tpr and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)

# Check the false positive rates
fpr

# %%
# Create a function for plotting ROC curves
import matplotlib.pyplot as plt

def plot_ROC(fpr, tpr):
    """
    Plots a ROC curve given the false positive rate (fpr)
    and true positive rate (tpr) of a model.
    """
    # Plot roc curve
    plt.plot(fpr, tpr, color="orange", label="ROC")
    # Plot line with no predictive power (baseline)
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Guessing")

    # Customize the plot
    plt.xlabel("False positive rate (fpr)")
    plt.ylabel("True positive rate (tpr)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()

# %%
plot_ROC(fpr, tpr)

# %%
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_probs_positive)

# %%
# Plot perfect ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_test)
plot_ROC(fpr, tpr)

# %%
# Perfect AUC score
roc_auc_score(y_test, y_test)

# %% [markdown]
# **Confusion matrix**
# 
# A confusion matrix is a quick way to compare the labels a model predicts and the actual labels it was supposed to predict.
# 
# In essence, giving you an idea of where the model is getting confused.

# %%
from sklearn.metrics import confusion_matrix

y_preds = clf.predict(X_test)
confusion_matrix(y_test, y_preds)

# %%
# Visualize confusion matrix with pd.crosstab()
pd.crosstab(
    y_test,
    y_preds,
    rownames=["Actual labels"],
    colnames=["Predicted labels"]
)

# %%
# Make our confusion matrix more visual with Seaborn's heatmap
import seaborn as sns

# Set the font scale
sns.set(font_scale = 1.5)

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_preds)

# Plot it using Seaborn
sns.heatmap(conf_mat)

# %% [markdown]
# **Classification Report**

# %%
from sklearn.metrics import classification_report
print(classification_report(y_test, y_preds))

# %%
# Where precision and recall become valuable
from sqlalchemy import true


disease_true = np.zeros(10_000)
disease_true[0] = 1 # only one positive case

disease_preds = np.zeros(10_000) # model predicts every case as zero
pd.DataFrame(classification_report(
    disease_true, 
    disease_preds,
    output_dict=true)
)

# %% [markdown]
# ### 4.2.2 Regression model evaluation metrics
# 
# Model evaluation metrics documentation - https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
# 
# The ones we're goign to cover are:
# 1. R^2 or coefficient of determination
# 2. Mean absolute error (MAE)
# 3. Mean squared error (MSE)

# %% [markdown]
# **R^2**
# 
# What R^2 does: Compares your models predictions to the mean of the targets. Values can range from negative infinity (a very porr model) to 1. For example, if all your model does is predict the mean of the targets, it's R^2 value would be 0. ANd if your model perfectly predicts a range of numbers it's R^2 value would be 1.

# %%
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
X = housing_df.drop("target", axis=1)
y = housing_df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# %%
model.score(X_test, y_test)

# %%
housing_df.head()

# %%
y_test

# %%
y_test.mean()

# %%
from sklearn.metrics import r2_score

# Fill an array with y_test mean
y_test_mean = np.full(len(y_test), y_test.mean())

# %%
y_test_mean[:10]

# %%
r2_score(
    y_true=y_test,
    y_pred=y_test_mean)

# %%
r2_score(
    y_true=y_test,
    y_pred=y_test
)

# %% [markdown]
# **Mean absolute error (MAE)**
# 
# MAE is the average of the absolute difference between predictions and actual values.
# It gives you an idea of how wrong your models predictions are.

# %%
from sklearn.metrics import mean_absolute_error

y_preds = model.predict(X_test)
mae = mean_absolute_error(
    y_test, 
    y_pred)
mae

# %%
df = pd.DataFrame(
    data = {"actual values": y_test,
    "predicted values": y_preds}
)
df["differences"] = df["predicted values"] - df["actual values"]
df.head(10)

# %%
# MAE using formulas and differences
np.abs(df["differences"]).mean()

# %% [markdown]
# **Mean squared error (MSE)**
# 
# MSE is the mean of the square of the errors between actual and predicted values.

# %%
from sklearn.metrics import mean_squared_error

y_preds = model.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
mse

# %%
df["squared_differences"] = np.square(df["differences"])
df.head()

# %%
# Calculate MSE by hand
squared = np.square(df["differences"])
squared.mean()

# %%
df_large_error = df.copy()
df_large_error.iloc[0]["squared_differences"] = 16

# %%
df_large_error.head(10)

# %%
# Calculate MSE with large error
df_large_error["squared_differences"].mean()

# %%
df_large_error.iloc[1:100] = 20
df_large_error

# %%
df_large_error["squared_differences"].mean()

# %% [markdown]
# ### 4.2.3 Finally using the scoring parameter

# %%
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

clf = RandomForestClassifier(n_estimators=100)

# %%
np.random.seed(42)

# Cross-validation accuracy
cv_acc = cross_val_score(clf, X, y, cv=5, scoring=None) 
    # if scoring = None, estimator's default scoring evaluating metric is used (accuracy for classification models)
print(cv_acc)
print(f"The cross-validated accuracy is: {np.mean(cv_acc)*100:.2f}%")

# %%
np.random.seed(42)
cv_acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy") # accuracy = default
print(f"The cross-validated accuracy is: {np.mean(cv_acc)*100:.2f}%")

# %%
# Precision
np.random.seed(42)
cv_precision = cross_val_score(clf, X, y, cv=5, scoring="precision")
print(cv_precision)
print(f"The cross-validated precision is: {np.mean(cv_precision)*100:.2f}%")

# %%
# Recall
np.random.seed(42)
cv_recall = cross_val_score(clf, X, y, cv=5, scoring="recall")
print(cv_recall)
print(f"THe cross-validated recall is: {np.mean(cv_recall)*100:.2f}%")

# %% [markdown]
# Let's see the `scoring` parameter being used for a regression model...

# %%
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

X = housing_df.drop("target", axis=1)
y = housing_df["target"]

model = RandomForestRegressor(n_estimators=100)

# %%
np.random.seed(42)
cv_r2 = cross_val_score(model, X, y, cv=5, scoring=None)
print(cv_r2)
print(f"r2: {np.mean(cv_r2)*100:.2f}%")

# %%
np.random.seed(42)
cv_mae = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
print(cv_mae)
print(f"mae: {np.mean(cv_mae)*100:.2f}%")

# %%
np.random.seed(42)
cv_mse = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
print(cv_mse)
print(f"mse: {np.mean(cv_mse)*100:.2f}%")

# %% [markdown]
# ## 4.3 Using different evaluation metrics as Scikit-Learn functions
# 
# The 3rd way to evaluate scikit-learn machine learning models/estimators is to use the `sklearn.metrics` module

# %%
from matplotlib import test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Create X & y
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
clf = RandomForestClassifier(n_estimators=100)

# Fit model
clf.fit(X_train, y_train)

# Evaluate model using evaluation functions
print("Classifier metrics on the test set")
print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test))*100:.2f}%")
print(f"Precision: {precision_score(y_test, clf.predict(X_test))*100:.2f}%")
print(f"Recall: {recall_score(y_test, clf.predict(X_test))*100:.2f}%")
print(f"f1: {f1_score(y_test, clf.predict(X_test))*100:.2f}%")

# %%
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Create X & y
X = housing_df.drop("target", axis=1)
y = housing_df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = RandomForestRegressor(n_estimators=100)

# Fit model
model.fit(X_train, y_train)

# Evaluate model using evaluation functions
print("Regressor metrics on the test set")
print(f"r2: {r2_score(y_test, model.predict(X_test))*100:.2f}%")
print(f"mae: {mean_absolute_error(y_test, model.predict(X_test))*100:.2f}%")
print(f"mse: {mean_squared_error(y_test, model.predict(X_test))*100:.2f}%")

# %% [markdown]
# ## 5. Improving a model
# 
# * First predictions = baseline predictions
# * First model = baseline model
# 
# From a data perspective:
# * Could we collect more data? (generally, the more data, the better)
# * Could we improve our data?
# 
# From a model perspective:
# * Is there a better model we could use?
# * Could we improve the current model?
# 
# Parameters = model find these patterns in data
# Hyperparameters = settings on a model you can adjust to (potentially) improve its ability to find patterns
# 
# Three ways to adjust hyperparameters:
# 1. By hand
# 2. Randomly with RandomSearchCV
# 3. Exhaustively with GridSearchCV

# %%
from random import Random
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

# %%
clf.get_params()

# %% [markdown]
# ### 5.1 Tuning hyperparameters by hand
# 
# Let's make 3 sets, training, validation and test.
# 
# We're going to try and adjust:
# 
# * `max_depth`
# * `max_features`
# * `min_samples_leaf`
# * `min_samples_split`
# * `n_estimators`

# %%
def evaluate_preds(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2)
    }
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"f1: {f1:.2f}")

    return metric_dict

# %%
from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)

# Shuffle the data
heart_disease_shuffled = heart_disease.sample(frac=1)

# Split into X & y
X = heart_disease_shuffled.drop("target", axis=1)
y = heart_disease_shuffled["target"]

# Split the data into train, validation & test sets
train_split = round(0.7 * len(heart_disease_shuffled)) # 70% of data
valid_split = round(train_split + 0.15 * len(heart_disease_shuffled))
X_train, y_train = X[:train_split], y[:train_split]
X_valid, y_valid = X[train_split:valid_split], y[train_split:valid_split]
X_test, y_test = X[valid_split:], y[valid_split:]

clf = RandomForestClassifier(n_estimators=10)
    # changing n_estimators to 10, since the default value in the video is 10 (older version of sklearn)

# Train on train data
clf.fit(X_train, y_train)

# Predict on validation data
y_preds = clf.predict(X_valid)

# Evaluate on validation set
baseline_metrics = evaluate_preds(y_valid, y_preds)
baseline_metrics

# %%
np.random.seed(42)

# Create a second classifier with different hyperparameters
clf_2 = RandomForestClassifier(n_estimators=100)
clf_2.fit(X_train, y_train)
y_preds_2 = clf_2.predict(X_valid)
clf_2_metrics = evaluate_preds(y_valid, y_preds_2)
clf_2_metrics

# %% [markdown]
# ### 5.2 Hyperparameter tuning with RandomizedSearchCV

# %%
from sklearn.model_selection import RandomizedSearchCV

grid = {
    "n_estimators": [10, 100, 200, 500, 1000, 1200],
    "max_depth": [None, 5, 10, 20, 30],
    "max_features": ["auto", "sqrt"],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 4]
}

np.random.seed(42)

# Split into X & y
X = heart_disease_shuffled.drop("target", axis=1)
y = heart_disease_shuffled["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate clf
clf = RandomForestClassifier(n_jobs=-1)

# Setup RandomizedSearchCV
rs_clf = RandomizedSearchCV(
    estimator=clf,
    param_distributions=grid,
    n_iter=50,
    cv=5,
    verbose=2
)

# %%
# Fit the RandomizedSearchCV version of clf
rs_clf.fit(X_train, y_train);

# %%
rs_clf.best_params_

# %%
# Make predictions with the best hyperparameters
rs_y_preds = rs_clf.predict(X_test)
rs_metrics = evaluate_preds(y_test, rs_y_preds)

# %% [markdown]
# ### 5.3 Hyperparameter tuning with GridSearchCV

# %%
grid

# %%
#Adjust search space according to result of RandomizedSearchCV, since GridSearchCV would test every combination of the grid
grid_2 = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_split': [6],
    'min_samples_leaf': [1, 2]
}
grid_2

# %%
from sklearn.model_selection import GridSearchCV, train_test_split
np.random.seed(42)

# Split into X & y
X = heart_disease_shuffled.drop("target", axis=1)
y = heart_disease_shuffled["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate clf
clf = RandomForestClassifier(n_jobs=-1)

# Setup GridSearchCV
gs_clf = GridSearchCV(
    estimator=clf,
    param_grid=grid_2,
    cv=5,
    verbose=2
)

# %%
# Fit the GridSearchCV version of clf
gs_clf.fit(X_train, y_train);

# %%
gs_clf.best_params_

# %%
gs_y_preds = gs_clf.predict(X_test)
gs_metrics = evaluate_preds(y_test, gs_y_preds)
gs_metrics

# %% [markdown]
# Let's compare our different models metrics

# %%
compare_metrics = pd.DataFrame({
    "baseline": baseline_metrics,
    "clf_2": clf_2_metrics,
    "random search": rs_metrics,
    "grid search": gs_metrics
})
compare_metrics.plot.bar(figsize=(10,7))

# %% [markdown]
# ## 6. Saving and loading trained machine learning models
# 
# Two ways to save and load machine learning models:
# 1. With Python's `pickle` module
# 2. With `joblib` module

# %% [markdown]
# **Pickle**

# %%
import pickle

# Save an existing model to file
pickle.dump(gs_clf, open("gs_random_forrest_model_1.pkl", "wb"))

# %%
# Load a saved model
loaded_pickle_model = pickle.load(open("gs_random_forrest_model_1.pkl", "rb"))

# %%
# Make some predictions
pickle_y_preds = loaded_pickle_model.predict(X_test)
evaluate_preds(y_test, pickle_y_preds)

# %% [markdown]
# **Joblib**

# %%
from joblib import dump, load

# Save model to file
dump(gs_clf, filename="gs_random_forrest_model_1.joblib")

# %%
# Import a saved joblib model
loaded_job_model = load(filename="gs_random_forrest_model_1.joblib")

# %%
# Make some predictions
joblib_y_preds = loaded_job_model.predict(X_test)
evaluate_preds(y_test, joblib_y_preds)

# %% [markdown]
# ## 7. Putting it all together

# %%
data = pd.read_csv("car-sales-extended-missing-data.csv")
data

# %%
data.isna().sum()

# %% [markdown]
# Steps we want to do (all in one cell):
# 1. Fill missing data
# 2. Convert data to numbers
# 3. Build a model on the data

# %%
# Getting data ready
import pandas as pdü
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Setup random seed
import numpy as np
np.random.seed(42)

# Import data and drop rows with missing labels
data = pd.read_csv("car-sales-extended-missing-data.csv")
data.dropna(subset=["Price"], inplace=True)

# Define different features and transformer pipeline
categorical_features = ["Make", "Colour"]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehote", OneHotEncoder(handle_unknown="ignore"))
    ]
)

door_feature = ["Doors"]
door_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=4))
    ]
)

numeric_feature = ["Odometer (KM)"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
    ]
)

# Setup preprocessing steps (fill missing values, then convert to numbers)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("door", door_transformer, door_feature),
        ("num", numeric_transformer, numeric_feature)
    ]
)

# Creating a preprocessing and modelling pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor())
    ]
)

# Split data
X = data.drop("Price", axis=1)
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit and score the model
model.fit(X_train, y_train)
model.score(X_test, y_test)

# %% [markdown]
# It's also possible to use `GridSearchCV` or `RandomizedSearchSV` with our `Pipeline`.

# %%
# Use GridSearchCV with our regression Pipeline
pipe_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "model__n_estimators": [100, 1000],
    "model__max_depth": [None, 5],
    "model__max_features": ["auto"],
    "model__min_samples_split": [2, 4]
}

gs_model = GridSearchCV(model, pipe_grid, cv=5, verbose=2)
gs_model.fit(X_train, y_train)

# %%
gs_model.score(X_test, y_test)

# %%



