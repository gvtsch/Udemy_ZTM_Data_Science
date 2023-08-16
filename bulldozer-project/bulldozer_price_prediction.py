# %% [markdown]
# # 🚜 Predicting the Sale Price of Bulldozers using Machine Learning
# 
# In this notebook, i am going to go through an example machine learning project with the goal of predicting the sale price of bulldozers.
# 
# ## 1. Problem definition
# > How well can we predict the future sale price of a bulldozer, given its characteristics and previous examples of how much similar bulldozers have been sold for?
# 
# ## 2. Data
# The data is downloaded from the Kaggle Bluebook for Bulldozers competition.
# 
# There are 3 main datasets.
# 
# * Train.csv is the training set, which contains data through the end of 2011.
# * Valid.csv is the validateion set, which contains data from January 1, 2012 - April 30, 2012. I make predictions on this set throughout the majority of the competition. 
# * Test.csv is the test set. It contains data from May 1, 2012 - November 2012.
# 
# ## 3. Evaluation
# The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.
# 
# For more on the evaluation of this project check Kaggle.
# 
# **Note:** The goal for most regression evaluation metrics is to minimize the error. For example, our goal for this project will be to build a machine learning model which minimises RMSLE.
# 
# ## 4. Features
# Kaggle provides a data dictionary detailing all of the features of the dataset. You can view this data dictionary...

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# %%
# Import training and validation sets
df = pd.read_csv(
    "bluebook-for-bulldozers/TrainAndValid.csv",
    low_memory=False
)

# %%
df.info()

# %%
df.isna().sum()

# %%
fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000]);

# %%
df.SalePrice.plot.hist()

# %% [markdown]
# ### Parsing dates
# 
# When we work with time series data, we want to enrich the time & date component as much as possible.
# 
# We can do that by telling pandas which of our columns has dates in it using the `parse_dates` parameter.

# %%
# Import data again but this time parse dates
df = pd.read_csv(
    "bluebook-for-bulldozers/TrainAndValid.csv",
    low_memory=False,
    parse_dates=["saledate"]
)

# %%
df.saledate.dtype

# %%
df.saledate[:1000]

# %%
fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])

# %%
df.head().T

# %%
df.saledate.head(20)

# %% [markdown]
# ### Sort dataframe by saledate
# 
# When working with timeseries data, it's a good idea to sort it by date.

# %%
# Sort DataFrame in date order
df.sort_values(by=["saledate"], inplace=True, ascending=True)
df.saledate.head(20)

# %%
df.head() # will show, that the indizes have changed, since the DataFrame has been sorted

# %% [markdown]
# ### Make a copy of the original DataFrame
# 
# Make a copy of the original DataFrame, so when you manipulate the copy, you'll still have got your original data.

# %%
# Make a copy
df_tmp = df.copy()

# %% [markdown]
# ### Add datetime parameters for `saledate` column

# %%
df_tmp["saleYear"] = df_tmp.saledate.dt.year
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saleDayOfWeel"] = df_tmp.saledate.dt.dayofweek
df_tmp["saleDayOfYear"] = df_tmp.saledate.dt.dayofyear

# %%
df_tmp.head().T

# %%
# Now we've enriched our DataFrame with datatime features, we can remove saledate
df_tmp.drop("saledate", axis=1, inplace=True)

# %%
# Check the values of different columns
df_tmp.state.value_counts()

# %% [markdown]
# ## 5. Modelling
# Done enough EDA (you can always do more) but let's start to do some model driven EDA

# %%
# Let's build a machine learning model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_jobs=-1, random_state=42
)

# Throwing errors:
#model.fit(
#    df_tmp.drop("SalePrice", axis=1),
#    df_tmp["SalePrice"]
#)

# %% [markdown]
# ### Convert strings to categories
# 
# One way we can turn all of our data into numbers is by converting them into pandas categories.

# %%
pd.api.types.is_string_dtype(df_tmp["UsageBand"])

# %%
# Find the columns which contain strings
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)

# %%
# If you're wondering what df.items() does, here's an example
random_dict = {
    "key1": "hello",
    "key2": "world"
}

for key, value in random_dict.items():
    print(f"this is a key: {key}")
    print(f"this is a value: {value}")

# %%
# This will turn all of the strin value into category values
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()

# %%
df_tmp.info()

# %%
df_tmp.state.cat.categories

# %%
df_tmp.state.cat.codes

# %% [markdown]
# Thanks to pandas Categories, we now have a way to acces all of our data in the form of numbers.
# 
# But we still have a bunch of missing data...

# %%
# Check missing data
df_tmp.isnull().sum()/len(df_tmp)

# %% [markdown]
# ### Save preprocessed data
# 

# %%
# Export current tmp dataframe
df_tmp.to_csv(
    "bluebook-for-bulldozers/train_tmp.csv",
    index=False
)

# %%
# Import preprocessed data
df_tmp = pd.read_csv(
    "bluebook-for-bulldozers/train_tmp.csv",
    low_memory=False
)

# %% [markdown]
# ## Fill missing values
# 
# ### Fill numeric missing values first

# %%
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)

# %%
# Check for which numeric columns have null values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column which tells us if the data was missing or not
            df_tmp[label+"_is_missing"] = pd.isnull(content)
            # Fill missing numeric values with median
            df_tmp[label] = content.fillna(content.median())

# %%
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)

# %%
# Check to see how many examples were missing
df_tmp.auctioneerID_is_missing.value_counts()

# %%
df_tmp.isna().sum()

# %% [markdown]
# ### Filling and turning categorical variables into numbers

# %%
# Check for columns which aren't numeric
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)

# %%
# Turn categorical variables into numbers and fill missing
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to indicate whether sample had missing value
        df_tmp[label+"_is_missing"] = pd.isnull(content)
        # Turn categories into numbers and add +1
        # Missing values are set to -1 if missing
        # But we want positive numbers
        df_tmp[label] = pd.Categorical(content).codes+1 

# %%
pd.Categorical(df_tmp["state"]).codes

# %%
df_tmp.info()

# %%
df_tmp.head().T

# %%
df_tmp.isna().sum()

# %% [markdown]
# Now that all of our data is numeric as well as our dataframe has no missing values we should be able to build a machine learning model

# %%
df_tmp.head()

# %%
%%time
# Instantiate model
model = RandomForestRegressor(
    n_jobs=-1,
    random_state=42
)

# Fit the model
model.fit(
    df_tmp.drop("SalePrice", axis=1),
    df_tmp["SalePrice"]
)


# %%
# Score the model
model.score(
    df_tmp.drop("SalePrice", axis=1),
    df_tmp["SalePrice"]
)

# %% [markdown]
# **Question:** Why doesn't the above metric hold water? (why isn't the metric reliable)

# %% [markdown]
# ### Splitting data into train/validation sets

# %%
df_tmp.saleYear

# %%
df_tmp.saleYear.value_counts()

# %%
# Split data into training and validation
df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]

len(df_val), len(df_train)

# %%
# Split data into X & y
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape

# %%
y_train

# %% [markdown]
# > **Note:** We worked on filling missing data before splitting it up into train and validation sets! So we used data from validation set to fill the train set.
# 
# The way to go:
# 1. Encode/transfrom all categorical variables of the data (on the entire dataset to ensure, that the categorical values are encoded the same across training and test sets. If this isn't possible make sure the datasets have the same column names).
# 2. Split the data into train and test
# 3. Fill the training set and test set numerical values **separately**.

# %% [markdown]
# ### Building an evaluation function

# %%
# Create evaluation function (the competition uses RMSLE)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_test, y_pred):
    """
    Calculates root mean squared log error between predictions and tru labels.
    """
    return np.sqrt(mean_squared_log_error(y_test, y_pred))

# Create function to evaluate model on a few different levels
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {
        "Training MAE": mean_absolute_error(y_train, train_preds),
        "Valid MAE": mean_absolute_error(y_valid, val_preds),
        "Training RMSLE": rmsle(y_train, train_preds),
        "Valid RMSLE": rmsle(y_valid, val_preds),
        "Training R^2": r2_score(y_train, train_preds),
        "Valid R^2": r2_score(y_valid, val_preds)
    }
    return scores

# %% [markdown]
# ## Testing our model on a subset (to tune the hyperparameters)

# %%
# Change max_samples value
model = RandomForestRegressor(
    n_jobs=-1,
    random_state=42,
    max_samples=10_000
)

model.fit(X_train, y_train)

# %%
show_scores(model)

# %% [markdown]
# ### Hyperparameter tuning with RandomizedSearchCV

# %%
%%time
from sklearn.model_selection import RandomizedSearchCV

# Different RandomForestRegressor hyperparameters
rf_grid = {
    "n_estimators": np.arange(10, 100, 10),
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2),
    "max_features": [0.5, 1, "sqrt", "auto"],
    "max_samples": [10_000]
}

# Instantiate
rs_model = RandomizedSearchCV(
    RandomForestRegressor(
        n_jobs=-1, 
        random_state=42),
    param_distributions=rf_grid,
    n_iter=5,
    cv=5,
    verbose=True
)

# %%
rs_model.fit(X_train, y_train)

# %%
# Find the best model hyperparameters
rs_model.best_params_

# %%
show_scores(rs_model)

# %% [markdown]
# ### Train a model with the best hyperparameters
# **Note:** These were found after 100 iterations of `RandomizedSearchCV`.

# %%
%%time

# Most ideal hyperparameters
ideal_model = RandomForestRegressor(
    n_estimators=40,
    min_samples_leaf=1,
    min_samples_split=14,
    max_features=0.5,
    n_jobs=-1,
    max_samples=None,
    random_state=42
)

# Fit
ideal_model.fit(X_train, y_train)

# %%
show_scores(ideal_model)

# %% [markdown]
# ## Make predictions on test data

# %%
# Import test data
df_test = pd.read_csv(
    "bluebook-for-bulldozers/Test.csv",
    low_memory=False,
    parse_dates=["saledate"]
)
df_test.head()

# %% [markdown]
# ### Preprocessing the data (getting the test dataset in the same format as training dataset)

# %%
def preprocess_data(df):
    """
    Performs transformation on df and returns transformed df
    """
    
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeel"] = df.saledate.dt.dayofweek
    df["saleDayOfYear"] = df.saledate.dt.dayofyear

    df.drop("saledate", axis=1, inplace=True)

    # Fill numeric rows with median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add binary column which tells us if the data was missing
                df[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                df[label] = content.fillna(content.median())

        # Fill categorical missing data und turn categories into numbers
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
            # Add +1 to the category code because pandas encodes missing cats as -1
            df[label] = pd.Categorical(content).codes + 1

    return df

# %%
df_test = preprocess_data(df_test)
df_test.head()

# %%
X_train.head()

# %% [markdown]
# Different columns! 101 vs 102

# %%
# Find the differences using sets
set(X_train.columns) - set(df_test.columns)

# %%
# Manually adjust df_test to have auctioneerID_is_missing column
df_test["auctioneerID_is_missing"] = False
df_test.head()

# %% [markdown]
# Finally the test dataframe has the same features as the training dataframe. We now can make predictions!

# %%
# Make predictions on test dataset
test_preds = ideal_model.predict(df_test)
test_preds, len(test_preds)

# %% [markdown]
# We've made some predictions but they're not yet in the same format Kaggle is asking for...

# %%
# Format predictions into the same format Kaggle is after
df_preds = pd.DataFrame()
df_preds["SalesID"] = df_test["SalesID"]
df_preds["SalesPrice"] = test_preds

# %%
df_preds

# %%
# Export prediction data
df_preds.to_csv("bluebook-for-bulldozers/test_preds.csv", index=False)

# %% [markdown]
# ### Feature importance
# 
# Feature importance seeks to figure out which different attributes of the data were most important when it comes to predicting the **target variable** (SalePrice).

# %%
# Find feature importance of our best model
len(ideal_model.feature_importances_)

# %%
# Helper function for plotting feature importance
def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({
            "features": columns,
            "feature_importances": importances
        })
        .sort_values("feature_importances", ascending=False)
        .reset_index(drop=True)
    )

    # Plot
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:n])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature importance")
    ax.invert_yaxis()

# %%
plot_features(X_train.columns, ideal_model.feature_importances_)

# %%



