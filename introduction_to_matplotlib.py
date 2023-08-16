# %% [markdown]
# # Introduction to Matplotlib

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
plt.plot();

# %%
plt.plot()
plt.show()

# %%
plt.plot([1, 2, 3, 4]);

# %%
x = [1, 2, 3, 4]
y = [11, 22, 33, 44]
plt.plot(x, y);

# %%
# 1st method
fig = plt.figure() # creates a figure
ax = fig.add_subplot()
plt.show()

# %%
# 2nd method
fig = plt.figure() 
ax = fig.add_axes([1, 1, 1, 1])
ax.plot(x, y)
plt.show()

# %%
# 3rd method (recommended)
fig, ax = plt.subplots()
ax.plot(x, [50, 100, 200, 250]);
type(fig), type(ax)

# %% [markdown]
# ## Matplotlib example workflow

# %%
# 0. import matplotlib abd get it ready for plotting in Jupyter
import matplotlib.pyplot as plt

# 1. Prepare data
x = [1, 2, 3, 4]
y = [11, 22, 33, 44]

# 2. Setup plot
fig, ax = plt.subplots(figsize=(10, 10)) # width, heigth

# 3. Plot the data
ax.plot(x, y)

# 4. Customize plot
ax.set(
    title="Simple Plot",
    xlabel="x-axis",
    ylabel="y-label"
)

# 5. Save & show figure
fig.savefig("Images/sample-plot.png")

# %% [markdown]
# ## Making figures with NumPy arrays
# 
# We want:
# * Line plot
# * Scatter plot
# * Bar Plot
# * Hist
# * Subplots

# %%
import numpy as np

# %%
# Create some data
x = np.linspace(0, 10, 100)

# %%
fig, ax = plt.subplots()
ax.plot(x, x**2);

# %%
# Use the same data to make a scatter
fig, ax = plt.subplots()
ax.scatter(x, np.exp(x));

# %%
# Another one
fig, ax = plt.subplots()
ax.scatter(x, np.sin(x));

# %%
# Make a plot from dictionary
nut_butter_prices = {
    "Almond butter": 10,
    "Peanut butter": 8,
    "Cashew butter": 12,
}
fig, ax = plt.subplots()
ax.bar(nut_butter_prices.keys(), nut_butter_prices.values());
ax.set(
    title="Dan's Nut Butter Store",
    ylabel="Price ($)",
);

# %%
fig, ax = plt.subplots()
ax.barh(list(nut_butter_prices.keys()), list(nut_butter_prices.values()))

# %%
# Make some data
x = np.random.randn(1000)
fig, ax = plt.subplots()
ax.hist(x);

# %% [markdown]
# ## Two options for subplots

# %%
# Subplot option 1
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(10, 5)
);
ax1.plot(x, x/2);
ax2.scatter(np.random.random(10), np.random.random(10));
ax3.bar(nut_butter_prices.keys(), nut_butter_prices.values());
ax4.hist(np.random.randn(1000));

# %%
# Subplot option 2
fig, ax = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(10, 5)
)
ax[0, 0].plot(x, x/2);
ax[0, 1].scatter(np.random.random(10), np.random.random(10));
ax[1, 0].bar(nut_butter_prices.keys(), nut_butter_prices.values());
ax[1, 1].hist(np.random.randn(1000));

# %% [markdown]
# ## Plotting from pandas DataFrame

# %%
import pandas as pd

# %%
# Make a DataFrame
import os
import wget
if not os.path.exists("car-sales.csv"):
    site_url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/car-sales.csv"
    file_name = wget.download(site_url)
    car_sales = pd.read_csv(file_name)
else:
    car_sales = pd.read_csv("car-sales.csv")
car_sales.head()

# %%
ts = pd.Series(
    np.random.randn(1000),
    index=pd.date_range("1/1/2022",
    periods=1000)
)
ts = ts.cumsum()
ts.plot();

# %%
car_sales["Price"] = car_sales["Price"].str.replace('[\$\,\.]', '')
car_sales

# %%
type(car_sales["Price"])

# %%
# Remoce last two zeros
car_sales["Price"] = car_sales["Price"].str[:-2]
car_sales

# %%
car_sales["Sale Date"] = pd.date_range("1/1/2020", periods=len(car_sales))
car_sales

# %%
car_sales["Total Sales"] = car_sales["Price"].astype(int).cumsum()
car_sales

# %%
# Let's plot total sales
car_sales.plot(
    x="Sale Date",
    y="Total Sales"
);

# %%
car_sales["Price"] = car_sales["Price"].astype(int)

car_sales.plot(
    x="Odometer (KM)",
    y="Price",
    kind="scatter"
);

# %%
# How about a bar graph
x = np.random.rand(10, 4)

# Turn it into a dataframe
df = pd.DataFrame(x, columns=['a', 'b', 'c', 'd'])
df

# %%
df.plot.bar();

# %%
df.plot(kind="bar");

# %%
car_sales

# %%
car_sales.plot(
    x="Make",
    y="Odometer (KM)",
    kind="bar"
);

# %%
# How about histograms
car_sales["Odometer (KM)"].plot.hist();

# %%
car_sales["Odometer (KM)"].plot(kind="hist");

# %%
car_sales["Odometer (KM)"].plot.hist(bins=8);

# %%
# Let's try on another dataset
import os
if not os.path.exists("heart-disease.csv"):
    site_url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
    file_name = wget.download(site_url)
    heart_disease = pd.read_csv(file_name)
else:
    heart_disease = pd.read_csv("heart-disease.csv")
heart_disease.head()

# %%
# Create a histogram
print(heart_disease.head())
heart_disease["age"].plot.hist(bins=50)

# %%
heart_disease.head()

# %%
heart_disease.plot.hist(
    subplots=True,
    figsize=(10, 30)
);

# %% [markdown]
# ### Which one should you use? (pyplot vs matplotlib OO method)
#  
# * When plotting something quickly, okay to use the pyplot method
# * When plotting something more advanced, use the OO method

# %%
over_50 = heart_disease[heart_disease["age"] > 50]
over_50.head(), len(over_50)

# %%
# Pyplot method
over_50.plot(
    kind="scatter",
    x="age",
    y="chol",
    c="target",
    figsize=(10, 6)
);

# %%
## OO method mixed with pyplot method
fig, ax = plt.subplots(figsize=(10, 6))
over_50.plot(
    kind="scatter",
    x="age",
    y="chol",
    c="target",
    ax=ax
);

ax.set_xlim([45, 100])

# %%
## OO Method from scratch
fig, ax = plt.subplots(figsize=(10, 7));

# Plot the data
scatter = ax.scatter(
    x=over_50["age"],
    y=over_50["chol"],
    c=over_50["target"]
);

# Customize
ax.set(
    title="Heart Disease and Cholesterol Levels",
    xlabel="Age",
    ylabel="Cholesterol"
);

# Legend
ax.legend(*scatter.legend_elements(), title="Target");

# Add horizontal line
ax.axhline(
    over_50["chol"].mean(),
    linestyle="--"
);

# %%
over_50.head()

# %%
# Subplot of chol, age, thalach
fig, (ax0, ax1) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 10),
    sharex=True
);

# Add data
scatter = ax0.scatter(
    x=over_50["age"],
    y=over_50["chol"],
    c=over_50["target"]
)

# Customize ax0
ax0.set(
    title="Heart Disease and Cholesterol Levels",
    ylabel="Cholesterol"
)

# Legend
ax0.legend(*scatter.legend_elements(), title="Target");

# Meanline
ax0.axhline(
    y=over_50["chol"].mean(),
    linestyle="--"
);

# Add data to ax1
scatter = ax1.scatter(
    x=over_50["age"],
    y=over_50["thalach"],
    c=over_50["target"]
);

# Customize ax1
ax1.set(
    title="Heart Disease and max heart rate",
    xlabel="Age",
    ylabel="Max Heart Rate"
);

# Meanline
ax1.axhline(
    y=over_50["thalach"].mean(),
    linestyle="--"
);

# Add a title to the figure
fig.suptitle("Heart Diseasse Analysis", fontsize=16, fontweight="bold");

# %% [markdown]
# ## Customizing Matplotlib plots and getting stylish

# %%
# See the different styles available
plt.style.available

# %%
car_sales["Price"].plot();

# %%
plt.style.use("seaborn-whitegrid")
car_sales["Price"].plot();

# %%
plt.style.use("seaborn")
car_sales["Price"].plot();

# %%
car_sales.plot(
    x="Odometer (KM)",
    y="Price",
    kind="scatter"
);

# %%
plt.style.use("ggplot")
car_sales["Price"].plot();

# %%
# Create some data
x = np.random.randn(10, 4)

# %%
df = pd.DataFrame(x, columns=["a", "b", "c", "d"])
df

# %%
ax = df.plot(
    kind="bar"
);

# %%
# Customize our plot with the set() method
ax = df.plot(kind="bar")
# Add some labels and a title
ax.set(
    title="Random Number Bar Graph from DataFrame",
    xlabel="Row number",
    ylabel="Random number"
);
# Make the legend visible
ax.legend().set_visible(True)


# %%
# Set the style
plt.style.use("seaborn")

## OO Method from scratch
fig, ax = plt.subplots(figsize=(10, 7));

# Plot the data
scatter = ax.scatter(
    x=over_50["age"],
    y=over_50["chol"],
    c=over_50["target"],
    cmap="winter"
);

# Customize
ax.set(
    title="Heart Disease and Cholesterol Levels",
    xlabel="Age",
    ylabel="Cholesterol"
);

# Legend
ax.legend(*scatter.legend_elements(), title="Target");

# Add horizontal line
ax.axhline(
    over_50["chol"].mean(),
    linestyle="--"
);

# %%
# Customizing the y and x axis limitations
# Subplot of chol, age, thalach
fig, (ax0, ax1) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 10),
    sharex=True
);

# Add data
scatter = ax0.scatter(
    x=over_50["age"],
    y=over_50["chol"],
    c=over_50["target"],
    cmap="winter"
)

# Customize ax0
ax0.set(
    title="Heart Disease and Cholesterol Levels",
    ylabel="Cholesterol"
)
ax0.set_xlim([50, 80])
#ax0.set_ylim([60, 200])

# Legend
ax0.legend(*scatter.legend_elements(), title="Target");

# Meanline
ax0.axhline(
    y=over_50["chol"].mean(),
    linestyle="--",
    c="red"
);

# Add data to ax1
scatter = ax1.scatter(
    x=over_50["age"],
    y=over_50["thalach"],
    c=over_50["target"],
    cmap="winter"
);

# Customize ax1
ax1.set(
    title="Heart Disease and max heart rate",
    xlabel="Age",
    ylabel="Max Heart Rate"
);

ax1.set_ylim([60, 200])

# Meanline
ax1.axhline(
    y=over_50["thalach"].mean(),
    linestyle="--",
    c="red"
);

# Legend
ax1.legend(*scatter.legend_elements(), title="Target");

# Add a title to the figure
fig.suptitle("Heart Diseasse Analysis", fontsize=16, fontweight="bold");

# %%
fig.savefig("heart-disease-analysis-plot-saved-with-code.png")

# %%



