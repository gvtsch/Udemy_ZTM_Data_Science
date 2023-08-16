# %% [markdown]
# # 🐶 Using Transfer Learning and TensorFlow 2.x to Classify Different Dog Breeds
# 
# Who's that doggy in the window?
# 
# Dogs are incredible. But have you ever been sitting at a cafe, seen a dog and not known what breed it is? I have. And then someone says, "it's an English Terrier" and you think, how did they know that?
# 
# In this project we're going to be using machine learning to help us identify different breeds of dogs.
# 
# To do this, we'll be using data from the [Kaggle dog breed identification competition](https://www.kaggle.com/c/dog-breed-identification/overview). It consists of a collection of 10,000+ labelled images of 120 different dog breeds.
# 
# This kind of problem is called multi-class image classification. It's multi-class because we're trying to classify mutliple different breeds of dog. If we were only trying to classify dogs versus cats, it would be called binary classification (one thing versus another).
# 
# Multi-class image classification is an important problem because it's the same kind of technology Tesla uses in their self-driving cars or Airbnb uses in atuomatically adding information to their listings.
# 
# Since the most important step in a deep learng problem is getting the data ready (turning it into numbers), that's what we're going to start with.
# 
# We're going to go through the following TensorFlow/Deep Learning workflow:
# 
# 1. Get data ready (download from Kaggle, store, import).
# 2. Prepare the data (preprocessing, the 3 sets, X & y).
# 3. Choose and fit/train a model ([TensorFlow Hub](https://www.tensorflow.org/hub), ´tf.keras.applications´, [TensorBoard](https://www.tensorflow.org/tensorboard), [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)).
# 4. Evaluating a model (making predictions, comparing them with the ground truth labels).
# 5. Improve the model through experimentation (start with 1000 images, make sure it works, increase the number of images).
# 6. Save, sharing and reloading your model (once you're happy with the results).
# 
# For preprocessing our data, we're going to use TensorFlow 2.x. The whole premise here is to get our data into Tensors (arrays of numbers which can be run on GPUs) and then allow a machine learning model to find patterns between them.
# 
# For our machine learning model, we're going to be using a pretrained deep learning model from TensorFlow Hub.
# 
# The process of using a pretrained model and adapting it to your own problem is called **transfer learning**. We do this because rather than train our own model from scratch (could be timely and expensive), we leverage the patterns of another model which has been trained to classify images.

# %%
# Import necessary tools
import tensorflow as tf
import tensorflow_hub as hub
print("TF version: ", tf.__version__)
print("TF hub version: ", hub.__version__)

# %%
tf.config.list_physical_devices()

# %%
# Check for GPU availability
print("GPU", "available (Yessss!!!)" if tf.config.list_physical_devices("GPU") else "Not availabe :/")

# %% [markdown]
# ## Get workspace ready
# * Import TensorFlow 2.x ✅
# * Import TensorFlow Hub ✅
# * Make sure we're using a GPU ✅

# %% [markdown]
# ## Get data ready (turn it into tensors)
# 
# With all machine learning model,s our data has to be in numerical format. So that's what we'll be doing first: Turning our images into tensors (numerical representation).
# 
# Let's start by accssing our data and checking out the labels.

# %%
# Checkout the labels of our data
import pandas as pd
labels_csv = pd.read_csv("labels.csv")
print(labels_csv.describe())
print(labels_csv.head())

# %%
labels_csv.head()

# %%
# How many images of each breed are there?
labels_csv["breed"].value_counts().plot.bar(figsize=(10,7));

# %%
labels_csv["breed"].value_counts().median()

# %%
# Let's view an image
from IPython.display import Image
Image("train/0a0c223352985ec154fd604d7ddceabd.jpg")

# %% [markdown]
# ### Getting images and their labels
# 
# Let's get a list of all of our image file pathnames.

# %%
# Create pathnames from image ID's
filenames = ["train/" + fname + ".jpg" for fname in labels_csv["id"]]

# Check the first 10
filenames[:10]

# %%
import os
os.listdir("train/")[:10]

# %%
# Check wheter number of filenames matches number of image files
if len(os.listdir("train/")) == len(filenames):
    print("Filenames match actual amount of files!!! Proceed.")
else:
    print("Filenames do not match actual amount of files, check the target directory.")

# %%
# One more check
Image(filenames[9_000])

# %%
labels_csv["breed"][9_000]

# %% [markdown]
# Since we've got our training image filepaths in a list, let's prepare our labels.

# %%
import numpy as np
labels = labels_csv["breed"]
# labels = labels_csv["breed"].to_numpy() does the same as below
labels = np.array(labels)
labels, len(labels)

# %%
# See if number of labels matches the number of filenames
if len(labels) == len(filenames):
    print("Number of labels matches number of filenames")
else:
    print("Number do not match!")

# %%
# Find the unique label values
unique_breeds = np.unique(labels)
unique_breeds[:10], len(unique_breeds)

# %%
# Turn a single label into an array of booleans
print(labels[0])
labels[0] == unique_breeds

# %%
# Turn every label into a boolean array
boolean_labels = [label == unique_breeds for label in labels]
boolean_labels[:2], len(boolean_labels)

# %%
# Example: Turning boolean array into integers
print(labels[0]) # original label
print(np.where(unique_breeds == labels[0])) # index where label occurs
print(boolean_labels[0].argmax()) # index where label occurs in boolean array
print(boolean_labels[0].astype(int)) # there will be a 1 where the sample label occurs

# %%
print(labels[2])
print(boolean_labels[2].astype(int))

# %% [markdown]
# ### Creating our on validation set
# SInce the dataset from Kaggle does not come with a validation set, we're going to create our own.

# %%
# Setup X & y variables
X = filenames
y = boolean_labels


# %% [markdown]
# We're going to start off experimenting with ~1000 images and increase as needed.

# %%
# Set number of images to use for experimenting
NUM_IMAGES = 1_000 

# %%
# Let's split data into train and validation
from sklearn.model_selection import train_test_split

# Split them into training and validation of total size NUM_IMAGES
X_train, X_val, y_train, y_val = train_test_split(
    X[:NUM_IMAGES], 
    y[:NUM_IMAGES],
    test_size=0.2,
    random_state=42
)
len(X_train), len(y_train), len(X_val), len(y_val)

# %%
# Let's have a geez at the training data
X_train[:2], y_train[:2]

# %% [markdown]
# ## Preprocessing images (turning images into Tensors)
# 
# To preprocess our images into Tensors we're going to write a function which does a few things:
# 1. Take an image filepath as input
# 2. Use TensorFlow to read the file and save it to a variable, `image`
# 3. Turn our `image` (a jpg) into Tensors
# 4. Normalize our image (convert color channel values from 0-255 to 0-1)
# 5. Resize the `image` to be a shape of (224, 224)
# 6. Return the modified `image`
# 
# ### Before we do, let's see what importing an image looks like

# %%
# Convert image to NumPy array
from matplotlib.pyplot import imread
image = imread(filenames[42])
image.shape, image.max(), image.min()

# %%
# Turn image into a Tensor
tf.constant(image)[:2]

# %% [markdown]
# Now we've seen what an image looks like as a Tensor, let's make a function to preprocess them.

# %%
# Define image size
IMG_SIZE = 224

# Create a function for preprocessing images
def process_image(image_path, img_size=IMG_SIZE):
    """
    Takes an image file path and turns it into a Tensor
    """
    # Read in an image file
    image = tf.io.read_file(image_path)
    # Turn the jpg into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert the colour channel values from 0-255 to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image
    image = tf.image.resize(image, size=[img_size, img_size])

    return image

# %% [markdown]
# ## Turning our data into batches
# 
# Why turn our data into batches?
# 
# Let's say you're trying to preocess 10_000+ images in one go... they all might not fit into memory.
# 
# So that's why we do about 32 (this is the batch size) images at a time (you can manually adjust the batch size if need be).
# 
# IN order to use TensorFlow effectively, we need our data in the form of Tensor tuples which look like this:
# `(image, labels)`

# %%
# Create a simple function to return a tuple (image, label)
def get_image_label(image_path, label):
    """
    Takes an image file path name and the associated label,
    precesses the image and returns a tuple if (image, label).
    """
    image = process_image(image_path)
    return image, label

# %%
(process_image(X[42]), y[42])

# %% [markdown]
# Now we've got a way to turn our data into tuples of Tensors in the form `(image, label)`, let's make a function to turn all of our data (`X` & `y`) into batches.

# %%
# Define the batch size, 32 is a good start
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
    Creates batches of data out of image (X) und label (y) pairs.

    Shuffles the data if it's training data but doesn't shuffle if it's validation data.
    
    Also accepts test data is input (no labels).
    """
    # If the data is a test dataset, we probably don't have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) # only filepaths (no labels)
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch

    # If the data is a valid dataset, we don't need to shuffle it
    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((
            tf.constant(X), # filepaths
            tf.constant(y)  # labels
        ))
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch

    # Otherwise it must be a training dataset
    else:
        print("Creating training data batches...")
        data = tf.data.Dataset.from_tensor_slices((
            tf.constant(X),
            tf.constant(y)
        ))
        # Shuffle the pathnames and labels before mapping image processor function is faster than shuffling images
        data = data.shuffle(buffer_size=len(X))
        # Create image label tuples (this also turns the image path into a preprocessed image)
        data = data.map(get_image_label)
        # Turn the data into batches
        data_batch = data.batch(BATCH_SIZE)
        return data_batch


# %%
# Create training and validation data batches
train_data = create_data_batches(
    X_train,
    y_train
)
val_data = create_data_batches(
    X_val,
    y_val,
    valid_data=True
)

# %%
# Check out the different attributes of data batches
train_data.element_spec, val_data.element_spec

# %% [markdown]
# ## Visualizing data batches
# Our data is now in batches, however these can be a little hard to understand/comprehend, let's visualize them.

# %%
import matplotlib.pyplot as plt

# Create a function for viewing images in a data batch
def show_25_images(images, labels):
    """
    Displays a plot of 25 images and their labels from a data batch
    """
    # Setup the figure
    plt.figure(figsize=(10,10))
    # Loop through 25 images
    for i in range(25):
        # Create subplots (5 rows, 5 cols)
        ax = plt.subplot(5, 5, i+1)
        # Display image
        plt.imshow(images[i])
        # Add the image label as title
        plt.title(unique_breeds[labels[i].argmax()])
        # Turn grid lines off
        plt.axis("off")

# %%
len(train_data)

# %%
train_images, train_labels = next(train_data.as_numpy_iterator())
train_images, train_labels, len(train_images), len(train_labels)

# %%
# Now let's visualize the data in a training batch
show_25_images(train_images, train_labels)

# %%
# Visualize validation set
val_images, val_labels = next(val_data.as_numpy_iterator())
show_25_images(val_images, val_labels)

# %% [markdown]
# ## Building a model
# 
# Before we build a model, there are a few things, we need to define:
# * The input shape (images shape in the form of Tensors) to the model
# * The output shape (images labels in the form of Tensors) of the model
# * The URL of the model to use

# %%
# Setup input shape
INPUT_SHAPE =[None, IMG_SIZE, IMG_SIZE, 3]
# Setup output shape
OUTPUT_SHAPE = len(unique_breeds)
# Setup model URL from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5"

# %% [markdown]
# Now we've got our inputs, outputs and model ready to go, let's put them together into a Keras deep learning model!
# 
# Knowing this, let's create a function which:
# * Takes the input shape, output shape and the model we've chosen as parameters.
# * Defines the layers in a Keras model in sequential fashion (do this first, then this, then that)
# * Compiles the model (says it should be evaluated and improved)
# * Build the model (tells the model the input shape it'll be getting)
# * Returns the model
# 
#  All of the steps can be found here: https://www.tensorflow.org/guide/keras/overview

# %%
# Create a function which builds a Keras model
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
    print("Building model with: ", MODEL_URL)

    # Setup model layers
    model = tf.keras.Sequential([
        hub.KerasLayer(MODEL_URL), # Layer 1: input layer
        tf.keras.layers.Dense(
            units=OUTPUT_SHAPE, 
            activation="softmax"
        ) # Layers 2: output layer
    ])

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.build(INPUT_SHAPE)

    return model

# %%
model = create_model()
model.summary()

# %% [markdown]
# ## Creating callbacks
# 
# Callbacks are helperfunctions a model can use during training to do such things as save its progress, check its progress or stop training early if a model stops improving.
# 
# We'll create two callbacks, one for TensorBoard which helps track our models progress and another for early stopping, which prevents our model from training for too long.
# 
# ### Tensorboard Callback
# To setup a TensorBoard callback, we need to do 3 things:
# 1. Load the TensorBoard notebook extension ✅
# 2. Create a TensorBoard callback which is able to save logs to a directory and pass it to our model's `fit()` function. ✅
# 3. Visualize our models training logs with the  `$tensorboard` magic function (we'll do this after model training).
# 

# %%
# Load TensorBoard notebook extension
%load_ext tensorboard

# %%
import datetime

# Create a function to build a TensorBoard callback
def create_tensorboard_callback():
    """
    Create a log directory for storing TensorBoard logs.
    """
    logdir = os.path.join(
        "Logs",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    return tf.keras.callbacks.TensorBoard(logdir)

# %% [markdown]
# ### Early stopping callback
# 
# Early stopping helps stop our model from overfitting by stopping training if a certain evaluation metric stops improving.

# %%
# Create early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3
)

# %% [markdown]
# ## Training a model (on subset of data)
# 
# Our first model is only going to train on 1000 images, to make sure everything is working.

# %%
NUM_EPOCHS = 100

# %% [markdown]
# Let's create a function which trains a model.
# 
# * Create a model using `create_model()`
# * Setup a TensorBoard callback using `create_tensorboard_callback()`
# * Call the `fit()` function on our model passing it the training data, validation data, number of epochs to train for (`NUM_EPOCHS`) and the callbacks we'd like to use
# * Return the model

# %%
# Build a function to train and return a trained model
def train_model():
    """
    Trains a given model and returns the trained version.
    """
    # Create a model
    model = create_model()
    model.summary()
    # Create new TensorBoard session everytime we train a model
    tensorboard = create_tensorboard_callback()
    # Fit the model to the data passing it the callbacks we created
    model.fit(
        x=train_data,
        epochs=NUM_EPOCHS,
        validation_data=val_data,
        validation_freq=1,
        callbacks=[tensorboard, early_stopping]
    )
    return model

# %%
# Fit the model to the data
model = train_model()

# %% [markdown]
# **Question:** Looks like our model is overfitting because it is performing far better on the training set than the validation set. What are some ways to prevent model overfitting in deep learning neural networks?
# 
# **Note:** Overfitting to begin wi is a good thing! It means our model  is learning!!!

# %% [markdown]
# ### Checking the TensorBoard logs
# 
# The TensorBoard magic function (`%tensorboard`) will acess the logs directory we created earlier and visualize its contents.

# %%
#%tensorboard --logdir C:\Selbststudium\Udemy\Udemy_ZTM_Data_Science\dog-breed\Logs

# %% [markdown]
# > Try to find out how to use TensorBoard within MS Code!

# %% [markdown]
# ## Making and evaluating predictions using a trained model

# %%
val_data

# %%
# Make predictions on the validation data (not used to train on)
predictions = model.predict(val_data, verbose=1)
predictions

# %%
print(f"Shape of predictions: {predictions.shape}")
print(f"Amount of unique breeds: {len(unique_breeds)}")
print(f"Length of one single prediction: {len(predictions[0])}")
print(f"Sum of predictions should be one: {np.sum((predictions[0]))}")

# %%
# First prediction
index = 42
print(predictions[index])
print(f"Max value (probability): {np.max(predictions[index])}, Index: {np.argmax(predictions[index])}")
print(f"Breed: {unique_breeds[np.argmax(predictions[index])]}")

# %% [markdown]
# Having the above functionality is great, but we want to be able to do it at scale.
# 
# It would be even better if we could see the image the prediction was made on.
# 
# **Note:** PRediction probabilities are also know as confidence levels.

# %%
# Turn prediction probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]

# %%
# Get a predicted label based on an array of prediction probabilities
pred_label = get_pred_label(predictions[81])
pred_label

# %% [markdown]
# Since our validation data is still in a batch dataset, we'll have to unbatchify it to make predictions on the validation images and then compare those predictions to the validation labels (truth labels).

# %%
# First try without function
images_ = []
labels_ = []

# Loop through unbatched data
for image, label in val_data.unbatch().as_numpy_iterator():
    images_.append(image)
    labels_.append(label)

images_[0], labels_[0]

# %%
# Create a function to unbatch
def unbatchify(data):
    """
    Takes a batched dataset of (image, label) Tensors and returns separate arrays of images and labels.
    """
    images = []
    labels = []
    # Loop through unbatched data
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(unique_breeds[np.argmax(label)])
    return images, labels

# %%
val_images, val_labels = unbatchify(val_data)

# %% [markdown]
# Now we've got ways to get:
# * Prediction labels
# * Validation labels (truth labels)
# * Validation images
# 
# Let's make some functions to make these all a bit more visualized. 
# 
# We'll create a function which:
# * Takes an array of prediction probabilities, an array of truth labels and an array of images an integer ✅
# * Convert the prediction probabilities to a predicted label. ✅
# * Plot the predicted label, its predicted probability, the truth label and the target image on a single plot. ✅

# %%
def plot_pred(prediction_probabilities, labels, images, n=1):
    """
    View the prediction, ground truth and image for sample n
    """
    pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

    # Get the pred label
    pred_label = get_pred_label(pred_prob)

    # Plot image and remove ticks
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    # Change the color of the title depending on if the prediction is right or wrong
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"

    # Change plot title to be predicted, probability of prediction and truth label
    plt.title(f"{pred_label} - {np.max(pred_prob)*100:2.0f}% - {true_label}", color=color)

# %%
plot_pred(prediction_probabilities=predictions, labels=val_labels, images=val_images, n=77)

# %% [markdown]
# No we've got one function to visualize our models top predictions, let's make another to view our models top 10 predictions.
# 
# This function will:
# * Take an input of prediction probabilities array and a ground truth array and an integer ✅
# * Find the prediction using `get_pred_label()` ✅
# * Find the top 10: 
#   * Prediction probabilities indexes ✅
#   * Prediction probabalities values ✅
#   * Prediction labels ✅
# * Plt the top 10 prediction probability values and labels, coloring the true label green

# %%
def plot_pred_conf(prediction_probabilities, labels, n=1):
    """
    Plot the top 10 hiughest prediction confidences aling with the thruth label for sample n.
    """
    pred_prob, true_label = prediction_probabilities[n], labels[n]

    # Get the predicted label
    pred_label = get_pred_label(pred_prob)

    # Find the top 10 prediction confidence indexes
    top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
    # Find the top 10 prediction confidence values
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    # Find the top 10 prediction labels
    top_10_pred_labels = unique_breeds[top_10_pred_indexes]

    # Setup plot
    top_plot = plt.bar(
        np.arange(len(top_10_pred_labels)), 
        top_10_pred_values,
        color="salmon"
    )
    plt.xticks(
        np.arange(len(top_10_pred_labels)),
        labels=top_10_pred_labels,
        rotation="vertical"
    )

    # Change color of true label
    if np.isin(true_label, top_10_pred_labels):
        top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")

# %%
plot_pred_conf(
    prediction_probabilities=predictions,
    labels=val_labels,
    n=9
)

# %% [markdown]
# Now we've got some functions to help us visualize our predictions and evaluate our model, let's check out a few.

# %%
# Let's check out a few predictions and their different values
i_multiplier = 0
num_rows = 10
num_cols = 1
num_images = num_rows * num_cols
plt.figure(figsize=(5*2*num_cols, 5*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_pred(
        prediction_probabilities=predictions,
        labels=val_labels,
        images=val_images,
        n=i+i_multiplier
    )
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_pred_conf(
        prediction_probabilities=predictions,
        labels=val_labels,
        n=i+i_multiplier
    )
plt.tight_layout(h_pad=1.0)
plt.show()

# %% [markdown]
# **Challenge:** How would you create a confusion matrix with our models predictions and true labels?

# %% [markdown]
# ## Saving and reloading a trained model

# %%
from tensorflow.keras.models import load_model 
# Create a function to save a model
def save_model_(model, suffix=None):
    """
    Saves a given model in a model's directory and appends a suffix (string).
    """
    # Create a model directory pathname with current time
    modeldir = os.path.join(
        "C:/Selbststudium/Udemy/Udemy_ZTM_Data_Science/dog-breed/Models",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    model_path = modeldir + "-" + suffix + ".h5"
    print(f"Saving model to: {model_path}...")
    model.save(model_path)
    return model_path

# %%
# Create a function to load a trained model
def load_model_(model_path):
    """
    Loads a saved model from a specified path.
    """
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"KerasLayer":hub.KerasLayer}
    )
    return model

# %% [markdown]
# Now we've got functions to save and load a trained model, let's make sure they work!

# %%
# Save our model trained on 1000 images
# save_model_(model, suffix="1000-images-Adam")
# model_1000_images_model = load_model("C:/Selbststudium/Udemy/Udemy_ZTM_Data_Science/dog-breed/Models\20221110-053425-1000-images-Adam.h5")
# loaded_1000_images_model.evaluate(val_data)

# %% [markdown]
# ## Training a big dog model 🐶 (on the full data)

# %%
# Create a data batch with the full data set
full_data = create_data_batches(X, y)

# %%
full_data

# %%
# Create a model for full model
full_model = create_model()

# %%
# Create full model callbacks
full_model_tensorboard = create_tensorboard_callback()
# No validation set when trainig on all the data, so we can't monitor validation accuracy
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy",
    patience=3
)

# %%
# Fit the full model to the full data
full_model.fit(
    x=full_data,
    epochs=NUM_EPOCHS,
    callbacks=[full_model_tensorboard, full_model_early_stopping]
)

# %% [markdown]
# > I now should save and load the model, since it took so long to train it.

# %% [markdown]
# ## Making predictions on the test dataset
# 
# Since our model has been trained on images in the form of Tensor batches, to make predictions on test data, we'll have to get it into the same format.
# 
# Luckily we created `create_data_batches()` earlier, which can take a list of filenames as input and convert them into Tensor batches.
# 
# To predictions on the test data, we'll:
# * Get the test image filenames ✅
# * Convert the filenames into test data batches using `create_data_batches()` and setting the `test_data` parameter to `True` (since the test data doesn't have labels). ✅
# * Make a predictions array by passing the test batches to the `predict()` method called on our model. ✅

# %%
# Load test image filenames
test_path = "C:/Selbststudium/Udemy/Udemy_ZTM_Data_Science/dog-breed/test/"
test_filenames = [test_path + fname for fname in os.listdir(test_path)]
test_filenames[:10]

# %%
len(test_filenames)

# %%
# Create test data batch
test_data = create_data_batches(test_filenames, test_data=True)

# %%
test_data

# %%
# Make preds on test data batch using full model
test_predictions = model.predict(
    test_data,
    verbose=1
)

# %%
# Save preds (NumPy array) to csv file (for access later)
np.savetxt("C:/Selbststudium/Udemy/Udemy_ZTM_Data_Science/dog-breed/preds.csv", test_predictions, delimiter=",")

# %%
# Load predictions from csv file
test_predictions = np.loadtxt("C:/Selbststudium/Udemy/Udemy_ZTM_Data_Science/dog-breed/preds.csv", delimiter=",")

# %%
test_predictions[:10], test_predictions.shape

# %% [markdown]
# ## Preparing test dataset predictions for Kaggle
# Looking at the Kaggle sample submissio, we find that it wants our models prediction probability outputs in a DataFrame with an ID and a column for each different dof breed.
# 
# To get the data in this format, we'll:
# * Create a pandas DataFrame with an ID column as well as a column for each dog breed
# * Add data to the ID column by extractin the test image ID's from theri filepaths
# * Add data (the prediction probabilities) to each of the dog breed columns
# * Export the DataFrame as a CSV to submit it to Kaggle.

# %%
# Create a pandas DataFrame with empty columns
preds_df = pd.DataFrame(
    columns=["id"] + list(unique_breeds)
)
preds_df.head()

# %%
# Append test iamge ID's to predictions DataFrame
test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df["id"] = test_ids
preds_df.head()

# %%
# Add the prediction probabilities to each dof breed column
preds_df[list(unique_breeds)] = test_predictions
preds_df.head()

# %%
preds_df.to_csv(
    "C:/Selbststudium/Udemy/Udemy_ZTM_Data_Science/dog-breed/full_model_preds_submission_1_mobilenetV2.csv",
    index=False
)

# %% [markdown]
# ## Making predictions on custom images
# To make predictions on custom images, we'll:
# * Get filepaths of our own images.
# * Turn the filepaths into data bates using `create_data_batches()`. And since our custom images won't have labels, we set the `test_data` parameter to `True`.
# * Pass the custom image data bratch to our model's `predict()` method.
# * Convert the prediction output probabilities to prediction labels.
# * Compare the predicted labels to the custom images.

# %%
# Get custom image filepaths
custom_path = "C:/Selbststudium/Udemy/Udemy_ZTM_Data_Science/dog-breed/Own_images/"
custom_image_paths = [custom_path + fname for fname in os.listdir(custom_path)]
custom_image_paths

# %%
# Turn custom images into batch datasets
custom_data = create_data_batches(custom_image_paths, test_data=True)
custom_data

# %%
# Make predictions on the custom data
custom_preds = model.predict(custom_data)

# %%
custom_preds, custom_preds.shape

# %%
# Get custom image prediction labels
custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
custom_pred_labels

# %%
# Get the custom images (unbatchify function work since there aren't labes... maybe we could fix this later)
custom_images = []
# Loop through unbatched data
for image in custom_data.unbatch().as_numpy_iterator():
    custom_images.append(image)

# %%
# Check custom image predictions
plt.figure(figsize=(10, 10))
for i, image in enumerate(custom_images):
    plt.subplot(2, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title(custom_pred_labels[i])
    plt.imshow(image)

# %%



