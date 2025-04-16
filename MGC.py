from google.colab import drive
drive.mount('/content/drive')

import librosa.display
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/content/drive/MyDrive/datasets/Data/features_3_sec.csv')
df.shape
df['tags'] = df['filename'].str.split('.').str[1]
df['tags1'] = df['filename'].str.split('.').str[0]
df['tags'].value_counts()

df.loc[(df['tags'] == '00002') & (df['tags1'] == 'blues')]

# Lets drop some unecessary features.
df.drop(['filename','tags'],axis =1 ,inplace = True)
df.head()

## Let's visualize the dataset and understand our data better.

# There are 8 main genres with 1000 sub-classes per
# genre, such as Electronic, Experimental, Folk, Hip-hop, Instrumental, International,
# Pop, Rock. And most of them are with sampling rate of 44,100 Hz, bit rate 320 kb/s,
# and in stereo. The GTZAN dataset consists of 1000 audio tracks of 30 seconds long,
# which contains 10 genres with 100 tracks per genre. All tracks are 22,050Hz, Mono
# 16-bit audio files in .wav format (Sturm, 2013).

# It returns
# y : np.ndarray [shape=(n,) or (..., n)]
#    audio time series. Multi-channel is supported.
#sr : number > 0 [scalar]
#   sampling rate of ``y``

rec_file = '/content/drive/MyDrive/datasets/Data/genres_original/metal/metal.00002.wav'
data,sr = librosa.load(rec_file)

librosa.load(rec_file,sr=44100) # Opening in sample rate of 44100

import matplotlib.pyplot as plt
plt.figure(figsize = (12,4))
librosa.display.waveshow(data)

# Lets Create A spectogram

# By converting to decibels (dB) the scale becomes logarithmic.
#This limits the numerical range, to something like 0-120 dB instead.
#The intensity of colors when this is plotted corresponds more closely to what we hear than if one used a linear scale.
n_fft = 2048
hop_length = 512
stft = librosa.stft(data,n_fft=2048,hop_length = 512)
stft_db = librosa.amplitude_to_db(abs(stft))
librosa.display.specshow(data = stft_db,sr = 22050, x_axis = 'time', y_axis = 'hz')
plt.colorbar(format="%+2.f dB")

#The vertical axis represents frequencies (from 0 to 10kHz), and the horizontal axis represents the time of the clip.

# # Rolloff - fequency
# Its a frequency below which a specified percentage of the total spectral lies / 85%


from sklearn.preprocessing import normalize

spectral_rolloff = librosa.feature.spectral_rolloff(y=data + 0.01, sr=sr)[0]

plt.figure(figsize=(14, 6))
librosa.display.waveshow(data, sr=sr, alpha=0.4, color="#2B4F72")


# # Chroma Feature.
# It is a powerful tool for analyzing music features whose pitches can be meaningfully categorized and whose tuning approximates to the equal-tempered scale. One main property of chroma features is that they capture harmonic and melodic characteristics of music while being robust to changes in timbre and instrumentation.

plt.figure(figsize=(12, 4))

# Pass 'y' explicitly instead of using positional argument
chroma = librosa.feature.chroma_stft(y=data, sr=45600)

# Ensure 'sr' is correctly passed in specshow()
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=45600)

plt.colorbar()
plt.title("Chroma Features")
plt.show()


chroma.mean()

# MFCC Extraction

plt.figure(figsize=(12, 4))

# Use 'y=data' instead of passing 'data' as a positional argument
mfcc = librosa.feature.mfcc(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

# Ensure correct parameter usage in specshow()
librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, x_axis="time")

plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.title("MFCC Features")
plt.show()


import os

# Use raw string or forward slashes for correct path handling
path = r"/content/drive/MyDrive/datasets/Data/genres_original"

# Verify if path exists
if not os.path.exists(path):
    print(f"Error: Path '{path}' does not exist!")
else:
    filenames_all = []
    labels = []

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path)):
        if dirpath != path:
            filenames_all.append(filenames)
            labels.append(len(filenames_all[i-1]) * [i-1])

    print(f"Total directories found: {len(filenames_all)}")

    # Ensure there's data before accessing indices
    if filenames_all and filenames_all[0]:
        i = 0
        j = 0
        print(filenames_all[i][j])
    else:
        print("Error: No files found in any subdirectory!")

i = 0
j = 0
filenames_all[i][j]


import math
import librosa
import json
import os

SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath != dataset_path:
            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = os.path.basename(dirpath)  # Use os.path.basename for cross-platform compatibility
            data["mapping"].append(semantic_label)

            # process all audio files in genre sub-dir
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # Load audio file
                try:
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue  # Skip corrupted files

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc (Fix: use y=signal)
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate,
                                                n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T  # Transpose to align features properly

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)  # Assign label

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


save_mfcc(dataset_path='/content/drive/MyDrive/datasets/Data/genres_original', json_path='data.json')


import json
with open('data.json') as fp:
        data = json.load(fp)
        X = np.array(data['mfcc'])
        y = np.array(data['labels'])
        
        
        
set(y) # These are unique values
#0-> Blues
#1-> classical
#2-> country
#3-> disco
#4-> hiphop
#5-> jazz
#6-> metal
#7-> pop
#8-> reggae
#9-> rock



# splitting the data into Train and Test sets.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

# Building the CNN
from tensorflow import keras
model = keras.Sequential(
    [   # Input
        keras.layers.Flatten(input_shape = (X.shape[1], X.shape[2])),
        # Hidden
        keras.layers.Dense(512, activation = "relu", kernel_regularizer = keras.regularizers.l2(0.001)),
        # Dropout to reduce overfitting.
        keras.layers.Dropout(0.3),
        # Hidden
        keras.layers.Dense(256, activation = "relu", kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # Hidden
        keras.layers.Dense(64, activation = "relu", kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # Ouput
        keras.layers.Dense(10,activation = "softmax")
    ]
)

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer = optimizer,
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"]
              )
model.fit(X_train,y_train,
          validation_data = (X_test,y_test),
          epochs = 50,
          batch_size = 32)


# **ANN** was not giving a good accuracy so let's see if we can use **CNN** for our project. For a CNN we will use mfcc spectogram 2-d array as it will be required for cnn to train as cnn will be able to extract features better  we can extract features which look like images and shape them in a way in order to feed them into a CNN.

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from tensorflow import keras
# ---------------------------
# 1. Load and Preprocess the Dataset
# ---------------------------

# Load MFCCs and labels from a JSON file
with open('data.json', 'r') as fp:
    data = json.load(fp)



# 'mfcc' should be a list (or nested lists) representing the MFCC features
# and 'labels' should be the corresponding integer labels.
X = np.array(data['mfcc'])  # Expected shape: (num_samples, time_steps, n_mfcc)
y = np.array(data['labels'])

# For scaling purposes we need to reshape the 3D input to 2D.
num_samples, time_steps, n_mfcc = X.shape
X_reshaped = X.reshape(num_samples, -1)  # Now shape is (num_samples, time_steps * n_mfcc)

# Create a DataFrame from the flattened inputs
df_inputs = pd.DataFrame(X_reshaped)

# Create and fit a MaxAbsScaler to scale each feature between -1 and 1
abs_scaler = MaxAbsScaler()
abs_scaler.fit(df_inputs)

# Transform the data using the fitted scaler
scaled_data = abs_scaler.transform(df_inputs)

# (Optional) Store the scaled data in a DataFrame and visualize the first few rows
df_scaled_input = pd.DataFrame(scaled_data, columns=df_inputs.columns)
print("Scaled data preview:\n", df_scaled_input.head())

# Reshape the scaled data back to its original 3D shape
X_scaled = df_scaled_input.values.reshape(num_samples, time_steps, n_mfcc)

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)


# ---------------------------
# 2. Build the Neural Network Model
# ---------------------------
# Define an L2 regularization factor (adjust as needed)
l2_reg = keras.regularizers.l2(1e-4)

# Build the model using a Flatten layer (to convert 2D MFCC input into 1D)
# followed by Dense layers with ReLU activations and Dropout layers to help prevent overfitting.
model = keras.Sequential([
    # Input layer: Flatten the (time_steps, n_mfcc) MFCC input
    keras.layers.Flatten(input_shape=(time_steps, n_mfcc)),

    # Hidden Layer 1: 512 neurons, ReLU, L2 regularization, and Dropout
    keras.layers.Dense(512, activation="relu", kernel_regularizer=l2_reg),
    keras.layers.Dropout(rate=0.3),

    # Hidden Layer 2: 256 neurons
    keras.layers.Dense(256, activation="relu", kernel_regularizer=l2_reg),
    keras.layers.Dropout(rate=0.3),

    # Hidden Layer 3: 64 neurons
    keras.layers.Dense(64, activation="relu", kernel_regularizer=l2_reg),
    keras.layers.Dropout(rate=0.3),

    # Output Layer: 10 neurons (assuming 10 classes) with softmax activation
    keras.layers.Dense(10, activation="softmax")
])




# Compile the model
# - Adam optimizer is chosen for its efficiency.
# - clipnorm=1 prevents exploding gradients.
# - Sparse categorical crossentropy is used since our targets are integer labels.
optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1)
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Print the model summary
model.summary()



# ---------------------------
# 3. Train the Model with Early Stopping
# ---------------------------
# Early stopping will monitor the validation loss and stop training
# if no improvement is seen after 7 epochs (patience), restoring the best weights.
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=128,  # Mini-batch training is a good compromise between speed and accuracy.
    callbacks=[early_stop]
)

# ---------------------------
# 4. Visualize Training History
# ---------------------------
def plot_history(history):
    fig, axis = plt.subplots(2, 1, figsize=(10, 8))

    # Accuracy subplot
    axis[0].plot(history.history["accuracy"], label="Train Accuracy")
    axis[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axis[0].set_ylabel("Accuracy")
    axis[0].set_title("Accuracy Evaluation")
    axis[0].legend(loc="lower right")

    # Loss subplot
    axis[1].plot(history.history["loss"], label="Train Loss")
    axis[1].plot(history.history["val_loss"], label="Validation Loss")
    axis[1].set_ylabel("Loss")
    axis[1].set_xlabel("Epochs")
    axis[1].set_title("Loss Evaluation")
    axis[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

plot_history(history)


# ---------------------------
# 5. Evaluate the Model
# ---------------------------

# Evaluate the model on the test set and retrieve the loss and accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Optionally, generate predictions and further analyze performance:
# predictions = model.predict(X_test)
# predicted_labels = np.argmax(predictions, axis=1)
# You could then use sklearn.metrics (e.g., confusion_matrix, classification_report) for more insights.


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Generate predictions
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Detailed classification report
print(classification_report(y_test, predicted_labels))


# # What else we could have done ?
# 1. Right now we are only using mfcc we can also use chroma values
# 2. We can also use other params like rollof frequency for our dataset creation but and with that our data would become big and would give our model more information

#Lets now go ahead and save our model and create a function to extract mfcc features from new audio file.

# 2. Save the entire model to Google Drive
model.save('/content/drive/MyDrive/music-gen-classify-v1.h5')

# 3. Later, load the model from Google Drive
from tensorflow import keras
loaded_model = keras.models.load_model('/content/drive/MyDrive/music-gen-classify-v1.h5')

# Optionally, evaluate the loaded model
test_loss, test_accuracy = loaded_model.evaluate(X_test, y_test, verbose=0)
print(f"Loaded Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

from tensorflow import keras
new_model = keras.models.load_model('/content/drive/MyDrive/music-gen-classify-v1.h5')

# Suppose you have extracted MFCCs for your new audio and got an array of shape (32, 13)
# For demonstration, we'll simulate an MFCC array:
mfcc_sample = np.random.rand(32, 13)  # Replace this with your actual MFCC extraction
def pad_truncate_mfcc(mfcc, target_length=259):
    current_length = mfcc.shape[0]
    if current_length > target_length:
        return mfcc[:target_length, :]
    elif current_length < target_length:
        pad_width = target_length - current_length
        mfcc_padded = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
        return mfcc_padded
    else:
        return mfcc

# Adjust the MFCC sample to match the expected shape (259, 13)
mfcc_fixed = pad_truncate_mfcc(mfcc_sample, target_length=259)


# Add a batch dimension: final shape should be (1, 259, 13)
X_predict = np.expand_dims(mfcc_fixed, axis=0)

# Now make predictions
pred = new_model.predict(X_predict)
predicted_class = np.argmax(pred, axis=1)
print("Predicted class:", predicted_class)