print("The University of Regina")
print("Faculty of Engineering and Applied Science")
print("Applied Machine Learning")
print("Final Project: K-NN Algorithm for Image Classification")
print("Student: Behnam Moradi")
print("SID: 200433555")

# Importing libraries and modules
import os
import cv2 # OpenCV
import numpy as np

import argparse
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Importing the dataset
print("Importing Dataset...")
def import_images_data(samples_directory):

    image_pixels = []
    image_labels = []

    for i, image_path in enumerate(samples_directory):
        sample_image = cv2.imread(image_path)
        sample_label = image_path.split(os.path.sep)[-2]

        sample_image = cv2.resize(sample_image, (32, 32), interpolation=cv2.INTER_AREA)

        image_pixels.append(sample_image)
        image_labels.append(sample_label)

    return (np.array(image_pixels), np.array(image_labels))

# Reading images from Image folder located inside the knn.py file
path_to_sample_images = 'Images'
# Directories for every single image
samples_directory = list(paths.list_images(path_to_sample_images))

# Seperating and saving image pixels and image labels
(image_pixels, image_labels) = import_images_data(samples_directory)

# Reshaping image pixels
image_pixels = image_pixels.reshape((image_pixels.shape[0], 3072))

# Every image label will be incoded to a single digit: Cats: 0, Dogs: 1, Panda: 2
encod_image_labels = LabelEncoder()
image_labels = encod_image_labels.fit_transform(image_labels)

# Number of neighbors to be manipulated by the programmer 
number_of_neighbors = 10

# Spliting dataset into 75% training data and 25% test data. It can be changed.
print("Splitting Data...")
(trainig_image_pixels, test_image_pixels, training_image_labels, test_image_labels) = train_test_split(image_pixels, image_labels, test_size=0.25, random_state=42)


# Total Metrics
total_precision_array = []
total_recal_array = []
total_f1_score_array = []

# Precision Metric
cats_precision_array = []
dogs_precision_array = []
pandas_precision_array = []

# Recall Metric
cats_recall_array = []
dogs_recall_array = []
pandas_recall_array = []

# F1-Score Metric
cats_f1_score_array = []
dogs_f1_score_array = []
pandas_f1_score_array = []

# Classification will strt from here
print("All good!")
print("Starting Classification...")
for i in range(1,31):
    # Creating KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=i)

    # Model fitting
    knn_classifier.fit(trainig_image_pixels, training_image_labels)

    # Making predictions
    prediction = knn_classifier.predict(test_image_pixels)

    # Saving correct predictions
    correct_predictions = np.sum(prediction == test_image_labels)

    # FYI: This is how percision in being done. I will use the library function to do this step
    precision = float(correct_predictions)/float(test_image_labels.size)

    # Outputing the report
    classifier_report = classification_report(test_image_labels, prediction, target_names=encod_image_labels.classes_, output_dict=True)

    # Saving results for precision
    cats_precision_array.append(classifier_report["cats"]["precision"])
    dogs_precision_array.append(classifier_report["dogs"]["precision"])
    pandas_precision_array.append(classifier_report["panda"]["precision"])

    # Saving results for recall
    cats_recall_array.append(classifier_report["cats"]["recall"])
    dogs_recall_array.append(classifier_report["dogs"]["recall"])
    pandas_recall_array.append(classifier_report["panda"]["recall"])

    # Saving results for f1-score
    cats_f1_score_array.append(classifier_report["cats"]["f1-score"])
    dogs_f1_score_array.append(classifier_report["dogs"]["f1-score"])
    pandas_f1_score_array.append(classifier_report["panda"]["f1-score"])


# FYI: Plotting a sample metric

# Data visualization - precision
# plt.plot(range(1, len(cats_precision_array)+1) ,cats_precision_array, c="green", label="cats")
# plt.plot(range(1, len(dogs_precision_array)+1) ,dogs_precision_array, c="blue", label="dogs")
# plt.plot(range(1, len(pandas_precision_array)+1) ,pandas_precision_array, c="red", label="panda")
# plt.xlabel("Number of Neighbors")
# plt.ylabel("Percision")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()

# # Data visualization - recall
# plt.plot(range(1, len(cats_precision_array)+1) ,cats_precision_array, c="green", label="cats")
# plt.plot(range(1, len(dogs_recall_array)+1) ,dogs_recall_array, c="blue", label="dogs")
# plt.plot(range(1, len(pandas_recall_array)+1) ,pandas_recall_array, c="red", label="panda")
# plt.xlabel("Number of Neighbors")
# plt.ylabel("Recall")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()

# Data visualization - f1_score
plt.plot(range(1, len(cats_f1_score_array)+1) ,cats_f1_score_array, c="green", label="cats")
plt.plot(range(1, len(dogs_f1_score_array)+1) ,dogs_f1_score_array, c="blue", label="dogs")
plt.plot(range(1, len(pandas_f1_score_array)+1) ,pandas_f1_score_array, c="red", label="panda")
plt.xlabel("Number of Neighbors")
plt.ylabel("F1_score")
plt.legend(loc="upper left")
plt.grid()
plt.show()