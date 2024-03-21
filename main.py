# Final Project Main Function: Finding Greatest Predictors for Dancability and Valence of the Most Popular Songs on Spotify
# Creators: Michael Yinka'Oke, Sam Clear, Diwas Dahal
# Start Date: March 21st, 2024

# This file currently acts as the main dataset inspecting file of the program, but will in the future just be
# used for running the main bulk of objects in the program.

# Essential Packages for inspecting the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# Main function
# Questions we want to answer: What features are the most effective at predicting whether a song is popular on Spotify?
# - We have seen that genres of songs have changed in popularity over time, but we may be able to pair genre with other
#   musical aspects to find a better predictor for popularity.
# - What features are the best at predicting popularity?
# This can be tested with giving machine learning models datasets of different features, then see which subset of features is the best for predicting this trait.
# However, we also have a prompt offered around predicting what songs people would be interested based on their danceability and valence scores:
# - What do we need to know what songs would fit a person's danceability and valence scores? What does this have to do with popularity scores?
# - A person isn't necessarily going to like a song based on what everyone else likes in a song, people's perception's
#   of a songs dancability can be variable for one particular song. This can't really be shown with our project, since
#   we have the average popularity scores for each popular song, but I believe it would confound our project's ability
#   to return songs that a person would actually like.


if __name__ == '__main__':
    data = pd.read_csv('data/song_data.csv').dropna()
    print(data.describe())
    print("Some of the data is unnamed, has no popularity, tempo, or time signature. This data is likely null.")

    # Between 1-3 different entries don't have an artist, album name, or track_name. Are these columns what we want to use to evaluate the relationship between different characteristics in songs, Popularity, and Dancability?
    # Another thing to note is that some tracks are entered into the dataset more than once. These songs will be
    # overrepresented in the dataset, so we can just take unique values by track ID.
    print(data.info())

    # unique_data = data[data['track_id'].unique()]
    # unique_data.info()

    # Plotting Danceability by Popularity
    # plt.hist(data=data)
    # plt.show()

    print(data.corr()['popularity'])

    # Split data into training and testing sets (adjust test_size and random_state as needed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Tokenization using CountVectorizer (you can also use TF-IDF vectorizer)
    X_train_vectorized, X_test_vectorized = create_bag_of_words(X_train, X_test)

    # #Try SMOTE to fix class imbalance -- *** tried, made precision, recall, accuracy worse
    # X_train_vectorized, y_train = use_smote(X_train_vectorized, y_train)

    # Train RF classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_classifier.fit(X_train_vectorized, y_train)

    # Predict the labels for test data
    y_pred = rf_classifier.predict(X_test_vectorized)

    # Evaluate this classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')
    print("Confusion Matrix:")
    print(conf_matrix)

    # for row in range(len(data)):
    #     if data['artists'][row].isnull():
    #         pass
    # pass
