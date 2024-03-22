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
from sklearn.cluster import KMeans

# Main function
# Questions we want to answer: What features are the most effective at predicting whether a song is popular on Spotify?
# - We have seen that genres of songs have changed in popularity over time, but we may be able to pair genre with other
#   musical aspects to find a better predictor for popularity.
# - What features are the best at predicting popularity?
# This can be tested with giving machine learning models datasets of different features, then see which subset of features is the best for predicting this trait.
# However, we also have a prompt offered around predicting what songs people would be interested based on their danceability and valence scores:
# - What do we need to know what songs would fit a person's danceability and valence scores? What does this have to do with popularity scores?
# - A person isn't necessarily going to like a song based on what everyone else likes in a song, people's perception's
#   of a songs danceability can be variable for one particular song. This can't really be shown with our project, since
#   we have the average popularity scores for each popular song, but I believe it would confound our project's ability
#   to return songs that a person would actually like.


if __name__ == '__main__':
    data = pd.read_csv('data/song_data.csv').dropna()
    # print(data.describe())
    print("Some of the data is unnamed, has no popularity, tempo, or time signature. This data is likely null.")

    # Between 1-3 different entries don't have an artist, album name, or track_name. Are these columns what we want to
    # use to evaluate the relationship between different characteristics in songs, Popularity, and Dancability?
    # Another thing to note is that some tracks are entered into the dataset more than once. These songs will be
    # overrepresented in the dataset, so we can just take unique values by track ID.
    # print(data.info())

    # Removing all columns with non-unique values
    unique_data = data.drop_duplicates('track_id')

    # TODO: Remove the column "Unnamed: 0" which acts as the index. We will have a unique key with the track_id section, and
    print(unique_data.corr())
    unique_data.drop(unique_data.columns[0])
    print(unique_data.columns)

    # What clusters are returned from clustering the data?
    clusters = KMeans(n_clusters=5, max_iter=1000, random_state=1).fit(unique_data[['danceability', 'explicit', 'instrumentalness']])
    print(clusters.labels_)

    unique_data["possible_groups"] = clusters.labels_
    print(unique_data.corr())

    if __name__ == '__main__':
        # Load data and drop rows with empty values

        # print(data.describe())

        # Load data and drop unnecessary columns and rows with null values
        data = pd.read_csv('data/song_data.csv').drop(columns=['track_id', 'key']).dropna()

        # Remove duplicates based on track name
        data = data.drop_duplicates(subset=['track_name'])

        # Filter data with "popularity" greater than 75
        '''we consider a song is popular if it's popularity determined by the number of time it was played is greater than 75
        * also if we want we can increase the number of popularity for better accuracy'''

        data = data[data['popularity'] > 75]

        # Calculate average of specified columns
        '''using that data we find the average the durations_ms, danceability, 
        energy, loudness and any other factors that can help us make a model to predict the popularity of a song'''

        average_data = data[['duration_ms', 'danceability', 'energy', 'loudness',
                             'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                             'time_signature']].mean()

        '''using the average we set the upper limit/threshold to predict the popularity of a song
        * once our model is created we can see what features make the song popular 
        * we would most likely need create some sort of classification to check the features '''

        print(average_data)


