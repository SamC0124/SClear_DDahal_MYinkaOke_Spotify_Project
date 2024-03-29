# Final Project Main Function: Finding Greatest Predictors for Dancability and Valence of the Most Popular Songs on Spotify
# Creators: Michael Yinka'Oke, Sam Clear, Diwas Dahal
# Start Date: March 21st, 2024

# This file currently acts as the main dataset inspecting file of the program, but will in the future just be
# used for running the main bulk of objects in the program.

# Essential Packages for inspecting the data
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import string
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn import tree

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

    ## Data Cleaning
    music_data = pd.read_csv('data/song_data.csv', index_col=0).dropna()

    # Try removing all popularity values that are equal to zero
    music_data = music_data[music_data['popularity'] > 0]

    cont_pop_data = music_data.drop_duplicates(subset=['track_name'])
    cont_pop_data['duration_s'] = cont_pop_data['duration_ms'] / 1000
    cont_pop_data = cont_pop_data.drop(columns=['track_id', 'key']).dropna()

    # Load data and drop unnecessary columns and rows with null values
    unique_data = music_data.drop(columns=['track_id', 'key']).dropna()

    '''using the average we can make model like linear regression, decision tree, or forest'''
    cleared_data = unique_data
    cleared_data['popularity'] = cleared_data['popularity'].apply(lambda x: 'popular' if x > 75 else 'not_popular')
    print(len(cleared_data[cleared_data['popularity'] == 'popular']),
          len(cleared_data[cleared_data['popularity'] == 'not_popular']))

    # Convert True to 1 and False to 0 in the "popularity" column
    cleared_data['explicit'] = cleared_data['explicit'].astype(int)
    # Remove duplicates based on track name
    cleared_data = cleared_data.drop_duplicates(subset=['track_name'])
    # Convert milliseconds to seconds
    cleared_data['duration_s'] = cleared_data['duration_ms'] / 1000

    print(music_data.describe())
    print("Some of the data is unnamed, has no popularity, tempo, or time signature. This data is likely null.")

    # Between 1-3 different entries don't have an artist, album name, or track_name. Are these columns what we want to
    # use to evaluate the relationship between different characteristics in songs, Popularity, and Dancability?
    # Another thing to note is that some tracks are entered into the dataset more than once. These songs will be
    # overrepresented in the dataset, so we can just take unique values by track ID.
    print(music_data.info())

    ## Prior Graphing, Gathering Insights
    # Computing correlations of data with highest popularity value
    '''we consider a song is popular if it's popularity determined by the number of time it was played is greater than 75
    * also if we want we can increase the number of popularity for better accuracy'''
    pop_data = music_data[music_data['popularity'] > 75]
    plt.imshow(pop_data.corr(), cmap="PuBu")
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
               ['artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 'explicit', 'danceability',
                'energy',
                'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                'time_signature', 'track_genre'], rotation=45)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
               ['artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 'explicit', 'danceability',
                'energy',
                'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                'time_signature', 'track_genre'])
    plt.title("Correlations between Characteristics of Songs with High Popularities")
    plt.tight_layout()
    plt.show()

    # List of columns you want to keep
    columns_to_keep = ['popularity', 'duration_s', 'explicit', 'danceability', 'energy', 'loudness', 'speechiness',
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    columns_to_get_mean = ['duration_s', 'explicit', 'danceability', 'energy', 'loudness', 'speechiness',
                           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

    ## SVM with Soft Margin (Allow for missclassificationat a low cost, essential for our imperfect dataset)
    ## With the current hour of this work, this program has been assisted by ChatGPT, plotting will be done on our own.
    # Generate noisy data
    X = cleared_data[columns_to_get_mean]
    y = cleared_data['popularity']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create SVM classifier with soft-margin (C=1)
    svm_classifier = SVC(kernel='poly', degree=3, C=1)

    # Train the classifier
    svm_classifier.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = svm_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    ## Decision Tree Modeling
    # First Decision Tree Model - All features
    # This Decision Tree Classifier Program was adapted from the following site: https://scikit-learn.org/stable/modules/tree.html
    print(cleared_data.columns)
    X = cleared_data[columns_to_get_mean]
    Y = cleared_data['popularity']
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    print(tree.plot_tree(clf))

    import graphviz

    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("spotify_pop_data1_no_lim")
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=columns_to_get_mean,
                                    class_names=['popular', 'not_popular'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)

    ## K-Means Clustering: Not Fit for Model
    pop_clusters = KMeans(n_clusters=5, max_iter=500, random_state=42).fit(cont_pop_data[['loudness', 'danceability']])
    cont_pop_data["popularity_groups"] = pop_clusters.labels_

    plt.scatter(data=unique_data, x="popularity", y="danceability", c="popularity_groups")
    plt.show()

    first_group = cont_pop_data[cont_pop_data["popularity_groups"] == 0]
    second_group = cont_pop_data[cont_pop_data["popularity_groups"] == 1]
    third_group = cont_pop_data[cont_pop_data["popularity_groups"] == 2]
    fourth_group = cont_pop_data[cont_pop_data["popularity_groups"] == 3]
    fifth_group = cont_pop_data[cont_pop_data["popularity_groups"] == 4]

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5)
    ax1.hist(x=first_group['popularity'], bins=8, density=True, histtype='bar')
    ax2.hist(x=second_group['popularity'], bins=8, density=True, histtype='bar')
    ax3.hist(x=third_group['popularity'], bins=8, density=True, histtype='bar')
    ax4.hist(x=fourth_group['popularity'], bins=8, density=True, histtype='bar')
    ax5.hist(x=fifth_group['popularity'], bins=8, density=True, histtype='bar')
    fig.tight_layout()
    plt.show()

    # Calculate average of specified columns
    '''using that data we find the average the durations_ms, danceability, 
    energy, loudness and any other factors that can help us make a model to predict the popularity of a song'''

    average_data = pop_data[['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
         'valence', 'tempo', 'time_signature']].mean()

    '''using the average we set the upper limit/threshold to predict the popularity of a song
    * once our model is created we can see what features make the song popular 
    * we would most likely need create some sort of classification to check the features '''

    plt.scatter(data=pop_data, x='tempo', y='popularity')
    plt.show()
    print(average_data)

    ## K-Nearest-Neighbors: Fit adequately for model
    # Filtering columns
    cleared_data = cleared_data[columns_to_keep]
    # Reshape the data to long format
    long_data = pd.melt(cleared_data, id_vars=['popularity'], var_name='feature', value_name='value')
    # Calculate the mean for each feature and diagnosis
    means = long_data.groupby(['feature', 'popularity'])['value'].mean().reset_index()
    # Reshape the data back to wide format
    wide_means = means.pivot(index='feature', columns='popularity', values='value')
    # Print the result
    print(wide_means)
    # Create a box plot
    sns.set(style="whitegrid")
    g = sns.FacetGrid(long_data, col='feature', col_wrap=2, margin_titles=True,
                      xlim=(long_data['value'].min(), long_data['value'].max()))
    g.map(sns.boxplot, 'value', 'popularity', 'popularity', order=['popular', 'not_popular'],
          hue_order=['popular', 'not_popular'], palette={"popular": "tomato", "not_popular": "cyan"})
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")

    # Remove the legend
    plt.legend().remove()
    plt.xlim(left=0, right=1)
    plt.show()

    # Select numeric columns
    numeric_cols = cleared_data.select_dtypes(include='number')
    # Min-max normalization
    pop_norm = (numeric_cols - numeric_cols.min()) / (numeric_cols.max() - numeric_cols.min())
    # Standardization
    pop_stan = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()
    # Create knn_results DataFrame
    knn_results = pd.DataFrame({'k': range(1, 11), 'pop_norm': [-1] * 10, 'pop_stan': [-1] * 10})
    # Displaying the knn_results DataFrame
    print(knn_results)
    # Convert 'pop_norm' and 'pop_stan' columns to float64
    knn_results['pop_norm'] = knn_results['pop_norm'].astype(float)
    knn_results['pop_stan'] = knn_results['pop_stan'].astype(float)
    # Fit KNN Algorithm for normalized data
    for i in range(len(knn_results)):
        knn = KNeighborsClassifier(n_neighbors=knn_results.loc[i, 'k'])
        loop_knn = cross_val_predict(knn, pop_norm, cleared_data['popularity'], cv=5)
        loop_norm_cm = confusion_matrix(loop_knn, cleared_data['popularity'])
        accuracy = round(accuracy_score(loop_knn, cleared_data['popularity']), 2)
        print(f"Accuracy for k={knn_results.loc[i, 'k']} with normalized data: {accuracy}")

        # Debugging print
        knn_results.loc[i, 'pop_norm'] = accuracy
        # Fit KNN Algorithm for standardized data
        knn = KNeighborsClassifier(n_neighbors=knn_results.loc[i, 'k'])
        loop_knn2 = cross_val_predict(knn, pop_stan, cleared_data['popularity'], cv=5)
        accuracy2 = round(accuracy_score(loop_knn2, cleared_data['popularity']), 2)
        print(f"Accuracy for k={knn_results.loc[i, 'k']} with standardized data: {accuracy2}")

        # Debugging print
        knn_results.loc[i, 'pop_stan'] = accuracy2
        # Displaying the first 10 rows of knn_results DataFrame
        # print(knn_results.head(10))
        # long_knn_results = knn_results.melt(id_vars='k', var_name='rescale_method', value_name='accuracy')

        # # # Create the plot
        # plt.figure(figsize=(10, 6))
        # sns.lineplot(data=long_knn_results, x='k', y='accuracy', hue='rescale_method')

        # # # Set labels and title
        # plt.xlabel('Choice of K') # plt.ylabel('Accuracy')
        # plt.title('KNN Algorithm Performance')
        # plt.legend(title='Rescale Method', labels=['Normalized', 'Standardized'])

        # # # Set scale for x-axis and y-axis # plt.xticks(range(1, 6))
        # plt.yticks(np.arange(0.95, 1.0, 0.01), labels=[f'{i:.2f}%' for i in np.arange(0.95, 1.0, 0.01)])

        # # # Adjust legend position
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # # # Show plot
        # plt.grid(True)
        # plt.show()
        long_knn_results = knn_results.melt(id_vars='k', var_name='rescale_method', value_name='accuracy')
        # Select the rows with the maximum accuracy for each rescale method
        max_accuracy_rows = long_knn_results.loc[long_knn_results.groupby('rescale_method')['accuracy'].idxmax()]

        # Displaying the knn_results DataFrame
        print(knn_results)

    long_knn_results = knn_results.melt(id_vars='k', var_name='rescale_method', value_name='accuracy')

    # Select the rows with the maximum accuracy for each rescale method
    max_accuracy_rows = long_knn_results.loc[long_knn_results.groupby('rescale_method')['accuracy'].idxmax()]
    print(max_accuracy_rows)
    # Ensure 'popularity' is included in pop_norm
    pop_norm['popularity'] = cleared_data['popularity']
    # Splitting data into features (X) and target (y)
    X = pop_norm.drop(columns=['popularity'])
    # Exclude 'popularity' from features
    y = pop_norm['popularity']
    # Use 'popularity' as the target column
    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initializing KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=2)
    # Fitting the classifier on the training data
    knn_classifier.fit(X_train, y_train)
    # Predicting on the test data
    y_pred = knn_classifier.predict(X_test)
    # Calculating confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # Displaying confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_classifier.classes_)
    disp.plot()

