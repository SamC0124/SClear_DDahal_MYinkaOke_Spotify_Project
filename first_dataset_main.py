# Final Project Main Function: Finding Greatest Predictors for Dancability and Valence of the Most Popular Songs on Spotify
# Creators: Michael Yinka'Oke, Sam Clear, Diwas Dahal
# Start Date: March 21st, 2024

# This file currently acts as the main dataset inspecting file of the program, but will eventually just be
# used for running the main bulk of objects in the program.

# Essential Packages for inspecting the data
import graphviz
import IPython
from IPython.core.display_functions import display
import joblib
from joblib import parallel_backend
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import string
import sklearn as sk
import xgboost
from random import randint
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn import tree
from sklearn.tree import export_graphviz

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

# Current Version of Main Function for Dataset 1
if __name__ == '__main__':

    music_data = pd.read_csv('data/song_data.csv', index_col=0).dropna()

    ## Data Cleaning
    # Try removing all popularity values that are equal to zero. These songs may have never been evaluated, which may
    # misrepresent the true metrics of an unpopular song.
    music_data = music_data[music_data['popularity'] > 0]

    # Fixing indexing
    music_data = music_data.reset_index()

    cont_pop_data = music_data.drop_duplicates(subset=['track_name'])
    cont_pop_data.loc[:, 'duration_s'] = cont_pop_data['duration_ms'] / 1000
    cont_pop_data = cont_pop_data.drop(columns=['track_id']).dropna()

    # Load data and drop unnecessary duplicate columns and rows with null values.
    unique_data = music_data.drop(columns=['track_id']).dropna()

    # Classifying popularity of data based off of whether it's popularity value is greater than 75 (1) or not (0).
    cleared_data = unique_data
    cleared_data['popularity'] = cleared_data['popularity'].apply(lambda x: 1 if x > 75 else 0)

    # Convert True to 1 and False to 0 in the "popularity" column
    cleared_data['explicit'] = cleared_data['explicit'].astype(int)
    # Remove duplicates based on track name
    cleared_data = cleared_data.drop_duplicates(subset=['track_name'])
    # Convert milliseconds to seconds
    cleared_data['duration_s'] = cleared_data['duration_ms'] / 1000

    # Select numeric columns for modeling (This removes any additional string columns that we haven't already
    # removed in this subset of the dataset for training ML algorithms).
    numeric_cols = cleared_data.select_dtypes(include='number')

    # Min-max normalization with entire dataset
    pop_norm = (numeric_cols - numeric_cols.min()) / (numeric_cols.max() - numeric_cols.min())

    # Standardization with standard deviation of all columns in the dataset?
    pop_stan = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()

    # List of columns you want to keep (Including Key)
    columns_to_keep = ['popularity', 'duration_s', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
    columns_to_get_mean = ['duration_s', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

    # Between 1-3 different entries don't have an artist, album name, or track_name. Are these columns what we want to
    # use to evaluate the relationship between different characteristics in songs, Popularity, and Dancability?
    # Another thing to note is that some tracks are entered into the dataset more than once. These songs will be
    # overrepresented in the dataset, so we can just take unique values by track ID.
    #cleared_data.info()

    ## Prior Graphing, Gathering Insights
    # Computing correlations of data with highest popularity value
    '''we consider a song is popular if it's popularity determined by the number of time it was played is greater than 75
    * also if we want we can increase the number of popularity for better accuracy'''
    pop_data = music_data[music_data['popularity'] > 75]
    pop_columns = ['artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 'explicit', 'danceability',
                   'energy',
                   'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
                   'tempo',
                   'time_signature', 'track_genre']
    plt.imshow(pop_data.corr(), cmap="PuBu")
    plt.xticks(ticks=range(len(pop_columns)), labels=pop_columns, rotation=90)
    plt.yticks(ticks=range(len(pop_columns)), labels=pop_columns)
    plt.title("Correlations between Characteristics of Songs with High Popularities")
    plt.tight_layout()
    plt.colorbar()
    plt.close()

    # Calculate average of specified columns, which we can use to predict the popularity of a song
    average_data = pop_data[
        ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
         'liveness', 'valence', 'tempo', 'time_signature']].mean()

    # Compare popularity to another characteristic
    plt.scatter(data=pop_data, x='duration_ms', y='popularity')
    plt.ylabel("Popularity Score of Song (0 to 1)")
    plt.xlabel("Duration of the Song (Milliseconds)")
    plt.title("Comparing popularity of song to Duration (To be deprecated to scatterplot graph in final document)")
    plt.close()
    print(average_data)

    ## Under-sampling the negative class for the data, picking from equal portions of the data throughout.
    print(f"Positive Class: {len(cleared_data[cleared_data['popularity'] == 1])}, Negative Class: {len(cleared_data[cleared_data['popularity'] == 0])}")
    floor_divisor = len(cleared_data[cleared_data['popularity'] == 0]) // len(cleared_data[cleared_data['popularity'] == 1])
    current_new_data = []
    for i in range(len(cleared_data[cleared_data["popularity"] == 1])):
        current_new_data.append(cleared_data[cleared_data['popularity'] == 0].iloc[(i * floor_divisor) + 5])
    undersampled_dataframe = pd.DataFrame(data=current_new_data, columns=columns_to_keep)
    all_data = undersampled_dataframe.append(cleared_data[cleared_data['popularity'] == 1])
    print(f"New Positive Class Size: {len(all_data[all_data['popularity'] == 1])}, Negative Class Size: {len(all_data[all_data['popularity'] == 0])}")
    print("Data should now be balanced, however a large amount of data was lost. We may decrease the minimum popularity score needed to be labeled \'popular\'.")

    all_numeric_data = all_data.select_dtypes(include='number').drop(columns=['duration_ms', 'index'])

    # Normalize the under-sampled data
    norm_undersampled = (all_numeric_data - all_numeric_data.min()) / (all_numeric_data.max() - all_numeric_data.min())

    # Standardization with standard deviation of all columns in the dataset?
    stan_undersampled = (all_numeric_data - all_numeric_data.mean()) / all_numeric_data.std()


    ## SVM with Soft Margin (Allow for missclassification at a low cost, essential for our imperfect dataset)
    ## The very initial version of this file was constructed with the support of ChatGPT, but has been built
    ## up significantly since then.

    # Generate noisy data
    X = norm_undersampled[columns_to_get_mean]
    y = norm_undersampled['popularity']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Finding average accuracy values of each model, comparing the models
    accuracies_linear = []
    accuracies_sgd = []
    accuracies_poly = []
    accuracies_sigmoid = []
    accuracies_rbf = []

    # Create SVM classifier with soft-margin (C=1)
    # Measure differences in SVM Accuracy by kernel
    print("Creating Classifiers...")
    svm_classifier_linear = SVC(kernel='linear', degree=3, C=1)
    svm_classifier_sgd = SGDClassifier(max_iter=2500)
    svm_classifier_poly = SVC(kernel='poly', degree=3, C=1)
    svm_classifier_sigmoid = SVC(kernel='sigmoid', degree=3, C=1)
    svm_classifier_rbf = SVC(degree=3, C=1)

    # Process the data through each of these five models 10 times to calculate the true accuracy of the model
    for _ in range(9):

        # Train the classifiers
        print("Training classifiers...")
        svm_classifier_linear.fit(X_train, y_train)
        svm_classifier_sgd.fit(X_train, y_train)
        svm_classifier_poly.fit(X_train, y_train)
        svm_classifier_sigmoid.fit(X_train, y_train)
        svm_classifier_rbf.fit(X_train, y_train)

        # Make predictions on test data
        print("Classifying unseen data based on trained classifiers...")
        y_pred_linear = svm_classifier_linear.predict(X_test)
        y_pred_sdg = svm_classifier_sgd.predict(X_test)
        y_pred_poly = svm_classifier_poly.predict(X_test)
        y_pred_sigmoid = svm_classifier_sigmoid.predict(X_test)
        y_pred_rbf = svm_classifier_rbf.predict(X_test)

        # Calculate accuracy of each model
        print("Processing the accuracy of each classifier, compared to actual data values...")
        accuracies_linear.append(accuracy_score(y_test, y_pred_linear))
        accuracies_sgd.append(accuracy_score(y_test, y_pred_sdg))
        accuracies_poly.append(accuracy_score(y_test, y_pred_poly))
        accuracies_sigmoid.append(accuracy_score(y_test, y_pred_sigmoid))
        accuracies_rbf.append(accuracy_score(y_test, y_pred_rbf))

    # Last iteration occurs outside of the loop for graphing purposes
    # Train the classifiers
    print("Training classifiers...")
    svm_classifier_linear.fit(X_train, y_train)
    svm_classifier_sgd.fit(X_train, y_train)
    svm_classifier_poly.fit(X_train, y_train)
    svm_classifier_sigmoid.fit(X_train, y_train)
    svm_classifier_rbf.fit(X_train, y_train)

    # Make predictions on test data
    print("Classifying unseen data based on trained classifiers...")
    y_pred_linear = svm_classifier_linear.predict(X_test)
    y_pred_sgd = svm_classifier_sgd.predict(X_test)
    y_pred_poly = svm_classifier_poly.predict(X_test)
    y_pred_sigmoid = svm_classifier_sigmoid.predict(X_test)
    y_pred_rbf = svm_classifier_rbf.predict(X_test)

    # Calculate accuracy of each model
    print("Processing the accuracy of each classifier, compared to actual data values...")
    accuracies_linear.append(accuracy_score(y_test, y_pred_linear))
    accuracies_sgd.append(accuracy_score(y_test, y_pred_sgd))
    accuracies_poly.append(accuracy_score(y_test, y_pred_poly))
    accuracies_sigmoid.append(accuracy_score(y_test, y_pred_sigmoid))
    accuracies_rbf.append(accuracy_score(y_test, y_pred_rbf))

    # Calculate the final average accuracies of each of the models, print results
    avg_acc_linear = np.mean(accuracies_linear)
    avg_acc_sgd = np.mean(accuracies_sgd)
    avg_acc_poly = np.mean(accuracies_poly)
    avg_acc_sigmoid = np.mean(accuracies_sigmoid)
    avg_acc_rbf = np.mean(accuracies_rbf)

    print("Model Accuracy by Kernel/Method")
    print(f"Linear Kernel: {avg_acc_linear}")
    print(f"Linear SGD Kernel: {avg_acc_sgd}")
    print(f"Polynomial Kernel: {avg_acc_poly}")
    print(f"Sigmoid Kernel: {avg_acc_sigmoid}")
    print(f"RBF Kernel: {avg_acc_rbf}")

    # What classification errors were made in each type of model? Collect true positives against false positives for each type of kernel
    linear_tn, linear_fp, linear_fn, linear_tp = confusion_matrix(y_test, y_pred_linear).ravel()
    sgd_tn, sgd_fp, sgd_fn, sgd_tp = confusion_matrix(y_test, y_pred_sgd).ravel()
    poly_tn, poly_fp, poly_fn, poly_tp = confusion_matrix(y_test, y_pred_poly).ravel()
    sigmoid_tn, sigmoid_fp, sigmoid_fn, sigmoid_tp = confusion_matrix(y_test, y_pred_sigmoid).ravel()
    rbf_tn, rbf_fp, rbf_fn, rbf_tp = confusion_matrix(y_test, y_pred_rbf).ravel()

    # Comparing TP/FP on Two-layer Bar Plot
    class_given = ("Linear", "Linear SGD", "Polynomial", "Sigmoid", "RBF")
    positive_class_data = {"True Positive": [linear_tp, sgd_tp, poly_tp, sigmoid_tp, rbf_tp],
                           "False Positive": [linear_fp, sgd_fp, poly_fp, sigmoid_fp, rbf_fp]}
    negative_class_data = {"False Negative": [linear_fn, sgd_fn, poly_fn, sigmoid_fn, rbf_fn],
                           "True Negative": [linear_tn, sgd_tn, poly_tn, sigmoid_tn, rbf_tn]}

    # This plot was inspired by Matplotlib's barplot example: "Grouped Bar Plots with Labels".
    x = np.arange(len(class_given))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    # Positive Class Plot
    fig, ax = plt.subplots()
    for attribute, measurement in positive_class_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=4)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel("Kernel Type for SVC, divided by TP and FP Totals")
    plt.ylabel("Totals Correct/Incorrect Classifications")
    plt.title("Comparing Kernel Classifications by True/False Positive Totals")
    ax.set_xticks(ticks=(x + width), labels=class_given)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 250)
    plt.show()
    plt.close()

    # Negative Class Plot
    fig, ax = plt.subplots()
    for attribute, measurement in negative_class_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=4)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel("Kernel Type for SVC, divided by TP and FP Totals")
    plt.ylabel("Totals Correct/Incorrect Classifications")
    plt.title("Comparing Kernel Classifications by True/False Positive Totals")
    ax.set_xticks(ticks=(x + width), labels=class_given)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 250)
    plt.show()

    # pred = svm_classifier_poly.predict_proba(X_test)[:, 1]
    #
    # tpr_svm, fpr_svm, thresholds = roc_curve(y_test, pred)

    # Finding the best classifier model's average accuracy of features, verified by RandomSearchCV
    # accuracy_scores = []
    # for _ in range(8):
    #     svm_classifier_best = SVC(kernel='linear', degree=16, C=3)
    #     svm_classifier_best.fit(X_train, y_train)
    #     y_pred_best = svm_classifier_best.predict(X_test)
    #     accuracy_scores.append(round(accuracy_score(y_test, y_pred_best), 2))
    # avg_acc_best_svc = np.mean(np.array(accuracy_scores))
    # print(f"Average Accuracy of Optimized Hyperparameters: {np.mean(np.array(accuracy_scores))}")

    # Use RandomSearchCV for remaining hyperparameters to automate
    # features = {"C": [0.5 * x for x in range(20)], "degree": [x for x in range(20)]}
    # cross_validate = RandomizedSearchCV(svm_classifier_linear, features)
    # params = cross_validate.fit(X_train, y_train)
    # print(params.best_params_) # Returns the values (degree: 16, C: 3)


    ## Decision Tree Modeling
    # First Decision Tree Model - All features
    # This Decision Tree Classifier Program was adapted from the following site:
    # https://scikit-learn.org/stable/modules/tree.html
    X = norm_undersampled[columns_to_get_mean]
    Y = norm_undersampled['popularity']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    decision_tree_matrix = confusion_matrix(y_test, y_pred)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=columns_to_get_mean,
                                    class_names=['popular', 'not_popular'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")

    pred = clf.predict_proba(X_test)[:, 0]

    tpr_decision_tree, fpr_decision_tree, thresholds = roc_curve(y_test, pred)


    ## XGBoosted Random Forest Classifier
    # How can we boost this RFC model?
    # Create train/test sets
    X = norm_undersampled[columns_to_get_mean]
    Y = norm_undersampled['popularity']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Define the hyperparameter space (possible values to try for each hyperparameter)
    param_dist = {
        'n_estimators': [x for x in range(1,25)],
        'max_depth': [x for x in range(1,30)]
    }

    # Train a Random Forest classifier
    # if we wanted to optimize the hyperparameters, we could use a RandomizedSearchCV
    # just define the classifier with no hyperparameters

    from numpy import mean
    from numpy import std
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from xgboost import XGBRFClassifier
    from xgboost import plot_tree

    # define the model
    model = XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2)
    # define the model evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    model.fit(X_train, y_train)
    plot_tree(model)
    plt.show()

    prediction_probabilities = model.predict_proba(X_test)[:, 0]
    tpr_rfc, fpr_rfc, _ = roc_curve(y_test, prediction_probabilities)

    print("How does our XGBRFClassifier do? Pretty bad to be honest, it's about as bad as random-guessing.")
    # pop_clusters = KMeans(n_clusters=5, max_iter=500, random_state=42).fit(cont_pop_data[['loudness', 'danceability']])
    # cont_pop_data["popularity_groups"] = pop_clusters.labels_
    #
    # plt.scatter(data=unique_data, x="popularity", y="danceability", c="popularity_groups")
    # plt.show()
    #
    # first_group = cont_pop_data[cont_pop_data["popularity_groups"] == 0]
    # second_group = cont_pop_data[cont_pop_data["popularity_groups"] == 1]
    # third_group = cont_pop_data[cont_pop_data["popularity_groups"] == 2]
    # fourth_group = cont_pop_data[cont_pop_data["popularity_groups"] == 3]
    # fifth_group = cont_pop_data[cont_pop_data["popularity_groups"] == 4]
    #
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5)
    # ax1.hist(x=first_group['popularity'], bins=8, density=True, histtype='bar')
    # ax2.hist(x=second_group['popularity'], bins=8, density=True, histtype='bar')
    # ax3.hist(x=third_group['popularity'], bins=8, density=True, histtype='bar')
    # ax4.hist(x=fourth_group['popularity'], bins=8, density=True, histtype='bar')
    # ax5.hist(x=fifth_group['popularity'], bins=8, density=True, histtype='bar')
    # fig.tight_layout()
    # plt.show()


    ## K-Nearest-Neighbors: Fit adequately for model

    # # Reshape the data to long format
    # long_data = pd.melt(cleared_data, id_vars=['popularity'], var_name='feature', value_name='value')
    # # Calculate the mean for each feature and diagnosis
    # means = long_data.groupby(['feature', 'popularity'])['value'].mean().reset_index()
    # # Reshape the data back to wide format
    # wide_means = means.pivot(index='feature', columns='popularity', values='value')
    # # Print the result
    # print(wide_means)
    # # Create a box plot
    # sns.set(style="whitegrid")
    # g = sns.FacetGrid(long_data, col='feature', col_wrap=2, margin_titles=True,
    #                   xlim=(long_data['value'].min(), long_data['value'].max()))
    # g.map(sns.boxplot, 'value', 'popularity', 'popularity', order=['popular', 'not_popular'],
    #       hue_order=['popular', 'not_popular'], palette={"popular": "tomato", "not_popular": "cyan"})
    # plt.title("")
    # plt.xlabel("")
    # plt.ylabel("")
    #
    # # Remove the legend
    # plt.legend().remove()
    # plt.xlim(left=0, right=1)
    # plt.show()

    # Min-max normalization completed above
    pop_norm = norm_undersampled[columns_to_get_mean]
    # Standardization completed above
    pop_stan = stan_undersampled[columns_to_get_mean]

    # Create knn_results DataFrame
    knn_results = pd.DataFrame({'k': range(1, 5), 'pop_norm': [-1] * 4, 'pop_stan': [-1] * 4})
    # Displaying the knn_results DataFrame
    # print(knn_results)
    # Convert 'pop_norm' and 'pop_stan' columns to float64
    knn_results['pop_norm'] = knn_results['pop_norm'].astype(float)
    knn_results['pop_stan'] = knn_results['pop_stan'].astype(float)
    # Fit KNN Algorithm for normalized data
    for i in range(len(knn_results)):
        knn = KNeighborsClassifier(n_neighbors=knn_results.loc[i, 'k'])
        loop_knn = cross_val_predict(knn, pop_norm, norm_undersampled['popularity'], cv=6)
        loop_norm_cm = confusion_matrix(loop_knn, norm_undersampled['popularity'])
        accuracy = round(accuracy_score(loop_knn, norm_undersampled['popularity']), 2)
        print(f"Accuracy for k={knn_results.loc[i, 'k']} with normalized data: {accuracy}")

        # Debugging print
        knn_results.loc[i, 'pop_norm'] = accuracy
        # Fit KNN Algorithm for standardized data
        knn = KNeighborsClassifier(n_neighbors=knn_results.loc[i, 'k'])
        loop_knn2 = cross_val_predict(knn, pop_stan, norm_undersampled['popularity'], cv=6)
        accuracy2 = round(accuracy_score(loop_knn2, norm_undersampled['popularity']), 2)
        print(f"Accuracy for k={knn_results.loc[i, 'k']} with standardized data: {accuracy2}")

        # Record Accuracy
        knn_results.loc[i, 'pop_stan'] = accuracy2

        long_knn_results = knn_results.melt(id_vars='k', var_name='rescale_method', value_name='accuracy')
        # Select the rows with the maximum accuracy for each rescale method
        max_accuracy_rows = long_knn_results.loc[long_knn_results.groupby('rescale_method')['accuracy'].idxmax()]

        # Displaying the knn_results DataFrame
        print(knn_results)

    long_knn_results = knn_results.melt(id_vars='k', var_name='rescale_method', value_name='accuracy')

    # # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=long_knn_results, x='k', y='accuracy', hue='rescale_method')

    # # Set labels and title
    plt.xlabel('Choice of K')  # plt.ylabel('Accuracy')
    plt.title('KNN Algorithm Performance')
    plt.legend(title='Rescale Method', labels=['Normalized', 'Standardized'])

    # # Set scale for x-axis and y-axis # plt.xticks(range(1, 6))
    plt.yticks(np.arange(0.95, 1.0, 0.01), labels=[f'{i:.2f}%' for i in np.arange(0.95, 1.0, 0.01)])

    # # Adjust legend position
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # # Show plot
    plt.grid(True)
    plt.show()

    # Select the rows with the maximum accuracy for each rescale method
    max_accuracy_rows = long_knn_results.loc[long_knn_results.groupby('rescale_method')['accuracy'].idxmax()]
    print(max_accuracy_rows)

    # Let's see how this model reacts to only being able to view an undersampled version of the data...
    # Ensure 'popularity' is included in pop_norm
    pop_norm['popularity'] = cleared_data['popularity']
    # Splitting data into features (X) and target (y)
    X = pop_norm.drop(columns=['popularity'])
    # Exclude 'popularity' from features
    y = pop_norm['popularity']
    # Use 'popularity' as the target column
    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Initializing KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=10)
    # Fitting the classifier on the training data
    knn_classifier.fit(X_train, y_train)
    # Predicting on the test data
    y_pred = knn_classifier.predict(X_test)
    pred = knn_classifier.predict_proba(X_test)[:, 1]

    tpr_knn, fpr_knn, thresholds = roc_curve(y_test, pred)

    # Calculating confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Displaying confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_classifier.classes_)
    #disp.plot()

    # We can collect each of the accuracy metrics and plot them on a scatterplot, along with
    # an expected ROC curve value and an identity line. This will show how each of them
    # matches in terms or effectivity of the positive class, and the overall accuracy of the
    # model if we use AUC.

    # ROC Curve: TP vs FP
    # Collecting accuracy statistics from each sample
    svm_tp = poly_tp
    svm_fp = poly_fp
    svm_tn = poly_tn
    svm_fn = poly_fn
    decision_tree_tn, decision_tree_fp, decision_tree_fn, decision_tree_tp = decision_tree_matrix.ravel()
    rfc_tn, rfc_fp, rfc_fn, rfc_tp = confusion_matrix(y_test, y_pred).ravel()
    knn_tn, knn_fp, knn_fn, knn_tp = cm.ravel()

    # Finding probabilities of SVM classifier
    svm_classifier_poly = SVC(kernel='poly', degree=3, C=1, probability=True)
    svm_classifier_poly.fit(X_train, y_train)
    svm_pred = svm_classifier_poly.decision_function(X_test)
    tpr_svm, fpr_svm, _ = roc_curve(y_test, svm_pred)


    # Plotting Layered ROC curves for each of the machine learning models
    plt.plot([0, 1], [0, 1], '--', color='red', label='Random Guessing')
    plt.plot(tpr_svm, fpr_svm, color='cadetblue', label='SVM with Polynomial Kernel')
    plt.plot(tpr_decision_tree, fpr_decision_tree, color='deepskyblue', label='Decision Tree')
    plt.plot(tpr_rfc, fpr_rfc, color='aqua', label='XGBoost Random Forest Classifier')
    plt.plot(tpr_knn, fpr_knn, color='darkcyan', label='K-Nearest-Neighbors')
    plt.title("ROC Curve Comparison between all tested Machine Learning Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # It appears that the KNN and Polynomial-kernel SVM models are close to being the most accurate. Is there
    # statistical significance in the accuracy values of each of these models?
    # Hypothesis Test
