# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:45:09 2024

@author: kalpavruksh_sjo
"""

# Importing required libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import re
import pickle

# Function to clean text data

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove punctuation and numbers
    text = text.lower().strip()  # Convert to lowercase and strip whitespaces
    return text

# Get the train data

#tweets_df = pd.read_csv('C:/Users/kalpavruksh_sjo/Downloads/Train.csv')
tweets_df = pd.read_csv("https://raw.githubusercontent.com/Surajjogu/Sentiment-Analysis/main/Train.csv")

# Select relevant columns

tweets_df = tweets_df[['text', 'airline_sentiment']]

# Encode the target labels

label_encoder = LabelEncoder()
tweets_df['sentiment_label'] = label_encoder.fit_transform(tweets_df['airline_sentiment'])

# Clean the text data

tweets_df['cleaned_text'] = tweets_df['text'].apply(clean_text)

# Split the data into training and testing sets

X = tweets_df['cleaned_text']
y = tweets_df['sentiment_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data

tfidf = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train)

# Apply SMOTE to handle class imbalance

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

# Train the model

rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, random_state=42)
rf.fit(X_train_smote, y_train_smote)

# Save the model and vectorizer to disk
with open('model.pkl', 'wb') as model_file:
    pickle.dump((rf, tfidf, label_encoder), model_file)

# Transform the test data

X_test_tfidf = tfidf.transform(X_test)

# Predict on the test set

y_pred_rf = rf.predict(X_test_tfidf)

# Convert the array into dataframe to store in excel

y_pred_rf_df = pd.DataFrame(y_pred_rf, columns=['Predicted_Sentiment_Label'])

# Evaluate the model

accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_)

#print("Accuracy: {:.2f}%".format(accuracy_rf * 100))
#print("Classification Report:\n", classification_rep_rf)

# Reset index to ensure they align properly

X_test_output = X_test.reset_index(drop=True)
y_test_output = y_test.reset_index(drop=True)
y_pred_rf_df_output = y_pred_rf_df.reset_index(drop=True)

test_output = pd.concat([X_test_output, y_test_output, y_pred_rf_df_output], axis=1)

#test output
#print(test_output)

def process_test_data():
    
    # get the test data

    test_df = pd.read_csv('C:/Users/kalpavruksh_sjo/Downloads/Test.csv')
    

    # Ensure the testing dataset has the same structure as the training dataset
    if 'airline_sentiment' in test_df.columns:
        test_df = test_df[['text', 'airline_sentiment']]
        test_df = test_df.dropna(subset=['text', 'airline_sentiment'])
        test_df['sentiment_label'] = label_encoder.transform(test_df['airline_sentiment'])
    else:
        test_df = test_df[['text']]
    
    # Clean the text data
    test_df['cleaned_text'] = test_df['text'].apply(clean_text)
    
    # Transform the test data
    X_test_new_tfidf = tfidf.transform(test_df['cleaned_text'])
    
    # Predict sentiments on the testing data
    predicted_sentiments = rf.predict(X_test_new_tfidf)
    
    # If true labels are available, calculate accuracy
    if 'sentiment_label' in test_df.columns:
        true_sentiments = test_df['sentiment_label']
        accuracy = accuracy_score(true_sentiments, predicted_sentiments)
        classification_rep = classification_report(true_sentiments, predicted_sentiments, target_names=label_encoder.classes_)
        print("Accuracy on new test data: {:.2f}%".format(accuracy * 100))
        print("Classification Report on new test data:\n", classification_rep)
        
    testcase_data = test_df[['cleaned_text', 'sentiment_label']]
    
    # Convert the array into dataframe to store in database
    predicted_sentiments_output = pd.DataFrame(predicted_sentiments, columns=['Predicted_Sentiment_Label'])
    
    # Reset index to ensure they align properly
    testcase_data_output = testcase_data.reset_index(drop=True)
    predicted_sentiments_output_output = predicted_sentiments_output.reset_index(drop=True)
    
    testcase_output = pd.concat([testcase_data_output, predicted_sentiments_output_output], axis=1)
    
    return testcase_output

def process_live_data(live_df):
    
    # Query the database and load into DataFrame

    #live_df = pd.read_csv('C:/Users/kalpavruksh_sjo/Downloads/testtext.csv')

    # Ensure the testing dataset has the same structure as the training dataset
    #if 'airline_sentiment' in live_df.columns:
        #live_df = live_df[['text', 'airline_sentiment']]
        #live_df = live_df.dropna(subset=['text', 'airline_sentiment'])
        #live_df['sentiment_label'] = label_encoder.transform(live_df['airline_sentiment'])
    #else:
        #live_df = live_df[['text']]
    
    # Clean the text data
    live_df['cleaned_text'] = live_df['text'].apply(clean_text)
    
    # Transform the text data
    X_test_new_tfidf = tfidf.transform(live_df['cleaned_text'])
    
    # Predict sentiments on the testing data
    predicted_sentiments = rf.predict(X_test_new_tfidf)
     
    testcase_data = live_df[['cleaned_text']]
    
    # Convert the array into dataframe to store in database
    predicted_sentiments_output = pd.DataFrame(predicted_sentiments, columns=['Predicted_Sentiment_Label'])
    
    # Reset index to ensure they align properly
    testcase_data_output = testcase_data.reset_index(drop=True)
    predicted_sentiments_output_output = predicted_sentiments_output.reset_index(drop=True)
    
    live_output = pd.concat([testcase_data_output, predicted_sentiments_output_output], axis=1)

    return live_output
