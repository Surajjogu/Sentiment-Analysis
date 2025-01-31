# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:23:37 2024

@author: kalpavruksh_sjo
"""

from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import pickle
import re

app = Flask(__name__)

# Load the sentiment analysis model
with open('model.pkl', 'rb') as model_file:
    rf, tfidf, label_encoder = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['csvFile']
        if not file:
            return redirect(url_for('home'))
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Check if the required column 'text' is in the dataframe
        if 'text' not in df.columns:
            return redirect(url_for('home'))
        
        # Process live data
        result, sentiment_counts = process_live_data(df)
        
        # Convert the dataframe to a list of dictionaries for rendering in the template
        prediction = result.to_dict(orient='records')
        
        return render_template('index.html', prediction=prediction, sentiment_counts=sentiment_counts)

def process_live_data(live_df):
    # Clean the text data
    live_df['cleaned_text'] = live_df['text'].apply(clean_text)
    
    # Transform the text data
    X_test_new_tfidf = tfidf.transform(live_df['cleaned_text'])
    
    # Predict sentiments on the data
    predicted_sentiments = rf.predict(X_test_new_tfidf)
    
    # Map predicted sentiment labels to sentiment strings
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_sentiments_mapped = [sentiment_map[label] for label in predicted_sentiments]
    
    
    # Convert the array into dataframe to store in database
    predicted_sentiments_output = pd.DataFrame(predicted_sentiments_mapped, columns=['Predicted_Sentiment_Label'])
    
    # Reset index to ensure they align properly
    live_df_output = live_df.reset_index(drop=True)
    predicted_sentiments_output_output = predicted_sentiments_output.reset_index(drop=True)
    
    live_output = pd.concat([live_df_output, predicted_sentiments_output_output], axis=1)
    
    # Calculate sentiment counts
    sentiment_counts = predicted_sentiments_output['Predicted_Sentiment_Label'].value_counts().to_dict()

    return live_output, sentiment_counts

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove punctuation and numbers
    text = text.lower().strip()  # Convert to lowercase and strip whitespaces
    return text

if __name__ == '__main__':
    app.run(debug=True)
