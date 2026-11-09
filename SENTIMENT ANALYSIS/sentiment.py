
import streamlit as st
import pickle
import pandas as pd
import numpy as np


st.title("Movie Sentiment Analyzer")
cv=pickle.load(open("count-vectorizer.pkl","rb"))
classifier=pickle.load(open("movie_review_classification.pkl","rb"))

               
def test_model(sentence):
    sen = cv.transform([sentence]).toarray()
    res = classifier.predict(sen)[0]
    if res == 1:
        return "Positive"
    else:
        return 'Negative'
               
               
def count_reviews(pdf_file):
    with open(pdf_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the content into reviews
    reviews = []
    current_review = ""
    review_number = 1
    for line in content.splitlines():
        if line.startswith("{}.".format(review_number)):
            if current_review:
                reviews.append(current_review)
            current_review = line
            review_number += 1
           
        elif line.startswith("{}.".format(review_number + 1)):
            current_review += " " + line
            reviews.append(current_review)
        else:
            current_review += line + " "
    if current_review:
        reviews.append(current_review)

    # Initialize the counters
    positive_sentiments = 0
    negative_sentiments = 0
     
    print(reviews)
               
               
    for review in reviews:
        # Test the sentiment of the review
        sentiment = test_model(review)

        # Count the number of positive and negative sentiments
        if sentiment == 'Positive':
            positive_sentiments += 1
        elif sentiment == 'Negative':
            negative_sentiments += 1

    return positive_sentiments, negative_sentiments


review_single = st.text_area("Enter a review:")
if st.button('Review Analyse'):
    if review_single is not None:
        sentiment_single = test_model(review_single)
        st.write(f'The sentiment of the review is: {sentiment_single}')
    
               

pdf_file = st.file_uploader('Upload a .txt file', type='txt')
if st.button('Analyse'):
    if pdf_file is not None:
        path = pdf_file.name
       
        positive_reviews, negative_reviews = count_reviews(path)
        st.write('Movie Sentiment is :')
        st.write(f'Positive reviews: {positive_reviews}')
        st.write(f'Negative reviews: {negative_reviews}')          
    
