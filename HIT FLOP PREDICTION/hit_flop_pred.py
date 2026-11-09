import streamlit as st
import pickle
import numpy as np
import pandas as pd


cv=pickle.load(open("count-vectorizer_final.pkl","rb"))
classifier=pickle.load(open("movie_sucess_classification_final.pkl","rb"))


               
def test_model(sentence):
    sen = cv.transform([sentence]).toarray()
    res = classifier.predict(sen)[0]
    
    if res == 0:
        return "AVERAGE MOVIE :)"
    elif res == 1:
        return "FLOP MOVIE :("
    elif res == 2:
        return "HIT MOVIE!!"

st.title("Movie Success Prediction")
st.subheader('ENTER MOVIE DETAILS BELOW')

movie_name = st.text_input("Name of the movie: ")

movie_overview = st.text_area("Movie Overview: ")

movie_tagline = st.text_input("Tagline: ")

movie_keywords = st.text_input("Keywords: ")

movie_genres =  st.text_input("Genres: ")

movie_director = st.text_input('Director:')

movie_actor_1 =  st.text_input('Main Actor:')

movie_actor_2 = st.text_input('Main Actresses')

ok=st.button('Predict')

if ok:
    movie_info =  movie_overview  + movie_tagline  + movie_keywords + movie_genres  + movie_director + movie_actor_1 +   movie_actor_2
    
    Prediction = test_model(movie_info.lower())
    st.subheader( f"{movie_name.upper()} IS PREDICTED TO BE A {Prediction}")


#     st.write(movie_info)











    
