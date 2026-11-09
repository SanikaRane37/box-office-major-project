

import requests
import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("Movie Recommendatation system")
movie_df=pickle.load(open("movie_recm1.pkl","rb"))
similarity=pickle.load(open("similarity1.pkl","rb"))
list_movie=np.array(movie_df["Title"])
option = st.selectbox(
"Select Movie ",
(list_movie))


def load_poster(poster_url):
    response = requests.get(poster_url)
    image = response.content
    return image

def movie_recommend_poster(movie,number=5):
    index = movie_df[movie_df['Title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    l=[]
    for i in distances[1:number+1]:
          l.append("{}".format(movie_df.iloc[i[0]].poster_img))

    return(l)


def movie_recommend_links(movie,number=5):
    index = movie_df[movie_df['Title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    l=[]
    for i in distances[1:number+1]:
        l.append("{}".format(movie_df.iloc[i[0]].Title_URL))
    return(l)


if st.button('Recommend Me'):
    st.write('Movies Recomended for you are:')
    movie_recommend_poster(option)
    movie_recommend_links(option)

    new_df= pd.DataFrame({
    'Movie posters': movie_recommend_poster(option),

    })
    
    new_df2= pd.DataFrame({
    'Movie Links': movie_recommend_links(option),

    })
    

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(load_poster(new_df.loc[0,'Movie posters']))
        url = new_df2.loc[0, 'Movie Links']
        st.markdown(f"<a href='{url}' target='_blank'> WATCH MOVIE </a>", unsafe_allow_html=True)
    with col2:
        st.image(load_poster(new_df.loc[1,'Movie posters']))
        url = new_df2.loc[1, 'Movie Links']
        st.markdown(f"<a href='{url}' target='_blank'> WATCH MOVIE </a>", unsafe_allow_html=True)

    with col3:
        st.image(load_poster(new_df.loc[2,'Movie posters']))
        url = new_df2.loc[2, 'Movie Links']
        st.markdown(f"<a href='{url}' target='_blank'> WATCH MOVIE </a>", unsafe_allow_html=True)
    with col4:
        st.image(load_poster(new_df.loc[3,'Movie posters']))
        url = new_df2.loc[3, 'Movie Links']
        st.markdown(f"<a href='{url}' target='_blank'> WATCH MOVIE </a>", unsafe_allow_html=True)
    with col5:
        st.image(load_poster(new_df.loc[4,'Movie posters']))
        url = new_df2.loc[4, 'Movie Links']
        st.markdown(f"<a href='{url}' target='_blank'> WATCH MOVIE </a>", unsafe_allow_html=True)