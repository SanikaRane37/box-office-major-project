import streamlit as st
from opening_pred import show_opening_pg
from domestic_perd import show_domestic_pg
from worldwide_pred import show_worldwide_pg 

page = st.sidebar.selectbox('SELECT A CATEGORY',('Opening Revenue','Domestic Revenue', 'Worldwide Revenue'))


if page =='Opening Revenue':
    show_opening_pg()
elif page =='Domestic Revenue':
    show_domestic_pg()
elif page =='Worldwide Revenue':
    show_worldwide_pg()
