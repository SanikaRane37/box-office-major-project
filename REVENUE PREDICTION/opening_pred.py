import streamlit as st
import pickle
import numpy as np
import pandas as pd
import array as arr

def load_model():
    with open ('saved_steps_opening.pkl', 'rb') as file:
        data = pickle.load(file)
        return data

    
data = load_model()


regressor_loaded = data['model']
oe_distributor = data['oe_distributor']
oe_rating = data['oe_rating']
oe_mgenre = data['oe_mgenre']
oe_sgenre = data['oe_sgenre']



def show_opening_pg():
    st.title('Opening Revenue Predictions')
  
    
  
    Distributor = (
 "Other","Warner Bros." ,                          
"Universal Pictures" ,                    
"Twentieth Century Fox"  ,                
"Lions Gate Films",                       
"Sony Pictures Entertainment (SPE)" ,    
"Walt Disney Studios Motion Pictures",    
"Paramount Pictures",                    
"Focus Features"   ,                      
"Fox Searchlight" ,                        
"Sony Pictures Classics",                  
"The Weinstein Company" ,                  
"Screen Gems"  ,       
"Roadside Attractions"  ,                  
"A24"             ,                        
"Open Road Films (II)" ,                      
    )
    
    rating = ('PG', 'PG-13', 'R',"G")
    
    main_genre = ("Action","Adventure","Comedy","Drama","Biography","Horror","Crime" , "Animation","Documentary","Fantasy","Other")

    side_genre = ("Action","Adventure","Comedy","Drama","Biography","Horror","Crime" , "Animation","Mystery" ,"Family","Sci-Fi"
                  ,"Fantasy","Music","Thriller","NA")
    
   
    
    Distributor_ = st.selectbox('Distributor' , Distributor  )
    
    opening_theaters_ =st.number_input(label="Opening Theaters",min_value=1, max_value=5000)
    
    rating_= st.selectbox('Rating' , rating )
    
    main_genre_ = st.selectbox('Main Genre' , main_genre )
    
    side_genre_ = st.selectbox('Side Genre' , side_genre )
    
    budget_ = st.number_input(label="Budget(USD)",min_value=1, max_value=500)
 
    prediction=''
    
    ok=st.button('Predict')


    if ok:
       

        x=np.array([[Distributor_  , rating_ ,opening_theaters_,budget_,main_genre_ , side_genre_ ]])
        
        x[:,0] = oe_distributor.transform(x[:,0].reshape(-1,1))
        x[:,1] = oe_rating.transform(x[:,1].reshape(-1,1))
        x[:,4] = oe_mgenre.transform(x[:,4].reshape(-1,1))
        x[:,5] = oe_sgenre.transform(x[:,5].reshape(-1,1))
        
        Domestic_Pred =''
     
        Domestic_Pred = regressor_loaded.predict(x)
      

        st.subheader(f'{Domestic_Pred}')
        
# show_opening_pg()
