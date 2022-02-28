import streamlit as st
import joblib
import gradio as gr 
import string
import re 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer

# the path to acces the model 
model_path ="./models/my_model2.pkl"
csv_file_path = "./data/SMSSpamCollection.tsv"

data = pd.read_csv(csv_file_path ,  names=['label', 'body_text'], sep='\t')
X_train,X_test,y_train,y_test = train_test_split(data["body_text"],data["label"], test_size = 0.2, random_state = 10)

classes  =  {
    1:"ham" ,
    0:"spam"}

# load, no need to initialize the loaded_rf
loaded_model = joblib.load(model_path)

def clean_text(text):

    text = text.lower()

    tokens = re.split('\W+', text)
    text = " ".join(word  for word in tokens  if word not in string.punctuation)
    return text 


def prediction (text):

    
    text_preprocessed  = clean_text(text)
    #print(text_preprocessed)

    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)

    
    X = vectorizer.transform([text_preprocessed])
    print(X)

     
    prediction = loaded_model.predict(X)
    return classes[prediction[0]]

  


def main():
    st.title("Spam Classifation Demo")
    #st.subheader("This application is about spam classification")


    txt = st.text_area('This application is about spam classification', placeholder='Enter your text here')

    if (st.button("Predict")):
        if txt:
            st.success(" {} is a  {}".format(txt , prediction(txt)))
        else:
            st.warning("Please enter some text to predict")

if __name__ == "__main__":
    main()

