from pyexpat import model
import joblib
import gradio as gr 
import string
import re 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# the path to acces the model 
model_path ="./models/my_model.pkl"

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

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([text_preprocessed])

    print(f"the len of X is {len(X.toarray()[0])}")
    
  #  y_pred = loaded_model.predict([X])

   # print(f"The text is {y_pred}")

  #  print(f"je suis la pour le moment {classes[str(y_pred[0])]}")

    return classes[0]

  
     

iface = gr.Interface(fn=prediction, inputs="text", outputs="text" , 
                    title="Spam Classifation Demo",
	                description = "This application is about spam classification",)


if __name__=="__main__":

    iface.launch(share=True)



