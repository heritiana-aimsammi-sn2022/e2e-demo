import joblib
import gradio as gr 
import string
import re 
from sklearn.feature_extraction.text import TfidfVectorizer

# the path to acces the model 
model_path ="./models/my_model.joblib"

classes  =  {
    "0":"ham" ,
    "1":"spam"}

# load, no need to initialize the loaded_rf
loaded_model = joblib.load(model_path)

def clean_text(text):

    text = text.lower()

    tokens = re.split('\W+', text)
    text = " ".join(word  for word in tokens  if word not in string.punctuation)
    return text 


def prediction (text):

    text_preprocessed  = clean_text(text)

    tfidf_vect = TfidfVectorizer(analyzer=clean_text)
    tfidf_vect_fit = tfidf_vect.fit(text_preprocessed)

    input = tfidf_vect_fit.transform(text_preprocessed) 

    output = loaded_model.predict(input)

    return classes[output]


iface = gr.Interface(fn=prediction, inputs="text", outputs="text" , 
                    title="Spam Classifation Demo",
	                description = "This application is about spam classification",)


if __name__=="__main__":

    iface.launch(share=True)



