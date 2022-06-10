from flask import Flask, request, redirect, url_for, flash, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from string import punctuation
import pickle as p
import re 
import json
import uvicorn
from os.path import dirname, join, realpath
import joblib
from typing import List
from typing import Optional
from fastapi import FastAPI, Query
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI(
    title="HTML Element Classifier Model API",
    description="API for ML model",
    version="0.1",
)

# load model
with open(
    join(dirname(realpath(__file__)), "finalized_svm_model.pkl"), "rb"
) as f:
    model = joblib.load(f)
    
    
def preprocess_text(text):

    stop_words = set(stopwords.words("english"))
    element = [entry.lower() for entry in text]
    
    #Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    element= [word_tokenize(entry) for entry in text]
    lemmatizer = WordNetLemmatizer()
    
    final_words = []
    for word in element:
        if word not in stopwords.words('english'):
            final_words.append(word)
            
    text_processed =[ ' '.join([lemmatizer.lemmatize(word) for word_list in element for word in word_list])]
    
    return text_processed


@app.get("/predict-element")
def predict(element: Optional[List[str]] = Query(["<input>type="text" type="text" id="search" name="q" </input>"])):
    
    ##Pre-process text
    processed_text = preprocess_text(element)
   
   # vect = CountVectorizer(decode_error=" Replace ", vocabulary=p.load(open("C:/Users/kelse/OneDrive/Documents/UOM-Imp Info/FYP/vector.pickel", "rb")))
    vect = CountVectorizer(ngram_range=(1,2), max_features=1000 , stop_words="english")
    test = vect.transform(processed_text)
    
   
    prediction = model.predict(test)[0]
    output = int(prediction[0])
   
    
    # output dictionary
    pred = {0: "AddToCart", 1: "Search"}
    
    # show results
    result = {"prediction": pred[prediction]}

    return result

@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )
