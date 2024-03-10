import streamlit as st
from config import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import sklearn
import pickle
import seaborn as sns

st.sidebar.header(PROJECT_NAME)
st.sidebar.write(AUTHOR)

Data = 'datasets/news.csv'

choice = st.sidebar.radio("Project Menu",MENU_OPTIONS)

def load_data(rows = None):
    data = pd.read_csv(Data)
    return data

dataNews = load_data()

if choice =='View data':

    st.title("View raw data")
    
    st.write('Dataset')
    st.write(dataNews)

if choice =='View stats':
    st.title('View Statistics in Dataset')
    
    st.write('News Dataset')
    describeData = dataNews.describe()
    st.write(describeData)

if choice =='Visualize':
    st.title("Graphs and charts")

    st.set_option('deprecation.showPyplotGlobalUse', False)

    dataNews.label.value_counts().plot(kind = 'pie', explode = (0,.1), figsize = (6,6), autopct = '%.2f%%')
    plt.title('Real and Fake News Pie Chart')
    plt.legend(['REAL','FAKE'])
    st.pyplot()

    st.subheader('WordCloud')
    st.image('images/wordcloud1.png')

    st.subheader('Model Prediction Graph')
    st.image('images/model prediction graph.png')

    st.subheader('Real Word Count')
    st.image('images/realWordCountPlot.png')

    st.subheader('Fake Word Count')
    st.image('images/fakeWordCountPlot.png')

if choice =='Prediction':
    st.title('Use AI to predict')
    st.subheader('fill the detail and get result')

    def load_model(path = 'models/fake_news_detection_model.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    st.title('Fake News Detection')
    with st.spinner('loading fake news dectection model'):
        model = load_model()
        vectorizer = load_model('models/tfidf_vector.pkl')
        st.success('models loaded into memory')

    message = st.text_area('Enter your news text here for analysis', value = 'hi there')
    btn = st.button('click to check')
    if btn and len(message) > 10:
        data = vectorizer.transform([message])
        prediction = model.predict(data)
        st.title('Prediction')
        if prediction[0] == 'REAL':
            st.success('TRUE NEWS')
        elif prediction[0] == 'FAKE':
            st.success('FAKE NEWS')
        else:
            st.error('Something is fishy')

if choice =='About':
    st.title('About the project')
    #st.image('img.png')
    st.write("""I have plotted and deployed model to predict if any text news is fake or real. The plots are informative and creative. Model that worked finally is PassiveAggressiveClassifier(). The tfidfVectorizer work just fine. The accuracy I got on PassiveAggressiveClassifier() is of 92%. It predicted well as it gave 92 percent accuracy. I plotted wordcloud chart, common word Count Graphs and Model Prediction Graph.
Comparing the results maximum accuracy achieved is 92%, when PassiveAggressionClassifier is used algorithm is used for classifier. The features selected are ‘text’ and ‘label’. So the given is best combination for the given specific data.
A result like 92 percent accuracy is very nice to have. I got 92 percent accuracy by using PassiveAggressionClassifier on the dataset I have. I have met the expectations on accuracy of the model. 
As PassiveAggressionClassifier model had the highest accuracy I decided to use this model for prediction. Although I have tried many model for best accuracy I finally go with PassiveAggressionClassifier model for my project that is Fake News Detection.
""")
    st.write("""Fake news” is a term that has come to mean different things to different people. At its core, we are defining “fake news” as those news stories that are false: the story itself is fabricated, with no verifiable facts, sources or quotes. Sometimes these stories may be propaganda that is intentionally designed to mislead the reader, or may be designed as “clickbait” written for economic incentives (the writer profits on the number of people who click on the story). In recent years, fake news stories have 
proliferated via social media, in part because they are so easily and quickly shared online.
""")