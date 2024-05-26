import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np

@st.cache_data
def load_data():
    tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")
    return tqa
    

st.title('🤖 Hello there, try to drop some data below and I will do my best to answer you questions :)')
model = load_data()

uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    if uploaded_file.type == 'text/csv':
        data = pd.read_csv(uploaded_file)
        st.write(data)
        option = st.selectbox("Choose an identifier (this will work as an index to help you understand the data)", data.columns)
        question = st.text_input("What is you question? :) (we are limiting the answer to be from the first 450 rows only to avoid wasting time)")
        if len(question) != 0:
            data = data
            for c in data.columns:
                data[c] = data[c].astype(str)
            data = data[:450]
            ans = model(table=data, query=question)
            columns = set()
            rows = []
            idx = []
            for x in ans['coordinates']:
                rows.append(x[0])
                columns.add(data.columns[x[1]])
                idx.append(data[option].iloc[x[0]])
            columns = list(columns)
            ans_pd = data.iloc[rows][columns]
            ans_pd = ans_pd.set_index(pd.Index(idx))
            st.write(ans_pd)
            
else:
    st.write("Please upload a csv file")