import streamlit as st
from transformers import TapexTokenizer, BartForConditionalGeneration, pipeline
import pandas as pd
import pandas as pd
import numpy as np

@st.cache_data
def load_data():
    tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")
    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    return tqa, model, tokenizer


st.title('🤖 Hello there, try to drop some data below and I will do my best to answer your questions :)')
google, microsoft, tokenizer = load_data()

option = st.selectbox(
    'Choose the model you prefer',
    ('Google TaPas', 'Microsoft TaPEx'))

uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    if uploaded_file.type == 'text/csv':
        data = pd.read_csv(uploaded_file)
        st.write(data)
        question = st.text_input("What is your question? :) (we are limiting the answer to be from the first 450 rows only; otherwise the model will take a long time to answer the question)")
        if len(question) != 0:

            data = data
            for c in data.columns:
                data[c] = data[c].astype(str)
            if option == 'Google TaPas':
                option = st.selectbox("Choose an identifier (this will work as an index to help you understand the data)", data.columns)
                data = data[:450]
                ans = google(table=data, query=question)
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
                encoding = tokenizer(table=data[:450], query=question, return_tensors="pt")
                outputs = microsoft.generate(**encoding)
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                st.write(outputs)

            
else:
    st.write("Please upload a csv file")
