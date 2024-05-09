# ----- importing dependencies -----
import os
import cohere
import streamlit as st

from dotenv                                          import load_dotenv
from langchain_community.embeddings                  import HuggingFaceEmbeddings
from langchain_community.vectorstores                import FAISS
from langchain_experimental.text_splitter            import SemanticChunker
from langchain_community.document_loaders.csv_loader import CSVLoader

# import requests
# from flask import Flask, jsonify

from api2 import docs2

load_dotenv()

# -----
# url = 'http://192.168.30.106/magento2/magento/pub/rest/V1/semantic/search'
# payload = {
#     'query': "hi"
# }
# response = requests.post(url, json=payload)

# data = jsonify(response.text)
# -----

# ----- initializing cohere client -----
api_key = os.getenv("COHERE_API_KEY")
llm = cohere.Client(api_key)

# ----- loading data and docs -----
def load_data_and_get_docs(user_query):
    loader = CSVLoader('datasets/dataset1.csv')
    data = loader.load()
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="standard_deviation")
    docs = text_splitter.split_documents(data)
    db = FAISS.from_documents(docs, embedding_model)
    docs = db.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(user_query)
    return docs

# ----- initializing session state -----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----- generating chat -----
def display_chat(user_input,response): 
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    USER = "user"
    ASSISTANT = "assistant"
    st.chat_message(USER).write(user_input)
    st.session_state.messages.append({'role':'user','content':user_input})
    st.chat_message(ASSISTANT).write(response)
    st.session_state.messages.append({'role':'assistant','content':response})

# ----- getting query from the user -----
def get_text():
    input_text = st.chat_input("Write your query here...", key="input")
    return input_text

# ----- main function -----
def main():        
    st.set_page_config(page_title="RAG chatbot",layout="centered")
    user_input = get_text()

    if user_input is None or user_input == "":
        st.markdown("<h2 style=color:#2149f8; text-align=center>I am your chatbot to do a little chat.</h2>",unsafe_allow_html=True)
    else:
        # -----
        # docs = load_data_and_get_docs(user_input)
        # documents = []
        # docs = data['documents']
        # for doc in docs:
        #     d = {"title":"", "snippet": doc.page_content}
        #     documents.append(d)
        # -----
        documents = []
        for dc in docs2:
            d = {"title":"", "snippet": dc}
            documents.append(d)

        client = llm.chat(model="command-r",message=user_input,documents=documents)
        answer = client.text
        display_chat(user_input,answer)
        
if __name__ == '__main__':
    main()

