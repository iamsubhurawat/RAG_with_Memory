# ----- importing dependencies -----
import os
import json
import cohere
import requests
import streamlit   as     st

from   cohere      import ChatMessage
from   dotenv      import load_dotenv

load_dotenv()

# ----- URL for realdata extraction -----
url = 'http://192.168.30.106/magento2/magento/pub/rest/V1/semantic/search'

# ----- initializing cohere client -----
api_key = os.getenv("COHERE_API_KEY")
llm = cohere.Client(api_key)

# ----- loading data and docs -----
def load_data_and_get_docs(user_query):
    payload = {"query": user_query}
    response = requests.post(url, json=payload)
    data = json.loads(response.text)    
    docs = data[0]['data']['metadatas'][0]
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

# ----- generating standalone question -----
def standalone_question(user_input):
    co = cohere.Client()
    if len(st.session_state.messages) > 0:
        last_index = len(st.session_state.messages)-1
        context = st.session_state.messages[last_index]['content']
    else:
        context = ""
    preamble = f"You are given a question from user and a context. You have to create a standalone question based on question and context given. If the question given by the user is any of type Hi, Hello then simply generate a greeting message. Question: {user_input} Context: {context}"
    ques = co.chat(message=user_input,preamble=preamble)
    ques = ques.text
    return ques

def single_ques(q1,q2):
    co = cohere.Client()
    preamble = f"You are given two questions from user and you have to create a single question by combining them without losing the actual meaning of the questions. If any one of the questions is of type Hello, Hi message then just generate a simple hello question. Question 1: {q1} Question 2: {q2}"
    single_question = co.chat(message=q1,preamble=preamble)
    single_question = single_question.text
    return single_question

def check_greeting_message(q):
    co = cohere.Client()
    preamble = f"You are an intelligent assistant and you have to classify the messages that if the message is a greeting message or not. Simply return 1 if greeting message or else return 0. message:{q}"
    greet_msg = co.chat(message=q,preamble=preamble)
    greet_msg = greet_msg.text
    return greet_msg

# ----- main function -----
def main():        
    st.set_page_config(page_title="RAG chatbot",layout="centered")
    user_input = get_text()
    if user_input is None or user_input == "":
        st.markdown("<h2 style=color:#2149f8; text-align=center>I am your chatbot to do a little chat.</h2>",unsafe_allow_html=True)
    else:
        value = int(check_greeting_message(user_input))
        print(value)
        user_query = standalone_question(user_input)
        docs = load_data_and_get_docs(user_query)
        documents = []
        for doc in docs:
            product_desc = ""
            for key in doc.keys():
                product_desc = product_desc + f"{key}:{doc[key]}" + "\n"
            product_name  = doc['name']
            d = {"title":product_name, "snippet": product_desc}
            documents.append(d)
        if value == 1:
            message = user_input
        elif value == 0:
            message = single_ques(user_query,user_input)
        print(message)
        model = llm.chat(model="command-r",message=message,documents=documents)
        answer = model.text
        display_chat(user_input,answer)
        
if __name__ == '__main__':
    main()


# streamlit run rag_with_memory.py --server.port 8080
# streamlit run rag_with_memory.py --server.fileWatcherType None --server.port 8080 