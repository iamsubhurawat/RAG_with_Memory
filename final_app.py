# ----- importing dependencies -----
import os
import json
import cohere
import requests
import streamlit   as     st

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

# ----- main function -----
def main():        
    st.set_page_config(page_title="RAG chatbot",layout="centered")
    user_input = get_text()
    if user_input is None or user_input == "":
        st.markdown("<h2 style=color:#2149f8; text-align=center>I am your chatbot to do a little chat.</h2>",unsafe_allow_html=True)
    else:
        docs = load_data_and_get_docs(user_input)
        documents = []
        for doc in docs:
            product_desc = ""
            for key in doc.keys():
                product_desc = product_desc + f"{key}:{doc[key]}" + "\n"
            product_name  = doc['name']
            d = {"title":product_name, "snippet": product_desc}
            documents.append(d)

        model = llm.chat(model="command-r-plus",message=user_input,documents=documents)
        answer = model.text
        display_chat(user_input,answer)
        
if __name__ == '__main__':
    main()