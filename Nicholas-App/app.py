# installed packages:
#   streamlit: used to build the web app
#   langchain: used to build the conversational AI (llm workflow)
#   openai: used to build the conversational AI (gpt-4 workflow)
#   wikipedia: used to get the summary of a wikipedia article (connect GPT to wikipedia)
#   chromadb: vector storage
#   tiktoken: used to tokenize the input (backend tokenizer fro openai)

import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI

# make api_key available to other modules
os.environ["OPENAI_API_KEY"] = apikey

#App framework
st.title("🦜️🔗 Peter's first langchain App")
prompt = st.text_input("Enter your prompt here:", "Write the title of a story about a robot that is trying to learn how to be human.")

#LLMs: Create an instance of the OpenAI server
llm = OpenAI(temperature=0.9)

#Show the response to the screen if the user has entered a prompt
if prompt: 
    response = llm(prompt)
    st.write(response)