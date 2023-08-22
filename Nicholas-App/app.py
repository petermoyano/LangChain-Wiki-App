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
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

# make api_key available to other modules
os.environ["OPENAI_API_KEY"] = apikey

#App framework
st.title("🦜️🔗 Peter's first langchain App")
prompt = st.text_input("Enter a topic to write about:")


#Prompt templates
title_template = PromptTemplate(
    #A list of the names of the variables the prompt template expects.
    input_variables=['topic'],
    template='Write me a youtube video title about {topic}.',
)

script_template = PromptTemplate(
    #A list of the names of the variables the prompt template expects.
    input_variables=['title'],
    template='Write me a youtube video script based on this TITLE: {title}.',
)


#LLMs: Create an instance of the OpenAI server
llm = OpenAI(temperature=0.9)
##Create chain to run queries against LLMs. 'llm' parameter is required (LLM you want to call).
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
script_chain= LLMChain(llm=llm, prompt=script_template, verbose=True)
sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True)

#Show the response to the screen if the user has entered a prompt
if prompt: 
    response = sequential_chain.run(prompt)
    st.write(response)