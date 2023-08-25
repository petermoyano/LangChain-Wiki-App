# installed packages:
#   streamlit: used to build the web app
#   langchain: used to build the conversational AI (llm workflow)
#   openai: used to build the conversational AI (gpt-4 workflow)
#   wikipedia: used to get the summary of a wikipedia article (connect GPT to wikipedia)
#   chromadb: vector storage
#   tiktoken: used to tokenize the input (backend tokenizer fro openai)

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# load .env file
load_dotenv()

#make api_key available to other moduless
apikey = os.getenv('OPENAI_API_KEY')

#App framework
st.title("🦜️🔗 Peter's first langchain App")
prompt = st.text_input("Enter a topic you want to learn about:")


#Prompt templates
title_template = PromptTemplate(
    #A list of the names of the variables the prompt template expects.
    input_variables=['topic'],
    template='Make a list with the 5 most important concepts to grasp about {topic}. The list should only contain the concepts, not the explanation.',
)

script_template = PromptTemplate(
    #A list of the names of the variables the prompt template expects.
    input_variables=['concepts'],
    template='explain the following concepts: {concepts}.',
)


#LLMs: Create an instance of the OpenAI server
llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.9)

#Create chain to run queries against LLMs. 'llm' parameter is required (LLM you want to call).
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='concepts')
script_chain= LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='explanation')
#The output of the first chain is the input of the second chain
sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['concepts', 'explanation'], verbose=True) 

#Show the response to the screen if the user has entered a prompt
if prompt: 
    response = sequential_chain({'topic': prompt})
    st.write(response['concepts'])
    st.write(response['explanation'])