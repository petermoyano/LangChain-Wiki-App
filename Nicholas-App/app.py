# installed packages:
#   streamlit: used to build the web app
#   langchain: used to build the conversational AI (llm workflow)
#   openai: used to build the conversational AI (gpt-4 workflow)
#   wikipedia: used to get the summary of a wikipedia article (connect GPT to wikipedia)
#   chromadb: vector storage
#   tiktoken: used to tokenize the input (backend tokenizer fro openai)

import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import  LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# load .env file
load_dotenv()

#Front end App framework
st.title("🦜️🔗 Peter's first langchain App")
prompt = st.text_input("Enter a topic you want to learn about:")

#Prompt templates
topic_template = PromptTemplate(
    input_variables=['topic'], #This is a list of the names of the variables the prompt template expects.
    template='Make a list with the 5 most important concepts to grasp about {topic}. The list should only contain the concepts, not the explanation.',
)
concepts_template = PromptTemplate(
    input_variables=['concepts', 'wikipedia_research', 'prompt'],
    template='explain the following concepts: {concepts} in the context of {prompt} and this wikipedi_reaserch: {wikipedia_research}.',
)
print("This is the ProoooompT!!!!", prompt)

#Memory
topic_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
concepts_memory = ConversationBufferMemory(input_key='concepts', memory_key='chat_history')

#LLMs: Create an instance of the OpenAI server
llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.9)
topic_chain = LLMChain(llm=llm, prompt=topic_template, verbose=True, output_key='topic', memory=topic_memory)
concepts_chain= LLMChain(llm=llm, prompt=concepts_template, verbose=True, output_key='concepts', memory=concepts_memory)
#The output of the first chain is the input of the second chain
wiki = WikipediaAPIWrapper()

#Show the response to the screen if the user has entered a prompt
if prompt: 
    concepts = topic_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    result = concepts_chain.run(prompt=prompt, concepts=concepts, wikipedia_research=wiki)

    st.write(concepts)
    st.write(result)

    with st.expander('Concepts History'):
        st.info(concepts_memory.buffer)

    with st.expander('Wikipedia research'):
        st.info(wiki_research)