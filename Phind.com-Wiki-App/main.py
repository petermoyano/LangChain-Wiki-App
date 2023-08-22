from langchain.llms import OpenAI
from langchain import ConversationChain
from langchain.agents import load_tools, initialize_agent
from langchain.agents.agent_types import AgentType
import openai
import wikipedia

llm = OpenAI(temperature=0.9, max_tokens=100)
tools = load_tools(["wikipedia", "llm-match"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
conversation = ConversationChain(llm=llm, verbose=True)

conversation.predict(input="Hey!")
conversation.predict(input="Can we have a talk?")
conversation.predict(input="I'm interested in learning AI.")

wikipedia.set_lang("en")
wikipedia.summary("Python (programming language)")

openai.api_key = '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'
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What is the OpenAI mission?"}]
)
print(completion)
