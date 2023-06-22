from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate

template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt,
    verbose=False,
    memory=memory,
)

response = llm_chain.predict(human_input="Hi there I'm Erez from Israel")
print(response)
response = llm_chain.predict(human_input="Who am?")
print(response)
response = llm_chain.predict(human_input="Where I'm from?")
print(response)
response = llm_chain.predict(human_input="I like playing with some AI based models, are you one of them?")
print(response)
response = llm_chain.predict(human_input="Which model are you?")
print(response)
response = llm_chain.predict(human_input="What is the name of the NLP model you are using?")
print(response)
response = llm_chain.predict(human_input="When was that model released?")
print(response)