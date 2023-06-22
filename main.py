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

response = llm_chain.predict(human_input="Hi there I'm Erez from Langchain")
print(response)
response = llm_chain.predict(human_input="Who am I and where I'm from?")
print(response)