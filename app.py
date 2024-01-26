import streamlit as st
from llama_index import (SimpleDirectoryReader, VectorStoreIndex, ServiceContext)

from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
  messages_to_prompt,
  completion_to_prompt,
)
from langchain.schema import(SystemMessage, HumanMessage, AIMessage)

st.set_page_config(
    page_title="Personal Chatbot"
  )
st.header("Personal Chatbot")
st.sidebar.title("Options")


def main():
  st.session_state.messages = [
      SystemMessage(
        content="you are a helpful AI assistant. Reply your answer in markdown format."
      )
  ]

  messages = st.session_state.get("messages", [])

  if user_input := st.chat_input("You"):
    st.session_state.messages.append(user_input)
    with st.spinner("Bot is typing ..."):
      import time
      time.sleep(1)
      answer = "Answer"
    st.session_state.messages.append(AIMessage(content=answer))

  for message in messages:
    with st.chat_message("AI"):
      st.markdown(message)

if __name__ == "__main__":
  main()