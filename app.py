from styles import css, bot_template, user_template
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from streamlit_option_menu import option_menu
from langchain.vectorstores import Qdrant
import os
from qdrant_client import QdrantClient


def get_conversation_chain():
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=OpenAIEmbeddings(),
        content_payload_key="curso",
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def clear_input():
    st.session_state["user_question"] = st.session_state["question_input"]
    st.session_state["question_input"] = ""


def main():
    load_dotenv()
    st.set_page_config(page_title="Pregunta a tus documentos!", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.session_state.conversation = get_conversation_chain()

    # Option Menu
    with st.sidebar:
        option_menu(
            "UBot",
            ["Resuelve tus dudas", "Documentos"],
            icons=["house", "cloud-upload"],
            default_index=0,
            key="selected_option",
        )
    if st.session_state["selected_option"] == "Documentos":
        # TODO Upload pdf_docs to Qdrant on upload
        st.file_uploader(
            "Sube aquí tus archivos PDF y haz click en 'Process'",
            accept_multiple_files=True,
        )

    else:
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
        if "question_input" not in st.session_state:
            st.session_state.question_input = ""
        if "user_question" not in st.session_state:
            st.session_state.user_question = ""

        st.header("Pregunta a tus documentos!")

        st.text_input(
            "Escribe tu pregunta a continuación:",
            key="question_input",
            on_change=clear_input,
        )
        if st.session_state["user_question"]:
            handle_userinput(st.session_state["user_question"])


if __name__ == "__main__":
    main()
