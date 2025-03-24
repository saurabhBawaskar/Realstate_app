from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
import pandas as pd
import logging
import time
# import plotly.express as px
import os
from utils import get_answer, text_to_speech, autoplay_audio, speech_to_text, get_audio_excel_chain
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *

# Initialize floating features for the interface
float_init()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for managing chat messages
def initialize_session_state():
    if "text_chat_history" not in st.session_state:
        st.session_state.text_chat_history = [
            AIMessage(content="Hello! I'm a real estate assistant. Ask me anything about your real estate data."),
        ]
    if "audio_chat_history" not in st.session_state:
        st.session_state.audio_chat_history = [
            AIMessage(content="Hello! I'm a real estate assistant. You can use audio input to ask me anything about your real estate data."),
        ]
    if "sessions" not in st.session_state:
        st.session_state.sessions = []
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Text Chatbot"

initialize_session_state()

st.title("Real Estate Chatbot 🤖")

def init_data(file) -> pd.DataFrame:
    if file.name.endswith('.xlsx'):
        return pd.read_excel(file, engine='openpyxl')
    elif file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        raise ValueError("Unsupported file type")

def get_text_excel_chain(df):
    template = """
    You are a data analyst at a real estate company. You are interacting with a user who is asking you questions about the real estate data.
    Based on the data schema below, write a Python Pandas query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the Pandas query and nothing else. Do not wrap the Pandas query in any other text, not even backticks.
    
    For example:
    Question: List the top 3 most expensive properties.
    Pandas Query: df.nlargest(3, 'price')
    Question: Show the details of properties in New York.
    Pandas Query: df[df['city'] == 'New York']
    
    Your turn:
    
    Question: {question}
    Pandas Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model="gpt-4o")
    
    def get_schema(_):
        return df.columns.tolist()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, df: pd.DataFrame, chat_history: list):
    excel_chain = get_text_excel_chain(df)
    
    template = """
    You are a data analyst at a real estate company. You are interacting with a user who is asking you questions about the real estate data.
    Based on the data schema below, question, pandas query, and pandas response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    Pandas Query: <PANDAS>{query}</PANDAS>
    User question: {question}
    Pandas Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model="gpt-4o")
    
    def execute_query(query, df):
        try:
            local_vars = {"df": df}
            exec(f"result = {query}", {}, local_vars)
            logger.info(f"Executed query: {query}")
            return local_vars["result"]
        except Exception as e:
            logger.error(f"Error executing query: {query}\n{e}")
            return f"Error executing query: {query}\n{e}"
    
    chain = (
        RunnablePassthrough.assign(query=excel_chain).assign(
            schema=lambda _: df.columns.tolist(),
            response=lambda vars: execute_query(vars["query"], df),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

    # Log the generated query and response
    logger.info(f"User question: {user_query}")
    logger.info(f"Generated response: {response}")

    return response

load_dotenv()

# Sidebar for data upload
with st.sidebar:
    # st.image("path/to/animated_logo.gif", use_column_width=True)  # Add the path to your animated logo GIF here
    st.subheader("Upload the data")
    
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            df = init_data(uploaded_file)
            st.session_state.df = df
            st.success("Data loaded successfully!")

        # Display a data summary
        st.subheader("Data Summary")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")

        if st.button("Show Data"):
            st.dataframe(df)

        # Add a download button
        st.download_button(
            label="Download data as CSV",
            data=df.to_csv().encode('utf-8'),
            file_name='real_estate_data.csv',
            mime='text/csv',
        )
    
    if "processing_time" in st.session_state:
        st.write(f"Last processing time: {st.session_state.processing_time:.2f} seconds")
    
    # Display previous sessions
    st.subheader("Chat Sessions")
    for i, session in enumerate(st.session_state.sessions):
        with st.expander(f"Session {i+1}"):
            for message in session:
                if isinstance(message, HumanMessage):
                    st.write(f"**You:** {message.content}")
                else:
                    st.write(f"**Bot:** {message.content}")


# Tabs for text and audio chatbots
tab1, tab2 = st.tabs(["Text Chatbot", "Audio Chatbot"])

# Text Chatbot Tab
with tab1:
    st.header("Text Chatbot")
    st.session_state.active_tab = "Text Chatbot"
    for message in st.session_state.text_chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

# Audio Chatbot Tab
with tab2:
    st.header("Audio Chatbot")
    st.session_state.active_tab = "Audio Chatbot"
    footer_container = st.container()
    with footer_container:
        audio_bytes = audio_recorder()
    if audio_bytes:
        with st.spinner("Transcribing..."):
            # Write the audio bytes to a temporary file
            webm_file_path = "temp_audio.mp3"
            with open(webm_file_path, "wb") as f:
                f.write(audio_bytes)

            # Convert the audio to text using the speech_to_text function
            transcript = speech_to_text(webm_file_path)
            if transcript:
                st.session_state.audio_chat_history.append(HumanMessage(content=transcript))
                with st.chat_message("Human"):
                    st.write(transcript)
                os.remove(webm_file_path)

    if not isinstance(st.session_state.audio_chat_history[-1], AIMessage) and st.session_state.active_tab == "Audio Chatbot":
        with st.chat_message("assistant"):
            with st.spinner("Thinking🤔..."):
                serialized_chat_history = [
                    {"role": "assistant" if isinstance(msg, AIMessage) else "user", "content": msg.content}
                    for msg in st.session_state.audio_chat_history
                ]
                final_response = get_answer(serialized_chat_history, st.session_state.df)
            with st.spinner("Generating audio response..."):
                audio_file = text_to_speech(final_response)
                autoplay_audio(audio_file)
            st.write(final_response)
            st.session_state.audio_chat_history.append(AIMessage(content=final_response))
            os.remove(audio_file)

# Input and processing for both chatbots
user_query = st.chat_input("Type a message...")

if user_query:
    if st.session_state.active_tab == "Text Chatbot":
        st.session_state.text_chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        if uploaded_file is not None:
            with st.chat_message("AI"):
                start_time = time.time()
                with st.spinner("Your query is being processed..."):  
                    serialized_chat_history = [
                        {"role": "assistant" if isinstance(msg, AIMessage) else "user", "content": msg.content}
                        for msg in st.session_state.text_chat_history
                    ]
                    response = get_response(user_query, st.session_state.df, serialized_chat_history)
                end_time = time.time()
                
                processing_time = end_time - start_time
                st.session_state.processing_time = processing_time
                
                st.markdown(response)
                    
                st.session_state.text_chat_history.append(AIMessage(content=response))
        else:
            st.warning("Please upload an Excel or CSV file first.")
    elif st.session_state.active_tab == "Audio Chatbot":
        st.session_state.audio_chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        if uploaded_file is not None:
            with st.chat_message("AI"):
                start_time = time.time()
                with st.spinner("Your query is being processed..."):  
                    serialized_chat_history = [
                        {"role": "assistant" if isinstance(msg, AIMessage) else "user", "content": msg.content}
                        for msg in st.session_state.audio_chat_history
                    ]
                    response = get_response(user_query, st.session_state.df, serialized_chat_history)
                end_time = time.time()
                
                processing_time = end_time - start_time
                st.session_state.processing_time = processing_time
                
                st.markdown(response)
                    
                st.session_state.audio_chat_history.append(AIMessage(content=response))
        else:
            st.warning("Please upload an Excel or CSV file first.")

# Save the session
if st.button("Save Session"):
    st.session_state.sessions.append(st.session_state.text_chat_history.copy())
    st.session_state.sessions.append(st.session_state.audio_chat_history.copy())
    st.success("Session saved!")
