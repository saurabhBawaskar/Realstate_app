import speech_recognition as sr
from gtts import gTTS
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import pandas as pd

def speech_to_text(audio_data):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_data) as source:
        audio = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio)
            return transcript
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Sorry, there was an error with the speech recognition service."

def text_to_speech(input_text):
    tts = gTTS(text=input_text, lang='en')
    webm_file_path = "temp_audio_play.mp3"
    tts.save(webm_file_path)
    return webm_file_path

def get_answer(messages, df):
    # Custom logic to handle real estate data
    if df is not None:
        data_chain = get_audio_excel_chain(df)
        response = data_chain.invoke({
            "question": messages[-1]['content'],
            "chat_history": messages[:-1]
        })
        # Extract the query from the response
        query = response.strip()
        result = execute_query(query, df)
        return format_response(result, query)

    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm(messages=messages)
    return response.choices[0].message['content']

def get_audio_excel_chain(df):
    template = """
    You are a data analyst at a real estate company. You are interacting with a user who is asking you questions about the real estate data.
    Based on the data schema below, write a detailed Python Pandas query that would answer the user's question. Take the conversation history into account.
    
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
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    def get_schema(_):
        return df.columns.tolist()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def execute_query(query, df):
    try:
        local_vars = {"df": df}
        exec(f"result = {query}", {}, local_vars)
        return local_vars["result"]
    except Exception as e:
        return f"Error executing query: {query}\n{e}"

def format_response(result, query):
    if isinstance(result, pd.DataFrame):
        return f"The result of your query is:\n{result.to_string(index=False)}"
    elif isinstance(result, (int, float)):
        return f"The result of your query is: {result}"
    else:
        return f"The result of your query is: {result}"

def autoplay_audio(audio_file_path):
    audio_html = f"""
    <audio autoplay="true" controls>
    <source src="{audio_file_path}" type="audio/mp3" />
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
