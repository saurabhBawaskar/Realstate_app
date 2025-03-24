# import openai
# import os
# import streamlit as st
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI

# # Ensure your OpenAI API key is set
# openai.api_key = os.getenv("OPENAI_API_KEY")

# def speech_to_text(audio_data):
#     with open(audio_data, "rb") as audio_file:
#         transcript = openai.Audio.transcribe(
#             model="whisper-1",
#             file=audio_file
#         )
#     return transcript["text"]

# def text_to_speech(input_text):
#     response = openai.Audio.create(
#         model="tts-1",
#         voice="nova",
#         input=input_text
#     )
#     webm_file_path = "temp_audio_play.mp3"
#     with open(webm_file_path, "wb") as f:
#         for chunk in response.iter_bytes():
#             f.write(chunk)
#     return webm_file_path

# def get_answer(messages, df):
#     # Custom logic to handle real estate data
#     if df is not None:
#         data_chain = get_audio_excel_chain(df)
#         response = data_chain.invoke({
#             "question": messages[-1]['content'],
#             "chat_history": messages[:-1]
#         })
#         return response

#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=messages
#     )
#     return response.choices[0]['message']['content']

# def get_audio_excel_chain(df):
#     template = """
#     You are a data analyst at a real estate company. You are interacting with a user who is asking you questions about the real estate data.
#     Based on the data schema below, write a Python Pandas query that would answer the user's question. Take the conversation history into account.
    
#     <SCHEMA>{schema}</SCHEMA>
    
#     Conversation History: {chat_history}
    
#     Write only the Pandas query and nothing else. Do not wrap the Pandas query in any other text, not even backticks.
    
#     For example:
#     Question: List the top 3 most expensive properties.
#     Pandas Query: df.nlargest(3, 'price')
#     Question: Show the details of properties in New York.
#     Pandas Query: df[df['city'] == 'New York']
    
#     Your turn:
    
#     Question: {question}
#     Pandas Query:
#     """
    
#     prompt = ChatPromptTemplate.from_template(template)
    
#     llm = ChatOpenAI(model="gpt-4o")
    
#     def get_schema(_):
#         return df.columns.tolist()
    
#     return (
#         RunnablePassthrough.assign(schema=get_schema)
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

# def autoplay_audio(audio_file_path):
#     audio_html = f"""
#     <audio autoplay="true">
#     <source src="{audio_file_path}" type="audio/mp3" />
#     </audio>
#     """
#     st.markdown(audio_html, unsafe_allow_html=True)








from openai import OpenAI
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st

client = OpenAI()

def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript["text"]

def text_to_speech(input_text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=input_text
    )
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
    return webm_file_path

def get_answer(messages, df):
    # Custom logic to handle real estate data
    if df is not None:
        data_chain = get_audio_excel_chain(df)
        response = data_chain.invoke({
            "question": messages[-1]['content'],
            "chat_history": messages[:-1]
        })
        return response

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message['content']

def get_audio_excel_chain(df):
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

def autoplay_audio(audio_file_path):
    audio_html = f"""
    <audio autoplay="true">
    <source src="{audio_file_path}" type="audio/mp3" />
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)