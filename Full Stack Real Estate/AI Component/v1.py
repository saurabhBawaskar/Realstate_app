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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chat_with_excel():

    def init_data(file) -> pd.DataFrame:
        if file.name.endswith('.xlsx'):
            return pd.read_excel(file, engine='openpyxl')
        elif file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            raise ValueError("Unsupported file type")

    def get_excel_chain(df):
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
        
        # llm = ChatOpenAI(model="gpt-4o")
        llm = ChatOpenAI(model = "gpt-4o-mini")
        
        def get_schema(_):
            return df.columns.tolist()
        
        return (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm
            | StrOutputParser()
        )
        
    def get_response(user_query: str, df: pd.DataFrame, chat_history: list):
        excel_chain = get_excel_chain(df)
        
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
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
          AIMessage(content="Hello! I'm a real estate assistant. Ask me anything about your real estate data."),
        ]

    load_dotenv()

    st.title("Real Estate Chatbot 🤖")

    with st.sidebar:
        st.subheader("Upload the data")
        # st.write("This is a simple chat application using Excel data. Upload the Excel file and start chatting.")
        
        uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])
        
        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                df = init_data(uploaded_file)
                st.session_state.df = df
                st.success("Data loaded successfully!")
        
        if "processing_time" in st.session_state:
            st.write(f"Last processing time: {st.session_state.processing_time:.2f} seconds")
        
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Type a message...")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        if uploaded_file is not None:
            with st.chat_message("AI"):
                start_time = time.time()
                with st.spinner("Your query is being processed..."):  
                    response = get_response(user_query, st.session_state.df, st.session_state.chat_history)
                end_time = time.time()
                
                processing_time = end_time - start_time
                st.session_state.processing_time = processing_time
                
                st.markdown(response)
                    
                st.session_state.chat_history.append(AIMessage(content=response))
        else:
            st.warning("Please upload an Excel or CSV file first.")

if __name__ == "__main__":
    chat_with_excel()

