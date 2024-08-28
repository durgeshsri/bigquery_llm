import os
import re
import streamlit as st
from langchain.agents import create_sql_agent
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.messages import AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain.sql_database import SQLDatabase
from langchain_community.utilities import SQLDatabase
from langchain.tools import Tool


#from few_shots import few_shots

#from dotenv import load_dotenv
#load_dotenv()  # take environment variables from .env (especially openai api key)


def get_few_shot_db_chain():
    service_account_file = r"C:\Users\Durgesh\Downloads\refined-signer-351817-7fe9f506b60f.json"
    os.environ["OPENAI_API_KEY"] = ("sk-ZuVmKn3gSrPRr9aPUJD-mqc7_y1UCxjPspugduol1oT3BlbkFJ1Gaof6CYZsQToNZ7tE2mTiCYxIDFNcDP5f6b8IvHQA")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
    model = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    project = "refined-signer-351817"
    dataset = "llm_dataset"
    sqlalchemy_url = (f"bigquery://{project}/{dataset}?credentials_path={service_account_file}")
    db = SQLDatabase.from_uri(sqlalchemy_url,sample_rows_in_table_info=3)
    from langchain.agents.agent_toolkits import SQLDatabaseToolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    agent_executor = create_sql_agent(llm=model,toolkit=toolkit,verbose=True,top_k=100)
    return agent_executor

