import os
import streamlit as st
import psycopg2
import redis
import numpy as np
from pydantic import BaseModel, Field
from utils.text_processing import processar_texto
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query



sql_developer_agent = Agent(
    role='Postgres analyst senior',
    goal="Sua função é fazer um gráfico por meio de código e retorna-lo",
    backstory ="""Você é um programador especialista em matplotlib e plotar gráficos por código em geral""",
    code_execution_mode="unsafe",
    allow_code_execution=True,
    allow_delegation=False,
    verbose=True,
)

sql_developer_task = Task(
    description=
    """Sua tarefa é fazer um gráfico utilizando code e matplotlib para as informações a seguir:
    {infos}
    E neste código salve o gráfico como png no caminho graph.png e retorne uma string luiz""",
    expected_output="""Um gráfico com as informações solicitadas""",
    agent=sql_developer_agent,
)

crew = Crew(
    agents=[sql_developer_agent],
    tasks=[sql_developer_task],
    verbose=True
)
infos = """O gráfico que compare os RDOs aprovados com o total de RDOs é baseado nos seguintes dados:
Total de RDOs: 131
Total de RDOs Aprovados: 103
Assim, o gráfico mostrará que a maioria dos RDOs foi aprovada. Se precisar de mais informações ou uma visualização específica, estou à disposição para ajudar!"""
result = crew.kickoff(inputs = {'infos': infos})