from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process, LLM

import os

#NÃO COPIAR A CHAVE DO PROFESSOR E SIM USAR A SUA!!!!!
os.environ["GOOGLE_API"] = "AIzaSyBLTS6pW-ggl7TqECjFRCNLsg-fFbaK_4c"

llm = LLM(model='gemini/gemini-2.0-flash-lite', 
    verbose=True, 
    temperature=0.4,
    api_key = os.environ["GOOGLE_API"]
    )

pesquisador = Agent(
    role="Pesquisador Acadêmico",
    goal="Investigar profundamente um tema proposto", 
    backstory="Especialista em pesquisas educacionais e inovação tecnologica",
    verbose=True,
    llm=llm
)

escritor = Agent(
    role="Redator Técnico",
    goal="Escrever um texto claro e coerente com base na pesquisa realizada",
    backstory="Profissional da área de comunicação com foco em TI",
    verbose=True,
    llm=llm
)

revisor = Agent(
    role="Revisor de textos",
    goal="Corrigir erros e melhorar a clareza e a fluidez do conteúdo",
    backstory="Experiente em revisão e padronização de textos",
    verbose=True,
    llm=llm
)

tarefa_pesquisa = Task(
    description = "Pesquisar os principais impactos da Inteligencia Artificial na educação brasileira",
    expected_output = "Um resumo com pelo menos 3 impactos relevantes escrito em PT-BR",
    agent = pesquisador
)

tarefa_redacao = Task(
    description = "Com base na pesquisa anterior, rescreva um artigo sobre o tema",
    expected_output = "Artigo técnico com introdução, desenvolvimento e conclusão em PT-BR",
    agent = escritor
)

tarefa_revisao = Task(
    description = "Revisar o artigo, corrigir erros e implementar melhorias",
    expected_output = "Artigo revisado, coeso e bem estruturado em PT-BR",
    agent = revisor
)

crew = Crew(
    agents = [pesquisador, escritor, revisor],
    tasks = [tarefa_pesquisa, tarefa_redacao, tarefa_revisao],
    process = Process.sequential,
    verbose = True
)

resultado = crew.kickoff()

print(resultado)