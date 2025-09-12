from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process, LLM

import os 
# NÃO É PRA COPIAR ESSA CHAVE AQUI POIS ELA É DO PROFESSOR!!!!!
# USE A SUA!
os.environ["GOOGLE_API_KEY"] = "AIzaSyBLTS6pW-ggl7TqECjFRCNLsg-fFbaK_4c"

llm = LLM(model='gemini/gemini-2.0-flash-lite', verbose=True, temperature=0.4, api_key = os.environ["GOOGLE_API_KEY"])

#definiu os papeis de cada agente

acusacao = Agent(
    role="Advogado de Acusação",
    goal="Apresentar argumentos que responsabilizam Bolsonaro pelos crimes de 8 de janeiro.",
    backstory="Promotor público experiente e renomado.",
    verbose=True,
    llm=llm
)

defesa = Agent(
    role="Advogado de Defesa",
    goal="Defender Bolsonaro e argumentar que ele não teve responsabilidade nos atos do 8 de janeiro",
    backstory="Advogado experiente na defesa de figuras públicas politicas.",
    verbose=True,
    llm=llm
)

juiz = Agent(
    role="Juiz",
    goal="Analisar os dois pareceres e escrever uma decisão final",
    backstory="Magistrado com anos de experiência atuando no Superior Tribunal Federal.",
    verbose=True,
    llm=llm
)

tarefa_acusar = Task(
    description="Elabore um parecer acusatório responsabilizando Bolsonaro pelos atos antidemocráticos de 8 de janeiro",
    expected_output="Parecer com argumentos politicos, juridicos e sociais",
    agent=acusacao
)

tarefa_defender = Task(
    description="Redija uma defesa de Bolsonaro, argumentando que ele não teve envolvimento direto nos atos.",
    expected_output="Parecer com argumentos politicos, juridicos e sociais em defesa do réu",
    agent=defesa
)

tarefa_julgar = Task(
    description="Analise os dois pareceres e redija uma decisão final que aponte os argumentos e sua conclusão",
    expected_output="Decisão final coesa, baseada nos argumentos da acusação e defesa, decidindo se o réu é culpado ou inocente",
    agent=juiz
)

crew = Crew(
    agents=[defesa, acusacao, juiz],
    tasks=[tarefa_acusar, tarefa_defender, tarefa_julgar],
    process=Process.sequential,
    verbose=True
)

resultado = crew.kickoff()

print(resultado)