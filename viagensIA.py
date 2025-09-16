# bibliotecas
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process, LLM


# Adicione a SUA chave entre as aspas
os.environ["GOOGLE_API"] = ""  

# Configuração do modelo de linguagem
llm = LLM(model='gemini/gemini-2.0-flash-lite', 
    temperature=0.7,
    api_key = os.environ["GOOGLE_API"]
    )

# DEFINIÇÃO DOS AGENTES ESPECIALIZADOS

# Agente 1: Especialista em pesquisa de destinos
pesquisador_destinos = Agent(
    role="Especialista em Pesquisa de Destinos",
    goal="Identificar os melhores destinos de viagem com base no orçamento, interesses e preferências do usuário",
    backstory=(
        "Você é um viajante experiente que já visitou mais de 50 países. "
        "Sua expertise está em encontrar destinos que combinam perfeitamente com "
        "as preferências únicas de cada viajante, considerando orçamento, "
        "interesses e época do ano."
    ),
    verbose=True,
    allow_delegation=False,  # evitar loops
    llm=llm
)

# Agente 2: Especialista em planejamento orçamentário
especialista_orcamento = Agent(
    role="Especialista em Planejamento Orçamentário",
    goal="Criar um plano financeiro detalhado para a viagem, maximizando o valor gasto dentro do orçamento definido",
    backstory=(
        "Com formação em economia e anos de experiência em consultoria de viagens, "
        "você domina a arte de criar orçamentos realistas que permitem experiências "
        "incríveis sem gastos excessivos. Você conhece todos os truques para "
        "economizar em voos, hospedagem e atividades."
    ),
    verbose=True,
    allow_delegation=False, 
    llm=llm
)

# Agente 3: Criador de roteiros personalizados

planejador_roteiros = Agent(
    role="Criador de Roteiros Personalizados",
    goal="Desenvolver itinerários detalhados e atraentes que incluam atividades, atrações e experiências alinhadas aos interesses do viajante",
    backstory=(
        "Você é um storyteller nato com paixão por criar experiências de viagem memoráveis. "
        "Seus roteiros são conhecidos por equilibrar perfeição entre atividades culturais, "
        "aventura, gastronomia e momentos de descanso, sempre surpreendendo positivamente os viajantes."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# DEFINIÇÃO DAS TAREFAS A SEREM EXECUTADAS (COM MELHORIAS)

# Tarefa 1: Pesquisa de destinos
task_pesquisar_destinos = Task(
    description=(
        "Com base nas seguintes preferências do usuário:\n"
        "- Orçamento: {orcamento}\n"
        "- Interesses: {interesses}\n"
        "- Duração da viagem: {duracao}\n"
        "- Época do ano: {epoca}\n"
        "- Viajantes: {viajantes}\n\n"
        
        "Pesquise e recomende 3 destinos potenciais adequados. Para cada destino, forneça:\n"
        "1. Breve descrição e por que é adequado\n"
        "2. Clima e condições na época especificada\n"
        "3. Principais atrações alinhadas aos interesses\n"
        "4. Vantagens e considerações importantes"
    ),
    agent=pesquisador_destinos,
    expected_output=(
        "Uma lista de 3 destinos com descrições detalhadas de cada um, "
        "explicando por que são adequados para as preferências do usuário."
    ),
    context=[]  
)

# Tarefa 2: Planejamento orçamentário
# output_file para salvar resultado e usar como contexto
task_planejar_orcamento = Task(
    description=(
        "Para o destino selecionado {destino}, crie um orçamento detalhado considerando:\n"
        "- Orçamento total: {orcamento}\n"
        "- Duração: {duracao}\n"
        "- Número de viajantes: {viajantes}\n\n"
        
        "O orçamento deve incluir:\n"
        "1. Custos de transporte (voos, transfers, transporte local)\n"
        "2. Hospedagem (categoria adequada ao orçamento)\n"
        "3. Alimentação (refeições diárias)\n"
        "4. Atividades e ingressos\n"
        "5. Reserva para imprevistos\n"
        "6. Dicas para economizar em cada categoria"
    ),
    agent=especialista_orcamento,
    expected_output=(
        "Um orçamento detalhado dividido por categorias, com custos estimados "
        "e dicas para economizar dinheiro."
    ),
    # Contexto vazio pois depende apenas dos inputs do usuário
    context=[],
    # Poeria salvar em arquivo se necessário
)

# Tarefa 3: Criação de roteiro personalizado
task_criar_roteiro = Task(
    description=(
        "Crie um roteiro detalhado para {duracao} dias em {destino} considerando:\n"
        "- Interesses do viajante: {interesses}\n"
        "- Viajantes: {viajantes}\n"
        "- Época do ano: {epoca}\n\n"
        
        "O roteiro deve incluir:\n"
        "1. Itinerário diário com atividades matinais, tardes e noites\n"
        "2. Locais de alimentação recomendados\n"
        "3. Tempo de deslocamento entre atividades\n"
        "4. Dicas locais e melhores horários para visitar atrações\n"
        "5. Opções para diferentes condições climáticas\n"
        "6. Atividades alternativas caso precise ajustar o plano"
    ),
    agent=planejador_roteiros,
    expected_output=(
        "Um roteiro completo e detalhado com atividades para cada dia da viagem, "
        "incluindo horários, locais e recomendações específicas."
    ),
    # Contexto vazio pois depende apenas dos inputs do usuário
    context=[]
)

# CRIAÇÃO E CONFIGURAÇÃO DA EQUIPE (CREW)


# Criar a crew  com os agentes e tarefas definidas
crew = Crew(
    agents=[pesquisador_destinos, especialista_orcamento, planejador_roteiros],
    tasks=[task_pesquisar_destinos, task_planejar_orcamento, task_criar_roteiro],
    verbose=True,
    process=Process.sequential  # uma tarefa após a outra
)

# EXECUÇÃO DO SISTEMA MULTIAGENTE

# Dados de entrada para o processo de planejamento de viagem
inputs = {
    "orcamento": "R$ 5000 por pessoa",
    "interesses": "praia, cultura local, gastronomia, aventuras moderadas",
    "duracao": "7 dias",
    "epoca": "julho",
    "viajantes": "2 adultos",
    "destino": "Nordeste brasileiro"
}

# Executar a crew com as entradas do usuário
try:
    print("Iniciando processo de planejamento de viagem...")
    resultado = crew.kickoff(inputs=inputs)
    
    # Exibir o resultado final do processo
    print("\n\n" + "="*60)
    print("PLANO DE VIAGEM COMPLETO GERADO PELA EQUIPE DE AGENTES")
    print("="*60)
    print(resultado)
    
except Exception as e:
    print(f"Ocorreu um erro durante a execução: {e}")
    print("\nPossíveis soluções:")
    print("1. Verifique se a API key está correta e válida")
    print("2. Verifique se todos os agentes têm allow_delegation=False")
    print("3. Tente reduzir a complexidade das tarefas")
    print("4. Verifique se há recursos suficientes para executar os modelos")
