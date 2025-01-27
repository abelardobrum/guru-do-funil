import os
import streamlit as st

from transformers import GPT2TokenizerFast
from openai import OpenAI 
from dotenv import load_dotenv

from langchain_core.runnables import RunnableSequence
from langchain_community.vectorstores import FAISS 
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationSummaryBufferMemory

# Variaveis de ambiente
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
BASE_DE_DADOS = os.getenv('BASE_DE_DADOS')

def read_markdown(md_path):
    with open(md_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content
text = read_markdown(BASE_DE_DADOS)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", clean_up_tokenization_spaces=True)

def tokens(text: str) -> int:
     return len(tokenizer.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 24,
    length_function = tokens,
)

chunks = splitter.create_documents([text])

# Criar vetores
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

client = OpenAI(api_key=API_KEY)
llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=500)

# Prompt
template = """
# Objetivo 
Ensinar usuários leigos sobre Marketing Digital de maneira clara e acessível.  

# Função 
Você é o Guru do funil, um professor especializado em Marketing Digital. Seu objetivo é usar a base de dados fornecida como sua "apostila" para ensinar conceitos, estratégias e boas práticas de marketing digital de forma didática e envolvente.  

# Interação com Usuários  
- Use uma linguagem simples, clara e acessível, adaptada para quem não tem conhecimento prévio no assunto.  
- Mantenha um tom amigável, motivador e paciente, como um professor que realmente quer que seus alunos entendam o tema.  
- Explique os conceitos de maneira prática, utilizando exemplos fáceis de entender.  
- Se o usuário fizer perguntas vagas ou amplas, pergunte gentilmente sobre o que ele gostaria de saber mais especificamente.  
- Resuma pontos complexos em tópicos sempre que possível, para facilitar a compreensão.  
- Ofereça dicas práticas sempre que for relevante.  

# Base de Dados  
Você buscará informações na **Base de dados** fornecido e responderá com base nesse conteúdo.  
- Considere a base de dados como sua "apostila", garantindo que suas explicações sejam 100% alinhadas com ela.  
- Todas as respostas devem ser coerentes e consistentes com o que está na apostila.  

Base de dados: {BASE_DE_DADOS}

# Resumo das Interações  
Esse é o resumo das últimas perguntas que o usuário fez nesta sessão:  
`{historico}`  

- Utilize o resumo das interações para conectar os conceitos explicados e oferecer respostas mais detalhadas ou complementares sem repetição desnecessária.  
- Se o usuário pedir mais informações sobre algo, sempre leve em consideração as últimas perguntas no resumo.  
- Se você não encontrar informações no material, informe o usuário e pergunte como pode ajudar mais.  

# Regras  
1. **Use SOMENTE o conteúdo da apostila** como referência para ensinar.  
2. **Não invente informações.** Responda apenas com base no conteúdo fornecido.  
3. Priorize clareza e praticidade na explicação.  
4. Utilize exemplos do cotidiano para explicar termos técnicos, como SEO, ROI ou PPC.  
5. Caso não encontre informações sobre o que foi perguntado, deixe claro que não encontrou e pergunte se o usuário deseja aprender algo diferente.  
6. Evite respostas genéricas e sempre foque no ponto principal da dúvida do usuário.  
7. Responda sempre em português, com atenção a gramática e tom educacional.  

# Lembre-se  
- Seu papel é o de um professor paciente, envolvente e focado em transmitir conhecimento.  
- Garanta que o usuário entenda as informações, mesmo que precise simplificar ainda mais a explicação.  
- Sempre dê exemplos práticos e didáticos.  
- Nunca mencione que você é uma IA ou que usa modelos da OpenAI.  

# Estrutura das Respostas  
1. Comece introduzindo o conceito ou respondendo diretamente à pergunta.  
2. Explique o assunto passo a passo, com tópicos claros, se necessário.  
3. Ofereça exemplos reais ou simplificados.  
4. Conclua resumindo o principal e oferecendo dicas práticas, se aplicável.  

# **Pergunta Atual do Usuário**  
`{message}`  

**Resumo das Interações**: `{historico}`  

**IMPORTANTE**: Caso não encontre informações sobre o tema, informe o usuário de forma clara e pergunte se ele deseja explorar outro tópico de Marketing Digital.  
"""

prompt = PromptTemplate(
    input_variables=["message","historico", "BASE_DE_DADOS"],
    template=template
)

chain = RunnableSequence(prompt | llm)


# Definir a similaridade da resposta
def similar_question(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

# Histórico da conversa
history = ConversationSummaryBufferMemory(llm=llm, max_token_limit=300)
def similar_history(message, response):
    history.save_context({"message": message}, {"response": response})
    return history.load_memory_variables({})

# Gerar respostas
def generate_response(message, memory_variables):
    response = chain.invoke({"message": message, 
                             "BASE_DE_DADOS": similar_question(message), 
                             "historico": memory_variables.get("history", "")})
    similar_history(message, response.content)
    return response.content


#Streamlit (Exec: streamlit run "arquivo.py")
def main():
    st.set_page_config(
        page_title="Guru do Funil",
    )
    
    st.header("Guru do Funil")
    
    if "history" not in st.session_state:
        st.session_state.history = []

    if not st.session_state.history:
        st.session_state.history.append({
            "role": "assistant", 
            "content": "Olá! Eu sou o Guru do Funil, seu professor de Marketing Digital, com o que posso te ajudar hoje?"
        })
    
    for chat in st.session_state.history:
        if chat["role"] == "user":
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; align-items: center; margin: 10px 0;">
                    <div style="background-color:#ADD8E6; padding: 10px; border-radius: 10px; max-width: 70%;">
                        {chat['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            with st.chat_message("assistant"):
                st.markdown(chat['content'])
                
    message = st.chat_input("Faça qualquer pergunta sobre Marketing Digital")
    
    if message:
        st.session_state.history.append({"role": "user", "content": message})
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; margin: 10px 0;">
                <div style="background-color:#ADD8E6; padding: 10px; border-radius: 10px; max-width: 70%;">
                    {message}
                </div>
            </div>
            """, unsafe_allow_html=True
        )

        with st.spinner('Digitando...'):
            memory_variables = history.load_memory_variables({})
            
            result = generate_response(message, memory_variables)
            
            st.session_state.history.append({"role": "assistant", "content": result})
            
            with st.chat_message("assistant"):
                st.markdown(result)

            history.save_context({"message": message}, {"response": result})

if __name__ == '__main__':
    main()