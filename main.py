#pip install streamlit --upgrade
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ( 
    ChatPromptTemplate, 
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
) 
import streamlit as st
from streamlit_chat import message
from utils import *

st.set_page_config(page_title="FISBot - Natalie Musilová", page_icon="🤖", initial_sidebar_state="collapsed")

# Text 
title = '<p style="font-family:sans-serif; color: #00957d; font-size: 42px; font-weight: bold;">FISBot</p>'
# Image
img = "https://raw.githubusercontent.com/NatalieMusilova/FISBot/main/imgs/FIS_1_logo_rgb.png" 

# Create columns
col1, col2 = st.columns(2)
col1, col2 = st.columns([4,1]) # 4:1 width ratio
# Display content 
with col1:
   st.markdown(title, unsafe_allow_html=True)
with col2:
   st.image(img, width=130)


st.markdown("""
<div style="background-color: #00957d; text-align: center;">
  <p style="color: white; font-family: verdana; font-size: 17px; position: relative;">
    Chatbot Fakulty informatiky a statistiky Vysoké školy ekonomické v Praze
  </p>
</div>
""", unsafe_allow_html=True)



# Initialize chat history
if "responses" not in st.session_state:
    st.session_state["responses"] = ["Zdravím! Jak mohu pomoct?"]

if "requests" not in st.session_state:
    st.session_state["requests"] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", 
openai_api_key=st.secrets['OPENAI_API_KEY'])


if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=2,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template("""
Odpovídej na otázky pouze v češtině a poskytuj pouze pravdivé informace jen na základě Context.
Pokud jsou v dotazu vulgární slova, řekni 'Sprostá slova na akademickou půdu nepatří, zkus to ještě jednou a slušně.'.
Pokud neznáš odpověď, řekni 'Na tuto otázku nemám odpověď, prosím, přeformuluj ji. Nebo napiš email na adresu fis@vse.cz.'.
Pokud máš odkaz na stránku s podrobnějšími informacemi souvisejícími s uživatelským dotazem, uveď jej v odpovědi.
""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


def on_message_change():
    if st.session_state.input:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            if st.session_state.input.lower() in who_are_you:  # kdo jsi, co umis
                response = who_are_you_res
            else:
                context = find_match(st.session_state.input)
                response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{st.session_state.input}")

    st.session_state.requests.append(st.session_state.input)
    st.session_state.responses.append(response)
    st.session_state.input = ""


with textcontainer:
    query = st.text_input("Zde můžeš zadat svůj dotaz: ", key="input", on_change=on_message_change)

with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
