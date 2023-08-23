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

llm = ChatOpenAI(model_name="gpt-3.5-turbo", 
openai_api_key=st.secrets['OPENAI_API_KEY'])


if "buffer_memry" not in st.session_state:
    st.session_state.buffer_memory=ConversationBufferWindowMemory (k=3,
    return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template("""
Odpovídej na otázky pouze v češtině a poskytuj pravdivé informace jen na základě Human Context.
Pokud neznáš odpověď, řekni 'Na tuto otázku nemám odpověď, prosím, přeformuluj ji'.
Pokud máš odkaz na stránku s podrobnějšími informacemi souvisejícími s uživatelským dotazem, uveď jej v odpovědi.
""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Zde můžeš zadat svůj dotaz: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)

            if query.lower() in who_are_you:  # kdo jsi, co umis
                response = who_are_you_res
            else:
                if len(st.session_state["responses"]) > 4: #Finding the closest matching questions based on a textual corpus will only be done if there are more than 4 responses in the conversation history
                    st.subheader("Zdá se, že chceš vědět:")
                    st.write(refined_query)

                context = find_match(refined_query)
                response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')


