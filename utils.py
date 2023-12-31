import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer

PINECODE_API_KEY=st.secrets['PINECODE_API_KEY']
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

pinecone.init(api_key=PINECODE_API_KEY,  environment="us-west4-gcp-free")
index = pinecone.Index('fisbot')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=3, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']+result['matches'][2]['metadata']['text']


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

who_are_you = [
"kdo jsi", "jsi bot", "kdo jsi?", "jsi bot?", "kdo jis?", "kdo jseš?", "ty si nějaký bot", "ty si nějaký bot?", "ty jsi robot?", "jsi člověk?", "jsi clovek?", 
"jsi člověk", "jsi clovek", "jsi AI?", "AI?", "bot?", "co umíš", "co umíš?", "co umis", "co umis?",  "coumíš", "coumis", "coumíš?", "coumis?", "co umíš dělat?", "co všechno umíš?", 
"jaké máš dovedností?", "co umiš", "co umiš?", "co umís", "co umís?", "kdo jseš", "kdo jses?", "kdo jses", "jsi robot?", "jsi robot", "co vše umíš?", "co všechno umíš", "co vše umíš"
]
who_are_you_res = """Jsem chatbot, vytvořený Fakultou informatiky a statistiky. Moje hlavní úloha spočívá v tom, abych reagoval na otázky a na základě dostupných zdrojů poskytoval informace. 

V současné době pracuji s omezeným množstvím informací, avšak ty se neustále rozšiřují a zlepšují, abych mohl nabídnout co nejlepší pomoc.

Pokud máte zájem o informace týkající se programů na Fakultě informatiky a statistiky, jsem schopen vám poskytnout některé základní informace o dostupných programech a jejich charakteristikách.

Co se týče bodových limitů pro přijetí na fakultu, ty se každý rok mohou lišit v závislosti na mnoha faktorech. Poslední informace, které mám ohledně bodových hranic pro přijetí, se týkají přijímacího řízení na akademický rok 2023/2024.

Jako chatbot mohu reagovat pouze v češtině, přestože uživatelé mají možnost klást otázky i v jiných jazycích. Odpovědi však budou k dispozici pouze v českém jazyce. Na druhou stranu mám schopnost zpracovávat více než 100 jazyků, včetně slovenštiny nebo ukrajinštiny. 
Vždy však výstupy budou v češtině.
"""
