import openai
import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer

openai.api_key=""
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key="259bd4db-6a71-4c28-9ea3-9c483300988b",  environment="us-west4-gcp-free")
index = pinecone.Index('fisbot')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Na základě uživatelského dotazu a konverzačního logu formuluj otázku, která by byla nejrelevantnější pro poskytnutí uživateli odpovědi z databáze znalostí.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
