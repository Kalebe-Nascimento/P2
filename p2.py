
import streamlit as st
import pandas as pd
import pydeck as pdk
import pickle
import keras


scaler = pickle.load(open("scaler.pickle", 'rb'))
model = pickle.load(open("model.pickle", 'rb'))
#botandos os dados em valores, dispobivel em: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset


st.title('Dados do Recursos Humanos')


st.sidebar.header("Variáveis")

age = st.sidebar.slider('Idade', min_value=18, max_value=99)

gender = st.sidebar.selectbox('Genero', ['Feminino', 'Masculino'])

distance_from_home = st.sidebar.number_input('Distancia de casa', step=1, min_value=0)

education_field = st.sidebar.selectbox('Nivel conhecimento', ['Cientista Pesquisador','Outro','Medicina', 'Marketing', 'Curso Tecnico', 
  'Recursos Humanos'])

education = st.sidebar.selectbox('Grau escolaridade', ["Abaixo do E.M.","Ensino médio","Bacharel","Mestre","Doutor"])

environment_satisfaction = st.sidebar.selectbox('Satisfacao ambiente ', ["baixo", "medio", "alto", "muito alto"])

job_involvement = st.sidebar.selectbox('Satisfacao Envolvimento', ["baixo", "medio", "alto", "muito alto"])

job_satisfaction = st.sidebar.selectbox('Satisfacao Trabalho', ["baixo", "medio", "alto", "muito alto"])

relationship_satisfaction = st.sidebar.selectbox('Vida balanceada', ["baixo", "medio", "alto", "muito alto"])

work_life_balance = st.sidebar.selectbox('Vida balanceada', ["pouco","medio","bom","muito bom"])


dados = {
    
  'Idade' :[age],
  'Genero':[gender],
  'Distancia de casa':[distance_from_home],
  'Nivel conhecimento':[education_field],
  'Genero':[education],
  'Satisfacao ambiente':[environment_satisfaction],
  'Satisfacao Envolvimento':[job_involvement],
  'Satisfacao Trabalho':[job_satisfaction],
  'Vida balanceada':[relationship_satisfaction],
  'Vida balanceada':[work_life_balance],

}

rh_data = pd.DataFrame(dados)
resultado_Test = scaler.transform(rh_data)

predicaoSaida =  "Deixara o emprego" if model.predict(resultado_Test) > 0.51 else "Nao deixara o emprego"