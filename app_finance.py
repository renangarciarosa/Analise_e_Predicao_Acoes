from pickletools import optimize
import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly,plot_components_plotly
from plotly import graph_objects as go

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
st.set_page_config(
    page_title="Previsão de Ações",
    page_icon="📈",
    layout="wide"
)

DATA_INICIO = '2017-01-01'
DATA_FIM = date.today().strftime('%Y-%m-%d')

st.title('Análise de Ações')

#criando sidebar
st.sidebar.write('Escolha a ação')

n_dias = st.slider('Quantidade de dias de previsão', 30, 360)


def pegar_dados_acoes():
    path = 'C:/Users/RROSA20/Desktop/scripts_ML/StreamLit/Financas/acoes.csv'
    
    return pd.read_csv(path,delimiter=';')

df = pegar_dados_acoes()
acao = df['snome']
nome_acao_escolhida = st.sidebar.selectbox('Escolha uma Ação: ', acao)
df_acao = df[df['snome'] == nome_acao_escolhida] #pegando a ação escolhida pelo usuário
acao_escolhida = df_acao.iloc[0]['sigla_acao']
acao_escolhida = acao_escolhida + '.SA'   

@st.cache
def pegar_valores_online(sigla_acao):
    df = yf.download(sigla_acao, DATA_INICIO, DATA_FIM)
    df.reset_index(inplace=True)
    return df

df_valores = pegar_valores_online(acao_escolhida)

st.subheader('Tabela de Valores - ' + nome_acao_escolhida)
st.write(df_valores.tail(10))

#criar grafico
st.subheader('Gráfico de Preço')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_valores['Date'], y=df_valores['Open'], name="Preço Abertura", line_color='blue'))
fig.add_trace(go.Scatter(x=df_valores['Date'], y=df_valores['Close'], name="Preço Fechamento", line_color='yellow'))
st.plotly_chart(fig)

#Previsão
df_treino = df_valores[['Date', 'Close']]

#retomando colunas
df_treino = df_treino.rename(columns={"Date": "ds", "Close":"y"})

modelo = Prophet()
modelo.fit(df_treino)
futuro = modelo.make_future_dataframe(periods=n_dias,freq='B')

previsao = modelo.predict(futuro)

st.subheader('Previsão da Ação nos Próximos dias')
st.write(previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_dias))

#grafico 
grafico1 = plot_plotly(modelo, previsao)
st.plotly_chart(grafico1)

grafico2 = plot_components_plotly(modelo, previsao)
st.plotly_chart(grafico2)


