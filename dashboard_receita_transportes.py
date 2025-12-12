import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dashboard | Receita Transportes", layout="wide")
st.title("🚚 Dashboard de Receita — Transportes")

# =============================
# Utilitários de conversão
# =============================
def brl_to_float(x):
    """Converte 'R$ 57.375,00' -> 57375.00. Se já for número, mantém."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s in ["", "-", "nan", "None"]:
        return np.nan
    s = s.replace("R$", "").replace(" ", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def pct_to_float(x):
    """Converte '32,9%' -> 0.329. Se vier 32.9, vira 0.329. Se vier 0.329, mantém."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        v = float(x)
        return v / 100 if v > 1.5 else v
    s = str(x).strip().replace(" ", "")
    if s in ["", "-", "nan", "None"]:
        return np.nan
    s = s.replace("%", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        v = float(s)
        return v / 100 if v > 1.5 else v
    except Exception:
        return np.nan

def fmt_brl(v):
    if pd.isna(v):
        return "—"
    s = f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

# =============================
# Sidebar: upload
# =============================
st.sidebar.header("⚙️ Configurações")

SHEETS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQGkFzhy469J3SQo4xoY7tHEEopnAJLdKThEtsFIXPeaUqUMjXkCOdddDsT3r9CUK2Wsnl_c4lbYLy4/pub?output=csv"

try:
    df_raw = pd.read_csv(SHEETS_CSV_URL)
except Exception as e:
    st.error("Erro ao carregar dados do Google Sheets (CSV).")
    st.exception(e)
    st.stop()


# Espera 8 colunas A..H conforme sua estrutura
# A Dia | B Faturado | C Acumulado | D % Meta | E Projeção | F Diferença | G % Projeção | H Comissão
if df_raw.shape[1] < 8:
    st.error("A planilha precisa ter pelo menos 8 colunas (A até H) conforme a estrutura combinada.")
    st.stop()

df = df_raw.iloc[:, :8].copy()
df.columns = [
    "dia",
    "faturado_dia",
    "acumulado_mes",
    "percentual_meta",
    "projecao_mes",
    "diferenca_meta",
    "percentual_projecao",
    "comissao",
]

# =============================
# Tratamento do DIA (coluna A)
# =============================
dia_raw = df["dia"]

# Caso 1: se já é datetime
if pd.api.types.is_datetime64_any_dtype(dia_raw):
    df["dia"] = pd.to_datetime(dia_raw, errors="coerce").dt.day

else:
    # tenta numérico
    dia_num = pd.to_numeric(dia_raw, errors="coerce")

    # Caso 2: número grande (serial do Excel ou data numérica)
    if dia_num.notna().any() and dia_num.max() > 31:
        dtv = pd.to_datetime(dia_num, unit="D", origin="1899-12-30", errors="coerce")
        df["dia"] = dtv.dt.day
    else:
        # Caso 3: string tipo "01/dez"
        df["dia"] = dia_raw.astype(str).str.extract(r"(\d{1,2})")[0]
        df["dia"] = pd.to_numeric(df["dia"], errors="coerce")

df = df.dropna(subset=["dia"])
df["dia"] = df["dia"].astype(int)

# Se o "dia" colapsou (todo mundo virou o mesmo dia), cria dia sequencial pela ordem das linhas
dias_unicos = df["dia"].nunique(dropna=True)
if dias_unicos <= 1 and len(df) > 1:
    st.warning("⚠️ A coluna 'dia' veio repetida. Vou usar a ordem das linhas como dia 1..N.")
    df = df.sort_index().reset_index(drop=True)
    df["dia"] = np.arange(1, len(df) + 1)


# =============================
# Conversão de valores (R$ e %)
# =============================
for col in ["faturado_dia", "acumulado_mes", "projecao_mes", "diferenca_meta", "comissao"]:
    df[col] = df[col].apply(brl_to_float)

df["percentual_meta"] = df["percentual_meta"].apply(pct_to_float)
df["percentual_projecao"] = df["percentual_projecao"].apply(pct_to_float)

# Ordena por dia
df = df.sort_values("dia").reset_index(drop=True)

# =============================
# Filtros
# =============================
st.sidebar.subheader("Filtros")

dmin = int(df["dia"].min())
dmax = int(df["dia"].max())

if dmin == dmax:
    # Só existe um dia no dataset (min == max), então slider quebra.
    st.sidebar.info(f"Apenas um dia disponível: {dmin}")
    d0, d1 = dmin, dmax
else:
    d0, d1 = st.sidebar.slider(
        "Dia do mês",
        min_value=dmin,
        max_value=dmax,
        value=(dmin, dmax),
    )

df_f = df[(df["dia"] >= d0) & (df["dia"] <= d1)].copy()

if df_f.empty:
    st.warning("Nenhum dado no período selecionado.")
    st.stop()

# =============================
# KPIs (última linha com acumulado)
# =============================
df_kpi = df_f.dropna(subset=["acumulado_mes"]).copy()
if df_kpi.empty:
    st.error("Não encontrei valores em 'acumulado_mes' (coluna C). Verifique se há valores nessa coluna.")
    st.stop()

ultimo = df_kpi.iloc[-1]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💰 Faturamento do mês", fmt_brl(ultimo["acumulado_mes"]))
k2.metric("🎯 % da Meta", "—" if pd.isna(ultimo["percentual_meta"]) else f"{ultimo['percentual_meta']*100:.1f}%")
k3.metric("📈 Projeção do mês", fmt_brl(ultimo["projecao_mes"]))
k4.metric("📊 Diferença da Meta", fmt_brl(ultimo["diferenca_meta"]))
k5.metric("🤝 Comissão acumulada", fmt_brl(ultimo["comissao"]))

st.divider()

# =============================
# Gráficos
# =============================
g1, g2 = st.columns(2, gap="large")

with g1:
    st.subheader("🧾 Faturamento diário")
    fat = df_f.groupby("dia", as_index=False)["faturado_dia"].sum()
    fig1 = px.bar(
        fat,
        x="dia",
        y="faturado_dia",
        labels={"dia": "Dia do mês", "faturado_dia": "Faturado (R$)"},
    )
    st.plotly_chart(fig1, use_container_width=True)

with g2:
    st.subheader("📈 Acumulado no mês")
    ac = df_f.groupby("dia", as_index=False)["acumulado_mes"].max()
    fig2 = px.line(
        ac,
        x="dia",
        y="acumulado_mes",
        markers=True,
        labels={"dia": "Dia do mês", "acumulado_mes": "Acumulado (R$)"},
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

g3, g4 = st.columns(2, gap="large")

with g3:
    st.subheader("🚀 Projeção × Meta")
    fig3 = px.line(
        df_f,
        x="dia",
        y="projecao_mes",
        markers=True,
        labels={"dia": "Dia do mês", "projecao_mes": "Projeção (R$)"},
    )
    fig3.add_hline(y=3000000, line_dash="dash", annotation_text="Meta (3.000.000)")
    st.plotly_chart(fig3, use_container_width=True)

with g4:
    st.subheader("📊 % da Meta (acumulado)")
    fig4 = px.line(
        df_f,
        x="dia",
        y="percentual_meta",
        markers=True,
        labels={"dia": "Dia do mês", "percentual_meta": "% da meta (base 1)"},
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# =============================
# Tabela
# =============================
st.subheader("📋 Dados detalhados")
st.dataframe(df_f, use_container_width=True, height=420)
