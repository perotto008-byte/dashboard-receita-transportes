import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Dashboard | Receita Transportes", layout="wide")

SHEETS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQGkFzhy469J3SQo4xoY7tHEEopnAJLdKThEtsFIXPeaUqUMjXkCOdddDsT3r9CUK2Wsnl_c4lbYLy4/pub?output=csv"
META_MENSAL_PADRAO = 3_000_000

# =============================
# CSS (layout parecido com seu mockup)
# =============================
st.markdown(
    """
<style>
/* Fundo com grid suave */
.stApp {
  background:
    radial-gradient(circle at 85% 20%, rgba(180,255,0,0.10), transparent 40%),
    radial-gradient(circle at 15% 80%, rgba(180,255,0,0.06), transparent 45%),
    linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px),
    #0b0f12;
  background-size: auto, auto, 48px 48px, 48px 48px, auto;
  background-position: center, center, top left, top left, center;
}

/* Sidebar "dock" */
section[data-testid="stSidebar"] {
  background: rgba(10,14,17,0.65);
  border-right: 1px solid rgba(255,255,255,0.08);
}

/* Botões da sidebar mais “ícone” */
div.sidebar-btn button {
  width: 52px !important;
  height: 52px !important;

  border-radius: 16px !important;
  border: 1px solid rgba(255,255,255,0.16) !important;

  background: rgba(255,255,255,0.05) !important;

  font-size: 18px !important;   /* 👈 padronizado */
  line-height: 1 !important;

  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

div.sidebar-btn button:hover {
  border-color: rgba(180,255,0,0.55) !important;
  background: rgba(180,255,0,0.10) !important;
}

/* Cards (métricas) */
.kpi-card {
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  border-radius: 18px;
  padding: 14px 16px;
  height: 100px;

  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 6px;

  overflow: hidden;
}

.kpi-title {
  color: rgba(255,255,255,0.75);
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
}

.kpi-value {
  color: rgba(255,255,255,0.95);
  font-size: 26px;          /* 👈 reduzido */
  font-weight: 700;
  line-height: 1.1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;  /* 👈 evita quebrar card */
}

.kpi-delta {
  font-size: 12px;
  line-height: 1;
}

.kpi-delta {
  font-size: 12px;
  color: rgba(255,255,255,0.65);
}
.kpi-good { color: rgba(140,255,160,0.95); }
.kpi-bad  { color: rgba(255,120,120,0.95); }

/* Painéis */
.panel {
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  border-radius: 18px;
  padding: 14px 16px 10px 16px;
}
.panel h3 {
  margin: 0 0 8px 0;
  font-size: 18px;
  color: rgba(255,255,255,0.92);
}
.small-muted {
  color: rgba(255,255,255,0.55);
  font-size: 12px;
}

/* Esconde header do Streamlit */
header[data-testid="stHeader"] { background: transparent; }

/* ===============================
   PADRONIZA ÍCONES DOS TÍTULOS
================================ */
h3 span.icon,
.kpi-title span.icon {
  font-size: 16px;
  margin-right: 6px;
  vertical-align: middle;
}

</style>
""",
    unsafe_allow_html=True,
)

# =============================
# Helpers: conversões BR / %
# =============================
def brl_to_float(x):
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

def compute_delta(curr, prev):
    """Retorna delta percentual (ex.: +0.12 = +12%)."""
    if prev is None or pd.isna(prev) or prev == 0 or pd.isna(curr):
        return None
    return (curr - prev) / abs(prev)

# =============================
# Load data (Google Sheets CSV)
# =============================
@st.cache_data(ttl=60)
def load_df_from_sheets(url: str) -> pd.DataFrame:
    df_raw = pd.read_csv(url)

    # Espera 8 colunas A..H, mas aceita se tiver mais
    if df_raw.shape[1] < 8:
        raise ValueError("CSV precisa ter pelo menos 8 colunas (A até H).")

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

    # DIA: extrair número do dia
    dia_raw = df["dia"]

    if pd.api.types.is_datetime64_any_dtype(dia_raw):
        df["dia"] = pd.to_datetime(dia_raw, errors="coerce").dt.day
    else:
        dia_num = pd.to_numeric(dia_raw, errors="coerce")
        if dia_num.notna().any() and dia_num.max() > 31:
            dtv = pd.to_datetime(dia_num, unit="D", origin="1899-12-30", errors="coerce")
            df["dia"] = dtv.dt.day
        else:
            df["dia"] = dia_raw.astype(str).str.extract(r"(\d{1,2})")[0]
            df["dia"] = pd.to_numeric(df["dia"], errors="coerce")

    df = df.dropna(subset=["dia"])
    df["dia"] = df["dia"].astype(int)

    # Valores
    for col in ["faturado_dia", "acumulado_mes", "projecao_mes", "diferenca_meta", "comissao"]:
        df[col] = df[col].apply(brl_to_float)

    df["percentual_meta"] = df["percentual_meta"].apply(pct_to_float)
    df["percentual_projecao"] = df["percentual_projecao"].apply(pct_to_float)

    df = df.sort_values("dia").reset_index(drop=True)

    # Se "dia" colapsou (ex.: tudo = 20), usa dia sequencial 1..N
    if df["dia"].nunique(dropna=True) <= 1 and len(df) > 1:
        df["dia"] = np.arange(1, len(df) + 1)

    return df

# =============================
# Sidebar: navegação por ícones
# =============================
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

st.sidebar.markdown("##")
st.sidebar.markdown("##")  # espaço

def nav_btn(icon: str, key: str, tooltip: str):
    with st.sidebar:
        st.markdown('<div class="sidebar-btn">', unsafe_allow_html=True)
        if st.button(icon, key=key, help=tooltip):
            st.session_state.page = key
        st.markdown("</div>", unsafe_allow_html=True)

nav_btn("⦿", "dashboard", "Dashboard (visão geral)")
nav_btn("👤", "motoristas", "Motoristas (em breve)")
nav_btn("🚚", "placas", "Placas (em breve)")
nav_btn("💬", "chat", "Chat (em breve)")

# =============================
# Topo: título + seletor de período
# =============================
top_l, top_r = st.columns([0.7, 0.3], vertical_alignment="center")

with top_l:
    st.markdown("## Dashboard")

# Carrega dados
try:
    df = load_df_from_sheets(SHEETS_CSV_URL)
except Exception as e:
    st.error("Não consegui carregar o Google Sheets (CSV).")
    st.write("Confere se o link termina com `pub?output=csv`.")
    st.exception(e)
    st.stop()

# Filtro de dia (modo intervalo ou dia único)
with top_r:
    # “Pill” no canto direito
    st.markdown(
        """
        <div style="display:flex; justify-content:flex-end; gap:10px; align-items:center;">
        </div>
        """,
        unsafe_allow_html=True,
    )

    modo = st.selectbox("Período", ["Intervalo", "Dia único"], label_visibility="collapsed")
    dmin = int(df["dia"].min())
    dmax = int(df["dia"].max())

    if modo == "Dia único":
        if dmin == dmax:
            dia_sel = dmin
        else:
            dia_sel = st.selectbox("Dia", list(range(dmin, dmax + 1)), index=len(list(range(dmin, dmax + 1))) - 1, label_visibility="collapsed")
        d0, d1 = dia_sel, dia_sel
    else:
        if dmin == dmax:
            d0, d1 = dmin, dmax
        else:
            d0, d1 = st.slider("Dia do mês", min_value=dmin, max_value=dmax, value=(dmin, dmax), label_visibility="collapsed")

df_f = df[(df["dia"] >= d0) & (df["dia"] <= d1)].copy()
if df_f.empty:
    st.warning("Sem dados no período selecionado.")
    st.stop()

# =============================
# Páginas (por enquanto só dashboard)
# =============================
if st.session_state.page != "dashboard":
    st.info("🚧 Esta aba está em construção. Por enquanto, refizemos a página principal (Dashboard).")
    st.stop()

# =============================
# KPIs (4 cards)
# =============================
# Último dia válido no recorte
df_kpi = df_f.dropna(subset=["acumulado_mes"]).copy()
if df_kpi.empty:
    st.error("Não encontrei valores em 'acumulado_mes' no período selecionado.")
    st.stop()
ultimo = df_kpi.iloc[-1]

meta = META_MENSAL_PADRAO
fat_mes = ultimo["acumulado_mes"]
pct_meta = (fat_mes / meta) if (not pd.isna(fat_mes) and meta > 0) else np.nan
proj = ultimo["projecao_mes"]
diff = ultimo["diferenca_meta"]
com = ultimo["comissao"]

# Delta (comparação simples: últimos 7 dias vs 7 anteriores dentro do dataset total)
def last7_vs_prev7(series_col: str):
    tmp = df.dropna(subset=[series_col]).copy()
    if tmp.empty or tmp["dia"].nunique() < 7:
        return None
    tmp = tmp.sort_values("dia")
    last7 = tmp.tail(7)[series_col].sum()
    prev7 = tmp.iloc[max(0, len(tmp)-14):max(0, len(tmp)-7)][series_col].sum()
    return compute_delta(last7, prev7)

delta_fat = last7_vs_prev7("faturado_dia")
delta_proj = compute_delta(proj, df.dropna(subset=["projecao_mes"]).tail(2)["projecao_mes"].head(1).values[0]) if df.dropna(subset=["projecao_mes"]).shape[0] >= 2 else None
delta_com  = compute_delta(com, df.dropna(subset=["comissao"]).tail(2)["comissao"].head(1).values[0]) if df.dropna(subset=["comissao"]).shape[0] >= 2 else None
delta_pct  = compute_delta(pct_meta, (df.dropna(subset=["acumulado_mes"]).tail(2)["acumulado_mes"].head(1).values[0] / meta) if df.dropna(subset=["acumulado_mes"]).shape[0] >= 2 else None)

def delta_text(delta):
    if delta is None:
        return "<span class='kpi-delta'>—</span>"
    sign = "+" if delta >= 0 else ""
    cls = "kpi-good" if delta >= 0 else "kpi-bad"
    return f"<span class='kpi-delta {cls}'>{sign}{delta*100:.1f}% desde a semana passada</span>"

c1, c2, c3, c4 = st.columns(4, gap="large")

with c1:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">Faturamento (mês)</div>
          <div class="kpi-value">{fmt_brl(fat_mes)}</div>
          {delta_text(delta_fat)}
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    pct_show = "—" if pd.isna(pct_meta) else f"{pct_meta*100:.1f}%"
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">% da Meta</div>
          <div class="kpi-value">{pct_show}</div>
          {delta_text(delta_pct)}
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">Projeção (mês)</div>
          <div class="kpi-value">{fmt_brl(proj)}</div>
          {delta_text(delta_proj)}
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">Comissão (acumulada)</div>
          <div class="kpi-value">{fmt_brl(com)}</div>
          {delta_text(delta_com)}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# =============================
# Painéis inferiores (donut + linha)
# =============================
p1, p2 = st.columns([0.42, 0.58], gap="large")

# Donut: meta atingida x restante (com base no acumulado)
with p1:
    st.markdown(
        """
        <div class="panel">
          <h3>📌 Meta (atingido x restante)</h3>
          <div class="small-muted">Baseado no acumulado do mês no último dia do período selecionado.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # container para o gráfico dentro do painel
    panel_placeholder = st.container()

    atingido = float(fat_mes) if not pd.isna(fat_mes) else 0.0
    restante = max(0.0, float(meta) - atingido)

    donut_df = pd.DataFrame(
        {"status": ["Atingido", "Restante"], "valor": [atingido, restante]}
    )

    fig_donut = px.pie(
        donut_df,
        names="status",
        values="valor",
        hole=0.70,
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    fig_donut.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.85)"),
    )
    panel_placeholder.plotly_chart(fig_donut, use_container_width=True)

# Linha: acumulado no mês por dia (max por dia)
with p2:
    st.markdown(
        """
        <div class="panel">
          <h3>📈 Evolução (acumulado no mês)</h3>
          <div class="small-muted">Linha do acumulado por dia (pega o maior valor do dia).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ac = df_f.groupby("dia", as_index=False)["acumulado_mes"].max()

    fig_line = px.line(
        ac,
        x="dia",
        y="acumulado_mes",
        markers=True,
        labels={"dia": "Dia do mês", "acumulado_mes": "Acumulado (R$)"},
    )
    fig_line.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.85)"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
    )
    st.plotly_chart(fig_line, use_container_width=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# =============================
# Extra: faturamento diário (bar) embaixo (opcional, mas útil)
# =============================
st.markdown(
    """
    <div class="panel">
      <h3>🧾 Faturamento diário</h3>
      <div class="small-muted">Soma do faturado por dia no período.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
fat = df_f.groupby("dia", as_index=False)["faturado_dia"].sum()
fig_bar = px.bar(
    fat,
    x="dia",
    y="faturado_dia",
    labels={"dia": "Dia do mês", "faturado_dia": "Faturado (R$)"},
)
fig_bar.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(255,255,255,0.85)"),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
)
st.plotly_chart(fig_bar, use_container_width=True)

# =============================
# Rodapé: tabela (debug/controle)
# =============================
with st.expander("📋 Ver dados (tabela)", expanded=False):
    st.dataframe(df_f, use_container_width=True, height=380)
