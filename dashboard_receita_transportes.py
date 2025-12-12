import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# =============================
# CONFIG (APENAS UMA VEZ)
# =============================
st.set_page_config(page_title="Dashboard | Receita Transportes", layout="wide")

st_autorefresh(interval=5_000, key="auto_refresh")


SHEETS_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQGkFzhy469J3SQo4xoY7tHEEopnAJLdKThEtsFIXPeaUqUMjXkCOdddDsT3r9CUK2Wsnl_c4lbYLy4/pub?output=csv"
META_MENSAL = 3_000_000.0


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

/* Botões da sidebar */
div.sidebar-btn button {
  width: 52px !important;
  height: 52px !important;
  border-radius: 16px !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  background: rgba(255,255,255,0.05) !important;
  font-size: 18px !important;
  line-height: 1 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}
div.sidebar-btn button:hover {
  border-color: rgba(180,255,0,0.55) !important;
  background: rgba(180,255,0,0.10) !important;
}

/* Cards (KPIs) */
.kpi-card {
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  border-radius: 18px;
  padding: 14px 16px;
  height: 104px;

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
  font-size: 26px;
  font-weight: 700;
  line-height: 1.1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.kpi-delta {
  font-size: 12px;
  line-height: 1;
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
    """'32,9%' -> 0.329 ; 32.9 -> 0.329 ; 0.329 -> 0.329"""
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
    if prev is None or pd.isna(prev) or prev == 0 or pd.isna(curr):
        return None
    return (curr - prev) / abs(prev)


def delta_text(delta):
    if delta is None:
        return "<span class='kpi-delta'>—</span>"
    sign = "+" if delta >= 0 else ""
    cls = "kpi-good" if delta >= 0 else "kpi-bad"
    return f"<span class='kpi-delta {cls}'>{sign}{delta*100:.1f}% desde a semana passada</span>"


# =============================
# Load data (Google Sheets CSV)
# =============================
@st.cache_data(ttl=60)
@st.cache_data(ttl=5)
def load_df_from_sheets(url: str) -> pd.DataFrame:

    df_raw = pd.read_csv(url)

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

    # DIA -> número (suporta datetime, serial Excel, "01/dez")
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

    df = df.dropna(subset=["dia"]).copy()
    df["dia"] = df["dia"].astype(int)

    # Converte valores BRL
    for col in ["faturado_dia", "acumulado_mes", "projecao_mes", "diferenca_meta", "comissao"]:
        df[col] = df[col].apply(brl_to_float)

    # Converte percentuais
    df["percentual_meta"] = df["percentual_meta"].apply(pct_to_float)
    df["percentual_projecao"] = df["percentual_projecao"].apply(pct_to_float)

    df = df.sort_values("dia").reset_index(drop=True)

    # Se colapsou tudo num mesmo dia (ruim), cria dia sequencial
    if df["dia"].nunique(dropna=True) <= 1 and len(df) > 1:
        df["dia"] = np.arange(1, len(df) + 1)

    return df


# =============================
# Sidebar: navegação por ícones
# =============================
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

st.sidebar.markdown("##")
st.sidebar.markdown("##")


def nav_btn(icon: str, key: str, tooltip: str):
    st.sidebar.markdown('<div class="sidebar-btn">', unsafe_allow_html=True)
    if st.sidebar.button(icon, key=key, help=tooltip):
        st.session_state.page = key
    st.sidebar.markdown("</div>", unsafe_allow_html=True)


nav_btn("⦿", "dashboard", "Dashboard (visão geral)")
nav_btn("👤", "motoristas", "Motoristas (em breve)")
nav_btn("🚚", "placas", "Placas (em breve)")
nav_btn("💬", "chat", "Chat (em breve)")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configurações")
show_debug = st.sidebar.checkbox("Mostrar Debug", value=False)


# =============================
# Header + Filtro no topo
# =============================
st.title("🚚 Dashboard de Receita — Transportes")

top_l, top_r = st.columns([0.70, 0.30], vertical_alignment="center")

with top_l:
    st.markdown("## Dashboard")

# Carrega dados (UMA VEZ)
try:
    df = load_df_from_sheets(SHEETS_CSV_URL)
except Exception as e:
    st.error("Não consegui carregar o Google Sheets (CSV).")
    st.write("Confere se o link termina com `pub?output=csv`.")
    st.exception(e)
    st.stop()

# Filtro no topo direito
with top_r:
    modo = st.selectbox("Período", ["Intervalo", "Dia único"], label_visibility="collapsed")

    dmin = int(df["dia"].min())
    dmax = int(df["dia"].max())

    if modo == "Dia único":
        if dmin == dmax:
            d0, d1 = dmin, dmax
        else:
            dia_sel = st.selectbox("Dia", list(range(dmin, dmax + 1)), index=(dmax - dmin), label_visibility="collapsed")
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
# Páginas (placeholder)
# =============================
if st.session_state.page != "dashboard":
    st.info("🚧 Esta aba está em construção. Por enquanto, a visão geral (Dashboard) está pronta.")
    if show_debug:
        st.write("Página atual:", st.session_state.page)
    st.stop()


# =============================
# KPIs (SEM DUPLICAR)
# =============================
# =============================
# KPIs – SEM depender do último dia
# =============================

# usa SEMPRE o maior acumulado disponível no período
fat_mes = (
    df_f["acumulado_mes"]
    .dropna()
    .max()
    if df_f["acumulado_mes"].notna().any()
    else np.nan
)

proj_mes = (
    df_f["projecao_mes"]
    .dropna()
    .max()
    if df_f["projecao_mes"].notna().any()
    else np.nan
)

com_acum = (
    df_f["comissao"]
    .dropna()
    .max()
    if df_f["comissao"].notna().any()
    else np.nan
)

dif_meta = (
    proj_mes - META_MENSAL
    if not pd.isna(proj_mes)
    else np.nan
)

pct_meta = (
    fat_mes / META_MENSAL
    if not pd.isna(fat_mes) and META_MENSAL
    else np.nan
)


# Deltas simples: últimos 7 dias vs 7 anteriores (pela coluna faturado_dia)
def last7_vs_prev7(series_col: str):
    tmp = df.dropna(subset=[series_col]).copy()
    if tmp.empty or len(tmp) < 14:
        return None
    tmp = tmp.sort_values("dia")
    last7 = tmp.tail(7)[series_col].sum()
    prev7 = tmp.iloc[-14:-7][series_col].sum()
    return compute_delta(last7, prev7)

delta_fat = last7_vs_prev7("faturado_dia")

# delta % meta (compara fat_mes atual vs fat_mes de 7 dias atrás se existir)
tmp_ac = df.dropna(subset=["acumulado_mes"]).sort_values("dia")
if len(tmp_ac) >= 8:
    prev_fat = tmp_ac.iloc[-8]["acumulado_mes"]
    delta_pct = compute_delta(pct_meta, (prev_fat / META_MENSAL) if META_MENSAL else None)
else:
    delta_pct = None

# delta projeção e comissão (último vs penúltimo)
tmp_proj = df.dropna(subset=["projecao_mes"]).sort_values("dia")
delta_proj = compute_delta(tmp_proj.iloc[-1]["projecao_mes"], tmp_proj.iloc[-2]["projecao_mes"]) if len(tmp_proj) >= 2 else None

tmp_com = df.dropna(subset=["comissao"]).sort_values("dia")
delta_com = compute_delta(tmp_com.iloc[-1]["comissao"], tmp_com.iloc[-2]["comissao"]) if len(tmp_com) >= 2 else None


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
          <div class="kpi-value">{fmt_brl(proj_mes)}</div>
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
          <div class="kpi-value">{fmt_brl(com_acum)}</div>
          {delta_text(delta_com)}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


# =============================
# Painéis: Donut + Linha
# =============================
p1, p2 = st.columns([0.42, 0.58], gap="large")

# Donut Meta
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

    atingido = 0.0 if pd.isna(fat_mes) else float(fat_mes)
    restante = max(0.0, float(META_MENSAL) - atingido)

    donut_df = pd.DataFrame({"status": ["Atingido", "Restante"], "valor": [atingido, restante]})

    fig_donut = px.pie(donut_df, names="status", values="valor", hole=0.70)
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    fig_donut.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.85)"),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# Linha Acumulado
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
# Painel extra: Faturamento diário (barra)
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
# Debug / Tabela (opcional)
# =============================
if show_debug:
    with st.expander("🔎 Debug (ver dados que o app está lendo)", expanded=False):
        st.write("df shape:", df.shape)
        st.write("df_f shape:", df_f.shape)
        st.write("colunas:", list(df.columns))
        st.dataframe(df.head(20), use_container_width=True)
        st.dataframe(df_f.head(20), use_container_width=True)

with st.expander("📋 Ver dados (tabela)", expanded=False):
    st.dataframe(df_f, use_container_width=True, height=420)
