import os
import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================================================
# CONFIG (SÓ UMA VEZ!)
# =========================================================
st.set_page_config(page_title="Dashboard | Receita Transportes", layout="wide")

META_MENSAL = 3_000_000.0
META_MOTORISTA_PADRAO = 100_000.0
DIAS_MES_PADRAO = 30

# Refresh / Cache
CACHE_TTL_SECONDS = 30

# =========================================================
# URLs (Google Sheets)
# =========================================================
SHEETS_CSV_URL_RECEITA = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQGkFzhy469J3SQo4xoY7tHEEopnAJLdKThEtsFIXPeaUqUMjXkCOdddDsT3r9CUK2Wsnl_c4lbYLy4"
    "/pub?output=csv"
)

SHEET_ID_MOTORISTAS = "1u4b4XrSLjkxRYbsc-Ytro6cTAOZRq0HoSROhpWtdKVw"
GID_MOTORISTAS = "114753046"
CSV_URL_MOTORISTAS = f"https://docs.google.com/spreadsheets/d/{SHEET_ID_MOTORISTAS}/export?format=csv&gid={GID_MOTORISTAS}"

# =========================================================
# CSS (layout mockup)
# =========================================================
st.markdown(
    """
<style>
/* Fundo */
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

/* Remove header padrão */
header[data-testid="stHeader"] { background: transparent; }

/* Sidebar estilo “dock” */
section[data-testid="stSidebar"] {
  background: rgba(10,14,17,0.65);
  border-right: 1px solid rgba(255,255,255,0.08);
}

/* Botões ícones sidebar */
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

/* Cards KPI */
.kpi-card{
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  border-radius: 18px;
  padding: 14px 16px;
  height: 112px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 6px;
}
.kpi-title{
  color: rgba(255,255,255,0.75);
  font-size: 13px;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.kpi-value{
  color: rgba(255,255,255,0.96);
  font-weight: 800;
  line-height: 1.05;
  font-size: clamp(18px, 2.3vw, 30px);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.kpi-delta{
  font-size: 12px;
  line-height: 1.1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: rgba(255,255,255,0.70);
}
.kpi-good{ color: rgba(140,255,160,0.95) !important; }
.kpi-bad { color: rgba(255,120,120,0.95) !important; }

/* Panels */
.panel{
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  border-radius: 18px;
  padding: 14px 16px 10px 16px;
}
.panel h3{
  margin: 0 0 8px 0;
  font-size: 18px;
  color: rgba(255,255,255,0.92);
}
.small-muted{
  color: rgba(255,255,255,0.55);
  font-size: 12px;
}

/* Texto IA (arrumado p/ não “esmagar”) */
.ai-text {
  font-size: 15px;
  line-height: 1.65;
  color: rgba(255,255,255,0.95);
  white-space: normal;
  word-wrap: break-word;
  overflow-wrap: anywhere;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def brl_to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s in ["", "-", "nan", "None", "R$ -"]:
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
    if v is None or pd.isna(v):
        return "—"
    s = f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

def fmt_brl_short(v):
    if v is None or pd.isna(v):
        return "—"
    v = float(v)
    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"R$ {v/1_000_000:.2f} mi".replace(".", ",")
    if abs_v >= 1_000:
        return f"R$ {v/1_000:.0f} mil"
    return fmt_brl(v)

def compute_delta(curr, prev):
    if prev is None or pd.isna(prev) or prev == 0 or curr is None or pd.isna(curr):
        return None
    return (curr - prev) / abs(prev)

def delta_text(delta, suffix=""):
    if delta is None:
        return "<span class='kpi-delta'>—</span>"
    sign = "+" if delta >= 0 else ""
    cls = "kpi-good" if delta >= 0 else "kpi-bad"
    return f"<span class='kpi-delta {cls}'>{sign}{delta*100:.1f}%{suffix}</span>"

def extrair_dia_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = re.search(r"(\d{1,2})", s)
    if not m:
        return np.nan
    d = int(m.group(1))
    return d if 1 <= d <= 31 else np.nan

# =========================================================
# OpenAI (IA real) — robusto, sem quebrar se não tiver chave
# =========================================================
def get_openai_key():
    # tenta secrets, senão variável de ambiente
    try:
        k = st.secrets.get("OPENAI_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")

def gerar_comentario_por_ia(contexto: dict) -> str:
    api_key = get_openai_key()
    if not api_key:
        return "⚠️ IA não configurada (sem OPENAI_API_KEY)."

    try:
        from openai import OpenAI
    except Exception:
        return "⚠️ Falta instalar a lib OpenAI. Rode: python -m pip install openai"

    client = OpenAI(api_key=api_key)

    prompt = f"""
Você é um analista de performance de transportes (linguagem executiva, direta e prática).
Gere um comentário ÚNICO e útil (3 a 6 linhas) baseado apenas nos dados.

Regras:
- Não invente números.
- Diga o que está bom, o que preocupa e uma ação prática.
- Se estiver abaixo da meta: diga quanto precisa por dia no restante do mês (use 30 dias).
- Se acima: diga como manter consistência.
- No máximo 1 emoji.

DADOS:
{contexto}
""".strip()

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.4,
        max_output_tokens=220,
    )
    return resp.output_text.strip()

# =========================================================
# LOADERS (robustos)
# =========================================================
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_receita_robusto(url: str) -> pd.DataFrame:
    url_cb = f"{url}&_cb={int(time.time())}" if "_cb=" not in url else url
    raw = pd.read_csv(url_cb, header=None)

    if raw.shape[0] < 5 or raw.shape[1] < 8:
        raise ValueError("CSV de Receita: esperado A..H e dados a partir da linha 4.")

    data = raw.iloc[3:, :8].copy()
    data.columns = [
        "dia",
        "faturado_dia",
        "acumulado_mes",
        "percentual_meta",
        "projecao_mes",
        "diferenca_meta",
        "percentual_projecao",
        "comissao",
    ]

    data["dia"] = data["dia"].apply(extrair_dia_num)
    data["dia"] = pd.to_numeric(data["dia"], errors="coerce")
    data = data.dropna(subset=["dia"]).copy()
    data["dia"] = data["dia"].astype(int)

    for col in ["faturado_dia", "acumulado_mes", "projecao_mes", "diferenca_meta", "comissao"]:
        data[col] = data[col].apply(brl_to_float)

    data["percentual_meta"] = data["percentual_meta"].apply(pct_to_float)
    data["percentual_projecao"] = data["percentual_projecao"].apply(pct_to_float)

    return data.sort_values("dia").reset_index(drop=True)

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_motoristas_matriz_robusto(csv_url: str) -> pd.DataFrame:
    url_cb = f"{csv_url}&_cb={int(time.time())}" if "_cb=" not in csv_url else csv_url
    raw = pd.read_csv(url_cb, header=None)

    if raw.shape[0] < 6 or raw.shape[1] < 6:
        raise ValueError("CSV Motoristas: poucas linhas/colunas; formato não bate.")

    header_row = raw.iloc[2].copy()       # linha 3 (D3.. datas)
    date_headers = header_row.iloc[3:].tolist()
    dias = [extrair_dia_num(x) for x in date_headers]

    valid_cols = [i for i, d in enumerate(dias) if not pd.isna(d)]
    dias_validos = [int(dias[i]) for i in valid_cols]

    data = raw.iloc[3:].reset_index(drop=True)  # linha 4+ (motoristas)

    base = data.iloc[:, :3].copy()
    base.columns = ["motorista", "cpf", "placa"]

    vals = data.iloc[:, 3:].copy()
    vals = vals.iloc[:, valid_cols].copy()
    vals.columns = [f"dia_{d:02d}" for d in dias_validos]

    df = pd.concat([base, vals], axis=1)

    # remove lixo
    df["motorista"] = df["motorista"].astype(str).str.strip()
    df = df[
        df["motorista"].notna()
        & (df["motorista"] != "")
        & (~df["motorista"].str.lower().isin(["nan", "none", "null"]))
    ].copy()

    for c in vals.columns:
        df[c] = df[c].apply(brl_to_float)

    return df

def motoristas_long(df_matriz: pd.DataFrame) -> pd.DataFrame:
    day_cols = [c for c in df_matriz.columns if c.startswith("dia_")]
    df_long = df_matriz.melt(
        id_vars=["motorista", "cpf", "placa"],
        value_vars=day_cols,
        var_name="dia_col",
        value_name="valor",
    )
    df_long["dia"] = df_long["dia_col"].str.extract(r"(\d{2})")[0].astype(int)
    df_long = df_long.drop(columns=["dia_col"])

    df_long["motorista"] = df_long["motorista"].astype(str).str.strip()
    df_long = df_long[
        df_long["motorista"].notna()
        & (df_long["motorista"] != "")
        & (~df_long["motorista"].str.lower().isin(["nan", "none", "null"]))
    ].copy()

    df_long["valor"] = pd.to_numeric(df_long["valor"], errors="coerce")
    df_long = df_long.dropna(subset=["valor"]).copy()
    df_long = df_long[df_long["valor"] > 0].copy()

    return df_long[["motorista", "cpf", "placa", "dia", "valor"]].copy()

# =========================================================
# SIDEBAR NAV
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

def nav_btn(icon: str, key: str, tooltip: str):
    st.sidebar.markdown('<div class="sidebar-btn">', unsafe_allow_html=True)
    if st.sidebar.button(icon, key=f"nav_{key}", help=tooltip):
        st.session_state.page = key
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("##")
st.sidebar.markdown("##")
nav_btn("⦿", "dashboard", "Dashboard (visão geral)")
nav_btn("👤", "motoristas", "Motoristas")
nav_btn("🚚", "placas", "Placas (em breve)")
nav_btn("💬", "chat", "Chat (em breve)")
st.sidebar.markdown("---")
show_debug = st.sidebar.checkbox("Mostrar Debug", value=False)

# =========================================================
# LOAD RECEITA (topo + dashboard)
# =========================================================
try:
    df_receita = load_receita_robusto(SHEETS_CSV_URL_RECEITA)
except Exception as e:
    st.error("Não consegui carregar a RECEITA (CSV).")
    st.write("Verifique se a estrutura está com legendas na linha 3 e dados na linha 4.")
    st.exception(e)
    st.stop()

# =========================================================
# TOPO: Título + Seletor de período
# =========================================================
top_l, top_r = st.columns([0.72, 0.28], vertical_alignment="center")
with top_l:
    st.title("🚚 Dashboard de Receita — Transportes")

with top_r:
    modo = st.selectbox("Período", ["Intervalo", "Dia único"], label_visibility="collapsed")

    dmin = int(df_receita["dia"].min())
    dmax = int(df_receita["dia"].max())

    if modo == "Dia único":
        if dmin == dmax:
            d0 = d1 = dmin
        else:
            dia_sel = st.selectbox("Dia", list(range(dmin, dmax + 1)), index=(dmax - dmin), label_visibility="collapsed")
            d0 = d1 = int(dia_sel)
    else:
        if dmin == dmax:
            d0 = d1 = dmin
        else:
            d0, d1 = st.slider("Dia do mês", min_value=dmin, max_value=dmax, value=(dmin, dmax), label_visibility="collapsed")

df_f = df_receita[(df_receita["dia"] >= d0) & (df_receita["dia"] <= d1)].copy().sort_values("dia")
if df_f.empty:
    st.warning("Sem dados no período selecionado.")
    st.stop()

if show_debug:
    with st.expander("🔎 Debug Receita", expanded=False):
        st.write("URL Receita:", SHEETS_CSV_URL_RECEITA)
        st.write("df_receita:", df_receita.shape)
        st.write("df_f:", df_f.shape)
        st.dataframe(df_receita.head(20), use_container_width=True)
        st.dataframe(df_f.head(20), use_container_width=True)

# =========================================================
# PAGE: DASHBOARD
# =========================================================
if st.session_state.page == "dashboard":
    st.markdown("## Dashboard")

    # Valores do período (robusto)
    fat_mes = float(df_f["acumulado_mes"].dropna().max()) if df_f["acumulado_mes"].dropna().shape[0] else np.nan
    proj_mes = float(df_f["projecao_mes"].dropna().iloc[-1]) if df_f["projecao_mes"].dropna().shape[0] else np.nan
    com_mes = float(df_f["comissao"].dropna().iloc[-1]) if df_f["comissao"].dropna().shape[0] else np.nan
    pct_meta = (fat_mes / META_MENSAL) if (META_MENSAL and not pd.isna(fat_mes)) else np.nan

    # deltas
    def last7_vs_prev7(col: str):
        tmp = df_receita.dropna(subset=[col]).sort_values("dia")
        if tmp.empty or tmp.shape[0] < 8:
            return None
        last7 = tmp.tail(7)[col].sum()
        prev7 = tmp.iloc[max(0, len(tmp)-14):max(0, len(tmp)-7)][col].sum()
        return compute_delta(last7, prev7)

    delta_fat = last7_vs_prev7("faturado_dia")

    proj_vals = df_receita["projecao_mes"].dropna()
    delta_proj = compute_delta(proj_vals.iloc[-1], proj_vals.iloc[-2]) if proj_vals.shape[0] >= 2 else None

    com_vals = df_receita["comissao"].dropna()
    delta_com = compute_delta(com_vals.iloc[-1], com_vals.iloc[-2]) if com_vals.shape[0] >= 2 else None

    acc_vals = df_receita["acumulado_mes"].dropna()
    delta_pct = compute_delta((acc_vals.iloc[-1]/META_MENSAL), (acc_vals.iloc[-2]/META_MENSAL)) if acc_vals.shape[0] >= 2 else None

    # KPIs
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Faturamento (mês)</div>
              <div class="kpi-value">{fmt_brl_short(fat_mes)}</div>
              {delta_text(delta_fat, " vs semana")}
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        pct_show = "—" if pd.isna(pct_meta) else f"{pct_meta*100:.1f}%"
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">% da Meta</div>
              <div class="kpi-value">{pct_show}</div>
              {delta_text(delta_pct, " vs dia anterior")}
            </div>
            """, unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Projeção (mês)</div>
              <div class="kpi-value">{fmt_brl_short(proj_mes)}</div>
              {delta_text(delta_proj, " vs último")}
            </div>
            """, unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Comissão (acumulada)</div>
              <div class="kpi-value">{fmt_brl_short(com_mes)}</div>
              {delta_text(delta_com, " vs último")}
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Painéis inferiores
    p1, p2 = st.columns([0.42, 0.58], gap="large")

    # Painel esquerdo: IA + Donut
    with p1:
        # Comentário IA (RECEITA)
        contexto_receita = {
            "tipo": "receita",
            "periodo": f"dias {d0} a {d1}",
            "faturamento_mes": None if pd.isna(fat_mes) else float(fat_mes),
            "percentual_meta": None if pd.isna(pct_meta) else float(pct_meta),
            "projecao_mes": None if pd.isna(proj_mes) else float(proj_mes),
            "comissao_mes": None if pd.isna(com_mes) else float(com_mes),
            "meta_mensal": float(META_MENSAL),
            "dias_mes_base": DIAS_MES_PADRAO,
        }
        comentario_receita = ""
        try:
            comentario_receita = gerar_comentario_por_ia(contexto_receita)
        except Exception:
            comentario_receita = ""

        st.markdown(
            f"""
            <div class="panel">
              <h3>🤖 Análise Inteligente</h3>
              <div class="ai-text">
                {(comentario_receita or "⚠️ Sem comentário gerado no momento.").replace("\\n", "<br>")}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Donut meta (atingido x restante)
        atingido = float(fat_mes) if not pd.isna(fat_mes) else 0.0
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

    # Painel direito: linha acumulado
    with p2:
        st.markdown(
            """
            <div class="panel">
              <h3>📈 Evolução (acumulado no mês)</h3>
              <div class="small-muted">Maior acumulado por dia no período.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        ac = df_f.groupby("dia", as_index=False)["acumulado_mes"].max()
        fig_line = px.line(ac, x="dia", y="acumulado_mes", markers=True,
                           labels={"dia": "Dia do mês", "acumulado_mes": "Acumulado (R$)"})
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

    # Bar faturamento diário
    st.markdown(
        """
        <div class="panel">
          <h3>🧾 Faturamento diário</h3>
          <div class="small-muted">Soma do faturado por dia no período.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    fat = df_f.groupby("dia", as_index=False)["faturado_dia"].sum()
    fig_bar = px.bar(fat, x="dia", y="faturado_dia",
                     labels={"dia": "Dia do mês", "faturado_dia": "Faturado (R$)"})
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

    if show_debug:
        with st.expander("📋 Tabela Receita (filtrada)", expanded=False):
            st.dataframe(df_f, use_container_width=True, height=420)

# =========================================================
# PAGE: MOTORISTAS
# =========================================================
elif st.session_state.page == "motoristas":
    st.markdown("## 👤 Motoristas")

    try:
        df_matriz = load_motoristas_matriz_robusto(CSV_URL_MOTORISTAS)
        df_m_long = motoristas_long(df_matriz)
    except Exception as e:
        st.error("Não consegui carregar a subplanilha Motoristas.")
        st.write("Verifique se a planilha está compartilhada (qualquer pessoa com link pode ver).")
        st.write("URL usada:", CSV_URL_MOTORISTAS)
        st.exception(e)
        st.stop()

    # Filtro pelo período do topo
    df_mf = df_m_long[(df_m_long["dia"] >= d0) & (df_m_long["dia"] <= d1)].copy()
    if df_mf.empty:
        st.warning("Sem lançamentos de motoristas no período selecionado.")
        st.stop()

    # Ranking global do período
    rank = (
        df_mf.groupby(["motorista", "placa"], as_index=False)["valor"].sum()
        .sort_values("valor", ascending=False)
        .reset_index(drop=True)
    )
    rank = rank[
        rank["motorista"].notna()
        & (rank["motorista"].astype(str).str.strip() != "")
        & (~rank["motorista"].astype(str).str.lower().isin(["nan", "none", "null"]))
    ].copy()
    rank["posicao"] = np.arange(1, len(rank) + 1)

    if rank.empty:
        st.warning("Não encontrei motoristas válidos (verifique a planilha).")
        st.stop()

    # Seletor motorista
    lista = sorted(rank["motorista"].astype(str).unique().tolist())
    motorista_sel = st.selectbox("Selecione o motorista", lista)

    # Filtra somente ele
    df_sel = df_mf[df_mf["motorista"] == motorista_sel].copy()
    if df_sel.empty:
        st.warning("Esse motorista não tem lançamentos no período selecionado.")
        st.stop()

    total = float(df_sel["valor"].sum())
    dias_com_valor = int(df_sel["dia"].nunique())
    media_dia = total / max(1, dias_com_valor)
    proj_30 = media_dia * DIAS_MES_PADRAO

    meta_m = META_MOTORISTA_PADRAO
    pct_meta_m = (total / meta_m) if meta_m else np.nan
    faltante = max(0.0, meta_m - total)
    dia_ultimo = int(df_sel["dia"].max())
    dias_restantes_aprox = max(1, DIAS_MES_PADRAO - dia_ultimo)
    necessario_dia = faltante / dias_restantes_aprox if faltante > 0 else 0.0

    # placa + posição
    row_rank = rank[rank["motorista"] == motorista_sel].sort_values("valor", ascending=False).iloc[0]
    placa_mais = str(row_rank["placa"])
    pos = int(row_rank["posicao"])
    total_rank = int(rank.shape[0])

    # KPIs do motorista (só ele)
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">{motorista_sel} — Total</div>
              <div class="kpi-value">{fmt_brl_short(total)}</div>
              <div class="kpi-delta">Placa: {placa_mais}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        pct_show = "—" if pd.isna(pct_meta_m) else f"{pct_meta_m*100:.1f}%"
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">% da Meta (R$ {int(meta_m/1000)}k)</div>
              <div class="kpi-value">{pct_show}</div>
              <div class="kpi-delta">Falta: {fmt_brl_short(faltante)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Projeção (30d)</div>
              <div class="kpi-value">{fmt_brl_short(proj_30)}</div>
              <div class="kpi-delta">Média/dia: {fmt_brl_short(media_dia)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Ranking</div>
              <div class="kpi-value">#{pos} / {total_rank}</div>
              <div class="kpi-delta">Precisa/dia: {fmt_brl_short(necessario_dia)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Comentário IA REAL (motorista)
    contexto_m = {
        "tipo": "motorista",
        "motorista": motorista_sel,
        "placa": placa_mais,
        "periodo": f"dias {d0} a {d1}",
        "total_no_periodo": float(total),
        "meta_motorista": float(meta_m),
        "percentual_da_meta": None if pd.isna(pct_meta_m) else float(pct_meta_m),
        "dias_com_lancamento": int(dias_com_valor),
        "media_por_dia": float(media_dia),
        "projecao_30_dias": float(proj_30),
        "faltante_para_meta": float(faltante),
        "necessario_por_dia_restante": float(necessario_dia),
        "posicao_no_ranking": int(pos),
        "total_no_ranking": int(total_rank),
        "dias_mes_base": DIAS_MES_PADRAO,
    }
    comentario_m = ""
    try:
        comentario_m = gerar_comentario_por_ia(contexto_m)
    except Exception:
        comentario_m = ""

    st.markdown(
        f"""
        <div class="panel">
          <h3>🧠 Comentário IA</h3>
          <div class="small-muted">Gerado a partir dos números do período selecionado.</div>
          <div class="ai-text" style="margin-top:10px;">
            {(comentario_m or "⚠️ Sem comentário gerado no momento.").replace("\\n", "<br>")}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Gráficos do motorista (somente ele)
    g1, g2 = st.columns([0.60, 0.40], gap="large")

    with g1:
        st.markdown(
            """
            <div class="panel">
              <h3>📊 Faturamento por dia (motorista)</h3>
              <div class="small-muted">Somente lançamentos do motorista selecionado.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        by_day = df_sel.groupby("dia", as_index=False)["valor"].sum().sort_values("dia")
        fig = px.bar(by_day, x="dia", y="valor", labels={"dia": "Dia", "valor": "R$"})
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.85)"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        st.markdown(
            """
            <div class="panel">
              <h3>🏁 Posição no ranking (contexto)</h3>
              <div class="small-muted">Comparação com a mediana do ranking.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        median = float(rank["valor"].median()) if rank.shape[0] else 0.0
        comp = pd.DataFrame({"item": [motorista_sel, "Mediana equipe"], "valor": [total, median]})
        fig2 = px.bar(comp, x="item", y="valor", labels={"item": "", "valor": "R$"})
        fig2.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.85)"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    if show_debug:
        with st.expander("🔎 Debug Motoristas", expanded=False):
            st.write("URL Motoristas:", CSV_URL_MOTORISTAS)
            st.write("df_mf:", df_mf.shape)
            st.dataframe(df_mf.head(40), use_container_width=True)
            st.dataframe(rank.head(30), use_container_width=True)

# =========================================================
# PLACEHOLDERS
# =========================================================
elif st.session_state.page == "placas":
    st.info("🚧 Aba Placas em construção (vamos usar a mesma base de Motoristas).")

elif st.session_state.page == "chat":
    st.info("🚧 Aba Chat em construção.")
