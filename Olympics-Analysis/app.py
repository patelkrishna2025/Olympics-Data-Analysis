"""
=============================================================
 Olympics Intelligence System — v2
 MODULE: Streamlit Dashboard
 Run: streamlit run app.py
 Tabs:
  1. 🏅 Overview        – KPIs, medals by year, type split
  2. 🌍 Countries       – Leaderboard, choropleth, gold rate
  3. 🏃 Athletes        – Top athletes, sport breakdown
  4. 🏊 Sports          – Sport trends, domination heatmap
  5. 👫 Gender Trends   – Participation over years
  6. 🔮 ML Predictor    – Medal type predictor
  7. 👁️ CV Analysis     – Sports image analyser
  8. 💬 Chatbot         – Olympics Q&A assistant
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os, io
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from chatbot.olympics_chatbot  import OlympicsChatbot
from cv_module.sports_cv       import OlympicsCVAnalyser
from models.olympics_models    import MedalPredictor, CountryScorer, SportDominationAnalyser

# ── Colours ───────────────────────────────────────────────
GOLD   = "#FFD700"
SILVER = "#C0C0C0"
BRONZE = "#CD7F32"
BLUE   = "#0085C7"
RED    = "#DF0024"
CARD   = "#1A1E2E"

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="🏅 Olympics Intelligence",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  .hero {{
    background: linear-gradient(90deg, #0a0a1a, #1a1a3e, #0a0a1a);
    border: 2px solid {GOLD};
    border-radius: 16px; padding: 24px 30px;
    margin-bottom: 22px;
    box-shadow: 0 0 40px rgba(255,215,0,0.25);
  }}
  .metric-card {{
    background: linear-gradient(135deg, {CARD}, #252840);
    border: 1px solid #333; border-radius: 12px;
    padding: 18px; text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.6);
  }}
  .metric-value {{ font-size: 2.1rem; font-weight: 900; color: {GOLD}; }}
  .metric-label {{ font-size: 0.82rem; color: #aaa; margin-top: 4px; }}
  .cb-user {{
    background: linear-gradient(135deg,{BLUE},{BLUE}cc);
    color: white; padding: 11px 15px;
    border-radius: 18px 18px 4px 18px;
    margin: 7px 0; max-width: 80%; float: right; clear: both;
  }}
  .cb-bot {{
    background: {CARD}; border: 1px solid {GOLD}44;
    color: #ddd; padding: 11px 15px;
    border-radius: 18px 18px 18px 4px;
    margin: 7px 0; max-width: 80%; float: left; clear: both;
  }}
  .cf {{ clear: both; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  DATA & MODEL LOADING
# ═══════════════════════════════════════════════
@st.cache_data(show_spinner="Loading Olympics data …")
def load_data() -> pd.DataFrame:
    candidates = [
        os.path.join(ROOT, "Summer-Olympic-medals-1976-to-2008.csv"),
        os.path.join(ROOT, "data", "Summer-Olympic-medals-1976-to-2008.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        st.error("❌ CSV file not found.")
        st.stop()

    df = pd.read_csv(path, encoding="latin-1")
    df = df.drop_duplicates().reset_index(drop=True)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    for col in ["Country", "Sport", "Athlete", "Medal", "Gender", "City"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["medal_weight"] = df["Medal"].map({"Gold": 3, "Silver": 2, "Bronze": 1}).fillna(0)
    return df


@st.cache_resource(show_spinner="Training ML models …")
def train_models(df: pd.DataFrame):
    predictor = MedalPredictor().fit(df)
    scorer    = CountryScorer().fit(df)
    dominator = SportDominationAnalyser()
    return predictor, scorer, dominator


def _dark_chart(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(t=20, b=20, l=10, r=10),
    )
    return fig


def main():
    st.markdown("""
    <div class="hero">
      <h1 style="color:#FFD700;margin:0;font-size:2rem;">🏅 Olympics Intelligence System</h1>
      <p style="color:#aac4ff;margin:8px 0 0 0;font-size:0.95rem;">
        1976–2008 Summer Olympics · EDA · ML Medal Predictor · CV Sports Analyser · AI Chatbot
      </p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    predictor, scorer, dominator = train_models(df)

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Controls")
        years = sorted(df["Year"].unique())
        year_range = st.select_slider(
            "📅 Year Range",
            options=years,
            value=(years[0], years[-1]),
        )
        all_sports = ["All"] + sorted(df["Sport"].unique())
        sel_sport  = st.selectbox("🏊 Sport Filter", all_sports)
        gender_sel = st.selectbox("👫 Gender", ["All", "Men", "Women"])
        top_n      = st.slider("📊 Top N", 5, 25, 10)
        st.divider()
        st.caption("🏅 Olympics Analytics v2")

    # ── Filter ───────────────────────────────────────────
    df_f = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
    if sel_sport != "All":
        df_f = df_f[df_f["Sport"] == sel_sport]
    if gender_sel != "All":
        df_f = df_f[df_f["Gender"] == gender_sel]

    # ── Tabs ─────────────────────────────────────────────
    tabs = st.tabs([
        "🏅 Overview",
        "🌍 Countries",
        "🏃 Athletes",
        "🏊 Sports",
        "👫 Gender Trends",
        "🔮 ML Predictor",
        "👁️ CV Analysis",
        "💬 Chatbot",
    ])
    (tab_ov, tab_country, tab_athlete, tab_sport,
     tab_gender, tab_ml, tab_cv, tab_chat) = tabs

    # ═══════════════════════════
    #  TAB 1 – OVERVIEW
    # ═══════════════════════════
    with tab_ov:
        total   = len(df_f)
        gold_n  = (df_f["Medal"] == "Gold").sum()
        countries_n = df_f["Country"].nunique()
        sports_n    = df_f["Sport"].nunique()

        k1, k2, k3, k4 = st.columns(4)
        for col, val, lbl in [
            (k1, f"{total:,}",      "🏅 Total Medals"),
            (k2, f"{gold_n:,}",     "🥇 Gold Medals"),
            (k3, f"{countries_n}",  "🌍 Countries"),
            (k4, f"{sports_n}",     "🏊 Sports"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("📈 Medals Per Olympic Year")
            yr_cnt = df_f.groupby("Year").size().reset_index(name="count")
            fig = px.area(yr_cnt, x="Year", y="count",
                          color_discrete_sequence=[GOLD],
                          markers=True,
                          labels={"count": "Medals"})
            st.plotly_chart(_dark_chart(fig, 320), use_container_width=True)

        with c2:
            st.subheader("🥇 Medal Type Distribution")
            mc = df_f["Medal"].value_counts()
            fig2 = go.Figure(go.Pie(
                labels=mc.index, values=mc.values,
                hole=0.52,
                marker_colors=[GOLD, SILVER, BRONZE],
            ))
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", height=320,
                showlegend=True, margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Year × Medal heatmap
        st.subheader("🔥 Year × Medal Type Heatmap")
        ym = df_f.groupby(["Year","Medal"]).size().unstack(fill_value=0)
        fig_hm = px.imshow(ym, color_continuous_scale="YlOrRd", aspect="auto",
                           labels={"color": "Count"})
        st.plotly_chart(_dark_chart(fig_hm, 280), use_container_width=True)

        # Host cities
        st.subheader("🏙️ Olympic Host Cities")
        city_df = df_f.drop_duplicates("Year")[["Year","City"]].sort_values("Year")
        st.dataframe(city_df, use_container_width=True, hide_index=True)

    # ═══════════════════════════
    #  TAB 2 – COUNTRIES
    # ═══════════════════════════
    with tab_country:
        st.subheader("🌍 Country Leaderboard")
        scorer_fit = CountryScorer().fit(df_f)
        board = scorer_fit.leaderboard(top_n)

        # Colour rows
        def _medal_style(val):
            if val == board["Score"].max():
                return f"background-color: {GOLD}33; color: {GOLD}"
            return ""

        st.dataframe(
            board.style.applymap(_medal_style, subset=["Score"]),
            use_container_width=True, height=380, hide_index=True
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"🏆 Top {top_n} — Total Medals")
            top_c = df_f["Country"].value_counts().head(top_n)
            fig = px.bar(x=top_c.values, y=top_c.index, orientation="h",
                         color=top_c.values, color_continuous_scale="YlOrRd",
                         labels={"x": "Medals", "y": ""})
            st.plotly_chart(_dark_chart(fig, 400), use_container_width=True)

        with c2:
            st.subheader(f"🥇 Top {top_n} — Gold Medals")
            top_gold = df_f[df_f["Medal"] == "Gold"]["Country"].value_counts().head(top_n)
            fig2 = px.bar(x=top_gold.values, y=top_gold.index, orientation="h",
                          color=top_gold.values, color_continuous_scale="Peach",
                          labels={"x": "Golds", "y": ""})
            st.plotly_chart(_dark_chart(fig2, 400), use_container_width=True)

        # Choropleth
        st.subheader("🗺️ Global Medal Map")
        country_cnt = df_f["Country"].value_counts().reset_index()
        country_cnt.columns = ["country", "medals"]
        fig_map = px.choropleth(
            country_cnt,
            locations="country",
            locationmode="country names",
            color="medals",
            color_continuous_scale="YlOrRd",
            hover_name="country",
        )
        fig_map.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False),
            height=480, margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Country medal type stacked bar
        st.subheader(f"🏅 Medal Breakdown — Top {min(top_n, 12)} Countries")
        top_list = df_f["Country"].value_counts().head(min(top_n, 12)).index.tolist()
        medal_breakdown = (
            df_f[df_f["Country"].isin(top_list)]
            .groupby(["Country","Medal"]).size().reset_index(name="count")
        )
        fig_stack = px.bar(
            medal_breakdown, x="Country", y="count", color="Medal",
            color_discrete_map={"Gold": GOLD, "Silver": SILVER, "Bronze": BRONZE},
            barmode="stack",
            labels={"count": "Medals"},
        )
        st.plotly_chart(_dark_chart(fig_stack, 380), use_container_width=True)

    # ═══════════════════════════
    #  TAB 3 – ATHLETES
    # ═══════════════════════════
    with tab_athlete:
        st.subheader(f"🏃 Top {top_n} Athletes by Medal Count")
        top_ath = df_f["Athlete"].value_counts().head(top_n)
        fig = px.bar(x=top_ath.values, y=top_ath.index, orientation="h",
                     color=top_ath.values, color_continuous_scale="Plasma",
                     labels={"x": "Medals", "y": ""})
        st.plotly_chart(_dark_chart(fig, 400), use_container_width=True)

        # Athlete-sport heatmap
        st.subheader("🔥 Top Athletes × Sports Heatmap")
        top_20_ath = df_f["Athlete"].value_counts().head(20).index.tolist()
        heat_df = (
            df_f[df_f["Athlete"].isin(top_20_ath)]
            .groupby(["Athlete","Sport"]).size().unstack(fill_value=0)
        )
        fig_hm = px.imshow(heat_df, color_continuous_scale="YlOrRd",
                           aspect="auto", labels={"color": "Medals"})
        st.plotly_chart(_dark_chart(fig_hm, 480), use_container_width=True)

        # Athlete search
        st.subheader("🔍 Athlete Search")
        ath_q = st.text_input("Type athlete name …", placeholder="e.g. Michael Phelps")
        if ath_q:
            res = df_f[df_f["Athlete"].str.lower().str.contains(ath_q.lower(), na=False)]
            if len(res) == 0:
                st.warning(f"No athletes found matching '{ath_q}'")
            else:
                ath_name = res.iloc[0]["Athlete"]
                g  = (res["Medal"] == "Gold").sum()
                s  = (res["Medal"] == "Silver").sum()
                b  = (res["Medal"] == "Bronze").sum()
                sp = res["Sport"].unique().tolist()
                c1, c2, c3 = st.columns(3)
                c1.metric("🥇 Gold",   g)
                c2.metric("🥈 Silver", s)
                c3.metric("🥉 Bronze", b)
                st.info(f"**{ath_name}** | Sports: {', '.join(sp)} | Country: {res.iloc[0]['Country']}")
                st.dataframe(res[["Year","City","Sport","Event","Medal"]].sort_values("Year"),
                             use_container_width=True, hide_index=True)

    # ═══════════════════════════
    #  TAB 4 – SPORTS
    # ═══════════════════════════
    with tab_sport:
        st.subheader(f"🏊 Top {top_n} Sports by Medal Count")
        sp_cnt = df_f["Sport"].value_counts().head(top_n)
        fig = px.bar(x=sp_cnt.values, y=sp_cnt.index, orientation="h",
                     color=sp_cnt.values, color_continuous_scale="Viridis",
                     labels={"x": "Medals", "y": ""})
        st.plotly_chart(_dark_chart(fig, 400), use_container_width=True)

        # Sport × year trend
        st.subheader("📈 Sport Medal Trend Over Years")
        top5_sp = df_f["Sport"].value_counts().head(5).index.tolist()
        trend = (
            df_f[df_f["Sport"].isin(top5_sp)]
            .groupby(["Year","Sport"]).size().reset_index(name="count")
        )
        fig2 = px.line(trend, x="Year", y="count", color="Sport",
                       markers=True,
                       color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(_dark_chart(fig2, 360), use_container_width=True)

        # Country × Sport heatmap
        st.subheader("🌡️ Country × Sport Domination Heatmap")
        top_countries = df_f["Country"].value_counts().head(15).index.tolist()
        top_sports    = df_f["Sport"].value_counts().head(12).index.tolist()
        cs_heat = (
            df_f[df_f["Country"].isin(top_countries) & df_f["Sport"].isin(top_sports)]
            .groupby(["Country","Sport"]).size().unstack(fill_value=0)
        )
        fig_cs = px.imshow(cs_heat, color_continuous_scale="YlOrRd",
                           aspect="auto", labels={"color": "Medals"})
        st.plotly_chart(_dark_chart(fig_cs, 480), use_container_width=True)

        # Sport domination table
        st.subheader("👑 Sport Domination — Who Rules Each Sport?")
        dom_df = dominator.analyse(df_f)
        st.dataframe(dom_df, use_container_width=True, hide_index=True)

    # ═══════════════════════════
    #  TAB 5 – GENDER TRENDS
    # ═══════════════════════════
    with tab_gender:
        st.subheader("👫 Gender Participation Over Years")
        gender_yr = (
            df_f.groupby(["Year","Gender"]).size().reset_index(name="count")
        )
        fig = px.line(gender_yr, x="Year", y="count", color="Gender",
                      markers=True,
                      color_discrete_map={"Men": BLUE, "Women": RED},
                      labels={"count": "Medal Entries"})
        st.plotly_chart(_dark_chart(fig, 380), use_container_width=True)

        # Stacked bar
        st.subheader("📊 Men vs Women — Per Year (Stacked)")
        fig2 = px.bar(gender_yr, x="Year", y="count", color="Gender",
                      barmode="stack",
                      color_discrete_map={"Men": BLUE, "Women": RED},
                      labels={"count": "Medals"})
        st.plotly_chart(_dark_chart(fig2, 320), use_container_width=True)

        # Gender share per sport
        st.subheader("🏊 Gender Balance Across Top Sports")
        top_sp = df_f["Sport"].value_counts().head(10).index.tolist()
        gs = (
            df_f[df_f["Sport"].isin(top_sp)]
            .groupby(["Sport","Gender"]).size().reset_index(name="count")
        )
        fig3 = px.bar(gs, x="Sport", y="count", color="Gender",
                      barmode="group",
                      color_discrete_map={"Men": BLUE, "Women": RED},
                      labels={"count": "Medals"})
        fig3.update_xaxes(tickangle=30)
        st.plotly_chart(_dark_chart(fig3, 360), use_container_width=True)

        # Women share over years
        st.subheader("📈 Women's Share (%) Over Years")
        gp = gender_yr.pivot(index="Year", columns="Gender", values="count").fillna(0)
        gp["Women_%"] = (gp.get("Women", 0) / (gp.get("Men", 0) + gp.get("Women", 1)) * 100).round(1)
        fig4 = px.area(gp.reset_index(), x="Year", y="Women_%",
                       color_discrete_sequence=[RED],
                       labels={"Women_%": "Women %"})
        st.plotly_chart(_dark_chart(fig4, 300), use_container_width=True)

    # ═══════════════════════════
    #  TAB 6 – ML PREDICTOR
    # ═══════════════════════════
    with tab_ml:
        st.subheader("🔮 ML Medal Type Predictor")
        st.info("Ensemble: Random Forest + Gradient Boosting · Predicts Gold / Silver / Bronze")

        if predictor.metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🌲 RF Accuracy",  f"{predictor.metrics['RF Accuracy']}%")
            m2.metric("🔥 GBM Accuracy", f"{predictor.metrics['GBM Accuracy']}%")
            m3.metric("📦 Train Size",   predictor.metrics["Train Size"])
            m4.metric("🧪 Test Size",    predictor.metrics["Test Size"])

        st.divider()
        st.subheader("🧮 Predict Medal for a New Entry")
        col1, col2 = st.columns(2)
        with col1:
            p_country = st.selectbox(
                "🌍 Country",
                sorted(df["Country"].unique()),
                index=list(sorted(df["Country"].unique())).index("United States")
                      if "United States" in df["Country"].values else 0,
                key="p_country"
            )
            p_sport = st.selectbox("🏊 Sport", sorted(df["Sport"].unique()), key="p_sport")
        with col2:
            p_gender = st.selectbox("👫 Gender", ["Men", "Women"], key="p_gender")
            p_year   = st.select_slider("📅 Year", options=sorted(df["Year"].unique()), key="p_year")

        if st.button("🔮 Predict Medal", type="primary"):
            result = predictor.predict(p_country, p_sport, p_gender, p_year)
            medal  = result["medal"]
            icon   = {"Gold": "🥇", "Silver": "🥈", "Bronze": "🥉"}.get(medal, "🏅")
            color  = {"Gold": GOLD, "Silver": SILVER, "Bronze": BRONZE}.get(medal, "#fff")
            st.markdown(
                f'<div style="background:{color}22;border:2px solid {color};border-radius:12px;'
                f'padding:20px;text-align:center;font-size:1.5rem;">'
                f'{icon} Predicted Medal: <strong>{medal}</strong> '
                f'— Confidence: {result["confidence"]}%</div>',
                unsafe_allow_html=True
            )
            # Probability bar
            probs = result["probabilities"]
            prob_df = pd.DataFrame({"Medal": list(probs.keys()), "Probability %": list(probs.values())})
            fig_p = px.bar(prob_df, x="Medal", y="Probability %",
                           color="Medal",
                           color_discrete_map={"Gold": GOLD, "Silver": SILVER, "Bronze": BRONZE})
            st.plotly_chart(_dark_chart(fig_p, 260), use_container_width=True)

        # Feature importance
        st.subheader("📊 Feature Importance")
        fi = predictor.feature_importance()
        if not fi.empty:
            fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale="YlOrRd")
            st.plotly_chart(_dark_chart(fig_fi, 280), use_container_width=True)

    # ═══════════════════════════
    #  TAB 7 – CV ANALYSIS
    # ═══════════════════════════
    with tab_cv:
        st.subheader("👁️ Computer Vision — Sports Image Analyser")
        st.markdown(
            "Upload any Olympics / sports image to detect mood, motion blur, "
            "crowd density, dominant colours, Olympic ring colour match, and more."
        )

        cv_mode = st.radio(
            "Mode",
            ["🔍 Full Analysis", "🎨 Filter Gallery"],
            horizontal=True
        )

        uploaded = st.file_uploader(
            "📤 Upload Sports Image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="cv_up"
        )

        if uploaded:
            try:
                import cv2
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("Could not decode image.")
                else:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    analyser = OlympicsCVAnalyser()

                    if "Full Analysis" in cv_mode:
                        col_img, col_res = st.columns(2)
                        with col_img:
                            st.markdown("**📸 Original**")
                            st.image(img_rgb, use_container_width=True)

                        result = analyser.analyse(frame)
                        with col_res:
                            st.markdown("**🔎 Analysis Results**")
                            st.metric("📐 Resolution",   f"{result.width} × {result.height}")
                            st.metric("☀️ Brightness",   f"{result.brightness}/255")
                            st.metric("🎛️ Contrast",    f"{result.contrast:.1f}")
                            st.metric("🔍 Edge Density", f"{result.edge_density:.4f}")
                            st.metric("💨 Motion Blur",  f"{result.motion_blur_score:.3f}")
                            st.markdown(f"**🏟️ Venue Mood:** {result.color_mood}")
                            st.markdown(f"**🏅 Sport Hint:** {result.sport_env_hint}")
                            st.markdown(f"**👥 Crowd:** {result.crowd_density}")
                            st.markdown(f"**⭕ Olympic Ring:** {result.olympic_ring_match}")

                            # Dominant colours
                            st.markdown("**🎨 Dominant Colours:**")
                            dcols = st.columns(3)
                            for wid, hx in zip(dcols, result.dominant_colors):
                                wid.markdown(
                                    f'<div style="background:{hx};height:40px;border-radius:6px;'
                                    f'text-align:center;line-height:40px;color:white;font-size:0.75rem;">'
                                    f'{hx}</div>',
                                    unsafe_allow_html=True
                                )

                        if result.annotated_frame is not None:
                            st.markdown("**✨ Edge-Annotated (Gold Overlay):**")
                            st.image(cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB),
                                     use_container_width=True)

                        # Histogram
                        st.markdown("### 🌈 RGB Color Histogram")
                        fig_h, ax = plt.subplots(figsize=(10, 3))
                        fig_h.patch.set_facecolor("#0a0a1a")
                        ax.set_facecolor("#1A1E2E")
                        for i, color in enumerate(["red", "green", "blue"]):
                            import cv2 as _cv2
                            hist = _cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                            ax.plot(hist, color=color, linewidth=1.5, alpha=0.9)
                        ax.set_title("RGB Color Histogram", color="white")
                        ax.tick_params(colors="white")
                        for sp in ax.spines.values():
                            sp.set_color("#555")
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0a0a1a")
                        buf.seek(0)
                        st.image(Image.open(buf), use_container_width=True)
                        plt.close()

                        # Pixel stats
                        st.markdown("### 📊 Pixel Statistics")
                        st.dataframe(
                            pd.DataFrame(analyser.pixel_stats(img_rgb)),
                            use_container_width=True, hide_index=True
                        )

                    else:
                        filters = analyser.apply_filters(img_rgb)
                        names   = list(filters.keys())
                        for row_start in range(0, len(names), 3):
                            cols = st.columns(3)
                            for j, name in enumerate(names[row_start:row_start+3]):
                                with cols[j]:
                                    st.markdown(f"**{name}**")
                                    img_out = filters[name]
                                    cmap = "gray" if len(img_out.shape) == 2 else None
                                    st.image(img_out, use_container_width=True, clamp=True)

            except ImportError:
                st.error("OpenCV not installed. Run: `pip install opencv-python`")
        else:
            st.info("📤 Upload any sports / Olympics image to run analysis.")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🔍 Analysis Features**")
                st.dataframe(pd.DataFrame([
                    {"Feature": "Brightness & Contrast",  "Method": "Grayscale Stats"},
                    {"Feature": "Dominant Colours",       "Method": "K-Means (k=3)"},
                    {"Feature": "Venue Mood",             "Method": "HSV Analysis"},
                    {"Feature": "Sport Environment Hint", "Method": "Rule-based Fusion"},
                    {"Feature": "Motion Blur Detection",  "Method": "Laplacian Variance"},
                    {"Feature": "Crowd Density Estimate", "Method": "Upper-Region Edge Density"},
                    {"Feature": "Olympic Ring Colour",    "Method": "HSV Hue Mapping"},
                    {"Feature": "Gold Edge Overlay",      "Method": "Canny + Weighted Blend"},
                ]), use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**🎨 Filter Gallery (11 filters)**")
                st.markdown("""
                - 🔳 Grayscale
                - ✏️ Edge Detection (Canny)
                - 🌫️ Gaussian Blur
                - 🔪 Sharpen
                - 🪨 Emboss
                - 🔄 Invert
                - 🟤 Sepia
                - 🥇 Olympic Gold Tint
                - ⬆️ High Contrast
                - ⬛ Binary Threshold
                """)

    # ═══════════════════════════
    #  TAB 8 – CHATBOT
    # ═══════════════════════════
    with tab_chat:
        st.subheader("💬 Olympics AI Assistant")
        st.markdown("Ask anything about Olympic medals, athletes, countries, sports, or trends!")

        if "ol_bot" not in st.session_state:
            st.session_state.ol_bot = OlympicsChatbot(df)
        if "ol_history" not in st.session_state:
            st.session_state.ol_history = [
                ("bot", (
                    "🏅 Hello! I'm your **Olympics Data Assistant**.\n\n"
                    "Try asking:\n"
                    "- *Top 5 countries?*\n"
                    "- *How many medals did USA win?*\n"
                    "- *Who dominated swimming?*\n"
                    "- *Gold medals by year?*\n"
                    "- *Gender participation trend?*"
                ))
            ]

        # Quick buttons
        st.markdown("**💡 Quick Questions:**")
        quick_qs = [
            "Top 5 countries?",
            "Top 5 athletes?",
            "How many total medals?",
            "USA medals?",
            "Who dominated swimming?",
            "Gold medals by year?",
            "Women participation?",
            "1996 Olympics?",
        ]
        qcols = st.columns(4)
        for i, q in enumerate(quick_qs):
            if qcols[i % 4].button(q, key=f"olqb_{i}", use_container_width=True):
                reply = st.session_state.ol_bot.answer(q)
                st.session_state.ol_history.append(("user", q))
                st.session_state.ol_history.append(("bot", reply))
                st.rerun()

        st.markdown("---")

        for role, msg in st.session_state.ol_history:
            css     = "cb-user" if role == "user" else "cb-bot"
            html_msg = msg.replace("\n", "<br>")
            st.markdown(
                f'<div class="{css}">{html_msg}</div><div class="cf"></div>',
                unsafe_allow_html=True
            )

        ci, cb = st.columns([5, 1])
        with ci:
            user_input = st.text_input(
                "Ask …", key="ol_input", label_visibility="collapsed",
                placeholder="e.g. China medals? / Top sports? / 2008 Olympics?"
            )
        with cb:
            send = st.button("Send 🚀", use_container_width=True)

        if send and user_input.strip():
            reply = st.session_state.ol_bot.answer(user_input)
            st.session_state.ol_history.append(("user", user_input))
            st.session_state.ol_history.append(("bot", reply))
            st.rerun()

        if st.button("🗑️ Clear Chat"):
            st.session_state.ol_history = [("bot", "👋 Chat cleared! Ask me about Olympics.")]
            st.rerun()

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align:center;color:#555;font-size:0.8rem;">
      🏅 Olympics Intelligence System v2 &nbsp;|&nbsp;
      Data: 1976–2008 Summer Olympics &nbsp;|&nbsp;
      ML: RF · GBM &nbsp;|&nbsp; CV: OpenCV &nbsp;|&nbsp; Chatbot: Rule-based NLP
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
