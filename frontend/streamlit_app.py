
import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────
API_BASE = "http://localhost:5000/api"
LABEL_COLORS = {"Positive": "#4ade80", "Neutral": "#facc15", "Negative": "#f87171"}
LABEL_EMOJI  = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}

# ── Session State ──────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💬 Sentiment Analysis App")
    st.markdown("---")

    # Health check
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        if resp.status_code == 200:
            st.success("✅ API Online")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
        st.caption("Start Flask: python backend/app.py")

    st.markdown("---")
    page = st.radio("Navigation", ["🔍 Predict", "📊 Batch", "📈 Model Metrics", "📜 History"])
    st.markdown("---")
    st.caption("Model: LSTM Sentiment Classifier v1.0")
    st.caption("Labels: Positive · Neutral · Negative")

# ── Helper Functions ───────────────────────────────────────────────────────
def predict_text(text: str) -> dict:
    resp = requests.post(f"{API_BASE}/predict",
                         json={"text": text}, timeout=10)
    resp.raise_for_status()
    return resp.json()

def predict_batch(texts: list) -> dict:
    resp = requests.post(f"{API_BASE}/predict/batch",
                         json={"texts": texts}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def gauge_chart(confidence: float, label: str) -> go.Figure:
    color = LABEL_COLORS.get(label, "#9ca3af")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Confidence — {label}", "font": {"size": 16, "color": "white"}},
        number={"suffix": "%", "font": {"size": 30, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
            "bar":  {"color": color},
            "bgcolor": "#1f2937",
            "steps": [
                {"range": [0, 50],  "color": "#111827"},
                {"range": [50, 75], "color": "#1f2937"},
                {"range": [75, 100],"color": "#374151"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": confidence * 100},
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font={"color": "white"}, height=250, margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig

def prob_bar_chart(probs: dict) -> go.Figure:
    labels = list(probs.keys())
    values = [v * 100 for v in probs.values()]
    colors = [LABEL_COLORS[l] for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="outside",
        textfont={"color": "white", "size": 13},
    ))
    fig.update_layout(
        title={"text": "Probability Distribution", "font": {"color": "white", "size": 14}},
        paper_bgcolor="#0f172a", plot_bgcolor="#1f2937",
        font={"color": "white"}, yaxis={"range": [0, 110], "gridcolor": "#374151"},
        xaxis={"gridcolor": "#374151"}, height=280,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Predict":
    st.header("Real-Time Sentiment Prediction")
    st.markdown("Enter any text below to analyze its sentiment instantly.")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_text = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="e.g. This product is absolutely amazing! I love it.",
            max_chars=5000,
        )
        char_count = len(user_text)
        st.caption(f"{char_count}/5000 characters")

    with col2:
        st.markdown("**Quick examples:**")
        examples = [
            ("😊 Positive", "This product is absolutely amazing! Exceeded all my expectations."),
            ("😞 Negative", "Terrible service. Broke after one day. Complete waste of money."),
            ("😐 Neutral",  "The product is okay. Does the job, nothing special."),
        ]
        for label, ex in examples:
            if st.button(label, use_container_width=True):
                st.session_state["example_text"] = ex

        if "example_text" in st.session_state:
            user_text = st.session_state["example_text"]

    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    result = predict_text(user_text)
                    pred   = result["prediction"]
                    label  = pred["label"]
                    conf   = pred["confidence"]
                    probs  = pred["probabilities"]
                    emoji  = LABEL_EMOJI[label]
                    color  = LABEL_COLORS[label]

                    # Result banner
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1e293b, #0f172a);
                                border: 2px solid {color}; border-radius: 16px;
                                padding: 24px; text-align: center; margin: 16px 0;">
                        <div style="font-size: 48px;">{emoji}</div>
                        <div style="font-size: 32px; font-weight: 800; color: {color}; margin: 8px 0;">{label}</div>
                        <div style="color: #9ca3af; font-size: 14px;">
                            Confidence: <span style="color: white; font-weight: 600;">{conf*100:.1f}%</span>
                            &nbsp;·&nbsp; Latency: <span style="color: white;">{result.get('latency_ms', 0):.1f}ms</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(gauge_chart(conf, label), use_container_width=True)
                    with c2:
                        st.plotly_chart(prob_bar_chart(probs), use_container_width=True)

                    # Save to history
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "text":      user_text[:80] + ("…" if len(user_text) > 80 else ""),
                        "label":     label,
                        "confidence": f"{conf*100:.1f}%",
                        "latency":   f"{result.get('latency_ms', 0):.1f}ms",
                    })

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to Flask API. Make sure `python backend/app.py` is running.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Batch":
    st.header("Batch Sentiment Analysis")
    st.markdown("Paste multiple texts (one per line) for bulk analysis.")

    batch_input = st.text_area(
        "Texts (one per line, max 50):",
        height=220,
        placeholder="I love this product!\nTerrible experience.\nJust okay I guess.",
    )
    if st.button("📊 Analyze All", type="primary", use_container_width=True):
        lines = [l.strip() for l in batch_input.split("\n") if l.strip()]
        if not lines:
            st.warning("Please enter at least one line of text.")
        elif len(lines) > 50:
            st.error("Maximum 50 texts per batch.")
        else:
            with st.spinner(f"Analyzing {len(lines)} texts…"):
                try:
                    result = predict_batch(lines)
                    rows = []
                    label_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
                    for r in result["results"]:
                        if "prediction" in r:
                            pred = r["prediction"]
                            rows.append({
                                "Text":       r["text"][:60] + "…" if len(r["text"]) > 60 else r["text"],
                                "Sentiment":  pred["label"],
                                "Confidence": f"{pred['confidence']*100:.1f}%",
                                "Positive":   f"{pred['probabilities']['Positive']*100:.1f}%",
                                "Neutral":    f"{pred['probabilities']['Neutral']*100:.1f}%",
                                "Negative":   f"{pred['probabilities']['Negative']*100:.1f}%",
                            })
                            label_counts[pred["label"]] += 1

                    st.success(f"✅ Analyzed {len(rows)} texts in {result['latency_ms']:.0f}ms")

                    # Pie chart summary
                    total = sum(label_counts.values())
                    if total > 0:
                        fig = go.Figure(go.Pie(
                            labels=list(label_counts.keys()),
                            values=list(label_counts.values()),
                            marker_colors=["#f87171", "#facc15", "#4ade80"],
                            hole=0.4,
                            textfont={"size": 14, "color": "white"},
                        ))
                        fig.update_layout(
                            paper_bgcolor="#0f172a", font={"color": "white"},
                            title={"text": "Sentiment Distribution", "font": {"color": "white"}},
                            legend={"font": {"color": "white"}},
                            height=320, margin=dict(t=50, b=0, l=0, r=0),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Results table
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button("⬇ Download CSV", csv, "sentiment_results.csv", "text/csv")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to Flask API.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Metrics":
    st.header("Model Performance Dashboard")
    try:
        resp = requests.get(f"{API_BASE}/metrics", timeout=5)
        m    = resp.json()["metrics"]

        # KPI cards
        cols = st.columns(4)
        for col, (label, val) in zip(cols, [
            ("Accuracy",  m["accuracy"]),
            ("Precision", m["precision"]),
            ("Recall",    m["recall"]),
            ("F1-Score",  m["f1_score"]),
        ]):
            col.metric(label, f"{val*100:.2f}%")

        st.markdown("---")

        # Confusion matrix heatmap
        import numpy as np
        cm = np.array(m["confusion_matrix"])
        labels = ["Negative", "Neutral", "Positive"]
        fig = px.imshow(
            cm, x=labels, y=labels,
            color_continuous_scale="Blues",
            text_auto=True,
            title="Confusion Matrix",
        )
        fig.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font={"color": "white", "size": 13},
            xaxis_title="Predicted", yaxis_title="Actual",
            coloraxis_showscale=False,
        )
        fig.update_traces(textfont={"size": 18, "color": "white"})

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig, use_container_width=True)

        # Per-class bar chart
        with c2:
            pc = m.get("per_class", {})
            bar_data = []
            for lbl in labels:
                if lbl in pc:
                    for metric in ["precision", "recall", "f1_score"]:
                        bar_data.append({
                            "Class": lbl, "Metric": metric.replace("_", " ").title(),
                            "Score": pc[lbl][metric],
                        })
            if bar_data:
                df_bar = pd.DataFrame(bar_data)
                fig2 = px.bar(df_bar, x="Class", y="Score", color="Metric",
                              barmode="group", title="Per-Class Metrics",
                              color_discrete_map={
                                  "Precision": "#f87171",
                                  "Recall":    "#facc15",
                                  "F1 Score":  "#4ade80",
                              })
                fig2.update_layout(
                    paper_bgcolor="#0f172a", plot_bgcolor="#1f2937",
                    font={"color": "white"}, yaxis_range=[0, 1.1],
                )
                st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load metrics: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📜 History":
    st.header("Prediction History")
    if not st.session_state.history:
        st.info("No predictions yet. Go to the Predict tab and analyze some text!")
    else:
        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()
        df = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False)
        st.download_button("⬇ Download History", csv, "prediction_history.csv", "text/csv")
