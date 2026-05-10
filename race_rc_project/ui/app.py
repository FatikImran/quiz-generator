"""
ui/app.py
Full RACE RC System — Streamlit UI
4 screens: Article Input | Quiz View | Hint Panel | Analytics Dashboard

Model A generates AND verifies questions/answers.
Model B generates distractors and graduated hints.
Feature representation: One-Hot Encoding (primary).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RACE RC Quiz System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #1565c0 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .correct-answer {
        background-color: #e8f5e9 !important;
        border: 2px solid #43a047 !important;
        border-radius: 8px; padding: 12px;
    }
    .wrong-answer {
        background-color: #ffebee !important;
        border: 2px solid #e53935 !important;
        border-radius: 8px; padding: 12px;
    }
    .ai-badge {
        background: #e3f2fd; border: 1px solid #1565c0;
        border-radius: 6px; padding: 4px 10px;
        font-size: 0.8em; color: #1565c0; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
defaults = {
    "current_screen": "article_input",
    "article": "",
    "question": "",            # displayed question (could be generated or RACE original)
    "question_source": "",     # "generated" | "race_original" | "manual"
    "generated_question_data": None,   # full output from generate_question()
    "options": {},
    "correct_label": "",
    "correct_text": "",
    "distractors": [],
    "hints": [],
    "verify_result": None,
    "user_answer": None,
    "checked": False,
    "hints_revealed": 0,
    "answer_revealed": False,
    "session_log": [],
    "race_row": None,          # the raw RACE row dict (for RACE-loaded samples)
    "use_generated_q": True,   # whether to prefer the AI-generated question
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load models (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_inference_module():
    try:
        from src.inference import (verify_answer, generate_distractors,
                                   generate_hints, generate_question,
                                   predict_from_race_row)
        return {
            "verify":       verify_answer,
            "distractors":  generate_distractors,
            "hints":        generate_hints,
            "gen_question": generate_question,
            "predict_row":  predict_from_race_row,
            "ok":           True,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@st.cache_data
def load_race_sample(n=200):
    try:
        df = pd.read_csv("data/raw/val.csv").head(n)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return None


inf       = load_inference_module()
models_ok = inf.get("ok", False)


# ── Helper: full inference on a loaded RACE row ───────────────────────────────
def run_full_inference(row_dict: dict) -> dict:
    """
    Runs both Model A (generation + verification) and Model B (distractors + hints).
    Falls back gracefully when models are not yet trained.
    """
    if not models_ok:
        correct = str(row_dict.get("answer", "A")).strip().upper()
        options = {k: str(row_dict.get(k, "")) for k in "ABCD"}
        return {
            "generated_question": {
                "question": str(row_dict.get("question", "What is the passage about?")),
                "method": "demo",
                "candidates": [],
                "latency_ms": 0,
            },
            "verify_result": {
                "predicted": correct,
                "probabilities": {k: 0.25 for k in "ABCD"},
                "latency_ms": 0,
                "feature_method": "demo",
            },
            "distractors": [options[k] for k in "ABCD" if k != correct][:3],
            "hints": [
                "Read the passage carefully for context clues.",
                "Focus on the paragraph most related to the question.",
                "The answer is directly stated in the passage.",
            ],
        }

    result = inf["predict_row"](row_dict)
    return {
        "generated_question": result["generated_question"],
        "verify_result":      result["verification"],
        "distractors":        result["distractors"],
        "hints":              result["hints"],
    }


# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 RACE RC System")
    st.markdown("*Intelligent Quiz Generation & Verification*")
    st.divider()

    screen = st.radio(
        "Navigate",
        ["📄 Article Input", "📝 Quiz View", "💡 Hint Panel", "📊 Analytics"],
        key="nav_radio",
    )
    screen_map = {
        "📄 Article Input": "article_input",
        "📝 Quiz View":     "quiz_view",
        "💡 Hint Panel":    "hint_panel",
        "📊 Analytics":     "analytics",
    }
    st.session_state.current_screen = screen_map[screen]

    st.divider()
    st.markdown("**Session Stats**")
    total   = len(st.session_state.session_log)
    correct = sum(1 for r in st.session_state.session_log if r.get("user_correct"))
    st.metric("Questions Attempted", total)
    if total > 0:
        st.metric("Correct Answers",
                  f"{correct}/{total} ({correct/total*100:.0f}%)")

    st.divider()
    if not models_ok:
        st.warning(f"⚠️ Demo mode\n\n`python src/train_all.py`\n\n"
                   f"Error: {inf.get('error','unknown')[:80]}")

    if st.button("🔄 Reset Session", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Article Input
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.current_screen == "article_input":
    st.markdown("""
    <div class="main-header">
        <h2>📄 Article Input</h2>
        <p>Paste a reading passage or load a random RACE sample.
           Model A will <strong>generate</strong> a question and <strong>verify</strong> answers.
           Model B will generate distractors and hints.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load from CSV or Random ────────────────────────────────────────────────
    upload_col, _ = st.columns([1, 1])
    with upload_col:
        uploaded_csv = st.file_uploader("Upload custom RACE-format CSV", type=["csv"])
        if uploaded_csv:
            df_custom = pd.read_csv(uploaded_csv)
            if st.button("Load random from custom CSV"):
                row = df_custom.sample(1).iloc[0].to_dict()
                st.session_state.race_row     = row
                st.session_state.article      = str(row.get("article", ""))
                st.session_state.question     = str(row.get("question", ""))
                st.session_state.options      = {k: str(row.get(k, "")) for k in "ABCD"}
                st.session_state.correct_label = str(row.get("answer", "A")).strip().upper()
                st.session_state.correct_text  = st.session_state.options.get(st.session_state.correct_label, "")
                # Clear inference
                st.session_state.verify_result = None
                st.session_state.checked = False

    st.markdown("##### Or load from included dataset:")
    col_btn, col_toggle = st.columns([2, 1])
    with col_btn:
        if st.button("🎲 Load Random RACE Sample", type="secondary",
                     use_container_width=True):
            race_df = load_race_sample()
            if race_df is not None:
                row = race_df.sample(1).iloc[0].to_dict()
                st.session_state.race_row     = row
                st.session_state.article      = str(row["article"])
                # Pre-fill RACE original question/options so user can edit
                st.session_state.question     = str(row["question"])
                st.session_state.options      = {k: str(row[k]) for k in "ABCD"}
                st.session_state.correct_label = str(row["answer"]).strip().upper()
                st.session_state.correct_text  = st.session_state.options[
                    st.session_state.correct_label]
                # Reset inference state
                st.session_state.verify_result        = None
                st.session_state.generated_question_data = None
                st.session_state.distractors          = []
                st.session_state.hints                = []
                st.session_state.checked              = False
                st.session_state.user_answer          = None
                st.session_state.hints_revealed       = 0
                st.session_state.answer_revealed      = False
                st.session_state.question_source      = "race_original"
            else:
                st.error("Could not load RACE data. Check data/raw/val.csv.")

    with col_toggle:
        st.session_state.use_generated_q = st.toggle(
            "Use AI-generated question", value=st.session_state.use_generated_q,
            help="When ON, Model A generates a question from the article instead of "
                 "using the RACE original.")

    # ── Article text area ──────────────────────────────────────────────────────
    st.markdown("#### Reading Passage")
    article_input = st.text_area(
        "Paste your article here:",
        value=st.session_state.article,
        height=220,
        placeholder="Paste a reading passage here, or click 'Load Random RACE Sample' →",
        key="article_ta",
    )
    st.session_state.article = article_input

    # ── Manual question + options (shown / editable when AI gen is OFF) ────────
    if not st.session_state.use_generated_q:
        st.markdown("#### Question (manual entry — AI generation is OFF)")
        q_input = st.text_input(
            "Question:", value=st.session_state.question, key="question_ta",
            placeholder="Enter your multiple-choice question..."
        )
        st.session_state.question = q_input

    st.markdown("#### Answer Options (A / B / C / D)")
    opt_cols = st.columns(2)
    for i, label in enumerate("ABCD"):
        with opt_cols[i % 2]:
            val = st.text_input(
                f"Option {label}:",
                value=st.session_state.options.get(label, ""),
                key=f"opt_{label}",
            )
            if st.session_state.options is None:
                st.session_state.options = {}
            st.session_state.options[label] = val

    correct_label_input = st.selectbox(
        "Correct answer label (required for verification):",
        ["A", "B", "C", "D"],
        index=["A", "B", "C", "D"].index(
            st.session_state.correct_label or "A"),
        key="correct_label_sel",
    )
    st.session_state.correct_label = correct_label_input
    st.session_state.correct_text  = st.session_state.options.get(
        correct_label_input, "")

    st.divider()

    # ── Submit ─────────────────────────────────────────────────────────────────
    if st.button("🚀 Submit — Run Model A & Model B", type="primary",
                 use_container_width=True):
        article = st.session_state.article.strip()
        options = st.session_state.options

        if not article:
            st.error("Please paste a reading passage before submitting.")
        elif not all(options.get(k, "").strip() for k in "ABCD"):
            st.error("Please fill in all 4 answer options.")
        else:
            row_dict = {
                "article":  article,
                "question": st.session_state.question or "What is this passage about?",
                "A": options["A"], "B": options["B"],
                "C": options["C"], "D": options["D"],
                "answer": st.session_state.correct_label or "A",
            }

            with st.spinner("Running Model A (question generation + verification) "
                            "and Model B (distractors + hints)…"):
                result = run_full_inference(row_dict)

            gen_q_data = result["generated_question"]
            st.session_state.generated_question_data = gen_q_data

            # Decide which question to display in the quiz
            if st.session_state.use_generated_q:
                st.session_state.question       = gen_q_data["question"]
                st.session_state.question_source = "generated"
            else:
                st.session_state.question_source = "manual"

            st.session_state.verify_result  = result["verify_result"]
            st.session_state.distractors    = result["distractors"]
            st.session_state.hints          = result["hints"]
            st.session_state.checked        = False
            st.session_state.user_answer    = None
            st.session_state.hints_revealed = 0
            st.session_state.answer_revealed = False

            # Show generated question preview
            if st.session_state.use_generated_q:
                st.success(
                    f"✅ Inference complete!\n\n"
                    f"**AI-Generated Question** ({gen_q_data['method']}):\n\n"
                    f"> {gen_q_data['question']}\n\n"
                    f"Navigate to **Quiz View** to answer."
                )
            else:
                st.success("✅ Inference complete! Navigate to **Quiz View** →")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Quiz View
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_screen == "quiz_view":
    st.markdown("""
    <div class="main-header">
        <h2>📝 Quiz View</h2>
        <p>Answer the question — Model A (OHE ensemble) verifies your response.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.article or not st.session_state.question:
        st.info("No question loaded yet. Go to **Article Input** and submit first.")
    else:
        with st.expander("📖 Reading Passage", expanded=True):
            art = st.session_state.article
            st.write(art[:1200] + ("…" if len(art) > 1200 else ""))

        st.divider()

        # Badge showing question source
        src = st.session_state.question_source
        if src == "generated":
            st.markdown('<span class="ai-badge">🤖 AI-Generated Question (Model A)</span>',
                        unsafe_allow_html=True)
        elif src == "race_original":
            st.markdown('<span class="ai-badge">📚 RACE Original Question</span>',
                        unsafe_allow_html=True)

        st.subheader(f"❓ {st.session_state.question}")

        # Show generated question alternatives
        gen_data = st.session_state.generated_question_data
        if gen_data and gen_data.get("candidates"):
            with st.expander("💡 Other AI-generated question candidates (Model A)"):
                for i, cand in enumerate(gen_data["candidates"], 1):
                    st.markdown(f"**{i}.** {cand['question']}  "
                                f"*(score: {cand['score']:.3f})*")

        options      = st.session_state.options or {}
        verify_result = st.session_state.verify_result

        # Model A prediction confidence chart
        if verify_result:
            with st.expander("🤖 Model A Prediction (expand to peek)", expanded=False):
                probs     = verify_result.get("probabilities", {})
                predicted = verify_result.get("predicted", "")
                fig = px.bar(
                    x=list(probs.keys()), y=list(probs.values()),
                    labels={"x": "Option", "y": "Confidence"},
                    title=f"Model A predicts: Option {predicted}  "
                          f"[{verify_result.get('feature_method', '')}]",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(showlegend=False, height=250)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Latency: {verify_result.get('latency_ms', 0):.1f} ms")

        # Answer selection
        if not st.session_state.checked:
            selected = st.radio(
                "Select your answer:",
                options=list(options.keys()),
                format_func=lambda k: f"**{k}**:  {options.get(k, '')}",
                key="user_answer_radio",
            )
            st.session_state.user_answer = selected

            col_check, col_hint = st.columns([2, 1])
            with col_check:
                if st.button("✅ Check Answer", type="primary",
                             use_container_width=True):
                    st.session_state.checked = True
                    correct_l    = st.session_state.correct_label
                    user_correct = (selected == correct_l)
                    model_pred   = (verify_result.get("predicted")
                                    if verify_result else None)
                    st.session_state.session_log.append({
                        "question":        st.session_state.question[:80],
                        "question_source": st.session_state.question_source,
                        "user_answer":     selected,
                        "correct_answer":  correct_l,
                        "user_correct":    user_correct,
                        "model_predicted": model_pred or "?",
                        "model_correct":   (model_pred == correct_l
                                            if model_pred else None),
                        "latency_ms":      (verify_result.get("latency_ms", 0)
                                            if verify_result else 0),
                    })
                    st.rerun()
            with col_hint:
                if st.button("💡 Get a Hint", use_container_width=True):
                    st.session_state.current_screen = "hint_panel"
                    st.rerun()

        else:
            user_ans     = st.session_state.user_answer
            correct_l    = st.session_state.correct_label
            correct_text = st.session_state.correct_text or options.get(correct_l, "")
            user_correct = (user_ans == correct_l)

            for label, text in options.items():
                if label == correct_l:
                    st.markdown(f"""
                    <div class="correct-answer">
                        ✅ <strong>{label}: {text}</strong> — Correct Answer
                    </div>
                    """, unsafe_allow_html=True)
                elif label == user_ans and not user_correct:
                    st.markdown(f"""
                    <div class="wrong-answer">
                        ❌ <strong>{label}: {text}</strong> — Your Answer
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"**{label}:** {text}")
                st.markdown("")

            if user_correct:
                st.success("🎉 Correct! Well done.")
            else:
                st.error(f"❌ Incorrect. Correct answer: **({correct_l}) {correct_text}**")
            
            # Simple explanation text generated by using our strongest hint (Hint 3)
            if st.session_state.hints and len(st.session_state.hints) >= 3:
                with st.expander("📝 View Explanation", expanded=True):
                    st.info(f"**Explanation extracted from passage:**\n\"{st.session_state.hints[2]}\"")

            if verify_result:
                model_pred  = verify_result.get("predicted", "?")
                model_right = (model_pred == correct_l)
                c1, c2 = st.columns(2)
                c1.metric("Your Answer", user_ans,
                          delta="✓ Correct" if user_correct else "✗ Wrong",
                          delta_color="normal" if user_correct else "inverse")
                c2.metric("Model A Predicted", model_pred,
                          delta="✓ Correct" if model_right else "✗ Wrong",
                          delta_color="normal" if model_right else "inverse")

            with st.expander("🎭 Model B — Generated Distractors"):
                distractors = st.session_state.distractors or []
                if distractors:
                    for i, d in enumerate(distractors, 1):
                        st.markdown(f"**Distractor {i}:** {d}")
                else:
                    st.info("No distractors generated.")

            if st.button("➡️ Next Question", use_container_width=True,
                         type="primary"):
                for k in ("checked", "user_answer", "hints_revealed",
                          "answer_revealed", "article", "question",
                          "generated_question_data", "verify_result",
                          "distractors", "hints"):
                    st.session_state[k] = defaults[k]
                st.session_state.current_screen = "article_input"
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — Hint Panel
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_screen == "hint_panel":
    st.markdown("""
    <div class="main-header">
        <h2>💡 Hint Panel</h2>
        <p>Graduated hints from Model B — most general → near-explicit. Use wisely!</p>
    </div>
    """, unsafe_allow_html=True)

    hints = st.session_state.hints or []
    if not hints:
        st.info("No hints available. Submit an article and question first.")
    else:
        st.subheader(f"❓ {st.session_state.question}")
        src = st.session_state.question_source
        if src == "generated":
            st.caption("🤖 AI-Generated Question (Model A)")
        st.divider()

        hint_labels = ["🌐 Hint 1 (General)",
                       "🔍 Hint 2 (Specific)",
                       "🎯 Hint 3 (Near-Explicit)"]

        tabs = st.tabs(hint_labels)

        for i, tab in enumerate(tabs):
            with tab:
                if i <= st.session_state.hints_revealed:
                    st.info(hints[i] if i < len(hints) else "Hint not available.")
                    if i == st.session_state.hints_revealed and i < 2:
                        if st.button(f"Unlock Next Hint", key=f"unlock_btn_{i}"):
                            st.session_state.hints_revealed = i + 1
                            st.rerun()
                else:
                    st.warning("🔒 *Hint locked* — reveal the previous hint first.")

        st.divider()

        if st.session_state.hints_revealed >= 3:
            if not st.session_state.answer_revealed:
                if st.button("🔑 Reveal Answer", type="primary",
                             use_container_width=True):
                    st.session_state.answer_revealed = True
                    st.rerun()
            else:
                cl   = st.session_state.correct_label
                ctxt = st.session_state.correct_text
                st.success(f"✅ Answer: **({cl})** {ctxt}")

                with st.expander("🎭 Model B — Generated Distractors"):
                    for i, d in enumerate(st.session_state.distractors or [], 1):
                        st.markdown(f"**Distractor {i}:** {d}")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 4 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_screen == "analytics":
    st.markdown("""
    <div class="main-header">
        <h2>📊 Analytics Dashboard</h2>
        <p>Model performance metrics, session stats, and inference log.</p>
    </div>
    """, unsafe_allow_html=True)

    model_a_path = "models/model_a/results.json"
    model_b_path = "models/model_b/results.json"
    eval_path    = "models/evaluation_report.json"

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 Model A Metrics",
        "🎭 Model B Metrics",
        "📈 BLEU/ROUGE/METEOR",
        "📋 Session Log",
        "⚙️ System Info",
    ])

    # ── Tab 1: Model A ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Model A — Answer Verification Performance")
        st.caption("Feature representation: **One-Hot Encoding (primary)**")
        if os.path.exists(model_a_path):
            with open(model_a_path) as f:
                res_a = json.load(f)

            rows = [
                {
                    "Model": name.replace("_", " ").title(),
                    "Accuracy": f"{m['accuracy']:.4f}",
                    "Macro F1": f"{m.get('macro_f1', 0):.4f}",
                }
                for name, m in res_a.items() if "accuracy" in m
            ]
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Accuracy",
                    x=[r["Model"] for r in rows],
                    y=[float(r["Accuracy"]) for r in rows],
                    marker_color="#1565c0"))
                fig.add_trace(go.Bar(
                    name="Macro F1",
                    x=[r["Model"] for r in rows],
                    y=[float(r["Macro F1"]) for r in rows],
                    marker_color="#0288d1"))
                fig.update_layout(barmode="group",
                                  title="Model A — Accuracy & Macro F1",
                                  yaxis_range=[0, 1], height=350)
                st.plotly_chart(fig, use_container_width=True)

            if "kmeans" in res_a:
                st.subheader("Unsupervised — K-Means Clustering (OHE features)")
                km = res_a["kmeans"]
                c1, c2 = st.columns(2)
                c1.metric("Silhouette Score", f"{km.get('silhouette', 0):.4f}")
                c2.metric("Cluster Purity",   f"{km.get('purity', 0):.4f}")

            if "label_propagation" in res_a:
                st.subheader("Semi-Supervised — Label Propagation (OHE features)")
                lp = res_a["label_propagation"]
                c1, c2 = st.columns(2)
                c1.metric("Accuracy (10% labeled)", f"{lp.get('accuracy', 0):.4f}")
                c2.metric("Macro F1",               f"{lp.get('macro_f1', 0):.4f}")

            if "question_ranker" in res_a:
                st.subheader("Question Generation Ranker (Model A — generation sub-task)")
                st.success("✅ Question ranker trained and available.")
        else:
            st.info("Train models first: `python src/train_all.py`")

    # ── Tab 2: Model B ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Model B — Distractor & Hint Generation")
        st.caption("Feature representation: **One-Hot Encoding (primary)**")
        if os.path.exists(model_b_path):
            with open(model_b_path) as f:
                res_b = json.load(f)

            if "distractor_ranker" in res_b:
                d = res_b["distractor_ranker"]
                st.markdown("**Distractor Ranker (Logistic Regression)**")
                cols = st.columns(4)
                cols[0].metric("Accuracy",  f"{d.get('accuracy', 0):.4f}")
                cols[1].metric("Macro F1",  f"{d.get('macro_f1', 0):.4f}")
                cols[2].metric("Precision", f"{d.get('precision', 0):.4f}")
                cols[3].metric("Recall",    f"{d.get('recall', 0):.4f}")

                if "confusion_matrix" in d:
                    cm  = np.array(d["confusion_matrix"])
                    fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                                       labels={"x": "Predicted", "y": "True"},
                                       title="Distractor Ranker — Confusion Matrix",
                                       color_continuous_scale="Blues")
                    st.plotly_chart(fig_cm, use_container_width=True)

            if "hint_scorer" in res_b:
                h = res_b["hint_scorer"]
                st.markdown("**Hint Scorer (Logistic Regression)**")
                cols = st.columns(3)
                cols[0].metric("Accuracy",  f"{h.get('accuracy', 0):.4f}")
                cols[1].metric("Macro F1",  f"{h.get('macro_f1', 0):.4f}")
                cols[2].metric("Precision", f"{h.get('precision', 0):.4f}")
        else:
            st.info("Train models first: `python src/train_all.py`")

    # ── Tab 3: BLEU/ROUGE/METEOR ───────────────────────────────────────────────
    with tab3:
        st.subheader("Generation Evaluation — BLEU / ROUGE / METEOR")
        st.caption("As per instructor requirements. Run `python src/evaluate.py` to populate.")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                eval_res = json.load(f)

            for section_key, section_title in [
                ("distractor_evaluation", "Distractor Generation"),
                ("hint_evaluation",       "Hint Generation"),
                ("distractor_diversity",  "Distractor Diversity"),
                ("hint_graduation",       "Hint Graduation Quality"),
            ]:
                if section_key in eval_res:
                    sec = eval_res[section_key]
                    st.markdown(f"**{section_title}**")
                    display = {k: round(v, 4) for k, v in sec.items()
                               if isinstance(v, float)}
                    st.dataframe(
                        pd.DataFrame([display]).T.rename(columns={0: "Score"}),
                        use_container_width=True,
                    )
                    st.divider()
        else:
            st.info("No evaluation report found. Run: `python src/evaluate.py`")

    # ── Tab 4: Session Log ─────────────────────────────────────────────────────
    with tab4:
        st.subheader("Session Inference Log")
        log = st.session_state.session_log
        if not log:
            st.info("No inferences yet this session.")
        else:
            df_log = pd.DataFrame(log)
            st.dataframe(df_log, use_container_width=True)

            cum_correct = np.cumsum([1 if r["user_correct"] else 0 for r in log])
            cum_acc     = cum_correct / np.arange(1, len(log) + 1)
            fig2 = px.line(
                x=list(range(1, len(log) + 1)), y=cum_acc,
                labels={"x": "Question #", "y": "Cumulative Accuracy"},
                title="Your Running Accuracy",
            )
            fig2.update_yaxes(range=[0, 1])
            st.plotly_chart(fig2, use_container_width=True)

            latencies = [r.get("latency_ms", 0) for r in log]
            st.metric("Avg Inference Latency", f"{np.mean(latencies):.1f} ms")

            csv = df_log.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Session Log (CSV)", csv,
                               "session_log.csv", "text/csv",
                               use_container_width=True)

    # ── Tab 5: System Info ─────────────────────────────────────────────────────
    with tab5:
        st.subheader("System Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model Files**")
            model_files = [
                ("OHE Vectorizer (primary)",   "models/ohe_vectorizer.pkl"),
                ("TF-IDF Vectorizer (optional)","models/tfidf_vectorizer.pkl"),
                ("Model A — LR",               "models/model_a/logistic_regression.pkl"),
                ("Model A — SVM",              "models/model_a/svm.pkl"),
                ("Model A — NB",               "models/model_a/naive_bayes.pkl"),
                ("Model A — RF",               "models/model_a/random_forest.pkl"),
                ("Model A — Ensemble",         "models/model_a/ensemble_bundle.pkl"),
                ("Model A — K-Means",          "models/model_a/kmeans.pkl"),
                ("Model A — Q-Ranker",         "models/model_a/question_ranker.pkl"),
                ("Model B — Distractor",       "models/model_b/distractor_ranker.pkl"),
                ("Model B — Hint",             "models/model_b/hint_scorer.pkl"),
            ]
            for name, path in model_files:
                exists = "✅" if os.path.exists(path) else "❌"
                st.markdown(f"{exists} `{name}`")

        with col2:
            st.markdown("**Data Files**")
            data_files = [
                ("Train CSV",        "data/raw/train.csv"),
                ("Val CSV",          "data/raw/val.csv"),
                ("Test CSV",         "data/raw/test.csv"),
                ("X_train (OHE)",    "data/processed/X_train_ohe.npy"),
                ("X_val   (OHE)",    "data/processed/X_val_ohe.npy"),
                ("Evaluation report","models/evaluation_report.json"),
            ]
            for name, path in data_files:
                exists = "✅" if os.path.exists(path) else "❌"
                st.markdown(f"{exists} `{name}`")

        st.divider()
        st.json({
            "Course":           "Artificial Intelligence (AL2002)",
            "University":       "FAST NUCES Islamabad",
            "Dataset":          "RACE (ReAding Comprehension from Examinations)",
            "UI Framework":     "Streamlit",
            "Feature Method":   "One-Hot Encoding — primary (TF-IDF optional)",
            "Models":           "LR, SVM, NB, RF, K-Means, LabelPropagation, Q-Ranker",
            "Evaluation":       "BLEU, ROUGE, METEOR (generation tasks); "
                                "Accuracy/F1 (verification tasks)",
        })
        st.caption("⚠️ This system uses AI-generated content. "
                   "All generated questions and distractors should be reviewed by a "
                   "human instructor before use in real assessments.")
