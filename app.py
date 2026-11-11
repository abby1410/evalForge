import json
import os
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from evalforge.config import settings
from evalforge.dataset import load_dataset, validate_dataset, export_dataset
from evalforge.runner import run_evaluation, DEFAULT_SYSTEM_PROMPT
from evalforge.storage import init_db, save_run, list_runs, get_run, build_run_record
from evalforge.regression import compare_runs
from evalforge.providers import ProviderError

st.set_page_config(page_title="EvalForge | LLM Evaluation Lab", page_icon="🧪", layout="wide")
init_db()

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; max-width: 1500px;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 14px; border-radius: 10px;}
.small-muted {color: #888; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)

if "prompt" not in st.session_state:
    st.session_state.prompt = DEFAULT_SYSTEM_PROMPT
if "last_run_id" not in st.session_state:
    st.session_state.last_run_id = None

def load_cases():
    try:
        return load_dataset()
    except Exception as exc:
        st.error(f"Could not load the bundled dataset: {exc}")
        return []

def render_run_detail(run, key_prefix="run_detail"):
    results = run["results"]
    df = pd.DataFrame([{
        "Case": r["case_id"], "Category": r["category"], "Score": r["overall_score"],
        "Passed": "PASS" if r["passed"] else "FAIL", "Latency (ms)": r["latency_ms"]
    } for r in results])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if results:
        selected = st.selectbox("Inspect test case", [r["case_id"] for r in results], key=f"{key_prefix}_detail_{run['id']}")
        item = next(r for r in results if r["case_id"] == selected)
        st.markdown(f"**Question**  \n{item['question']}")
        a, b = st.columns(2)
        with a:
            st.markdown("**Candidate response**")
            st.info(item.get("response") or item.get("error", "No response"))
        with b:
            st.markdown("**Reference answer**")
            st.success(item["reference_answer"])
        if item.get("scores"):
            st.markdown("**Judge scores**")
            score_df = pd.DataFrame([{"Metric": k.replace("_"," ").title(), "Score": v, "Judge rationale": item.get("rationale",{}).get(k,"")} for k,v in item["scores"].items()])
            st.dataframe(score_df, use_container_width=True, hide_index=True)
        st.markdown("**Deterministic checks**")
        st.json(item.get("deterministic_checks", {}))
        if item.get("error"):
            st.error(item["error"])

st.title("🧪 EvalForge")
st.caption("A repeatable evaluation and regression-testing harness for LLM applications.")
with st.sidebar:
    st.header("Evaluation setup")
    provider_options = ["demo", "openai", "gemini", "ollama"]
    default_provider = settings.provider if settings.provider in provider_options else "demo"
    provider = st.selectbox("Target provider", provider_options, index=provider_options.index(default_provider))
    if provider == "openai":
        model = st.text_input("Target model", value=settings.model)
    elif provider == "gemini":
        model = st.text_input("Target model", value=settings.gemini_model)
    elif provider == "ollama":
        model = st.text_input("Target model", value=settings.ollama_model)
    else:
        model = st.text_input("Target model", value="evalforge-demo-v1")
    judge_mode = st.selectbox("Judge", ["demo", "same", "openai", "gemini", "ollama"], index=0,
                              help="Demo is a deterministic heuristic, not an LLM judge. Select same or a provider for real LLM judging.")
    judge_model = st.text_input("Judge model (optional)", value=settings.judge_model)
    threshold = st.slider("Pass threshold (average / 5)", 1.0, 5.0, float(settings.pass_threshold), 0.1)
    st.divider()
    st.caption("Credentials are read from your local .env file. They are never stored in the SQLite run history.")

tab_run, tab_dataset, tab_history, tab_compare, tab_about = st.tabs(
    ["▶ Run evaluation", "📚 Dataset", "🕘 Run history", "⚖ Compare runs", "ℹ About"]
)

with tab_run:
    cases = load_cases()
    c1, c2, c3 = st.columns(3)
    c1.metric("Golden test cases", len(cases))
    c2.metric("Target", f"{provider} / {model}")
    c3.metric("Judge", "Demo heuristic" if judge_mode == "demo" else judge_mode)
    st.subheader("System prompt")
    st.caption("Edit this prompt, run the same dataset again, and compare the results against a previous run.")
    prompt = st.text_area("Prompt under test", key="prompt", height=190)
    run_label = st.text_input("Run label", value="Prompt experiment")
    prompt_version = st.text_input("Prompt version", value="v1")
    col1, col2 = st.columns([1, 4])
    with col1:
        run_clicked = st.button("Run all evaluations", type="primary", use_container_width=True, disabled=not cases)
    with col2:
        st.caption(f"Runs {len(cases)} cases. Real model calls may incur API costs.")
    if run_clicked:
        if not prompt.strip():
            st.error("System prompt cannot be empty.")
        else:
            progress = st.progress(0, text="Starting evaluation…")
            status = st.empty()
            try:
                with st.spinner("Running test cases and scoring responses…"):
                    results, errors = run_evaluation(
                        cases, prompt, provider_name=provider, model=model,
                        judge_mode=judge_mode, judge_model=judge_model or None,
                        threshold=threshold
                    )
                record = build_run_record(run_label, prompt_version, prompt, provider, model,
                                          "demo-heuristic" if judge_mode == "demo" else (judge_model or judge_mode),
                                          "bundled_golden_dataset_v1", results)
                run_id = save_run(record)
                st.session_state.last_run_id = run_id
                progress.progress(1.0, text="Evaluation complete.")
                if errors:
                    st.warning(f"{len(errors)} test case(s) failed during execution. See details below.")
                st.success(f"Saved evaluation run #{run_id}.")
                st.rerun()
            except ProviderError as exc:
                progress.empty()
                st.error(str(exc))
            except Exception as exc:
                progress.empty()
                st.exception(exc)

    if st.session_state.last_run_id:
        latest = get_run(st.session_state.last_run_id)
        if latest:
            st.divider()
            st.subheader(f"Latest run: #{latest['id']} — {latest['label']}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Average score", f"{latest['average_score']:.2f}/5")
            m2.metric("Pass rate", f"{latest['passed_cases']}/{latest['total_cases']}", f"{latest['passed_cases']/max(1,latest['total_cases']):.0%}")
            m3.metric("Average latency", f"{latest['average_latency_ms']:.0f} ms")
            m4.metric("Failed cases", latest["total_cases"]-latest["passed_cases"])
            render_run_detail(latest, key_prefix="latest")

with tab_dataset:
    st.subheader("Golden dataset")
    st.write("A versioned set of fixed questions and reference answers. Edit locally or upload your own JSON dataset.")
    cases = load_cases()
    st.metric("Bundled test cases", len(cases))
    if cases:
        ds = pd.DataFrame([{"ID":x["id"],"Category":x["category"],"Question":x["question"],"Reference answer":x["reference_answer"]} for x in cases])
        st.dataframe(ds, use_container_width=True, hide_index=True)
        st.download_button("Download golden dataset JSON", json.dumps(cases, indent=2), "golden_dataset.json", "application/json")
    uploaded = st.file_uploader("Upload custom JSON dataset", type=["json"])
    if uploaded:
        try:
            custom = json.loads(uploaded.getvalue())
            validate_dataset(custom)
            st.success(f"Valid dataset: {len(custom)} test cases.")
            st.json(custom[:2])
            st.info("To use this dataset in the current run, replace data/golden_dataset.json with your validated file. Persistent upload management is intentionally not implicit.")
        except Exception as exc:
            st.error(f"Invalid dataset: {exc}")

with tab_history:
    st.subheader("Saved evaluation runs")
    runs = list_runs()
    if not runs:
        st.info("No runs yet. Start an evaluation from the first tab.")
    else:
        hist = pd.DataFrame([{"Run":r["id"],"Created":r["created_at"][:19].replace("T"," "),"Label":r["label"],
                              "Prompt":r["prompt_version"],"Provider":r["provider"],"Model":r["model"],
                              "Score":r["average_score"],"Passed":f"{r['passed_cases']}/{r['total_cases']}"} for r in runs])
        st.dataframe(hist, use_container_width=True, hide_index=True)
        chosen = st.selectbox("Open run", [r["id"] for r in runs], format_func=lambda x: f"#{x} — {next(r['label'] for r in runs if r['id']==x)}")
        opened = get_run(chosen)
        if opened:
            render_run_detail(opened, key_prefix="history")
            st.download_button("Export run JSON", json.dumps(opened, indent=2), f"evalforge_run_{chosen}.json", "application/json")

with tab_compare:
    st.subheader("Prompt regression comparison")
    runs = list_runs()
    if len(runs) < 2:
        st.info("You need at least two saved runs to compare prompts.")
    else:
        ids = [r["id"] for r in runs]
        baseline_id = st.selectbox("Baseline run", ids, index=min(1, len(ids)-1), format_func=lambda x: f"#{x} — {next(r['label'] for r in runs if r['id']==x)}")
        candidate_id = st.selectbox("Candidate run", ids, index=0, format_func=lambda x: f"#{x} — {next(r['label'] for r in runs if r['id']==x)}")
        tolerance = st.slider("Regression tolerance", 0.05, 2.0, float(settings.regression_tolerance), 0.05)
        if baseline_id == candidate_id:
            st.warning("Choose two different runs.")
        else:
            base, cand = get_run(baseline_id), get_run(candidate_id)
            comparison = compare_runs(base, cand, tolerance)
            a,b,c = st.columns(3)
            a.metric("Average score change", f"{comparison['average_delta']:+.3f}")
            b.metric("Regressions", len(comparison["regressions"]))
            c.metric("Improvements", len(comparison["improvements"]))
            if comparison["regressions"]:
                st.error("Potential regressions detected")
                st.dataframe(pd.DataFrame(comparison["regressions"]), use_container_width=True, hide_index=True)
            else:
                st.success("No regressions detected at the selected tolerance.")
            if comparison["improvements"]:
                st.success("Improvements")
                st.dataframe(pd.DataFrame(comparison["improvements"]), use_container_width=True, hide_index=True)
            score_data = []
            for run, name in [(base,"Baseline"),(cand,"Candidate")]:
                for result in run["results"]:
                    score_data.append({"Case":result["case_id"],"Run":name,"Score":result["overall_score"]})
            fig = px.scatter(pd.DataFrame(score_data), x="Case", y="Score", color="Run", title="Per-case score comparison", range_y=[0,5])
            st.plotly_chart(fig, use_container_width=True)

with tab_about:
    st.subheader("What this project demonstrates")
    st.markdown("""
- **Evaluation-driven development:** treat prompts and model behavior as testable artifacts.
- **Golden datasets:** fixed inputs and references make repeated runs comparable.
- **LLM-as-a-judge:** structured, dimension-level scoring with rationale.
- **Deterministic assertions:** schema, required phrases, refusal expectations, and response constraints.
- **Regression testing:** detect case-level quality drops between prompt versions.
- **Run provenance:** persist prompt, model, provider, dataset label, scores, and outputs.
- **Provider abstraction:** demo, OpenAI-compatible APIs, Gemini, and Ollama.

### Important limitations
- LLM judge scores are noisy estimates, not ground truth. Calibrate rubrics against human review.
- A run is only strictly comparable when model, model settings, dataset, and judge configuration are held constant.
- Demo mode uses a deterministic fake support assistant and heuristic scoring. It proves the workflow, not model quality.
- Cost is not fabricated: provider token pricing varies and is not estimated unless you add a pricing table.
""")
    st.code("streamlit run app.py", language="bash")

