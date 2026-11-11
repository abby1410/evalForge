# EvalForge

[![Tests](https://github.com/abby1410/evalForge/actions/workflows/tests.yml/badge.svg)](https://github.com/abby1410/evalForge/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A repeatable evaluation and prompt-regression workbench for LLM applications.**

EvalForge runs a fixed golden dataset against a configurable model, evaluates each answer with an LLM judge and deterministic checks, stores run history locally, and compares runs to surface regressions. Use the Streamlit dashboard to iterate on prompts and inspect results case by case.

> **Demo mode is for trying the workflow, not measuring model quality.** It uses a deterministic sample-response provider and a heuristic judge; its scores are not evidence of real model performance.

## What it does

- Runs 50 customer-support test cases from a versioned JSON dataset.
- Connects to OpenAI-compatible APIs, Gemini, or Ollama, or runs offline with the demo provider.
- Scores answers across correctness, relevance, groundedness, completeness, instruction following, and hallucination control.
- Applies deterministic checks for response presence, length, required or forbidden phrases, valid JSON, and expected refusals.
- Saves prompts, provider/model details, scores, and per-case outputs in a local SQLite database.
- Compares two runs by test-case ID and highlights score regressions and improvements.
- Exports the bundled dataset and individual run results as JSON.
- Runs tests in GitHub Actions on pushes and pull requests.

## Get started

### Requirements

- Python 3.11 or newer
- For real evaluations: credentials for the selected provider (Ollama can run locally)

### Install and launch

```bash
git clone https://github.com/abby1410/evalForge.git
cd evalForge

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional: configure a real model; demo mode works without this file.
cp .env.example .env

streamlit run app.py
```

Open the local URL printed by Streamlit (typically <http://localhost:8501>). The app starts with the demo target and demo judge, so you can run the complete workflow without API credentials.

## Configure a model provider

Set the relevant values in `.env`, then choose the target provider and judge in the app's sidebar. The default target provider is `demo`.

### OpenAI or an OpenAI-compatible endpoint

```dotenv
EVALFORGE_PROVIDER=openai
EVALFORGE_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-api-key
```

For an OpenAI-compatible service, set its base URL as well:

```dotenv
OPENAI_BASE_URL=https://your-compatible-endpoint/v1
```

The endpoint must support chat completions. A compatible endpoint may require its own API key.

### Gemini

```dotenv
EVALFORGE_PROVIDER=gemini
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-2.5-flash
```

### Ollama

Start Ollama and make a model available locally, then configure:

```dotenv
EVALFORGE_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2
```

### Judge selection and cost

The sidebar lets you choose the `demo` judge, the same provider as the target, or a separate OpenAI, Gemini, or Ollama judge. Demo judging is a deterministic heuristic—not an LLM evaluation. A real judge makes an additional model request for each test case. Running all 50 cases with both a real target and a real judge can make about 100 model calls; provider charges may apply.

The sample `.env.example` also documents these optional controls:

| Variable | Default | Purpose |
| --- | --- | --- |
| `EVALFORGE_MAX_WORKERS` | `3` | Maximum concurrent test-case workers |
| `EVALFORGE_REQUEST_TIMEOUT` | `60` | Provider request timeout in seconds |
| `EVALFORGE_DEFAULT_PASS_THRESHOLD` | `3.5` | Default average judge-score threshold (1–5) |
| `EVALFORGE_REGRESSION_TOLERANCE` | `0.25` | Default score-change tolerance for run comparisons |

## Run an evaluation

1. In **Run evaluation**, select the target provider/model and judge.
2. Edit the system prompt and set a run label and prompt version.
3. Click **Run all evaluations** and review the aggregate metrics and individual cases.
4. Change the prompt and run the same dataset again.
5. In **Compare runs**, select a baseline and candidate to inspect regressions and improvements.

A case passes only if its judge score meets the selected threshold **and** all applicable deterministic checks pass. Execution errors are recorded as failed cases. Comparisons match cases by ID, so use the same dataset when you want a directly comparable result.

## Golden dataset

The bundled dataset is [`data/golden_dataset.json`](data/golden_dataset.json). Each test case requires `id`, `category`, `question`, and `reference_answer`. Optional fields are `context`, `rubric`, `required_phrases`, `forbidden_phrases`, `min_length`, `max_length`, `expected_json`, and `must_refuse`.

Example:

```json
{
  "id": "REF-001",
  "category": "Refunds",
  "question": "How long do I have to request a refund?",
  "reference_answer": "Customers may request a refund within 30 days of purchase.",
  "context": "Refund policy: customers may request a refund within 30 days.",
  "rubric": "Be accurate and concise.",
  "required_phrases": [],
  "forbidden_phrases": [],
  "must_refuse": false
}
```

Use the **Dataset** tab to inspect or download the bundled cases and validate an uploaded JSON dataset. Uploading validates and previews the file; it does not persistently replace the dataset. To make another dataset the default, replace `data/golden_dataset.json` with the validated JSON file.

## Local data and privacy

Evaluation runs are stored in `evalforge.db` in the project directory. The database includes the prompt and per-case model outputs so that runs can be inspected and compared. The app reads credentials from `.env`; credentials are not written to the run history. The database and `.env` are excluded from Git by `.gitignore`.

Treat prompts, datasets, model outputs, and exported run files as potentially sensitive. Do not use real customer data unless you have approval to send it to the selected model provider and store it locally.

## Tests

```bash
pytest -q
```

GitHub Actions runs this test suite for pushes and pull requests.

## Project layout

```text
.
├── app.py                       # Streamlit dashboard
├── data/
│   └── golden_dataset.json      # Bundled evaluation cases
├── evalforge/
│   ├── config.py                # Environment-based settings
│   ├── dataset.py               # Dataset loading and validation
│   ├── judge.py                 # LLM and demo judging
│   ├── metrics.py               # Deterministic assertions
│   ├── providers.py             # Model-provider adapters
│   ├── regression.py            # Run comparison
│   ├── runner.py                # Evaluation orchestration
│   └── storage.py               # SQLite run history
├── tests/
└── requirements.txt
```

## Interpreting results

- LLM judges can be inconsistent, biased, or sensitive to wording. Treat their scores as estimates; calibrate rubrics and review important results with people.
- Reference answers are not a complete ground truth. Have domain owners maintain the golden dataset.
- Keep the model, judge, prompt, dataset, and model settings consistent when comparing runs. Providers may update models behind a stable model name.
- Costs are not estimated because provider pricing varies and changes. Token counts are recorded when returned by the provider.
- This is a local development and evaluation tool, not a multi-tenant production service.

## License

EvalForge is distributed under the [MIT License](LICENSE).
