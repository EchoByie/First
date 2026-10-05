# AI Core Console

A small Python framework that runs preset **protocols** against a local LLM
(through Ollama on `localhost:11434`) and shows the results in a full-screen
text dashboard.

## The core rule: code collects, the model interprets

- **Collectors** are plain Python. They read a file or run one fixed,
  read-only, allow-listed command.
- **The model gets no tools, no shell and no network.** It only receives the
  collected text and must answer in JSON.
- All collected data is treated as **untrusted**. Every answer is checked
  against a JSON schema before it is used.
- **Code checks the evidence.** Every finding must quote the evidence it is
  based on. The framework searches for those quotes in the collected data and
  labels each finding:
  - **✔ SURE**: "I'm sure that X". The finding is observed and all its evidence was found.
  - **? THINK**: "I think X, but I'm not sure". The finding is inferred, or its evidence could not be found.
- Network use: Ollama on localhost only. The one exception is the **Update**
  section, which downloads reference data (such as the IEEE MAC vendor
  registry) from a fixed list of URLs, and only when you ask it to.
- It isn't tied to any one model. Protocols ask for *roles* (`analyst`,
  `worker`), and `config/console.toml` decides which model fills each role.

## Status

Built in steps, with a review after each one.

| Step | What | State |
|---|---|---|
| 1 | Project skeleton, config, test setup | ✅ |
| 2 | Protocol manifests + discovery + `aicore list` | ✅ |
| 3 | Schema validation + evidence verification (SURE / THINK) | ✅ |
| 4 | Model backend (Ollama, localhost only) + role resolver | ✅ |
| 5 | Pipeline, audit log, reports, `run` / `dry-run` | ✅ |
| 6 | Digest protocol | ⏳ |
| 7 | Chunking (worker model → analyst model) | |
| 8 | Local reference database (SQLite) | |
| 9 | Passive network map protocol | |
| 10 | Health checks | |
| 11 | Update section (reference data downloads) | |
| 12 | Prompt-injection test suite | |
| 13 | Full-screen dashboard (Textual) | |
| 14 | Tuning against your real Ollama | |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                             # runs all tests; no Ollama needed
aicore version
aicore list                        # protocols found in protocols/
aicore models                      # installed models + which one plays each role
aicore dry-run <protocol> --file x # show exactly what would be sent; sends nothing
aicore run <protocol> --file x     # run it; report saved to reports/<protocol>/
```

Every run (including dry-runs, refusals and failures) adds one line to
`logs/audit.jsonl`. The log stores fingerprints (SHA-256) of your data and
the model's answer, never the data itself.

## Adding a protocol

Copy `protocols/_template/`, rename the folder, and edit `manifest.toml`
(every option is explained in comments). It is picked up automatically.
If something is wrong, `aicore list` tells you exactly what.

## Layout

```
config/console.toml   your settings (roles → models, paths, UI options)
core/                 the framework
protocols/            one folder per protocol (auto-discovered)
data/                 local reference database (created by the importer/updater)
tests/                tests using fake data, no Ollama, no network
```
