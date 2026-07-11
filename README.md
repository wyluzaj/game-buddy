# game-buddy

Terminal chatbot for gaming discussions.

Built with [Pydantic AI](https://pydantic.dev/docs/ai/) and Python 3.10+.

## Setup

1. Clone the repository and enter the folder:
   ```
   git clone <repo-url>
   cd game-buddy
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate         # Windows
   source .venv/bin/activate      # Linux/Mac
   ```

3. Install the package (editable) with dev tools:
   ```
   pip install -e ".[dev]"
   ```
   This registers `buddy` as an importable package and installs pytest.

4. Copy the env template and fill in your API keys:
   ```
   copy .env.example .env         # Windows
   cp .env.example .env           # Linux/Mac
   ```
   Get the keys from:
   - [Google AI Studio](https://aistudio.google.com/apikey)
   - [OpenRouter](https://openrouter.ai/keys)

5. Run the chatbot (from the repo root):
   ```
   python -m buddy.main
   ```
   or, if the console script was installed:
   ```
   buddy
   ```
   On first run, a `data/chatbot.db` file will be created to persist conversations.

## Models

The default configuration in `buddy/config.py` uses free-tier models from two providers:

- **Google AI Studio** — Gemini 3.5 Flash (`high` tier), Gemini 3.1 Flash Lite (`medium` tier)
- **OpenRouter** — GPT-OSS 120B free (`medium` tier), GPT-OSS 20B free (`low` tier)

Both API keys (`GOOGLE_API_KEY`, `OPENROUTER_API_KEY`) are required — each authorizes
its own provider.

Models with peers in the same tier automatically fall back to each other on API errors
(rate-limits, 5xx). Models alone in a tier have no fallback.

To use different providers or paid models, edit the `Model` enum and `MODEL_TIERS`
dict in `buddy/config.py`. Any model string supported by Pydantic AI works — see the
[supported providers](https://pydantic.dev/docs/ai/models/overview/) list.

## Running tests

From the repo root:
```
pytest -v
```

## Project structure

```
game-buddy/
├── buddy/                    # Application package
│   ├── __init__.py
│   ├── config.py             # Model enum, tier map, history filter settings
│   ├── database.py           # SQLite: conversations + messages tables
│   ├── agent.py              # Pydantic AI Agent + keep_recent history processor
│   └── main.py               # Terminal loop, glues everything together
├── tests/                    # pytest suite (in-memory sqlite, monkeypatched config)
│   ├── conftest.py           # Shared `db` fixture
│   ├── test_config.py        # build_model + tier fallback logic
│   ├── test_database.py      # SQL round-trip and conversation isolation
│   └── test_agent.py         # keep_recent floor / passthrough behaviour
├── pyproject.toml            # Package metadata, dependencies, pytest config
├── README.md
├── .env.example              # Template for API keys (never commit real .env)
└── .gitignore
```
