# game-buddy

Terminal AI companion for casual conversations while gaming. The agent uses Pydantic AI,
keeps several layers of memory, compacts long conversations, can search the web, and
optionally sends traces to Pydantic Logfire.

## Features

- Interactive model selection with same-tier fallback models.
- Raw conversation persistence in SQLite.
- Time-based short-term memory compaction with a recent-message floor.
- Persistent long-term Harness Memory backed by SQLite.
- Automatic consolidation of completed sessions into long-term memory.
- Same-tier fallback models for chat, short-term summarization, and long-term analysis.
- Old tool-result cleanup with `ClearToolResults`.
- DuckDuckGo search for questions the agent cannot answer reliably.
- Optional Pydantic AI tracing with Logfire.

## Memory and compaction

The application uses three levels of conversation context:

1. **Immediate memory**: the current session's processed Pydantic AI message history is
   held in RAM and passed to the next agent run. Raw message batches are also written to
   `data/chatbot.db`.
2. **Short-term memory**: `SummarizingCompactionTimed`, a custom time-based variant of
   Harness [`SummarizingCompaction`](https://pydantic.dev/docs/ai/harness/compaction/),
   replaces expired messages with a structured summary. By default it keeps everything
   from the last 30 minutes and always keeps at least 15 recent messages. Existing summaries
   are updated incrementally using only newly expired messages. Its configured summarizer
   uses the same tier-based fallback policy as the main chat model.
3. **Long-term memory**: Harness [`Memory`](https://pydantic.dev/docs/ai/harness/memory/)
   stores a logical `main/MEMORY.md` notebook inside `data/memory.db`. Its bounded content is
   injected into later model requests and remains available between application sessions.
   The model that consolidates completed sessions also uses the tier-based fallback policy.

The long-term flow is:

```text
current chat
  -> raw messages and final session snapshot in data/chatbot.db
  -> session is marked as completed
  -> next application startup analyzes unprocessed completed sessions
  -> durable facts, preferences, goals, and decisions update main/MEMORY.md
  -> Harness Memory injects the updated notebook into future requests
```

The long-term analyzer treats user statements as the source of facts. Assistant-only claims,
greetings, temporary details, and empty sessions are not promoted to durable memory. A failed
session remains pending so it can be retried without blocking later sessions.

The raw database
is retained for persistence and as a fallback source for memory analysis.

### Tool-result compaction

Harness `ClearToolResults` activates when the estimated history exceeds 50,000 tokens. It
clears old tool outputs while preserving the three most recent tool-call/tool-return pairs.
This reduces context usage without orphaning tool calls.

### Timed compaction rules

Before a model request, `SummarizingCompactionTimed`:

- treats messages older than `HISTORY_MINUTES` as summary candidates;
- treats a missing message timestamp as old;
- clamps the cutoff so at least `MIN_RECENT_MESSAGES` remain unchanged;
- moves the cutoff backward when necessary to preserve tool-call/tool-return integrity;
- does nothing when no new expired messages need summarizing.


## Web search

The agent has Pydantic AI's `duckduckgo_search` tool. Its instructions tell it to search when
it does not know a factual answer instead of guessing. DuckDuckGo search does not require a
separate API key.

## Setup

1. Clone the repository and enter the directory:

   ```bash
   git clone <repo-url>
   cd game-buddy
   ```

2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

   Linux and macOS:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the application with development dependencies:

   ```bash
   python -m pip install -e ".[dev]"
   ```

4. Copy the environment template and add provider keys:

   ```powershell
   Copy-Item .env.example .env
   ```

   Linux and macOS:

   ```bash
   cp .env.example .env
   ```

   Get keys from:

   - [Google AI Studio](https://aistudio.google.com/apikey)
   - [OpenRouter](https://openrouter.ai/keys)

   `GOOGLE_API_KEY` is required by the default chat model and by the configured short- and
   long-term memory models. `OPENROUTER_API_KEY` is required when an OpenRouter model or
   fallback is used.

5. Run the application:

   ```bash
   python -m buddy.main
   ```

   The installed console entry point is equivalent:

   ```bash
   buddy
   ```

The application creates `data/chatbot.db` and `data/memory.db` automatically. Both contain
local user data and are ignored by Git.

## Models and configuration

The current model catalog is defined in `buddy/config.py`:

| Role | Primary/default | Fallback |
| --- | --- | --- |
| Main chat model | `google:gemini-3.5-flash` | None for the default `high` tier; selected models use a same-tier peer when available |
| Short-term summarizer | `google:gemini-3.1-flash-lite` | `openrouter:openai/gpt-oss-120b:free` |
| Long-term memory analyzer | `google:gemini-3.1-flash-lite` | `openrouter:openai/gpt-oss-120b:free` |

Available chat models also include Gemini 3.1 Flash Lite and the OpenRouter-hosted free
GPT-OSS 120B and 20B models. The main chat, short-term summarizer, and long-term analyzer all
use the shared `build_model()` function. It preserves the configured model as primary and
adds other models from the same tier as ordered fallbacks. Models alone in a tier have no
fallback.

Important settings:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `HISTORY_MINUTES` | `30` | Time window preserved as raw short-term context |
| `MIN_RECENT_MESSAGES` | `15` | Minimum number of recent messages kept unchanged |
| `SHORT_TERM_MEMORY_MODEL` | Gemini 3.1 Flash Lite | Model generating timed summaries |
| `LONG_TERM_MEMORY_MODEL` | Gemini 3.1 Flash Lite | Model curating durable memory |
| `MEMORY_DATABASE_PATH` | `data/memory.db` | Harness Memory SQLite store |

Prompts and model instructions are centralized in `buddy/prompts.py`.

## Optional Logfire tracing

The application always instruments Pydantic AI, but exports traces only when a Logfire token
is available:

```python
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
```

The application works normally without a Logfire account. To enable tracing, configure your
own Logfire project with the [Logfire CLI](https://logfire.pydantic.dev/docs/reference/cli/)
or provide `LOGFIRE_TOKEN` as an environment variable. Local `.logfire` credentials and
`.env` are not committed.

## Storage

| Path | Contents |
| --- | --- |
| `data/chatbot.db` | Conversations, raw Pydantic AI message batches, completed-session snapshots, and memory-analysis status |
| `data/memory.db` | Harness SQLite memory store containing the logical `main/MEMORY.md` notebook |

`MEMORY.md` is a logical Harness memory path stored in SQLite; it is not a standalone Markdown
file in the repository.

## Running tests

Run the complete suite from the repository root:

```bash
python -m pytest -q
```

The tests cover model fallback selection, database persistence and migration, timed
compaction rules, incremental summaries, Harness memory persistence and injection, and
completed-session consolidation.

## Project structure

```text
game-buddy/
├── buddy/
│   ├── agent.py               # Agent, capabilities, and DuckDuckGo tool
│   ├── config.py              # Models and memory/compaction settings
│   ├── database.py            # Raw conversations and session-state SQLite access
│   ├── long_term_memory.py    # Harness Memory with a SQLite store
│   ├── main.py                # Terminal loop, startup consolidation, and Logfire
│   ├── memory_analysis.py     # Completed-session long-term memory curator
│   ├── prompts.py             # Centralized prompts and model instructions
│   └── timed_compaction.py    # Custom time-based summarizing compaction
├── tests/
│   ├── conftest.py
│   ├── test_agent.py
│   ├── test_config.py
│   ├── test_database.py
│   ├── test_long_term_memory.py
│   └── test_memory_analysis.py
├── data/                      # Runtime SQLite databases (ignored by Git)
├── .env.example
├── pyproject.toml
└── README.md
```
