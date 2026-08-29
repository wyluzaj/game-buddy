# game-buddy

Terminal AI companion for casual conversations while gaming. The agent uses Pydantic AI,
keeps several layers of memory, compacts long conversations, can search the web, and
can turn recorded gameplay into source-linked context. It optionally sends traces to
Pydantic Logfire.

## Features

- Interactive model selection with same-tier fallback models.
- Raw conversation persistence in SQLite.
- Time-based short-term memory compaction with a recent-message floor.
- Persistent long-term Harness Memory backed by SQLite.
- Automatic consolidation of completed sessions into long-term memory.
- Searchable session reports and game-specific memory profiles.
- Source-aware memory recall reduced by a dedicated Context Butler.
- Gemini overview and on-demand detailed analysis of local gameplay clips.
- Bounded Video Context Butler state injected into the correct chat session.
- Video observations preserved in session reports and game-specific memory.
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
  -> global durable facts update main/MEMORY.md
  -> detailed reports remain under main/sessions/session-<id>.md
  -> game-specific progress and goals update main/games/<game>.md
  -> a SQLite index links sessions to their game profiles
  -> Harness injects bounded global memory; recall_memory retrieves older detail
```

The long-term analyzer treats user statements as the source of facts. Assistant-only claims,
greetings, temporary details, and empty sessions are not promoted to durable memory. A failed
session remains pending so it can be retried without blocking later sessions.

The raw database
is retained for persistence and as a fallback source for memory analysis.

## Gameplay video analysis

Video support has two analysis modes:

1. **Overview** uploads a completed local clip to Gemini and stores a structured report
   containing a summary, timestamped events, visible text, confidence, and uncertainties.
2. **Detail** is invoked by the main agent only when the user asks about a specific visual
   or audible detail and a `video://segment/...` source is available. It re-reads the
   requested clip or timestamp range and persists the question-specific answer.

The overview flow is:

```text
local clip
  -> video_segments metadata in data/chatbot.db
  -> Gemini VideoOverview in video_reports
  -> Video Context Butler replacement snapshot in video_context_updates
  -> next run of the matching conversation receives one explicit screen observation
  -> successful injection sets injected_at
```

The Butler keeps a single replacement state instead of appending every clip. Ordinary
movement and duplicate observations can produce `changed=false`. Timestamped source links
are generated and validated by the application; the model cannot invent arbitrary local
paths. The `analyze_video_detail` tool accepts only application-owned
`video://segment/<id>` URIs and caches identical question/range reports.

Important video observations are added to the completed session report with their source
URIs. They may update the profile of a game explicitly associated with the conversation,
but they are not treated as user statements and cannot create global facts in `MEMORY.md`.

### Manual video flow

Start a chat and send one message so the conversation exists. In another terminal, obtain
the active ID:

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/chatbot.db'); print(c.execute('SELECT id FROM conversations WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1').fetchone())"
```

Analyze a completed recording and associate it with that conversation:

```powershell
python -m buddy.video_cli "C:\path\to\recording.mp4" --duration-seconds 120 --conversation-id 1
```

The next user turn receives the compact observed-game state. If the user asks for a detail
and the state contains a source URI, the agent can call `analyze_video_detail` automatically.

Automatic window-specific recording, segmentation, and time-driven agent rounds are not
part of this branch; they belong to the follow-up capture/automation task.

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
| Main chat model | `openrouter:nvidia/nemotron-3-ultra-550b-a55b:free` | Other configured models in the `high` tier |
| Short-term summarizer | `google:gemini-3.1-flash-lite` | Other configured models in the `medium` tier |
| Long-term and game-memory analyzers | `google:gemini-3.1-flash-lite` | Other configured models in the `medium` tier |
| Memory and video Context Butlers | `google:gemini-3.1-flash-lite` | Other configured models in the `medium` tier |
| Native overview/detail video analysis | `gemini-3.1-flash-lite` | Provider errors are surfaced for retry; no cross-provider video fallback |

The main chat and text analyzers use the shared `build_model()` function. It preserves the
configured model as primary and adds other models from the same tier as ordered fallbacks.
Native Gemini video upload/generation uses `google-genai` directly and is retried by the
application workflow instead of falling back to a text-only provider.

Important settings:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `HISTORY_MINUTES` | `30` | Time window preserved as raw short-term context |
| `MIN_RECENT_MESSAGES` | `15` | Minimum number of recent messages kept unchanged |
| `SHORT_TERM_MEMORY_MODEL` | Gemini 3.1 Flash Lite | Model generating timed summaries |
| `LONG_TERM_MEMORY_MODEL` | Gemini 3.1 Flash Lite | Model curating durable memory |
| `CONTEXT_BUTLER_MODEL` | Gemini 3.1 Flash Lite | Model reducing recalled documents |
| `VIDEO_OVERVIEW_MODEL` | Gemini 3.1 Flash Lite | Native fast clip analysis |
| `VIDEO_DETAIL_MODEL` | Gemini 3.1 Flash Lite | Native question-specific clip analysis |
| `VIDEO_CONTEXT_BUTLER_MODEL` | Gemini 3.1 Flash Lite | Model maintaining bounded current gameplay state |
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
| `data/chatbot.db` | Conversations, raw messages, memory index/relations, video segment metadata, overview/detail reports, and compact video-context updates |
| `data/memory.db` | Harness SQLite store containing logical `main/MEMORY.md`, `main/sessions/*.md`, and `main/games/*.md` documents |
| Local recording path | Original video bytes; the database stores the normalized path and stable `video://` ID, not the video blob |

`MEMORY.md` is a logical Harness memory path stored in SQLite; it is not a standalone Markdown
file in the repository.

## Running tests

Run the complete suite from the repository root:

```bash
python -m pytest -q
```

The tests cover model fallback selection, database migrations, timed compaction, Harness
memory, session/game indexing and recall, provider cleanup, idempotent video analysis,
source validation, context replacement/injection, and detailed video report caching.

## Project structure

```text
game-buddy/
├── buddy/
│   ├── agent.py               # Agent, capabilities, search, recall, and video tools
│   ├── config.py              # Models and memory/compaction settings
│   ├── database.py            # Raw conversations and session-state SQLite access
│   ├── long_term_memory.py    # Harness Memory with a SQLite store
│   ├── main.py                # Terminal loop, startup consolidation, and Logfire
│   ├── memory_analysis.py     # Completed-session long-term memory curator
│   ├── memory_index.py        # Session/game search index and relations
│   ├── memory_recall.py       # Bounded historical recall with Context Butler
│   ├── prompts.py             # Centralized prompts and model instructions
│   ├── timed_compaction.py    # Custom time-based summarizing compaction
│   ├── video_cli.py           # Manual end-to-end clip analysis command
│   ├── video_pipeline.py      # Overview -> Video Context Butler orchestration
│   ├── video_storage.py       # Video metadata and reports in SQLite
│   ├── video_context_*.py     # Bounded state selection and chat injection
│   └── video_detail_*.py      # On-demand detail service and agent tool
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
