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
- English canonical memory and Porter-stemmed search with multilingual chat input.
- Independently configurable response language that explicit user requests override.
- Detailed Gemini episode timelines and on-demand re-analysis of one selected clip.
- Searchable timestamped episode observations and durable brief descriptions.
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

### Language model

Internal prompts and model-generated memory use English as the canonical language.
`MEMORY.md`, finalized session reports, game profiles, episode timelines, episode summaries,
and Butler context are therefore generated in English. Raw user/assistant messages remain
unchanged in `data/chatbot.db`; exact OCR and proper names also retain their original
spelling.

The response language is independent. `LANGUAGE.default_response_language` defaults to
`Polish`, but the user's latest explicit request in the current conversation takes priority
and remains active until changed again. For example, after “speak English” the agent replies
in English even though its internal memory policy does not change.

For historical retrieval, `recall_memory` keeps the complete `question` in the user's
language and receives short translated English terms in `query_en`. Canonical memory and
video FTS5 indexes use the English Porter stemmer, so related forms such as `star` and
`stars` match. Schema version 6 creates these indexes and backfills them from the retained
legacy indexes without deleting existing data.

## Gameplay video analysis

Video support stores two descriptions of every newly analyzed episode and offers a
third, on-demand mode:

1. **Full episode description** uploads a completed local clip to Gemini once and stores a
   detailed chronological timeline: actions, locations, objects, screen text, uncertainties,
   a possible game name, and confidence. Each observation is indexed for later text search.
2. **Brief episode description** is produced from that timeline by Video Butler. It preserves
   important places, events, discoveries, and problems for historical recall. Separately,
   Butler maintains a short current-state snapshot for the next chat turn.
3. **Detail re-analysis** is invoked only when stored observations do not answer a concrete
   visual or audible question and one `video://segment/...` candidate is available. It
   re-reads only that segment or timestamp range and caches the question-specific answer.

The episode flow is:

```text
local clip
  -> video_segments metadata in data/chatbot.db
  -> Gemini EpisodeAnalysis timeline in video_reports
  -> normalized observations in video_observations + FTS5 search index
  -> Video Butler brief description + replacement snapshot in video_context_updates
  -> next run of the matching conversation receives one explicit screen observation
  -> successful injection sets injected_at
```

The full timeline and brief description remain durable even when the current-state decision
is `changed=false`. The Butler keeps a single replacement state instead of appending every
clip; it includes at most six important facts and is bounded to 1,000 characters. Ordinary
movement still belongs in the timeline when it identifies a route, landmark, or location,
but duplicate or irrelevant movement need not change current chat context.

For questions about older gameplay, `recall_memory(memory_type="video")` searches timestamped
observations first. A matching observation selects a candidate episode; recall then returns
that episode's complete stored timeline up to the configured bound, including exact
timestamp sources. If no text match exists and a game is known, it returns a bounded set of
recent brief episode descriptions. The main agent may request detail analysis only after
stored evidence is insufficient and only for one concrete source. It does not ask Gemini to
scan an arbitrary two-hour recording or all stored clips. There is intentionally no visual
frame/embedding index in this version, so an object omitted from the detailed timeline can
only be recovered by re-analyzing a reasonably narrowed candidate episode.

Episode search first uses all supplied keywords for precision. If that produces no result,
it retries with a bounded any-keyword query while preserving game and date filters. This
makes retrieval tolerant of agent-added words such as a game title or `timestamp`.

Timestamped source links are generated and validated by the application; the model cannot
invent arbitrary local paths. The `analyze_video_detail` tool accepts only application-owned
`video://segment/<id>` URIs and caches identical question/range reports. Provider observations
outside the requested range are discarded. If none remain, the tool returns an uncertain,
controlled answer instead of trusting invalid timestamps. Detail prompts require absolute
decimal seconds (`00:07` means `7.0`, never `0.07`), and the service repairs provider output
when its textual answer proves that `mm.ss` notation was returned.

Brief episode descriptions are added to the completed session report with their source
URIs. They may update the profile of a game explicitly associated with the episode, but
they are not treated as user statements and cannot create global facts in `MEMORY.md`.

### Manual video flow

Start a chat and send one message so the conversation exists. In another terminal, obtain
the active ID:

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/chatbot.db'); print(c.execute('SELECT id FROM conversations WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1').fetchone())"
```

Analyze a completed recording and associate it with that conversation and game:

```powershell
python -m buddy.video.cli "C:\path\to\recording.mp4" --conversation-id 1 --game "Big Walk"
```

`--game` is optional. When supplied, it creates or reuses that confirmed game profile and
links the episode to it. Without `--game`, a high-confidence provider result (at least 0.9)
may link the episode only to an exact, already existing game profile; it never creates a new
profile from an unconfirmed guess. Use `--memory-database` together with `--database` when
testing against non-default database files.
For MP4, MOV, and M4V files the CLI reads duration directly from the container. Use
`--duration-seconds` only to override incorrect metadata or for another supported format;
unsupported containers fall back to `VIDEO.manual_duration_seconds` (120 seconds).

The next user turn receives the compact observed-game state. If the user asks for a detail
and the state contains a source URI, the agent can call `analyze_video_detail` automatically.

### UTF-8 audit export

Use the application exporter instead of PowerShell output redirection when inspecting
Unicode data:

```powershell
python -m buddy.audit --output-directory audit
```

The equivalent installed command is `buddy-audit`. It reads the databases without changing
them and writes `chat-database-audit.txt` plus `memory-documents-audit.txt` directly as UTF-8.
Custom databases can be selected with `--database` and `--memory-database`.

Automatic window-specific recording, segmentation, folder watching, active-game profile
injection, and time-driven agent rounds are not part of this branch; they belong to the
follow-up capture/automation task.

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

Python 3.10 through 3.14 is supported and tested in CI on 3.10 and 3.14. The project is
developed on Windows; the Python application and tests also run on Linux in CI. Video file
paths and the activation command are the only platform-specific parts of the documented
workflow.

1. Clone the repository and enter the directory:

   ```bash
   git clone https://github.com/wyluzaj/game-buddy.git
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

   For the exact dependency set verified in this branch, apply the lock file:

   ```bash
   python -m pip install -c requirements-lock.txt -e ".[dev]"
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

   `GOOGLE_API_KEY` is required by the configured short- and long-term memory models and
   native video analysis. `OPENROUTER_API_KEY` is required by the default chat model and
   whenever an OpenRouter fallback is used.

   | Variable | Required | Meaning |
   | --- | --- | --- |
   | `GOOGLE_API_KEY` | Yes for configured memory models and video | Gemini text and uploaded-video analysis |
   | `OPENROUTER_API_KEY` | Yes for default chat/fallbacks | OpenRouter chat models |
   | `LOGFIRE_TOKEN` | No | Exports Pydantic AI traces to the configured Logfire project |
   | `BUDDY_DEBUG` | No | Set to `true` to print tracebacks at terminal process boundaries |

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
| Native episode/detail video analysis | `gemini-3.1-flash-lite` | Provider errors are surfaced for retry; no cross-provider video fallback |

The main chat and text analyzers use the shared `build_model()` function. It preserves the
configured model as primary and adds other models from the same tier as ordered fallbacks.
Native Gemini video upload/generation uses `google-genai` directly and is retried by the
application workflow instead of falling back to a text-only provider.

Important settings:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `AGENT.history_minutes` | `30` | Time window preserved as raw short-term context |
| `AGENT.min_recent_messages` | `15` | Minimum number of recent messages kept unchanged |
| `LANGUAGE.default_response_language` | `Polish` | Fallback reply language when the user has not selected another |
| `MODELS.short_term_memory` | Gemini 3.1 Flash Lite | Model generating timed summaries |
| `MODELS.long_term_memory` | Gemini 3.1 Flash Lite | Model curating durable and game memory |
| `MODELS.memory_context_butler` | Gemini 3.1 Flash Lite | Model reducing recalled documents |
| `MODELS.video_episode` | Gemini 3.1 Flash Lite | Native full episode-timeline analysis |
| `MODELS.video_detail` | Gemini 3.1 Flash Lite | Native question-specific clip analysis |
| `MODELS.video_context_butler` | Gemini 3.1 Flash Lite | Model maintaining current gameplay state |
| `STORAGE.memory_database_path` | `data/memory.db` | Harness Memory SQLite store |
| `VIDEO.episode_summary_chars` | `1500` | Maximum durable brief-description size |
| `VIDEO.context_chars` | `1000` | Maximum current-state context size |
| `VIDEO.game_detection_confidence` | `0.9` | Minimum confidence for linking an existing game profile |
| `VIDEO.recall_candidate_multiplier` | `4` | Extra observation hits considered when selecting distinct episodes |
| `VIDEO.recall_episode_chars` | `20000` | Maximum stored timeline payload returned per candidate episode |

All code-level settings are immutable dataclass sections in `buddy/config.py`: `MODELS`,
`AGENT`, `RUNTIME`, `LANGUAGE`, `STORAGE`, `MEMORY`, and `VIDEO`. Provider credentials and runtime-only
diagnostics stay in environment variables. Prompts, instructions, and injection markers are
centralized in `buddy/prompts.py`.

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
| `data/chatbot.db` | Conversations, raw messages, memory index/relations, segment metadata, full episode/detail reports, searchable observations, brief descriptions, and compact video context |
| `data/memory.db` | Harness SQLite store containing logical `main/MEMORY.md`, `main/sessions/*.md`, and `main/games/*.md` documents |
| Local recording path | Original video bytes; the database stores the normalized path and stable `video://` ID, not the video blob |

`MEMORY.md` is a logical Harness memory path stored in SQLite; it is not a standalone Markdown
file in the repository.

### Privacy, backup, and deletion

- Conversations, memory documents, reports, and normalized recording paths are stored
  locally in the two SQLite databases. Original video bytes remain in their source files.
- Video episode and detail requests upload the selected clip to Google's Gemini service.
  The application requests deletion of the provider-side temporary file after generation,
  including failure paths, but provider retention is ultimately governed by Google's terms.
- Logfire receives traces only when `LOGFIRE_TOKEN` or local Logfire credentials are present.
  Do not enable it for sensitive sessions without reviewing the trace destination.
- For a consistent backup, exit Buddy first and copy `data/chatbot.db`, `data/memory.db`, and
  any recordings you need. Restore both databases as a pair because the chat database holds
  indexes and processing checkpoints for documents in the memory database.
- To delete all local application history, exit Buddy and remove both database files. This
  cannot delete copies already sent to an external model or tracing provider.

## Running tests

Run the complete suite from the repository root:

```bash
python -m pytest -q
```

Run the same local quality gates as CI:

```bash
ruff check buddy tests
pyright
pytest --cov=buddy --cov-report=term-missing --cov-fail-under=80 -q
```

The tests cover model fallback selection, database migrations, timed compaction, Harness
memory, session/game/video recall, provider cleanup, idempotent episode analysis, searchable
observations, game association, source validation, context replacement/injection, and
detailed video report caching.
CI runs these checks on Python 3.10 and 3.14. Pyright checks production code; test doubles
are validated through execution. When changing dependencies, update `pyproject.toml`, rebuild
the environment, regenerate `requirements-lock.txt`, and run all three commands.

## Troubleshooting

- **Missing provider key:** copy `.env.example` to `.env` and fill the key required by the
  selected model. OpenRouter fallback models can require `OPENROUTER_API_KEY` even when the
  primary model is Gemini.
- **Video analysis fails immediately:** verify that the file exists, has a supported suffix,
  is no longer being written, and that `GOOGLE_API_KEY` is valid. Retry with
  `BUDDY_DEBUG=true` for a traceback.
- **An old clip has only an overview:** reports made before the episode-timeline migration
  remain readable, but they do not gain a searchable full timeline without re-analysis.
- **Polish characters are broken only in redirected PowerShell logs:** the SQLite value may
  still be correct. Prefer `python -m buddy.audit`, which writes UTF-8 directly. For other
  redirected commands, run `chcp 65001` and set `[Console]::OutputEncoding =
  [System.Text.UTF8Encoding]::new()` first.
- **SQLite reports a lock:** close other Buddy or video-analysis processes and retry. Do not
  copy or delete either database while a process is writing it.
- **Memory was not updated:** finish the chat cleanly, then restart Buddy. Failed jobs remain
  checkpointed and are retried on a later startup.
- **Dependency behavior differs:** recreate `.venv` and install with
  `-c requirements-lock.txt`; the bounded ranges in `pyproject.toml` intentionally allow
  compatible upgrades.

For a detailed guide to module relationships, storage ownership, and runtime call flows, see
[`map-project.md`](map-project.md).

## Project structure

```text
game-buddy/
├── buddy/
│   ├── agent.py               # Main agent composition
│   ├── audit.py               # Read-only UTF-8 database and memory export
│   ├── config.py              # Grouped model, agent, memory, storage, and video settings
│   ├── database.py            # Raw conversations and database initialization
│   ├── main.py                # Terminal loop, startup consolidation, and Logfire
│   ├── prompts.py             # All LLM prompts, instructions, and injection markers
│   ├── memory/                # Compaction, durable memory, indexing, and recall
│   │   ├── analysis.py
│   │   ├── compaction.py
│   │   ├── context_butler.py
│   │   ├── game.py
│   │   ├── game_analysis.py
│   │   ├── index.py
│   │   ├── journal.py
│   │   ├── long_term.py
│   │   ├── recall.py
│   │   └── session.py
│   └── video/                 # Video domain, provider adapter, storage, and workflows
│       ├── analysis_service.py
│       ├── analyzer.py
│       ├── cli.py
│       ├── context_butler.py
│       ├── context_injection.py
│       ├── context_service.py
│       ├── detail_service.py
│       ├── detail_tool.py
│       ├── gemini.py
│       ├── models.py
│       ├── pipeline.py
│       ├── session_memory.py
│       └── storage.py
├── tests/
│   ├── conftest.py             # Shared in-memory database
│   ├── test_*memory*.py        # Memory, index, recall, and compaction tests
│   └── test_*video*.py         # Video domain, services, pipeline, storage, and CLI tests
├── data/                      # Runtime SQLite databases (ignored by Git)
├── .github/workflows/ci.yml   # Python 3.10/3.14 quality gates
├── .env.example
├── map-project.md             # Detailed module and runtime dependency map
├── pyproject.toml
├── requirements-lock.txt      # Exact resolved development environment
└── README.md
```
