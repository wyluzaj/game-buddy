# Project Map

This document maps the production modules, runtime flows, storage layers, and tests of
`game-buddy`.

## Runtime flows

### Chat startup and shutdown

```text
buddy.main.main
  -> config.choose_model + config.build_model
  -> database.init_db
       -> memory.index.init_memory_index
       -> video.storage.init_video_storage
  -> memory.long_term.create_long_term_memory
  -> memory.analysis.process_pending_memories
       -> memory.journal.get_or_create_memory_job
       -> memory.session.persist_session_document
       -> journal checkpoint: SESSION_PERSISTED
       -> memory.analysis._apply_long_term_memory
       -> journal checkpoint: GLOBAL_APPLIED + structured analyzer result
       -> memory.game.link_session_to_games
       -> memory.game_analysis.update_game_profiles
       -> journal checkpoint: GAME_PROFILES_APPLIED
       -> mark conversation analyzed and remove completed journal row
  -> agent.build_agent
       -> memory.compaction.SummarizingCompactionTimed
       -> memory.long_term.InjectionOnlyMemory
       -> memory.recall.recall_memory
       -> video.detail_tool.analyze_video_detail
       -> DuckDuckGo search
  -> terminal chat loop
       -> video.storage.get_latest_pending_video_context
       -> video.context_injection.inject_video_context
       -> Agent.run_sync
       -> database.save_messages
       -> video.storage.mark_video_context_injected_through
  -> session shutdown
       -> database.finish_conversation
       -> video.session_memory.build_session_video_observations
       -> memory.session.persist_session_document
```

Completed sessions are processed at the next application startup. A failed memory update
remains pending and does not block later sessions.

### Episode video analysis

```text
buddy.video.cli
  -> database.init_db
  -> video.gemini.GeminiVideoAnalyzer
  -> video.pipeline.process_video_file
       -> video.analysis_service.register_and_analyze_video
            -> SHA-256 recording key from bytes, UTC times, and conversation ID
            -> video.storage.create_video_segment
            -> video.storage.claim_video_segment_for_analysis
            -> GeminiVideoAnalyzer.analyze_episode
            -> validate every timestamp against segment duration
            -> video.storage.save_video_report(mode=episode)
            -> normalize observations into video_observations + FTS5
       -> video.context_service.process_video_report_context
            -> video.context_butler.build_video_context_prompt
            -> Video Context Butler Agent
            -> persist an episode summary for historical recall
            -> decide whether current gameplay state changed
            -> source validation
            -> video.storage.save_video_context_update
```

The next turn of the matching conversation consumes the newest pending context update:

```text
video_context_updates
  -> get_latest_pending_video_context
  -> one replaceable [game-buddy.video-context.v1] system message
  -> main chat agent
  -> injected_at is set only after a successful agent run
```

The replacement message stays separate from user statements. Timed compaction preserves
only its newest version and never folds stale screen state into the rolling chat summary.
Unassigned clips (`conversation_id=None`) never receive a previous replacement state and
therefore cannot leak context between unrelated manual analyses.

### Detailed video analysis

```text
user asks about a visible or audible detail
  -> main agent finds a video://segment/... source in context
  -> analyze_video_detail tool
       -> video.detail_service.parse_video_source
       -> validate segment ownership and timestamp range
       -> build a canonical request key and acquire its in-process lock
       -> reuse an identical indexed detail report when available
       -> GeminiVideoAnalyzer.analyze_detail
       -> discard provider evidence outside the requested range
       -> if no valid evidence remains, return a controlled uncertain answer
       -> video.storage.save_video_report(mode=detail, request_key=...)
  -> compact answer + validated video:// sources return to the main agent
```

The tool never accepts an arbitrary filesystem path from the model. It resolves only
application-owned `video://segment/<id>` identifiers through SQLite.

### Historical memory and recall

```text
completed conversation
  -> data/chatbot.db
       -> conversations
       -> raw message batches
       -> final compacted snapshot
       -> video observations linked by conversation_id
  -> data/memory.db
       -> main/sessions/session-<id>.md
       -> main/MEMORY.md
       -> main/games/<game-slug>.md
  -> data/chatbot.db memory index
       -> memory_documents
       -> memory_search (FTS5)
       -> memory_relations (session -> game)
```

Future recall uses a bounded retrieval pipeline:

```text
recall_memory
  -> memory_type=video
       -> video.storage.search_episode_observations (FTS5, timestamped evidence)
       -> if no exact match and a game is known:
            video.storage.get_recent_episode_summaries (bounded fallback)
  -> memory_type=game/session/all
       -> memory.index.search_memory_documents
       -> MemoryRecallService reads selected logical Markdown documents
  -> Context Butler reduces the retrieved documents and/or video evidence
  -> source allow-list validation
  -> a small source-linked answer returns to the main agent
```

The main agent must search stored episode evidence before requesting a new Gemini
detail analysis. A detail request is allowed only for one concrete `video://` source
returned by current context or recall; the agent never scans an unbounded recording set.

Global memory is based on explicit user statements. Video observations may enrich a
confirmed game's profile and session report, but they cannot become global user facts by
themselves. Raw structured `ModelMessage` objects are the preferred evidence source;
complete user prompts are JSON-encoded one per line before they reach either memory
analyzer, so multiline content and role-like text inside a user message are preserved.
The global analyzer returns a structured replacement list in `global_facts` plus separate
`games_with_user_state`; `memory.analysis.render_global_memory` is the only component that
turns global facts into Markdown. This removes substring-based deletion of model output.

## Storage map

### `data/chatbot.db`

```text
conversations
  1 -> many messages
  1 -> many video_segments

video_segments
  recording_key = SHA-256(bytes + capture metadata)
  optional game_document_id -> memory_documents (game)
  1 -> many video_reports

video_reports (overview or episode)
  1 -> 0..1 video_context_updates

video_reports (episode)
  1 -> many video_observations
  indexed by video_observation_search (FTS5)

video_reports (detail)
  unique request_key -> one cached answer

memory_documents
  session --belongs_to_game--> game
  indexed by memory_search (FTS5)

memory_processing_jobs
  conversation_id -> ordered cross-database checkpoint + retry payload
```

Main responsibilities:

- raw conversation persistence and completion state;
- searchable metadata for logical memory documents;
- session-to-game relations;
- normalized paths to local video files and optional confirmed game profiles;
- overview/episode/detail reports, searchable observations, and compact context updates.

The database uses `PRAGMA user_version`. Schema version 2 migrated the original path-unique
video tables to recording keys and added indexed detail request keys. Version 3 adds the
memory-processing journal. Version 4 adds detailed episode reports, normalized searchable
observations, and optional segment-to-game links. Version 5 adds durable episode summaries
to context updates. Existing segment/report/context IDs are preserved. Storage reads
validate report JSON and context-source JSON before returning domain objects.

### `data/memory.db`

Harness `SqliteMemoryStore` contains logical Markdown paths rather than repository files:

```text
main/MEMORY.md
main/sessions/session-<conversation-id>.md
main/games/<normalized-game-name>.md
```

- `main/MEMORY.md` contains bounded global preferences, facts, and long-term goals.
- session documents contain the final short-memory report and source-linked gameplay
  observations;
- game documents contain game-specific character/build information, preferences, current
  goals, progress, and important events.

### Local video files

The original video bytes remain at their local filesystem path. SQLite stores metadata and
stable `video://` identifiers; it does not store video blobs. Gemini uploads are temporary
and are deleted after generation whenever cleanup succeeds.

## Production module map

| Module | Responsibility | Main dependencies / consumers |
| --- | --- | --- |
| `buddy/main.py` | Application composition, terminal loop, startup consolidation, session shutdown | Coordinates chat, memory, video injection, and persistence |
| `buddy/config.py` | Central immutable sections for models, runtime diagnostics, agent, storage, memory, video, and MIME types | Used by all factories, services, models, and entry points |
| `buddy/agent.py` | Main Pydantic AI agent factory | Registers compaction, global memory, recall, video detail, and web search |
| `buddy/prompts.py` | Central LLM instructions and prompt templates | Used by chat, memory, Butler, and video analyzers |
| `buddy/database.py` | SQLite connection, conversations, and raw message batches | Initializes memory-index and video-storage schemas |
| `buddy/memory/compaction.py` | Time-window short-memory compaction and history formatting | Main agent capability and session-memory formatter |
| `buddy/memory/long_term.py` | Injection-only Harness Memory backed by SQLite | Used by chat, recall, and post-session analyzers |
| `buddy/memory/session.py` | Logical `main/sessions/*.md` reports and indexing | Called at shutdown and pending-memory processing |
| `buddy/memory/index.py` | Memory metadata, FTS5 search, and session-to-game relations | Used by session/game memory and recall |
| `buddy/memory/analysis.py` | Completed-session global consolidation coordinator | Updates global memory, session documents, and game profiles |
| `buddy/memory/journal.py` | Durable cross-database checkpoints and retry payloads | Lets pending-memory processing resume after session, global, or game stages |
| `buddy/memory/game.py` | Game-name normalization, collision-safe paths, documents, and relations | Used by memory analysis and recall; adds a title hash only when two distinct titles share a slug |
| `buddy/memory/game_analysis.py` | Source-aware updates of allowed `main/games/*.md` profiles | Receives explicit user evidence and video observations |
| `buddy/memory/recall.py` | Bounded historical retrieval and reduction for game profiles, sessions, episode summaries, and timestamped video observations | Uses the memory/video indexes, Harness store, and Context Butler |
| `buddy/memory/context_butler.py` | Small agent reducing recalled memory documents | Used only by `memory.recall.py` |
| `buddy/video/models.py` | Video enums, dataclasses, legacy overview models, and detailed Episode Pydantic models | Shared contract for all video modules |
| `buddy/video/analyzer.py` | Provider-neutral episode/detail protocols and domain errors | Implemented by Gemini; consumed by application services |
| `buddy/video/gemini.py` | Gemini upload, polling, minimal provider-compatible JSON Schema generation, Pydantic parsing, and cleanup | Native provider adapter for episode, legacy overview, and detail modes; reports the failing provider stage |
| `buddy/video/storage.py` | Versioned video schema, validated report/observation persistence, FTS5 episode search, game links, state transitions, and indexed request cache | Repository used by every video service and the chat loop |
| `buddy/video/analysis_service.py` | Content-aware registration, atomic claim, episode timeline sanitization, retry, legacy-overview reuse, and persistence | Connects `VideoAnalyzer` to `video.storage` without losing an otherwise valid episode |
| `buddy/video/context_butler.py` | Episode-summary and replacement-state prompt construction plus source validation | Used by `video.context_service.py` |
| `buddy/video/context_service.py` | Idempotent episode-to-summary/current-context reduction, including late-report handling | Connects storage to Video Context Butler |
| `buddy/video/pipeline.py` | End-to-end episode and context-update orchestration | Used by the manual CLI and future capture worker |
| `buddy/video/context_injection.py` | Replaces prior gameplay state in chat history | Used by the terminal loop |
| `buddy/video/detail_service.py` | Internal URI parsing, range validation, cache, and detailed reports | Used by the detail tool |
| `buddy/video/detail_tool.py` | Main-agent tool for question-specific clip inspection | Creates a Gemini analyzer per invocation |
| `buddy/video/session_memory.py` | Bounded source-aware episode summaries for completed sessions | Used by shutdown and memory processing |
| `buddy/video/cli.py` | Manual end-to-end episode analysis command | Composes SQLite, Gemini, pipeline, and terminal output |

## Test map

| Test module | Main coverage |
| --- | --- |
| `tests/conftest.py` | Fresh shared SQLite `:memory:` fixture |
| `tests/test_config.py` | Model roles, tiers, and fallback selection |
| `tests/test_timed_compaction.py` | Timed compaction, incremental summaries, and history filtering |
| `tests/test_long_term_memory.py` | Harness store, injection-only behavior, and agent capabilities |
| `tests/test_database.py` | Conversations, raw messages, completion, schema versioning, and legacy migrations |
| `tests/test_memory_index.py` | FTS5 retrieval, relations, path validation, and dates |
| `tests/test_memory_recall.py` | Retrieval fallback, Context Butler, and source validation |
| `tests/test_memory_analysis.py` | Session consolidation, retry, games, and video evidence |
| `tests/test_game_memory_analysis.py` | Allowed-profile and evidence isolation |
| `tests/test_video_models.py` | Video domain validation |
| `tests/test_gemini_video_analyzer.py` | SDK adapter, structured modes, provider errors, and cleanup |
| `tests/test_video_storage.py` | State machine, validated persistence, corruption handling, and lifecycle |
| `tests/test_video_analysis_service.py` | Content identity, timeline bounds, atomic claim, retry, and overview reuse |
| `tests/test_video_context_service.py` | Replacement state, no-change decisions, sources, and ordering |
| `tests/test_video_context_injection.py` | Context replacement, acknowledgement, and session isolation |
| `tests/test_video_detail_service.py` | Safe URIs, time ranges, evidence validation, and concurrent detail cache |
| `tests/test_video_pipeline.py` | Overview-to-context pipeline and partial-failure recovery |
| `tests/test_video_cli.py` | Parser defaults, input validation, exit codes, and CLI orchestration |

## Quality and dependency controls

`pyproject.toml` is the source of supported dependency ranges and configures pytest,
Ruff, and Pyright. `requirements-lock.txt` records the fully resolved development
environment for reproducible installation. `.github/workflows/ci.yml` runs linting,
production-code type checks, and the complete test suite on Python 3.10 and 3.14; total
coverage must remain at or above 80%.

## Current scope boundary

Operational setup, environment variables, privacy boundaries, paired-database backup and
deletion, troubleshooting, and contributor quality commands are maintained in `README.md`.
This map remains the source for module ownership and code-to-storage relationships.

Implemented:

- manual analysis of completed local video files;
- detailed structured episode timelines with searchable normalized observations;
- durable brief descriptions plus bounded current gameplay state;
- bounded current-screen context;
- explicit injection into the matching chat session;
- source-linked session and game memory;
- detailed re-analysis through an agent tool.

Not implemented yet:

- OBS integration or any other capture backend;
- selecting and recording a specific game window;
- automatic clip rotation and folder watching;
- recording pause/resume controls;
- autonomous time-driven agent rounds.
