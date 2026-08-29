# Project Map

This document maps the production modules, runtime flows, storage layers, and tests of
`game-buddy`. It describes the current implementation on the `feature/video-analysis`
branch. Automatic game-window recording and autonomous time-driven agent rounds are not
implemented yet.

## Runtime flows

### Chat startup and shutdown

```text
buddy.main.main
  -> config.choose_model + config.build_model
  -> database.init_db
       -> memory_index.init_memory_index
       -> video_storage.init_video_storage
  -> long_term_memory.create_long_term_memory
  -> memory_analysis.process_pending_memories
       -> session_memory.persist_session_document
       -> memory_analysis._apply_long_term_memory
       -> game_memory.link_session_to_games
       -> game_memory_analysis.update_game_profiles
  -> agent.build_agent
       -> timed_compaction.SummarizingCompactionTimed
       -> long_term_memory.InjectionOnlyMemory
       -> memory_recall.recall_memory
       -> video_detail_tool.analyze_video_detail
       -> DuckDuckGo search
  -> terminal chat loop
       -> video_storage.get_latest_pending_video_context
       -> video_context_injection.inject_video_context
       -> Agent.run_sync
       -> database.save_messages
       -> video_storage.mark_video_context_injected_through
  -> session shutdown
       -> database.finish_conversation
       -> video_session_memory.build_session_video_observations
       -> session_memory.persist_session_document
```

Completed sessions are processed at the next application startup. A failed memory update
remains pending and does not block later sessions.

### General video analysis

```text
buddy.video_cli
  -> database.init_db
  -> gemini_video_analyzer.GeminiVideoAnalyzer
  -> video_pipeline.process_video_file
       -> video_analysis_service.register_and_analyze_video
            -> video_storage.create_video_segment
            -> video_storage.claim_video_segment_for_analysis
            -> GeminiVideoAnalyzer.analyze_overview
            -> video_storage.save_video_report(mode=overview)
       -> video_context_service.process_video_report_context
            -> video_context_butler.build_video_context_prompt
            -> Video Context Butler Agent
            -> source validation
            -> video_storage.save_video_context_update
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

### Detailed video analysis

```text
user asks about a visible or audible detail
  -> main agent finds a video://segment/... source in context
  -> analyze_video_detail tool
       -> video_detail_service.parse_video_source
       -> validate segment ownership and timestamp range
       -> reuse an identical cached detail report when available
       -> GeminiVideoAnalyzer.analyze_detail
       -> validate evidence timestamps
       -> video_storage.save_video_report(mode=detail)
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
  -> memory_index.search_memory_documents
  -> MemoryRecallService reads selected logical Markdown documents
  -> Context Butler reduces the retrieved documents
  -> source allow-list validation
  -> a small source-linked answer returns to the main agent
```

Global memory is based on explicit user statements. Video observations may enrich a
confirmed game's profile and session report, but they cannot become global user facts by
themselves.

## Storage map

### `data/chatbot.db`

```text
conversations
  1 -> many messages
  1 -> many video_segments

video_segments
  1 -> many video_reports

video_reports (overview)
  1 -> 0..1 video_context_updates

memory_documents
  session --belongs_to_game--> game
  indexed by memory_search (FTS5)
```

Main responsibilities:

- raw conversation persistence and completion state;
- searchable metadata for logical memory documents;
- session-to-game relations;
- normalized paths to local video files;
- overview/detail reports and compact context updates.

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
| `buddy/config.py` | Model catalog, fallback tiers, memory/video model roles, compaction settings | Used by all agent and analyzer factories |
| `buddy/agent.py` | Main Pydantic AI agent factory | Registers compaction, global memory, recall, video detail, and web search |
| `buddy/prompts.py` | Central LLM instructions and prompt templates | Used by chat, memory, Butler, and video analyzers |
| `buddy/database.py` | SQLite connection, conversations, and raw message batches | Initializes memory-index and video-storage schemas |
| `buddy/timed_compaction.py` | Time-window short-memory compaction and history formatting | Main agent capability and session-memory formatter |
| `buddy/long_term_memory.py` | Injection-only Harness Memory backed by SQLite | Used by chat, recall, and post-session analyzers |
| `buddy/session_memory.py` | Logical `main/sessions/*.md` reports and indexing | Called at shutdown and pending-memory processing |
| `buddy/memory_index.py` | Memory metadata, FTS5 search, and session-to-game relations | Used by session/game memory and recall |
| `buddy/memory_analysis.py` | Completed-session global consolidation coordinator | Updates global memory, session documents, and game profiles |
| `buddy/game_memory.py` | Game-name normalization, paths, documents, and relations | Used by memory analysis and recall |
| `buddy/game_memory_analysis.py` | Source-aware updates of allowed `main/games/*.md` profiles | Receives explicit user evidence and video observations |
| `buddy/memory_recall.py` | Bounded historical retrieval and reduction | Uses the memory index, Harness store, and Context Butler |
| `buddy/context_butler.py` | Small agent reducing recalled memory documents | Used only by `memory_recall.py` |
| `buddy/video_models.py` | Video enums, dataclasses, and Pydantic result models | Shared contract for all video modules |
| `buddy/video_analyzer.py` | Provider-neutral overview/detail protocols and domain errors | Implemented by Gemini; consumed by application services |
| `buddy/gemini_video_analyzer.py` | Gemini upload, polling, structured generation, parsing, and cleanup | Native provider adapter for overview and detail modes |
| `buddy/video_storage.py` | Video schema and persistence operations | Repository used by every video service and the chat loop |
| `buddy/video_analysis_service.py` | Atomic claim, overview persistence, retry, and reuse lifecycle | Connects `VideoAnalyzer` to `video_storage` |
| `buddy/video_context_butler.py` | Replacement-state prompt construction and source validation | Used by `video_context_service.py` |
| `buddy/video_context_service.py` | Idempotent overview-to-context reduction | Connects storage to Video Context Butler |
| `buddy/video_pipeline.py` | End-to-end overview and context-update orchestration | Used by the manual CLI and future capture worker |
| `buddy/video_context_injection.py` | Replaces prior gameplay state in chat history | Used by the terminal loop |
| `buddy/video_detail_service.py` | Internal URI parsing, range validation, cache, and detailed reports | Used by the detail tool |
| `buddy/video_detail_tool.py` | Main-agent tool for question-specific clip inspection | Creates a Gemini analyzer per invocation |
| `buddy/video_session_memory.py` | Bounded source-aware video evidence for completed sessions | Used by shutdown and memory processing |
| `buddy/video_cli.py` | Manual end-to-end analysis command | Composes SQLite, Gemini, pipeline, and terminal output |

## Test map

| Test module | Main coverage |
| --- | --- |
| `tests/conftest.py` | Fresh shared SQLite `:memory:` fixture |
| `tests/test_config.py` | Model roles, tiers, and fallback selection |
| `tests/test_agent.py` | Timed compaction, incremental summaries, and history filtering |
| `tests/test_long_term_memory.py` | Harness store, injection-only behavior, and agent capabilities |
| `tests/test_database.py` | Conversations, raw messages, completion, and migrations |
| `tests/test_memory_index.py` | FTS5 retrieval, relations, path validation, and dates |
| `tests/test_memory_recall.py` | Retrieval fallback, Context Butler, and source validation |
| `tests/test_memory_analysis.py` | Session consolidation, retry, games, and video evidence |
| `tests/test_game_memory_analysis.py` | Allowed-profile and evidence isolation |
| `tests/test_video_models.py` | Video domain validation |
| `tests/test_gemini_video_analyzer.py` | SDK adapter, structured modes, provider errors, and cleanup |
| `tests/test_video_storage.py` | Segment/report/context persistence and lifecycle |
| `tests/test_video_analysis_service.py` | Atomic claim, failure retry, and overview reuse |
| `tests/test_video_context_service.py` | Replacement state, no-change decisions, sources, and ordering |
| `tests/test_video_context_injection.py` | Context replacement, acknowledgement, and session isolation |
| `tests/test_video_detail_service.py` | Safe URIs, time ranges, evidence validation, and detail cache |
| `tests/test_video_pipeline.py` | Overview-to-context pipeline and partial-failure recovery |

## Current scope boundary

Implemented:

- manual analysis of completed local video files;
- general structured reports;
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
