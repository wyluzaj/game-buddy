"""Terminal entry point: builds the agent, opens the database, runs the chat loop."""
import asyncio

import logfire
from dotenv import load_dotenv

load_dotenv()
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()

from buddy.config import choose_model, build_model
from buddy.agent import build_agent
from buddy.database import (
    create_conversation,
    finish_conversation,
    get_conversations_pending_memory,
    init_db,
    save_messages,
)
from buddy.long_term_memory import create_long_term_memory
from buddy.game_memory_analysis import build_game_memory_analyzer
from buddy.memory_analysis import build_memory_analyzer, process_pending_memories
from buddy.session_memory import (
    parse_database_timestamp,
    persist_session_document,
)
from buddy.timed_compaction import format_messages
from buddy.video_context_injection import inject_video_context
from buddy.video_storage import (
    get_latest_pending_video_context,
    mark_video_context_injected_through,
)
from buddy.video_session_memory import build_session_video_observations


def main():
    """Main function to run the chatbot session."""
    chosen = choose_model()
    model_object = build_model(chosen)
    conn = init_db()
    memory = create_long_term_memory()

    pending = get_conversations_pending_memory(conn)
    if pending:
        try:
            analyzer = build_memory_analyzer()
            game_analyzer = build_game_memory_analyzer()
            report = asyncio.run(
                process_pending_memories(
                    conn,
                    analyzer,
                    memory,
                    game_analyzer,
                )
            )
            if report.processed:
                print(
                    "Long-term memory: "
                    f"processed {report.processed} session(s), "
                    f"global updates {report.updated}, "
                    f"game profiles updated {report.game_profiles_updated}."
                )
            if report.failed_conversation_ids:
                for conversation_id in report.failed_conversation_ids:
                    reason = report.failure_reasons.get(
                        conversation_id,
                        "unknown error",
                    )
                    print(
                        "[Memory analysis failed for conversation "
                        f"{conversation_id}: {reason}]"
                    )
        except Exception as error:
            print(f"[Memory analysis unavailable: {error}]")

    agent = build_agent(
        model_object,
        memory=memory,
        memory_connection=conn,
    )
    conversation_id: int | None = None
    history = []

    try:
        print("Ready to go. Ctrl + C to end session.\n")

        while True:
            try:
                user_input = input("Ty: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye")
                break

            if not user_input:
                continue

            try:
                run_history = history
                pending_video_context = None
                if conversation_id is not None:
                    pending_video_context = get_latest_pending_video_context(
                        conn,
                        conversation_id=conversation_id,
                    )
                    if pending_video_context is not None:
                        run_history = inject_video_context(
                            history,
                            pending_video_context,
                        )
                result = agent.run_sync(user_input, message_history=run_history)
                print(f"\nBot: {result.output}\n")
                if conversation_id is None:
                    conversation_id = create_conversation(conn)
                save_messages(conn, conversation_id, result.new_messages_json().decode())
                # Keep the capability-processed history (including its summary) in RAM.
                history = result.all_messages()
                if pending_video_context is not None:
                    mark_video_context_injected_through(
                        conn,
                        conversation_id=conversation_id,
                        update_id=pending_video_context.id,
                    )
            except Exception as e:
                print(f"\n[Error: {e}]\n")
    finally:
        try:
            if conversation_id is not None:
                compacted_history = format_messages(history)
                finished = finish_conversation(
                    conn,
                    conversation_id,
                    compacted_history,
                )
                try:
                    asyncio.run(
                        persist_session_document(
                            conn,
                            memory,
                            conversation_id=conversation_id,
                            started_at=parse_database_timestamp(
                                finished["created_at"]
                            ),
                            ended_at=parse_database_timestamp(
                                finished["ended_at"]
                            ),
                            session_context=compacted_history,
                            video_observations=build_session_video_observations(
                                conn,
                                conversation_id,
                            ),
                        )
                    )
                except Exception as error:
                    print(f"[Session memory save failed: {error}]")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
