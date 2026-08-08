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
from buddy.memory_analysis import build_memory_analyzer, process_pending_memories
from buddy.timed_compaction import format_messages


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
            report = asyncio.run(process_pending_memories(conn, analyzer, memory))
            if report.processed:
                print(
                    "Long-term memory: "
                    f"processed {report.processed} session(s), "
                    f"updated by {report.updated}."
                )
            if report.failed_conversation_ids:
                failed = ", ".join(map(str, report.failed_conversation_ids))
                print(f"[Memory analysis failed for conversation(s): {failed}]")
        except Exception as error:
            print(f"[Memory analysis unavailable: {error}]")

    agent = build_agent(model_object, memory=memory)
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
                result = agent.run_sync(user_input, message_history=history)
                print(f"\nBot: {result.output}\n")
                if conversation_id is None:
                    conversation_id = create_conversation(conn)
                save_messages(conn, conversation_id, result.new_messages_json().decode())
                # Keep the capability-processed history (including its summary) in RAM.
                history = result.all_messages()
            except Exception as e:
                print(f"\n[Error: {e}]\n")
    finally:
        try:
            if conversation_id is not None:
                compacted_history = format_messages(history)
                finish_conversation(conn, conversation_id, compacted_history)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
