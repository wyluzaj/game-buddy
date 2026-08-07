"""Terminal entry point: builds the agent, opens the database, runs the chat loop."""
from dotenv import load_dotenv

load_dotenv()
from buddy.config import choose_model, build_model
from buddy.agent import build_agent
from buddy.database import init_db, create_conversation, save_messages, get_messages
from pydantic_ai.messages import ModelMessagesTypeAdapter


def main():
    """Main function to run the chatbot session."""
    chosen = choose_model()
    model_object = build_model(chosen)
    agent = build_agent(model_object)
    conn = init_db()
    conversation_id = create_conversation(conn)

    history = []
    for raw in get_messages(conn, conversation_id):
        messages = ModelMessagesTypeAdapter.validate_json(raw)
        history.extend(messages)

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
                save_messages(conn, conversation_id, result.new_messages_json().decode())
                # Keep the capability-processed history (including its summary) in RAM.
                history = result.all_messages()
            except Exception as e:
                print(f"\n[Error: {e}]\n")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
