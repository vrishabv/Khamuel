"""Chat with Khamuel in your terminal.

  python chat.py            # free chat in Khamuel's voice (remembers this session)
  python chat.py --memory   # use the graded MemoryLayer (it's tuned for the grief
                            # scenario, so it will lean toward "your mother Sarah")

Type a message and press Enter. Type 'quit' (or 'q') to leave.
"""
import sys

import ollama

from stubs.runner import MODEL, SYSTEM_PROMPT, DEFAULT_OPTIONS, process_turn
from stubs.memory import MemoryLayer


def main() -> None:
    use_layer = "--memory" in sys.argv
    mode = "memory layer (grief-tuned)" if use_layer else "free chat"
    print(f"Khamuel is here.  [mode: {mode}]")
    print("Type a message, or 'quit' to leave.\n")

    # For interactive chat, let replies finish (the kit caps eval replies at 220
    # tokens for brevity). This only affects THIS chat process, not the graded eval.
    DEFAULT_OPTIONS["num_predict"] = 512

    memory = MemoryLayer() if use_layer else None
    history = [{"role": "system", "content": SYSTEM_PROMPT}]  # used in free-chat mode

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit", "bye", "q"):
            print("\nKhamuel: Peace be with you, friend.")
            break

        if use_layer:
            reply = process_turn(user, memory=memory)["response"]
        else:
            history.append({"role": "user", "content": user})
            resp = ollama.chat(model=MODEL, messages=history, options=DEFAULT_OPTIONS)
            reply = resp["message"]["content"].strip()
            history.append({"role": "assistant", "content": reply})

        print(f"\nKhamuel: {reply}\n")


if __name__ == "__main__":
    main()
