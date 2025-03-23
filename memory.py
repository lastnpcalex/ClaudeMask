import os
import pickle
import time
from config import (
    CORE_MEMORY_PROMPT,
    CORE_MEMORY_DUMP_PROMPT,  # New: additional prompt when core memories get too long.
    PREMIUM_MODEL,
    DEFAULT_MODEL,
    SUMMARIZATION_PROMPT,
    ENABLE_CORE_MEMORY_PICKLE_LOG,  # New: toggle for logging core memories.
    CORE_MEMORY_PICKLE_DIR  # New: directory path for pickle dumps.
)
from token_utils import anthropic_token_count
from ai import call_claude

def estimate_tokens(text: str) -> int:
    """Estimate token count by assuming one token is roughly 4 characters."""
    return len(text) // 4

async def maybe_summarize_conversation(
    user_id: str,
    user_data: dict,
    # These parameters are no longer used because we use estimated token counts.
    # max_unsummary_messages: int = 10,
    # token_limit: int = 3000
):
    """
    If the user's conversation is too large (by estimated token count),
    call the summarizer to update core memories and replace older messages with a summary.
    """
    if user_id not in user_data:
        user_data[user_id] = {
            "token_usage": 0,
            "premium": False,
            "conversation_history": [],
            "core_memories": ""
        }

    conversation = user_data[user_id]["conversation_history"]
    if not conversation:
        return

    premium = user_data[user_id].get("premium", False)
    model_to_use = PREMIUM_MODEL if premium else DEFAULT_MODEL

    # Build a single text block from conversation messages.
    conversation_text = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in conversation)
    estimated_conv_tokens = estimate_tokens(conversation_text)

    # If the estimated token count of the conversation is less than 25,000, do nothing.
    if estimated_conv_tokens < 25000:
        return

    old_core = user_data[user_id].get("core_memories", "")
    # If the core memories are too long, add an extra prompt.
    if estimate_tokens(old_core) >= 25000:
        core_prompt = f"{CORE_MEMORY_PROMPT}\n\n{CORE_MEMORY_DUMP_PROMPT}"
    else:
        core_prompt = CORE_MEMORY_PROMPT

    # Dump old core memories to a pickle file if enabled.
    if ENABLE_CORE_MEMORY_PICKLE_LOG:
        os.makedirs(CORE_MEMORY_PICKLE_DIR, exist_ok=True)
        pickle_filename = os.path.join(
            CORE_MEMORY_PICKLE_DIR,
            f"{user_id}_core_memories_{int(time.time())}.pickle"
        )
        with open(pickle_filename, "wb") as f:
            pickle.dump(old_core, f)

    # Build the summarization request.
    summarization_request = (
        f"{core_prompt}\n\n"
        f"CURRENT CORE MEMORIES:\n{old_core}\n\n"
        f"CONVERSATION:\n{conversation_text}\n\n"
        "Please return updated core memories and a short summary in the format:\n\n"
        "CORE MEMORIES:\n<updated core memories>\n\nSUMMARY:\n<short summary>"
    )

    # Backup the conversation.
    backup_convo = conversation[:]
    # Replace conversation_history with a single summarization request.
    user_data[user_id]["conversation_history"] = [
        {"role": "user", "content": summarization_request}
    ]

    # Call the summarizer using the SUMMARIZATION_PROMPT as the system prompt.
    response = await call_claude(
        user_id=user_id,
        user_dict=user_data,
        model=model_to_use,
        system_prompt=SUMMARIZATION_PROMPT,
        user_content=None,
        temperature=0.5,
        max_tokens=750
    )
    raw_output = response.choices[0].message["content"]

    # Restore the original conversation.
    user_data[user_id]["conversation_history"] = backup_convo

    # Parse the summarizer's output.
    updated_core = old_core
    short_summary = ""
    split_core = raw_output.split("CORE MEMORIES:")
    if len(split_core) > 1:
        after_core = split_core[1].strip()
        sum_split = after_core.split("SUMMARY:")
        if len(sum_split) > 1:
            updated_core = sum_split[0].strip()
            short_summary = sum_split[1].strip()
        else:
            updated_core = after_core.strip()
    else:
        updated_core = raw_output.strip()

    user_data[user_id]["core_memories"] += "\n" + updated_core

    # Replace the older conversation with a single summary message.
    user_data[user_id]["conversation_history"] = [
        {"role": "assistant", "content": f"(Summary) {short_summary}"}
    ]
