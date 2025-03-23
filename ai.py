# ai.py
import time
import json
from utils import log_error
import anthropic
from config import (
    OAI_TOKEN,
    ENABLE_API_CALL_LOGGING,
    COST_PER_TOKEN_HAIKU,
    COST_PER_TOKEN_SONNET,
    DEFAULT_MODEL,
    PREMIUM_MODEL
)
from utils import log_error
from token_utils import anthropic_token_count


def log_api_call(user_id: str, payload: dict, response_json: dict):
    from config import ENABLE_API_CALL_LOGGING
    if not ENABLE_API_CALL_LOGGING:
        return
    try:
        # Create sanitized copies of the payload and response
        sanitized_payload = payload.copy()
        if "messages" in sanitized_payload:
            # Replace message content with length indicators
            sanitized_payload["messages"] = [
                {
                    "role": msg["role"],
                    "content": f"[{len(msg['content'])} chars]"
                } for msg in sanitized_payload["messages"]
            ]

        sanitized_response = response_json.copy()
        if "completion" in sanitized_response:
            sanitized_response["completion"] = f"[{len(sanitized_response['completion'])} chars]"

        with open("anthropic_api_calls.log", "a", encoding="utf-8") as f:
            f.write("=== Anthropic API Call ===\n")
            f.write(f"User ID: {user_id}\n")
            f.write(f"Model: {payload.get('model', 'unknown')}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Messages count: {len(payload.get('messages', []))}\n")
            f.write(f"Temperature: {payload.get('temperature', 'default')}\n")
            f.write(f"Max tokens: {payload.get('max_tokens', 'default')}\n")
            f.write(f"Usage - Prompt tokens: {response_json.get('usage', {}).get('prompt_tokens', 'unknown')}\n")
            f.write("\n\n")
    except Exception as e:
        log_error(f"Error logging API call: {e}")



async def call_claude(
        user_id: str,
        user_dict: dict,
        model: str,
        system_prompt: str,
        user_content: str = None,
        temperature: float = 1.0,
        max_tokens: int = 1000,
        verbose: bool = False
):
    """
    Calls Anthropic's messages.create endpoint.
      - system: top-level system prompt.
      - messages: conversation history (only user/assistant roles).
      - If user_content is provided, appends it as a user message.
    Returns an object with .choices[0].message["content"] containing a plain text string.
    """
    if user_id not in user_dict:
        user_dict[user_id] = {
            "token_usage": 0,
            "premium": False,
            "conversation_history": []
        }
    conversation = user_dict[user_id]["conversation_history"]

    # Append new user message if provided.
    if user_content:
        conversation.append({"role": "user", "content": user_content})

    # Count prompt tokens.
    prompt_tokens = anthropic_token_count(model, system_prompt, conversation)

    try:
        client = anthropic.Anthropic(api_key=OAI_TOKEN)

        # Create the request payload for logging
        payload = {
            "model": model,
            "system": system_prompt,
            "messages": conversation,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1
        }

        msg_obj = client.messages.create(
            model=model,
            system=system_prompt,
            messages=conversation,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1
        )

        # Extract completion_text from msg_obj
        completion_text = ""
        if hasattr(msg_obj, "content"):
            if isinstance(msg_obj.content, list):
                completion_text = "\n".join([
                    getattr(block, "text", str(block)) for block in msg_obj.content
                ])
            else:
                completion_text = str(msg_obj.content)

        # Create a serializable response object for logging
        response_json = {
            "id": getattr(msg_obj, "id", "unknown"),
            "model": getattr(msg_obj, "model", model),
            "completion": completion_text,
            "usage": {
                "prompt_tokens": prompt_tokens
            }
        }

        # Log the API call with serializable objects
        log_api_call(user_id, payload, response_json)
    except Exception as e:
        log_error(f"Error in call_claude: {e}")
        return _fake_response("Error calling Anthropic. Please try again later.")

    # Extract plain text if completion_text is a list of TextBlocks or has a 'text' attribute.
    if isinstance(completion_text, list):
        # Assume each block is an object with a .text attribute.
        completion_text = "\n".join(
            [getattr(tb, "text", str(tb)) for tb in completion_text]
        )
    elif hasattr(completion_text, "text"):
        completion_text = completion_text.text
    else:
        # Otherwise, assume it's already a string.
        completion_text = str(completion_text)

    # Count completion tokens.
    completion_tokens = anthropic_token_count(
        model,
        "",
        [{"role": "assistant", "content": completion_text}]
    )
    total_tokens = prompt_tokens + completion_tokens

    # Calculate cost.
    if model == PREMIUM_MODEL:
        cost = total_tokens * COST_PER_TOKEN_SONNET
    else:
        cost = total_tokens * COST_PER_TOKEN_HAIKU

    # Update user's cumulative token usage.
    user_dict[user_id]["token_usage"] = user_dict[user_id].get("token_usage", 0) + total_tokens

    if verbose:
        log_error(
            f"[Verbose] API Call to model: {model}\n"
            f"System prompt length: {len(system_prompt)} chars\n"
            f"Conv history length: {len(conversation)} messages\n"
            f"Response length: {len(completion_text)} chars\n" 
            f"Tokens used: {prompt_tokens} + {completion_tokens} = {total_tokens}\n"
            f"Estimated cost: ${cost:.6f}"
        )

    return _fake_response(completion_text)


def _fake_response(text: str):
    """
    Returns an object with .choices[0].message["content"].
    This mimics the OpenAI-like response style.
    """

    class FakeChoice:
        def __init__(self, content):
            self.message = {"content": content}

    class FakeResponse:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    return FakeResponse(text)
