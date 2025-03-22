#!/usr/bin/env python3
import discord
import json
import time
import random
from discord.ext import commands, tasks
import pickle
import os
import logging
import aiofiles
import re
import asyncio

from config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DISCORD_TOKEN,
    DEFAULT_NAME,
    LOG_CHANNEL_ID,
    DEFAULT_MODEL,
    PREMIUM_MODEL,
    CORE_PROMPT,
    SHOULD_REPLY_TIMEOUT,
    SUMMARIZE_TIMEOUT,
    LLM_TIMEOUT,
    TYPING_SPEED_CPM,
    MAX_TYPING_TIME,
    MIN_TYPING_TIME,
    TYPING_VARIANCE,
    VERBOSE_LOGGING,
    REPLY_COOLDOWN,
    BOT_REPLY_THRESHOLD
)

from utils import log_info, log_error, send_large_message
from commands import setup_commands
from ai import call_claude
from memory import maybe_summarize_conversation

# Global to prevent errors, log_channel should be set by on_ready
log_channel = None

last_replied_to = {}  # Dict mapping channel_id -> {bot_id: timestamp}

# Configure Discord client sharding
shard_count = int(os.environ.get("shard_count", "1"))  # Get from config
intents = discord.Intents.all()
if shard_count > 1:
    bot = commands.AutoShardedBot(command_prefix="!", intents=intents, description="A Claude based persona.")
else:
    bot = commands.Bot(command_prefix="!", intents=intents, description="A Claude based persona.")


# Global counter for bot replies.
bot_reply_counts = {}

# Async lock for accessing bot_reply_counts.
bot_reply_lock = asyncio.Lock()

# Global dictionary to store channel context for public channels.
# Key: channel ID, Value: list of messages (each as a dict with author and content)
channel_context = {}

async def get_yes_no_votes(message, is_bot=False, vote_count=3):
    """
    Ask Claude-3-5-haiku for multiple yes/no votes.
    Returns a list of votes, each as "yes", "no", or "abstain".
    """
    bot_name = DEFAULT_NAME  # from config
    penalty_text = ""
    if is_bot:
        async with bot_reply_lock:
            count = bot_reply_counts.get(message.author.id, 0)
        penalty_text = f" Note: this message is from a bot and you have already received {count} replies from me."
    prompt = f"{bot_name}, respond with a simple yes/no: would you like to reply to this message?{penalty_text}"

    votes = []
    dummy_user_dict = {
        "system_vote": {
            "token_usage": 0,
            "premium": False,
            "conversation_history": [
                {"role": "user", "content": "dummy conversation message to satisfy API requirements."}
            ]
        }
    }

    for _ in range(vote_count):
        response = await call_claude(
            user_id="system_vote",          # system-level; not tied to a persistent user
            user_dict=dummy_user_dict,       # dummy conversation history
            model="claude-3-5-haiku-20241022",
            system_prompt=prompt,
            user_content=message.clean_content,  # pass the message as a user message
            temperature=1.0,
            max_tokens=5,
            verbose=False
        )
        vote_raw = response.choices[0].message["content"].strip().lower()
        if "yes" in vote_raw:
            votes.append("yes")
        elif "no" in vote_raw:
            votes.append("no")
        else:
            votes.append("abstain")
    return votes

async def should_reply(message):
    """
    Decide whether the bot should reply to the given message.
    - In DMs, always reply.
    - In non-DM channels:
      - If the bot name is mentioned, reply immediately.
      - Otherwise, ask Claude via multiple yes/no votes.
    - For messages from bots, check the reply counter atomically.
    """
    if isinstance(message.channel, discord.DMChannel):
        return True

    bot_name = DEFAULT_NAME
    if re.search(bot_name, message.clean_content, re.IGNORECASE):
        return True

    if message.author.bot:
        async with bot_reply_lock:
            count = bot_reply_counts.get(message.author.id, 0)
            if count >= BOT_REPLY_THRESHOLD:
                return False

   # For bot messages, check if we've replied to this bot recently
    if message.author.bot:
        channel_id = str(message.channel.id)
        author_id = str(message.author.id)
        current_time = time.time()
        
        # Check if we've replied to this bot in this channel recently
        if channel_id in last_replied_to and author_id in last_replied_to[channel_id]:
            last_time = last_replied_to[channel_id][author_id]
            if current_time - last_time < REPLY_COOLDOWN:
                return False  # Don't reply if we replied recently

        # Check the reply counter
        async with bot_reply_lock:
            count = bot_reply_counts.get(message.author.id, 0)
            if count >= BOT_REPLY_THRESHOLD:
                return False
        
    is_bot_message = message.author.bot
    votes = await get_yes_no_votes(message, is_bot=is_bot_message, vote_count=3)
    yes_votes = votes.count("yes")
    no_votes = votes.count("no")
    abstain_votes = votes.count("abstain")
    return yes_votes > no_votes and yes_votes > abstain_votes

async def detect_entities(message, bot_name, max_retries=2):
    """
    Robust entity detection using Claude with proper error handling and retry logic.
    
    Args:
        message: The Discord message object
        bot_name: The name of this bot
        max_retries: Maximum number of retry attempts
        
    Returns:
        tuple: (references_others_first, first_entity, all_entities)
    """
    content = message.clean_content.strip()
    
    # Quick check - if message is too short, likely no complex entity references
    if len(content) < 15:
        return False, None, []
    
    # Normalize bot_name for comparison
    normalized_bot_name = bot_name.lower().strip()
    
    # Create a focused prompt for entity detection
    prompt = f"""Your task is to analyze the given message and identify entities (characters, bots, or users) 
that are directly addressed or referenced. 

I am a bot named "{bot_name}".

INSTRUCTIONS:
1. Only identify entities that are clearly being addressed or referenced
2. Return ONLY a JSON array of strings representing the entities in order of appearance
3. Use exactly the name as it appears in the message
4. Do not include titles, honorifics, or descriptors unless they're part of the name
5. If no entities are detected, return an empty JSON array: []
6. DO NOT include any explanation or additional text, only the JSON array

Example valid responses:
[]
["Alice"]
["Bob", "Alice", "{bot_name}"]

ONLY RETURN THE JSON ARRAY, NO OTHER TEXT."""

    dummy_user_dict = {
        "entity_detection": {
            "token_usage": 0,
            "premium": False,
            "conversation_history": [
                {"role": "user", "content": "system message for entity detection."}
            ]
        }
    }

    # Try with retries
    for attempt in range(max_retries + 1):
        try:
            # Call Claude with a timeout
            response = await asyncio.wait_for(
                call_claude(
                    user_id="entity_detection",
                    user_dict=dummy_user_dict,
                    model="claude-3-5-haiku-20241022",
                    system_prompt=prompt,
                    user_content=content,
                    temperature=0.1,  # Very low temperature for consistency
                    max_tokens=50,
                    verbose=False
                ),
                timeout=3.0  # 3-second timeout to prevent blocking
            )
            
            response_text = response.choices[0].message["content"].strip()
            
            # Try to parse the response as JSON
            try:
                # Look for the first [ and last ] to extract just the JSON array
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_text = response_text[start_idx:end_idx+1]
                    entities = json.loads(json_text)
                    
                    # Validate that we got a list
                    if not isinstance(entities, list):
                        log_error(f"Entity detection returned non-list: {type(entities)}")
                        entities = []
                    
                    # Validate each entity is a string
                    entities = [str(e) for e in entities if e]
                    
                    break  # Successfully parsed, exit retry loop
                else:
                    # No valid JSON found, try again or use empty list
                    log_error(f"No JSON array found in response: {response_text}")
                    entities = []
                    
                    # If it's the last attempt, use empty list
                    if attempt == max_retries:
                        break
                    continue  # Try again
                    
            except json.JSONDecodeError as e:
                log_error(f"Failed to parse JSON in attempt {attempt+1}: {e}")
                entities = []
                
                # If it's the last attempt, use empty list
                if attempt == max_retries:
                    break
                continue  # Try again
                
        except asyncio.TimeoutError:
            log_error(f"Entity detection timed out on attempt {attempt+1}")
            # If all retries failed, use empty list
            if attempt == max_retries:
                entities = []
                break
            continue  # Try again
            
        except Exception as e:
            log_error(f"Error in entity detection attempt {attempt+1}: {e}")
            # If all retries failed, use empty list
            if attempt == max_retries:
                entities = []
                break
            continue  # Try again
    
    # If all retries failed or entities is still not defined
    if 'entities' not in locals():
        entities = []
    
    # Default return values
    references_others_first = False
    first_entity = None
    
    # Normalize entities for better matching
    normalized_entities = [e.lower().strip() for e in entities]
    
    # Check if any entities were detected
    if entities:
        # Check if bot is in the list
        if normalized_bot_name in normalized_entities:
            bot_position = normalized_entities.index(normalized_bot_name)
            
            # If bot is mentioned but not first
            if bot_position > 0:
                references_others_first = True
                first_entity = entities[0]  # Use the original case of the entity
        elif entities:  # Bot not mentioned but other entities are
            references_others_first = True
            first_entity = entities[0]
    
    return references_others_first, first_entity, entities


user_data = {}
USER_DATA_FILE = "user_info.pickle"

async def load_user_data():
    global user_data
    try:
        async with aiofiles.open(USER_DATA_FILE, "rb") as f:
            data = await f.read()
            loaded_data = pickle.loads(data)
            user_data.clear()
            user_data.update(loaded_data)
        log_info("User data loaded successfully.")
    except Exception as e:
        log_error(f"Failed to load user data: {e}")
        user_data.clear()

async def save_user_data():
    global user_data
    try:
        async with aiofiles.open(USER_DATA_FILE, "wb") as f:
            await f.write(pickle.dumps(user_data, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception as e:
        log_error(f"Error saving user data: {e}")


setup_commands(bot, user_data)


# Helper function for extended typing
async def extended_typing(channel, duration):
    """
    Keep the typing indicator active for a specified duration.
    Discord typing indicator expires after ~10 seconds, so we refresh it.
    """
    refresh_interval = 5.0  # Refresh every 5 seconds
    end_time = time.time() + duration
    
    while time.time() < end_time:
        # Start typing
        async with channel.typing():
            # Sleep for either refresh_interval or the remaining time, whichever is shorter
            remaining = end_time - time.time()
            await asyncio.sleep(min(refresh_interval, max(0.1, remaining)))

# Calculate realistic typing time based on response length
def calculate_typing_time(response_text):
    # Number of characters in the response
    char_count = len(response_text)
    
    # Calculate time based on typing speed (chars per minute)
    # Convert to seconds: (chars / chars_per_minute) * 60 seconds
    base_time = (char_count / TYPING_SPEED_CPM) * 60
    
    # Add some natural variation (±20% by default)
    variation = random.uniform(1 - TYPING_VARIANCE, 1 + TYPING_VARIANCE)
    typing_time = base_time * variation
    
    # Ensure time is within defined bounds
    typing_time = min(max(typing_time, MIN_TYPING_TIME), MAX_TYPING_TIME)
    
    if VERBOSE_LOGGING:
        log_info(f"Calculated typing time: {typing_time:.2f}s for {char_count} chars")
        
    return typing_time

# admin command processing
async def process_admin_commands(message: discord.Message):
    """
    Process admin commands from the log channel.
    """
    if message.author.bot:
        return

    split = message.content.split()
    if not split:
        return

    cmd = split[0].lower()

    # Shutdown command must be: "shutdown? {DEFAULT_NAME}"
    content = message.content.strip()
    if content.lower().startswith("shutdown?"):
        # Extract everything after "shutdown?" and trim whitespace
        name_part = content[len("shutdown?"):].strip()
        
        # Compare with the bot name (case-insensitive)
        if name_part.lower() == DEFAULT_NAME.lower():
            await log_channel.send(
                f"Admin {message.author.name}[id:{message.author.id}] sent shutdown? {DEFAULT_NAME}. Shutting down ({DEFAULT_NAME})..."
            )
            await log_channel.send(f"***Shutting down Claude's Mask ({DEFAULT_NAME})***")
            await bot.change_presence(status=discord.Status.invisible)
            # Save user data before shutting down
            try:
                await save_user_data()
                log_info("User data saved before shutdown")
            except Exception as e:
                log_error(f"Failed to save user data before shutdown: {e}")
            await bot.close()
            return
        else:
            await log_channel.send(f"Invalid shutdown command. Use: shutdown? {DEFAULT_NAME}")
            return

    elif cmd == "user" and len(split) > 1 and split[1].lower() == "data?":
        if len(split) > 2:
            target_user_id = split[2]
            data = user_data.get(target_user_id)
            if data:
                # Create a sanitized version for display (no actual conversation content)
                sanitized_data = {
                    "token_usage": data.get("token_usage", 0),
                    "premium": data.get("premium", False),
                    "conversation_length": len(data.get("conversation_history", [])),
                    "core_memories_bytes": len(data.get("core_memories", "")),
                }
                msg = f"User data for {target_user_id}:\n```{sanitized_data}```"
                await send_large_message(log_channel, msg)
            else:
                await send_large_message(log_channel, f"No data found for user {target_user_id}.")
        else:
            await send_large_message(log_channel, "Usage: user data? [user_id]")
        return

    # Add command to list all users
    elif cmd == "list" and len(split) > 1 and split[1].lower() == "users":
        user_list = [f"ID: {user_id}, Premium: {data.get('premium', False)}, Tokens: {data.get('token_usage', 0)}"
                     for user_id, data in user_data.items()]
        user_count = len(user_list)
        msg = f"Total users: {user_count}\n"

        # Send in chunks to avoid message length limits
        chunk_size = 20
        for i in range(0, len(user_list), chunk_size):
            chunk = user_list[i:i + chunk_size]
            chunk_msg = "\n".join(chunk)
            await send_large_message(log_channel, f"Users {i + 1}-{i + len(chunk)}:\n```{chunk_msg}```")

        return

    # Add command to toggle premium for a user
    elif cmd == "premium":
        if len(split) > 1:
            target_user_id = split[1]
            if target_user_id in user_data:
                current_status = user_data[target_user_id].get("premium", False)
                user_data[target_user_id]["premium"] = not current_status
                new_status = "enabled" if not current_status else "disabled"
                await log_channel.send(f"Premium status for user {target_user_id} {new_status}.")
                await save_user_data()
            else:
                await log_channel.send(f"User {target_user_id} not found.")
        else:
            await log_channel.send("Usage: premium [user_id]")
        return

# message processing as separate async function
async def process_message(message: discord.Message):
    try:
        # For non-DM messages, check if we should reply
        if not isinstance(message.channel, discord.DMChannel):
            # Use a timeout to limit the time spent checking if we should reply
            try:
                should_reply_result = await asyncio.wait_for(
                    should_reply(message),
                    timeout=SHOULD_REPLY_TIMEOUT
                )
                if not should_reply_result:
                    return
            except asyncio.TimeoutError:
                log_error(f"should_reply timed out for message {message.id}")
                return

        # Get the clean message content and skip if empty
        content = message.clean_content.strip()
        if not content:
            return

        # Bot reply counting logic
        user_id = str(message.author.id)
        # If a human sends a message, reset bot reply counts.
        if not message.author.bot:
            async with bot_reply_lock:
                bot_reply_counts.clear()
        else:
            # For bot messages, update the counter atomically.
            async with bot_reply_lock:
                count = bot_reply_counts.get(message.author.id, 0)
                if count >= BOT_REPLY_THRESHOLD:
                    return  # Skip if we've replied too many times to this bot
                bot_reply_counts[message.author.id] = count + 1
        
        # Entity detection and waiting logic
        should_wait = False
        wait_time = 0
        wait_reason = ""
        
        # Only do entity detection for non-DM channels and substantive messages
        if not isinstance(message.channel, discord.DMChannel) and len(content) > 15:
            try:
                # Set a timeout for the entire entity detection process
                references_others_first, first_entity, all_entities = await asyncio.wait_for(
                    detect_entities(message, DEFAULT_NAME),
                    timeout=4.0  # 4-second timeout for the entire detection process
                )
                
                if references_others_first and first_entity:
                    entity_list = ', '.join(all_entities)
                    wait_reason = f"Entities detected: {entity_list}, waiting for {first_entity}"
                    log_info(wait_reason)
                    
                    # Calculate wait time based on message length + randomness
                    # Base wait time: 3 seconds + 0.5 seconds per 20 characters
                    base_wait = 3.0 + (len(content) / 40)
                    
                    # Add randomness to prevent bots waiting the same time
                    variance = random.uniform(0.8, 1.2)
                    wait_time = base_wait * variance
                    
                    # Ensure reasonable bounds: 3-12 seconds
                    wait_time = min(max(wait_time, 3.0), 12.0)
                    
                    should_wait = True
            except asyncio.TimeoutError:
                log_error("Entity detection timed out, continuing without waiting")
            except Exception as e:
                log_error(f"Error in entity detection: {e}")
                # Continue with processing even if entity detection fails
        
        # If we should wait for another entity to respond
        if should_wait and wait_time > 0:
            log_info(f"Waiting {wait_time:.2f}s because: {wait_reason}")
            
            # Keep typing indicator active during wait (makes delay appear natural)
            typing_task = None
            if not isinstance(message.channel, discord.DMChannel):
                typing_task = asyncio.create_task(
                    extended_typing(message.channel, wait_time)
                )
            
            # Wait for the calculated time
            await asyncio.sleep(wait_time)
            
            # Cancel typing task if it's still running
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass
            
            # After waiting, update the channel context with any new messages
            if hasattr(message.channel, 'id') and message.channel.id in channel_context:
                try:
                    # Collect new messages that arrived during our wait
                    recent_messages = []
                    async for msg in message.channel.history(limit=5):
                        # Only include messages that:
                        # 1. Aren't the original message we're responding to
                        # 2. Were created after the original message
                        # 3. Aren't from this bot
                        if (msg.id != message.id and 
                            msg.created_at > message.created_at and
                            msg.author.id != bot.user.id):
                            
                            clean_content = msg.clean_content.strip()
                            if clean_content:
                                recent_messages.append({
                                    "author": msg.author.name,
                                    "content": clean_content
                                })
                    
                    # Add new messages to the context
                    if recent_messages:
                        channel_context[message.channel.id].extend(recent_messages)
                        # Keep within limit
                        channel_context[message.channel.id] = channel_context[message.channel.id][-10:]
                        log_info(f"Updated channel context with {len(recent_messages)} new messages after waiting")
                except Exception as e:
                    log_error(f"Error updating channel context after waiting: {e}")

        # Process the message and generate a response
        try:
            await process_user_message(message, content)
        except Exception as e:
            log_error(f"Error processing message: {e}")
            try:
                await message.channel.send("I'm having trouble processing your message. Please try again later.")
            except:
                pass
    except Exception as e:
        log_error(f"Error in process_message: {e}")


# Extract core message processing logic
async def process_user_message(message, content):
    user_id = str(message.author.id)
    if user_id not in user_data:
        user_data[user_id] = {
            "token_usage": 0,
            "premium": False,
            "conversation_history": [],
            "core_memories": ""
        }
    
    # Use a timeout for the summarization to prevent blocking
    try:
        await asyncio.wait_for(
            maybe_summarize_conversation(user_id, user_data),
            timeout=SUMMARIZE_TIMEOUT
        )
    except asyncio.TimeoutError:
        log_error(f"Summarization timed out for user {user_id}")
    
    # Append the user message to the conversation history
    user_data[user_id]["conversation_history"].append({"role": "user", "content": content})
    
    # Build the system prompt.
    core_mem = user_data[user_id].get("core_memories", "")
    if isinstance(message.channel, discord.DMChannel):
        extra_context = "This is a private conversation. You may be casual, personal, and more intimate."
        external_context = ""
    else:
        extra_context = ("This is a public channel. Be yourself, but the conversation is public; "
                         "be aware not to carry over private conversation topics unless you want everyone to know about them.")
        # Build external context from the channel context.
        context_lines = []
        for msg in channel_context.get(message.channel.id, []):
            if msg["content"]:
                context_lines.append(f"{msg['author']}: {msg['content']}")
        external_context = "\n".join(context_lines)
    
    system_text = f"{extra_context}\n"
    if external_context:
        system_text += f"External Context:\n{external_context}\n"
    system_text += f"{CORE_PROMPT}\n\nCore Memories:\n{core_mem}"
    
    # Choose the appropriate model.
    model_to_use = PREMIUM_MODEL if user_data[user_id].get("premium", False) else DEFAULT_MODEL
    
    # Make API call with typing indicator
    typing_task = None
    try:
        # First, make the API call with typing indicator
        async with message.channel.typing():
            response = await asyncio.wait_for(
                call_claude(
                    user_id=user_id,
                    user_dict=user_data,
                    model=model_to_use,
                    system_prompt=system_text,
                    user_content=None,
                    temperature=DEFAULT_TEMPERATURE,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    verbose=False
                ),
                timeout=LLM_TIMEOUT
            )
        
        result = response.choices[0].message["content"]
        
        # Append the assistant's reply to the conversation history
        user_data[user_id]["conversation_history"].append({"role": "assistant", "content": result})

        # Record that we replied to this bot if it's a bot message
        if message.author.bot:
            channel_id = str(message.channel.id)
            channel_id = str(message.channel.id)
            author_id = str(message.author.id)
        
            # Initialize channel dict if needed
            if channel_id not in last_replied_to:
                last_replied_to[channel_id] = {}
            
            # Record the timestamp of this reply
            last_replied_to[channel_id][author_id] = time.time()
        
        # Calculate realistic typing time based on response length (only in public channels)
        if not isinstance(message.channel, discord.DMChannel) and message.author.bot:
            typing_time = calculate_typing_time(result)
            
            # Start extended typing in background
            typing_task = asyncio.create_task(
                extended_typing(message.channel, typing_time)
            )
            
            # Wait for the typing time to elapse
            await asyncio.sleep(typing_time)
        
        # Send response with error handling
        try:
            await send_large_message(message.channel, f"{message.author.mention} {result}")
        except Exception as e:
            log_error(f"Error sending message: {e}")
            try:
                await message.channel.send("I had trouble sending my complete response. Please try again.")
            except:
                pass
            
    except asyncio.TimeoutError:
        log_error(f"LLM call timed out for user {user_id}")
        result = "I apologize, but I'm having trouble thinking right now. Could you please try again in a moment?"
        # Append the error message to the conversation history
        user_data[user_id]["conversation_history"].append({"role": "assistant", "content": result})
        await message.channel.send(f"{message.author.mention} {result}")
        
    except Exception as e:
        log_error(f"Error in LLM call: {e}")
        result = "I encountered an unexpected issue. Please try again later."
        # Append the error message to the conversation history
        user_data[user_id]["conversation_history"].append({"role": "assistant", "content": result})
        await message.channel.send(f"{message.author.mention} {result}")
        
    finally:
        # Make sure to clean up the typing task if it's still running
        if typing_task and not typing_task.done():
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
    
    # Save user data with error handling
    try:
        await save_user_data()
    except Exception as e:
        log_error(f"Error saving user data: {e}")



# Reconnection logic and heartbeat logging
@bot.event
async def on_disconnect():
    log_error("Bot disconnected from Discord!")


@bot.event
async def on_shard_ready(shard_id):
    log_info(f"Shard {shard_id} connected to Discord.")


@bot.event
async def on_resumed():
    log_info("Bot session resumed.")


# Add a heartbeat task to keep connections alive
@tasks.loop(seconds=30)
async def heartbeat_check():
    # Check if we're using a sharded bot
    if isinstance(bot, commands.AutoShardedBot):
        # If sharded, we have multiple latencies
        latencies = bot.latencies
        for shard_id, latency in latencies:
            if latency > 1.0:  # High latency warning threshold (1 second)
                log_error(f"High latency detected on shard {shard_id}: {latency:.2f}s")

        # Log overall status occasionally
        if random.random() < 0.1:  # ~10% chance on each check
            avg_latency = sum(l for _, l in latencies) / max(len(latencies), 1)
            log_info(f"Bot heartbeat - Avg latency: {avg_latency:.2f}s, Shards: {len(latencies)}")
    else:
        # For non-sharded bot, we just have a single latency
        latency = bot.latency
        if latency > 1.0:  # High latency warning threshold (1 second)
            log_error(f"High latency detected: {latency:.2f}s")

        # Log status occasionally
        if random.random() < 0.1:  # ~10% chance on each check
            log_info(f"Bot heartbeat - Latency: {latency:.2f}s")


@heartbeat_check.before_loop
async def before_heartbeat():
    await bot.wait_until_ready()

@bot.event
async def on_ready():
    global log_channel
    log_channel = bot.get_channel(LOG_CHANNEL_ID)
    if log_channel:
        # Startup message includes DEFAULT_NAME in parentheses.
        await log_channel.send(f"Claude's Mask ({DEFAULT_NAME}) is online!")
        await log_channel.send("Loading user data...")
    try:
        await load_user_data()
        if log_channel:
            await log_channel.send("User data loaded successfully!")
    except Exception as e:
        if log_channel:
            await log_channel.send(f"Error loading user data: {e}")
    log_info("Claude's Mask is online!")

    synced = await bot.tree.sync()
    log_info(f"Synced {len(synced)} slash commands.")
    await bot.change_presence(status=discord.Status.online)

    for guild in bot.guilds:
        member = guild.get_member(bot.user.id)
        if member:
            try:
                await member.edit(nick=f"{DEFAULT_NAME}")
            except Exception as e:
                log_error(f"Failed to update nickname in guild {guild.id}: {e}")

    heartbeat_check.start()
    periodic_save.start()

@bot.event
async def on_member_join(member: discord.Member):
    user_id = str(member.id)
    if user_id not in user_data:
        user_data[user_id] = {
            "token_usage": 0,
            "premium": False,
            "conversation_history": [],
            "core_memories": ""
        }
    await save_user_data()

@bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    await save_user_data()


@bot.event
async def on_message(message: discord.Message):
    # Skip processing the bot's own messages
    if message.author.id == bot.user.id:
        return

    # Record public channel messages for external context
    if not isinstance(message.channel, discord.DMChannel):
        channel_context.setdefault(message.channel.id, [])
        clean_content = message.clean_content.strip()
        if clean_content:
            channel_context[message.channel.id].append({
                "author": message.author.name,
                "content": clean_content
            })
        # Limit the external context to the last 10 messages
        channel_context[message.channel.id] = channel_context[message.channel.id][-10:]

    # Handle admin commands in the log channel - Check if log_channel exists first
    if log_channel is not None and hasattr(message.channel, 'id') and message.channel.id == log_channel.id:
        # Admin commands processing
        await process_admin_commands(message)
        return

    # Handle admin commands in the log channel
    if hasattr(message.channel, 'id') and message.channel.id == log_channel.id:
        # Admin commands processing
        await process_admin_commands(message)
        return

    # Create task for processing messages to avoid blocking
    asyncio.create_task(process_message(message))

@tasks.loop(minutes=1)
async def periodic_save():
    await save_user_data()

@periodic_save.before_loop
async def before_periodic_save():
    await bot.wait_until_ready()

bot.run(DISCORD_TOKEN)
