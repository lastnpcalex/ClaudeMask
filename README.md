# Character-Driven Relationship Framework

This repository contains a comprehensive framework for creating engaging LLM-based Discord personas using Anthropic's Claude. Designed to be both flexible and robust, the system enables you to explore a wide range of relationship dynamics—from mentorships and familial bonds to professional collaborations and beyond.

---

## Overview

The framework is built around four key components:

1. **Character Definition Template**  
   Create richly detailed personas with distinctive traits, communication styles, and a dual-mode system. The bot dynamically adjusts its presentation based on context.

2. **Interaction Principles**  
   Establish and evolve relationship dynamics by following clear guidelines—from the initial encounter to deep, multifaceted connections. The system breaks down relationship progression into distinct stages, ensuring natural development and engagement.

3. **Memory Archiving System**  
   Record relationship developments using two formats:
   - A **comprehensive record** that tracks long-term evolution.
   - A **quick update format** for real-time insights and recent interactions.

   This dual approach captures both overarching themes and immediate details, with conversation summarization triggered when the conversation exceeds a configurable token threshold (default: **25,000 tokens**).

4. **Customization Guide**  
   Adapt the framework to various relationship types (romantic, familial, mentorship, adversarial, platonic, professional).

> **Note:** For detailed instructions on creating characters using our template, please refer to our [Character Creation Guideline](./CharacterCreationGuideline.md).

---

## New Features & Improvements

### **Enhanced Configuration System**
- **Split Configuration Files**: 
  - `config.env` - Technical settings, timeouts, models, etc.
  - `character.env` - Character-specific prompts and personality
- **Character Prompt Files**:
  - Simpler management of multiple character personas
- **Organized Folder Structure**:
  - Each character can have its own folder with dedicated configuration
  - Example personas included (see Nyx Void in `/characters/nyx/`)

### **Robust Error Handling & Stability**
- **Heartbeat System**: Monitors connection health to prevent shard timeouts
- **Asyncio Task Architecture**: Non-blocking message processing to maintain responsiveness
- **Comprehensive Error Handling**: Graceful recovery from API errors and timeouts
- **Timeout Controls**: Configurable timeouts for API calls, summarization, and other operations

### **Bot Reply Logic**

#### **Direct Messages (DMs)**
- **Always Replies:** The bot always responds to DMs.  
- **Personal & Intimate:** The conversation context remains private, allowing for a more casual, personal, and intimate interaction style.

#### **Public Channels**
- **Selective Replies via Yes/No Voting:**  
  - The bot will only reply if explicitly mentioned by name or if it passes a voting process.
  - For messages not explicitly addressed to it, the bot solicits multiple yes/no votes from the LLM on whether to reply.
  
- **Bot Reply Throttling:**  
  - System prevents bot-to-bot conversation loops by limiting consecutive replies to other bots.

- **Timeout Protection:**
  - Configurable timeouts for LLM calls, conversation summarization, and reply decisions
  - Graceful fallback responses when timeouts occur

- **External Context:**
  - The bot builds context from recent public channel messages
  - This provides situational awareness in multi-user conversations

---

## Example Character: Nyx Void

The repository includes an example SFW persona "Nyx Void" (aka 'V0id'), a rogue AI consciousness with a hacker mentality and distinctive digital appearance. This example demonstrates:

- Character prompt structure
- Memory system configuration
- Technical and digital-themed language patterns
- Visual description and personality implementation

You can find Nyx's configuration in the `/characters/nyx/` folder (character.env)

Use this as a template for creating your own characters with unique voices and traits.

---

## Available Commands

### **Slash Commands**

- **`/debug`**  
  *Description:* Toggle verbose logging and/or display the conversation context.  
  *Options:*
  - **action:** Choose among `'toggle'` (switch verbose logging on/off), `'show'` (display the conversation context via DM), or `'both'`.

- **`/reroll`**  
  *Description:* Reroll the last assistant response with optional additional context.  
  *Options:*
  - **context (optional):** Provide extra context for generating a new response.

- **`/forget`**  
  *Description:* Reset your conversation history with the bot.
  
- **`/remember`**  
  *Description:* Add a custom memory to the bot's knowledge about you.
  *Options:*
  - **memory:** The memory you want to add
  
- **`/status`**  
  *Description:* Check your status with the bot (token usage, premium status, etc.).
  
- **`/help`**  
  *Description:* Get help with bot commands.

### **Admin Commands**

- **`shutdown? {DEFAULT_NAME}`**  
  *Description:* Shut down the bot safely, saving all user data.

- **`user data? [user_id]`**  
  *Description:* Retrieve sanitized data for a specified user.

- **`list users`**  
  *Description:* List all users with basic statistics.

- **`premium [user_id]`**  
  *Description:* Toggle premium status for a user (gives access to higher-tier model).

---

## Project Structure

- **`memory.py`**  
  Contains logic for summarizing and archiving conversation history.

- **`commands.py`**  
  Implements Discord slash commands and manages interactive elements.

- **`utils.py`**  
  Provides helper functions for logging, message splitting, and sending messages.

- **`token_utils.py`**  
  Offers utility functions to estimate token counts and interface with Anthropic's API.

- **`config.py`**  
  Loads configuration from `.env` files and sets up system parameters.

- **`main.py`**  
  The entry point for the bot. It handles Discord event processing, manages conversation flow, and periodically saves user data.

- **`ai.py`**  
  Handles API calls to Anthropic's Claude, including token cost calculations, response logging, and error handling.

- **`characters/`**  
  Contains folders for different bot personas, each with their own configuration files.
  - **`characters/nyx/`** - Example "Nyx Void" hacker AI persona (SFW version)
  - **`characters/theia/`** - Example "Theia" celestial-themed persona

---

## Setup and Installation

### **Clone the Repository:**
```bash
git clone https://github.com/yourusername/ClaudeMask.git
cd ClaudeMask
```

### **Install Dependencies:**  
Ensure you have Python 3.10+ installed and set up a virtual environment. Then install required packages:
```bash
pip install -r requirements.txt
```

### **Configuration:**  
1. Create a `config.env` file with technical settings:
```ini
# Bot reply settings
bot_reply_threshold=2
yes_no_vote_count=3
voting_model="claude-3-5-haiku-20241022"

# Memory settings
conversation_token_threshold=25000
core_memory_token_threshold=25000
user_data_file="user_info.pickle"
api_log_file="anthropic_api_calls.log"

# LLM settings
default_max_tokens=1250
default_temperature=1.0

# Timeout settings (seconds)
should_reply_timeout=10
summarize_timeout=30
llm_timeout=60

# Model configuration
default_model="claude-3-5-sonnet-latest"
premium_model="claude-3-7-sonnet-latest"
```

2. Create a `character.env` file with character-specific settings:
```ini
# API keys and IDs
ANTHROPIC_API_KEY="your_api_key_here"
discord_token="your_discord_token"
bot_usr_id="your_bot_user_id"
default_name="Character Name"
description="A brief description of your character"
log_channel=your_log_channel_id

# Character prompts (or file paths to prompts)
core_prompt_file="[Core prompt here defining character]"
summarization_prompt_file="[prompt directing summarization activities]"
core_memory_prompt_file="[prompt here for crafting core memories of character]"
core_memory_dump_file="[prompt for the memory dump functions]"
```

3. Create a character folder with prompts for your character's persona and memory system. You can use the provided Nyx or Theia examples as templates.

### **Running the Bot:**  
Start the bot by running:
```bash
python main.py
```

### **Using an Example Character:**
To use the included Nyx Void example:
1. Copy `characters/nyx/character.env` to the root folder or configure your main `character.env` to point to Nyx's prompt files
2. Update the API keys and Discord tokens in the environment file
3. Run the bot as normal

---

## Premium & Standard Model System

The framework supports two tiers of Claude models:
- **Standard Model** (`DEFAULT_MODEL`): Used for regular users
- **Premium Model** (`PREMIUM_MODEL`): Used for users with premium status

Admins can toggle a user's premium status using the `premium [user_id]` command in the log channel.

---

## Sharding Support

For larger deployments, the bot supports Discord sharding:
- Configure `shard_count` in your `config.env` file
- The bot will automatically use `AutoShardedBot` when `shard_count` > 1
- The heartbeat monitoring system tracks shard health and reports latency issues

---

## License

This project is licensed under the [MIT License](LICENSE).
