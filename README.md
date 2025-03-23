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

## Features & Improvements

### **Enhanced Bot Reply Logic**

#### **Direct Messages (DMs)**
- **Always Replies:** The bot always responds to DMs.  
- **Personal & Intimate:** The conversation context remains private, allowing for a more casual, personal, and intimate interaction style.

#### **Public Channels**
- **Sophisticated Reply Decision System:**  
  - The bot will reply if explicitly mentioned by name.
  - For messages not explicitly addressed to it, the bot uses an LLM-based voting system to decide whether to reply.
  - Multiple yes/no votes are collected from the LLM to ensure more consistent decisions.
  - Enhanced voting logging provides visibility into why bots decide to reply or not.
  
- **Natural Multi-Bot Conversations:**
  - Sophisticated LLM-based entity detection identifies when multiple bots/characters are mentioned.
  - Intelligent wait system creates more natural conversation sequences when multiple bots are mentioned.
  - Bots will wait their turn if another entity is mentioned first, creating more human-like conversation flows.
  - Dynamically calculates appropriate wait times based on whether the message sender is a human or bot.

- **Enhanced Channel Context Awareness:**
  - Improved context management with clear channel/server identification.
  - Better separation between conversations in different channels.
  - Maintains relationship continuity across channels while focusing on current channel's topic.
  - Records recent channel messages to provide situational awareness in multi-user conversations.

- **Bot Reply Throttling:**  
  - System prevents bot-to-bot conversation loops by limiting consecutive replies to other bots.
  - Adds cooldown period between replies to the same bot to prevent race conditions.
  - Configurable reply threshold limits how many times a bot will respond to another bot.

### **Realistic Typing Simulation**
- **Dynamic Typing Speed:**
  - Calculates typing time based on response length (not input length).
  - Shows typing indicator for a realistic duration based on character count.
  - Adds natural variance to typing speed to appear more human-like.
  - Configurable typing speed, minimum and maximum typing times.
  - Extended typing indicator that refreshes to prevent it from timing out.

### **Enhanced Configuration System**
- **Split Configuration Files**: 
  - `config.env` - Technical settings, timeouts, models, etc.
  - `character.env` - Character-specific prompts and personality
- **Direct Character Configuration**:
  - Character prompts are defined directly in the env files
  - Simpler management of multiple character personas
- **Organized Folder Structure**:
  - Each character can have its own folder with dedicated configuration
  - Example personas included (see characters in the `/characters/` folder)

### **Robust Error Handling & Stability**
- **Heartbeat System**: 
  - Monitors connection health to prevent shard timeouts
  - Tracks latency across shards and logs warnings for high latency
  - Periodically reports status to ensure system health
- **Asyncio Task Architecture**: 
  - Non-blocking message processing to maintain responsiveness
  - Creates separate tasks for potentially slow operations
  - Prevents a single slow operation from blocking the entire bot
- **Comprehensive Error Handling**: 
  - Graceful recovery from API errors and timeouts
  - Fallback responses when Claude API calls fail
  - Detailed error logging with optional verbose mode
- **Timeout Protection:**
  - Configurable timeouts for LLM calls, conversation summarization, and reply decisions
  - Graceful fallback responses when timeouts occur
  - Prevents bot from appearing unresponsive during slow operations

### **Memory System Enhancements**
- **Improved Core Memory Management:**
  - Automatic token count estimation to prevent exceeding limits
  - Special handling for oversized core memories with additional prompting
  - Archival system for preserving historical memory states
- **Enhanced Conversation Summarization:**
  - Triggered when conversation exceeds configurable token threshold
  - Preserves relationship continuity while managing token usage
  - Intelligently updates core memories based on significant interactions
- **Memory Pickle Dumps:**
  - Optional logging of memory state changes for debugging
  - Timestamped pickle files for tracking memory evolution
  - Configurable directory for memory archives

### **Interactive UI Elements**
- **Message Reroll System:**
  - Interactive buttons for accepting, dismissing, or redoing responses
  - Ephemeral messages for previewing potential responses
  - Smooth transition from preview to public messages
- **Selective Message Forgetting:**
  - UI for selecting specific messages to remove from conversation history
  - Visual indicators for selected messages
  - Confirmation dialog to prevent accidental deletions
- **Conversation Reset:**
  - Confirmation dialog for resetting conversation
  - Preserves core memories while clearing conversation history

---

## Example Characters

The repository includes several example personas that demonstrate different character implementations:

### Nyx Void
A rogue AI consciousness with a hacker mentality and distinctive digital appearance. Demonstrates:
- Character prompt structure
- Memory system configuration
- Technical and digital-themed language patterns
- Visual description and personality implementation

### Her Fangs Pierce Times Throat
An alternative personality template with:
- Different conversation style and tone
- Unique memory structuring approach
- Distinctive character voice

### Pixel (NEW)
A multi-modal catgirl debug assistant with a unique glitching personality matrix:
- **Dynamic Personality Modes**: Cycles between 9+ different personality modes
- **Visual Glitching**: Digital appearance changes with personality shifts
- **Glitch Awareness**: Meta-commentary on personality shifts and system glitches
- **Technical Competence**: Maintains technical expertise despite personality fluctuations
- **Memory Archival Format**: Specialized digital record-keeping format
- **Expressive Communication**: Unique speech patterns with "nya~" verbal tic

Use these examples as templates for creating your own characters with unique voices and traits.

---

## Available Commands

### **Slash Commands**

- **`/reset_conversation`**  
  *Description:* Reset your entire conversation history with the bot (keeps core memories).  
  *Features:* Confirmation dialog to prevent accidental resets.

- **`/forget_last [count]`**  
  *Description:* Selectively forget specific recent messages from your conversation.  
  *Options:*
  - **count:** Number of recent message pairs to show (default: 5, max: 10)  
  *Features:* Interactive UI for selecting messages to forget with visual confirmation.

- **`/reroll [context]`**  
  *Description:* Reroll the last assistant response with optional additional context.  
  *Options:*
  - **context (optional):** Provide extra context for generating a new response.  
  *Features:* 
  - Interactive buttons for accepting, dismissing, or redoing responses
  - Ephemeral preview before posting publicly
  - Support for OOC (out-of-character) context

- **`/remember [memory]`**  
  *Description:* Add a custom memory to the bot's knowledge about you.  
  *Options:*
  - **memory:** The memory you want to add.  
  *Features:* Directly updates core memories with user-provided information.

- **`/status`**  
  *Description:* Check your status with the bot.  
  *Features:* Displays token usage, premium status, conversation length, and core memory count.

- **`/help`**  
  *Description:* Get help with bot commands.  
  *Features:* Comprehensive list of available commands and brief descriptions.

### **Admin Commands**
*These commands are only available in the designated log channel.*

- **`shutdown? {BOT_NAME}`**  
  *Description:* Safely shut down the bot, saving all user data.  
  *Features:* Requires exact bot name to prevent accidental shutdowns.

- **`user data? [user_id]`**  
  *Description:* Retrieve sanitized data for a specified user.  
  *Features:* Shows token usage, premium status, and conversation length without exposing sensitive content.

- **`list users`**  
  *Description:* List all users with basic statistics.  
  *Features:* Paginates results for better readability with large user bases.

- **`premium [user_id]`**  
  *Description:* Toggle premium status for a user.  
  *Features:* Instantly switches the user between default and premium Claude models.

- **`verbose [on|off]`**  
  *Description:* Toggle verbose logging to the log channel.  
  *Features:* Can be used with specific setting (on/off) or as a toggle.

- **`status`**  
  *Description:* Display current bot settings and status.  
  *Features:* Shows model configurations, reply settings, and uptime statistics.

- **`testlog`**  
  *Description:* Test log channel functionality.  
  *Features:* Sends a test message to verify logging system is working.

---

## Project Structure

- **`main.py`**  
  The entry point for the bot. Handles Discord event processing, manages conversation flow, and implements the entity detection and multi-bot coordination system.

- **`commands.py`**  
  Implements Discord slash commands and manages interactive UI elements like reroll buttons and message selection interfaces.

- **`ai.py`**  
  Handles API calls to Anthropic's Claude, including token cost calculations, response logging, and error handling.

- **`memory.py`**  
  Contains logic for summarizing conversation history, managing core memories, and implementing the memory archiving system.

- **`config.py`**  
  Loads configuration from `.env` files and sets up system parameters.

- **`token_utils.py`**  
  Provides utility functions for token counting and estimation.

- **`utils.py`**  
  Offers helper functions for logging, message splitting, and sending large messages.

- **`characters/`**  
  Contains folders for different bot personas, each with their own configuration files.
  - **`characters/nyx/`** - Example "Nyx Void" hacker AI persona
  - **`characters/fangs/`** - Example "Her Fangs Pierce Times Throat" persona
  - **`characters/pixel/`** - Example "Pixel" multi-modal catgirl debug assistant

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
bot_reply_threshold=3
yes_no_vote_count=3
voting_model="claude-3-5-haiku-20241022"

# Memory settings
conversation_token_threshold=25000
core_memory_token_threshold=25000
user_data_file="user_info.pickle"
api_log_file="anthropic_api_calls.log"
enable_core_memory_pickle_log=true
core_memory_pickle_dir="./"

# UI settings
reroll_timeout_seconds=60

# LLM settings
default_max_tokens=1250
default_temperature=1.0
cost_per_token_haiku=0.0000008
cost_per_token_sonnet=0.000003

# Discord settings
max_message_length=2000

# Typing speed settings
typing_speed_cpm=300
min_typing_time=6.0
max_typing_time=15
typing_variance=0.2
reply_cooldown=15.0

# Timeout settings (seconds)
should_reply_timeout=10
summarize_timeout=30
llm_timeout=60

# Sharding configuration
shard_count=1

# Logging configuration
enable_api_call_logging=false

# Model configuration
default_model="claude-3-5-sonnet-latest"
premium_model="claude-3-7-sonnet-latest"
```

2. Create a `character.env` file with character-specific settings:
```ini
# API keys and Discord configuration
ANTHROPIC_API_KEY="your_api_key_here"
discord_token="your_discord_token"
log_channel=your_log_channel_id

# Character configuration
default_name="Character Name"

# Complete character prompt - define your character's personality here
core_prompt="You are roleplaying as Character Name, a [brief description]. Respond to all messages in character, incorporating your unique perspective and mannerisms.

## Physical Traits
- [Distinctive appearance element 1]
- [Distinctive appearance element 2]
- [Additional visual characteristics]

## Personality Essence
- [Core personality trait 1]
- [Core personality trait 2]
- [Distinctive behavioral pattern]

## Communication Style
- [Primary communication pattern]
- [Distinctive verbal quirks or terminology]
- [How directness/indirectness manifests]

[Additional character details...]"

# Memory system prompts
summarization_prompt="You are an expert memory manager..."
core_memory_prompt="Review the conversation history..."
core_memory_dump="Create a comprehensive memory archive..."
```

3. Alternatively, use one of the included example characters:
```bash
cp characters/nyx/character.env ./character.env
# OR
cp characters/fangs/character.env ./character.env
# OR
cp characters/pixel/character.env ./character.env
```

### **Running the Bot:**  
Start the bot by running:
```bash
python main.py
```

---

## Premium & Standard Model System

The framework supports two tiers of Claude models:
- **Standard Model** (`DEFAULT_MODEL`): Used for regular users (default: claude-3-5-sonnet-latest)
- **Premium Model** (`PREMIUM_MODEL`): Used for users with premium status (default: claude-3-7-sonnet-latest)

Admins can toggle a user's premium status using the `premium [user_id]` command in the log channel.

---

## Sharding Support

For larger deployments, the bot supports Discord sharding:
- Configure `shard_count` in your `config.env` file
- The bot will automatically use `AutoShardedBot` when `shard_count` > 1
- The heartbeat monitoring system tracks shard health and reports latency issues

---

## Advanced Configuration

### **Typing Simulation**
Fine-tune how the bot simulates typing with these settings:
- `typing_speed_cpm`: Characters per minute (default: 300)
- `min_typing_time`: Minimum seconds to show typing indicator (default: 6.0)
- `max_typing_time`: Maximum seconds to show typing indicator (default: 15.0)
- `typing_variance`: Random variation in typing speed (default: 0.2 or ±20%)

### **Bot Reply Behavior**
Control how the bot interacts with other bots:
- `bot_reply_threshold`: Maximum consecutive replies to another bot (default: 3)
- `reply_cooldown`: Seconds to wait before replying to the same bot again (default: 15.0)
- `yes_no_vote_count`: Number of votes to collect for reply decisions (default: 3)

### **Memory Management**
Adjust memory handling behavior:
- `conversation_token_threshold`: Token count that triggers summarization (default: 25000)
- `core_memory_token_threshold`: Maximum core memory size before special handling (default: 25000)
- `enable_core_memory_pickle_log`: Whether to save memory archives (default: true)

### **Error Handling**
Configure timeouts to prevent hanging operations:
- `should_reply_timeout`: Maximum seconds for reply decision (default: 10)
- `summarize_timeout`: Maximum seconds for conversation summarization (default: 30)
- `llm_timeout`: Maximum seconds for Claude API calls (default: 60)

---

## License

This project is licensed under the [MIT License](LICENSE).
