# config.py
import os
import json
from dotenv import load_dotenv

# Load system configuration first (base settings)
load_dotenv("config.env")

# Load character-specific configuration (these will override any duplicate settings)
load_dotenv("character.env", override=True)

# Use a valid Anthropic API key
OAI_TOKEN = os.environ.get("ANTHROPIC_API_KEY")

# Discord configuration
DISCORD_TOKEN = os.environ.get("discord_token")
LOG_CHANNEL_ID = int(os.environ.get("log_channel", "0"))

# Model configuration
DEFAULT_MODEL = os.environ.get("default_model", "claude-3-5-sonnet-latest")
PREMIUM_MODEL = os.environ.get("premium_model", "claude-3-7-sonnet-latest")

# Pricing configuration
COST_PER_TOKEN_HAIKU = float(os.environ.get("cost_per_token_haiku", "0.0000008"))
COST_PER_TOKEN_SONNET = float(os.environ.get("cost_per_token_sonnet", "0.000003"))


# Typing speed settings
TYPING_SPEED_CPM = int(os.environ.get("typing_speed_cpm", "250"))  # Characters per minute
MIN_TYPING_TIME = float(os.environ.get("min_typing_time", "2.0"))  # Minimum seconds
MAX_TYPING_TIME = float(os.environ.get("max_typing_time", "60.0"))  # Maximum seconds
TYPING_VARIANCE = float(os.environ.get("typing_variance", "0.2"))  # ±20% random variance
REPLY_COOLDOWN = float(os.environ.get("reply_cooldown", "10.0")) # 10.0 s reply cooldonw for bots

# Character configuration
DEFAULT_NAME = os.environ.get("default_name", "Assistant")
CORE_PROMPT = os.environ.get("core_prompt", "You are a helpful AI assistant.")

# Summarization prompts
SUMMARIZATION_PROMPT = os.environ.get("summarization_prompt", "Summarize the conversation.")
CORE_MEMORY_PROMPT = os.environ.get("core_memory_prompt", "Update core memories.")
CORE_MEMORY_DUMP_PROMPT = os.environ.get("core_memory_dump", "Create comprehensive memory update.")

# Memory settings
ENABLE_CORE_MEMORY_PICKLE_LOG = os.environ.get("enable_core_memory_pickle_log", "true").lower() == "true"
CORE_MEMORY_PICKLE_DIR = os.environ.get("core_memory_pickle_dir", "./")
CONVERSATION_TOKEN_THRESHOLD = int(os.environ.get("conversation_token_threshold", "25000"))
CORE_MEMORY_TOKEN_THRESHOLD = int(os.environ.get("core_memory_token_threshold", "25000"))

# Bot reply settings
BOT_REPLY_THRESHOLD = int(os.environ.get("bot_reply_threshold", "3"))
YES_NO_VOTE_COUNT = int(os.environ.get("yes_no_vote_count", "3"))
VOTING_MODEL = os.environ.get("voting_model", "claude-3-5-haiku-20241022")

# File paths
USER_DATA_FILE = os.environ.get("user_data_file", "user_info.pickle")
API_LOG_FILE = os.environ.get("api_log_file", "anthropic_api_calls.log")
VERBOSE_LOGGING = os.environ.get("VERBOSE_LOGGING", False)

# UI settings
REROLL_TIMEOUT_SECONDS = int(os.environ.get("reroll_timeout_seconds", "60"))

# LLM settings
DEFAULT_MAX_TOKENS = int(os.environ.get("default_max_tokens", "1250"))
DEFAULT_TEMPERATURE = float(os.environ.get("default_temperature", "1.0"))

# Discord settings
MAX_MESSAGE_LENGTH = int(os.environ.get("max_message_length", "2000"))

# Periodic tasks
SAVE_INTERVAL_MINUTES = int(os.environ.get("save_interval_minutes", "1"))

# Timeout settings
SHOULD_REPLY_TIMEOUT = float(os.environ.get("should_reply_timeout", "10"))
SUMMARIZE_TIMEOUT = float(os.environ.get("summarize_timeout", "30"))
LLM_TIMEOUT = float(os.environ.get("llm_timeout", "60"))

# Sharding configuration
SHARD_COUNT = int(os.environ.get("shard_count", "1"))

# Logging configuration
ENABLE_API_CALL_LOGGING = os.environ.get("enable_api_call_logging", "false").lower() == "true"
