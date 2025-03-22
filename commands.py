import discord
from discord import app_commands
from discord.ext import commands
from discord.ui import View, Button
from config import DEFAULT_MODEL, PREMIUM_MODEL, CORE_PROMPT
from utils import log_error, toggle_verbose
from ai import call_claude  # Import needed for reroll

# Global dictionary to track active reroll views by user ID.
active_reroll_views = {}

def disable_previous_views(user_id: str):
    if user_id in active_reroll_views:
        for view in active_reroll_views[user_id]:
            for child in view.children:
                child.disabled = True
            view.stop()
        active_reroll_views[user_id] = []

class RerollView(View):
    def __init__(self, result: str, user_id: str, system_text: str, model: str, temp_user_data: dict, reroll_callback, original_message: discord.Message):
        """
        :param result: The initial rerolled output.
        :param user_id: The ID of the user invoking reroll.
        :param system_text: The system prompt built for the LLM call.
        :param model: The model to use.
        :param temp_user_data: The temporary user data for the LLM call.
        :param reroll_callback: Callback function to re-run the reroll logic.
        :param original_message: The ephemeral message sent initially.
        """
        super().__init__(timeout=60)
        self.result = result
        self.user_id = user_id
        self.system_text = system_text
        self.model = model
        self.temp_user_data = temp_user_data
        self.reroll_callback = reroll_callback
        self.original_message = original_message

    async def disable_buttons(self, interaction: discord.Interaction):
        for child in self.children:
            child.disabled = True
        self.stop()
        try:
            await self.original_message.edit(view=self)
        except discord.NotFound:
            pass

    @discord.ui.button(label="Accept", style=discord.ButtonStyle.success, emoji="✅")
    async def accept_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Accept: update the ephemeral message and then send result publicly.
        await self.original_message.edit(content="Message accepted.")
        await interaction.response.send_message(f"{self.result}")
        self.stop()
        if self.user_id in active_reroll_views and self in active_reroll_views[self.user_id]:
            active_reroll_views[self.user_id].remove(self)

    @discord.ui.button(label="Dismiss", style=discord.ButtonStyle.danger, emoji="❌")
    async def dismiss_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Dismiss: update the ephemeral message.
        await self.original_message.edit(content="Reroll dismissed.")
        self.stop()
        if self.user_id in active_reroll_views and self in active_reroll_views[self.user_id]:
            active_reroll_views[self.user_id].remove(self)

    @discord.ui.button(label="Redo", style=discord.ButtonStyle.primary, emoji="🎲")
    async def redo_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Disable current view's buttons
        for child in self.children:
            child.disabled = True
        try:
            await self.original_message.edit(view=self)
        except discord.NotFound:
            await interaction.response.send_message("The original message was deleted. Please try again.",
                                                    ephemeral=True)
            return
        except discord.HTTPException as e:
            log_error(f"Error updating message: {e}")
            await interaction.response.send_message("Failed to update the message. Please try again.", ephemeral=True)
            return

        # Add a deferred response while we wait for the reroll
        await interaction.response.defer(ephemeral=True)

        # Re-run the reroll callback
        try:
            new_result = await self.reroll_callback(self.user_id, self.system_text, self.model, self.temp_user_data)
            self.result = new_result
        except Exception as e:
            log_error(f"Error during reroll: {e}")
            await interaction.followup.send("An error occurred during reroll. Please try again.", ephemeral=True)
            return

        # Create a new view instance with fresh buttons
        new_view = RerollView(
            result=new_result,
            user_id=self.user_id,
            system_text=self.system_text,
            model=self.model,
            temp_user_data=self.temp_user_data,
            reroll_callback=self.reroll_callback,
            original_message=self.original_message
        )

        # Disable previous views for this user
        disable_previous_views(self.user_id)
        active_reroll_views.setdefault(self.user_id, []).append(new_view)

        # Edit the original ephemeral message with the new result and view
        try:
            await self.original_message.edit(content=new_result, view=new_view)
            await interaction.followup.send("Reroll complete!", ephemeral=True)
        except discord.HTTPException as e:
            log_error(f"Error updating message with reroll: {e}")
            await interaction.followup.send("Failed to update the message with the new response. Please try again.",
                                            ephemeral=True)

class ForgetLastView(View):
    def __init__(self, user_id: str, messages_to_forget: list, user_data: dict, original_message: discord.Message):
        """
        View for forgetting the last N messages.
        
        :param user_id: The ID of the user.
        :param messages_to_forget: List of indexes to potentially forget.
        :param user_data: The global user data dictionary.
        :param original_message: The ephemeral message this view is attached to.
        """
        super().__init__(timeout=120)  # 2 minute timeout
        self.user_id = user_id
        self.messages_to_forget = messages_to_forget
        self.user_data = user_data
        self.original_message = original_message
        self.selected_indexes = set()  # Track which messages are selected to forget

    @discord.ui.button(label="Select All", style=discord.ButtonStyle.secondary, emoji="🔠")
    async def select_all_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Toggle whether all messages are selected
        if len(self.selected_indexes) == len(self.messages_to_forget):
            # If all are selected, unselect all
            self.selected_indexes.clear()
        else:
            # Otherwise, select all
            self.selected_indexes = set(range(len(self.messages_to_forget)))
        
        # Update the message to reflect the new selection state
        await self.update_message(interaction)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary, emoji="🔙")
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.original_message.edit(content="Operation cancelled. No messages were forgotten.", view=None)
        self.stop()

    @discord.ui.button(label="Forget Selected", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def forget_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        conversation = self.user_data.get(self.user_id, {}).get("conversation_history", [])
        
        if not self.selected_indexes:
            await interaction.response.send_message("No messages selected to forget.", ephemeral=True)
            return
            
        # Get the actual indexes in the conversation history
        indexes_to_remove = [self.messages_to_forget[i] for i in self.selected_indexes]
        indexes_to_remove.sort(reverse=True)  # Sort in reverse to remove from the end first
        
        # Remove the selected messages
        for idx in indexes_to_remove:
            if 0 <= idx < len(conversation):
                del conversation[idx]
        
        # Update the conversation history
        self.user_data[self.user_id]["conversation_history"] = conversation
        
        # Update the message and disable the view
        await self.original_message.edit(
            content=f"✅ Successfully removed {len(indexes_to_remove)} messages from your conversation history.", 
            view=None
        )
        self.stop()

    async def update_message(self, interaction: discord.Interaction):
        """Update the message to show which messages are selected."""
        conversation = self.user_data.get(self.user_id, {}).get("conversation_history", [])
        
        # Build the message content
        content = "Select messages to forget (click on message numbers to toggle selection):\n\n"
        
        for i, msg_idx in enumerate(self.messages_to_forget):
            if msg_idx < len(conversation):
                msg = conversation[msg_idx]
                # Format each message with selection status
                selected = "✅" if i in self.selected_indexes else "⬜"
                # Truncate message content if too long
                msg_content = msg["content"]
                if len(msg_content) > 100:
                    msg_content = msg_content[:97] + "..."
                content += f"**[{selected}] {i+1}.** {msg['role'].upper()}: {msg_content}\n\n"
        
        content += "\nClick 'Forget Selected' to remove the selected messages, or 'Cancel' to cancel."
        
        await interaction.response.defer()
        await self.original_message.edit(content=content)
    
    @discord.ui.button(label="1", style=discord.ButtonStyle.primary, row=1)
    async def button_1(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.toggle_selection(interaction, 0)
    
    @discord.ui.button(label="2", style=discord.ButtonStyle.primary, row=1)
    async def button_2(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.toggle_selection(interaction, 1)
    
    @discord.ui.button(label="3", style=discord.ButtonStyle.primary, row=1)
    async def button_3(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.toggle_selection(interaction, 2)
    
    @discord.ui.button(label="4", style=discord.ButtonStyle.primary, row=1)
    async def button_4(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.toggle_selection(interaction, 3)
    
    @discord.ui.button(label="5", style=discord.ButtonStyle.primary, row=1)
    async def button_5(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.toggle_selection(interaction, 4)
    
    @discord.ui.button(label="6", style=discord.ButtonStyle.primary, row=2)
    async def button_6(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.toggle_selection(interaction, 5)
    
    @discord.ui.button(label="7", style=discord.ButtonStyle.primary, row=2)
    async def button_7(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.toggle_selection(interaction, 6)
    
    @discord.ui.button(label="8", style=discord.ButtonStyle.primary, row=2)
    async def button_8(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.toggle_selection(interaction, 7)
    
    @discord.ui.button(label="9", style=discord.ButtonStyle.primary, row=2)
    async def button_9(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.toggle_selection(interaction, 8)
    
    @discord.ui.button(label="10", style=discord.ButtonStyle.primary, row=2)
    async def button_10(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.toggle_selection(interaction, 9)
    
    async def toggle_selection(self, interaction: discord.Interaction, index: int):
        """Toggle the selection state of a message."""
        # Check if the index is valid
        if index >= len(self.messages_to_forget):
            await interaction.response.send_message("Invalid selection.", ephemeral=True)
            return
            
        # Toggle selection
        if index in self.selected_indexes:
            self.selected_indexes.remove(index)
        else:
            self.selected_indexes.add(index)
            
        # Update the message
        await self.update_message(interaction)


def setup_commands(bot: commands.Bot, user_data: dict):
    
    @bot.tree.command(name="reset_conversation", description="Reset your entire conversation history with the bot (keeps core memories).")
    async def reset_conversation(interaction: discord.Interaction):
        user_id = str(interaction.user.id)
        if user_id not in user_data:
            await interaction.response.send_message("No conversation history found.", ephemeral=True)
            return
        # Create confirmation buttons
        class ConfirmView(discord.ui.View):
            def __init__(self):
                super().__init__(timeout=30)  # 30 second timeout
            @discord.ui.button(label="Yes, Reset Everything", style=discord.ButtonStyle.danger)
            async def confirm_button(self, interaction: discord.Interaction, button: discord.ui.Button):
                # Keep core memories but clear conversation history
                user_data[user_id]["conversation_history"] = []
                await interaction.response.send_message(
                    "✅ Your conversation history has been reset. Core memories remain intact.", ephemeral=True)
                self.stop()
                
            @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
            async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
                await interaction.response.send_message("Operation cancelled. Your conversation history remains intact.", ephemeral=True)
                self.stop()
              
        # Send confirmation message
        await interaction.response.send_message(
            "⚠️ **Are you sure?** This will permanently delete your entire conversation history with me.\n"
            "Note: Your core memories will be preserved.", 
            view=ConfirmView(),
            ephemeral=True
        )
    
    @bot.tree.command(name="forget_last", description="Forget specific recent messages from your conversation.")
    @app_commands.describe(count="Number of recent message pairs to show (default: 5)")
    async def forget_last(interaction: discord.Interaction, count: int = 5):
        user_id = str(interaction.user.id)
        if user_id not in user_data:
            await interaction.response.send_message("No conversation history found.", ephemeral=True)
            return
            
        conversation = user_data[user_id].get("conversation_history", [])
        if not conversation:
            await interaction.response.send_message("No conversation history found.", ephemeral=True)
            return
            
        # Limit count to a reasonable number (max 10 for UI buttons)
        count = min(max(1, count), 10)
        
        # Get the last 'count' messages (pairs of user/assistant)
        total_msgs = len(conversation)
        start_idx = max(0, total_msgs - (count * 2))
        
        # Collect indexes of messages to potentially forget
        messages_to_forget = list(range(start_idx, total_msgs))
        
        # If no messages to forget, return
        if not messages_to_forget:
            await interaction.response.send_message("No recent messages found to forget.", ephemeral=True)
            return
            
        # Create initial content
        content = f"Select messages to forget (click on message numbers to toggle selection):\n\n"
        
        for i, msg_idx in enumerate(messages_to_forget[:10]):  # Only show up to 10 messages
            if msg_idx < len(conversation):
                msg = conversation[msg_idx]
                msg_content = msg["content"]
                if len(msg_content) > 100:
                    msg_content = msg_content[:97] + "..."
                content += f"**[⬜] {i+1}.** {msg['role'].upper()}: {msg_content}\n\n"
        
        content += "\nClick 'Forget Selected' to remove the selected messages, or 'Cancel' to cancel."
        
        # Send ephemeral message with the view
        await interaction.response.defer(ephemeral=True)
        ephemeral_msg = await interaction.followup.send(content=content, ephemeral=True)
        
        # Create and add the view
        view = ForgetLastView(
            user_id=user_id,
            messages_to_forget=messages_to_forget[:10],  # Only use up to 10 messages
            user_data=user_data,
            original_message=ephemeral_msg
        )
        
        # Add the view to the message
        await ephemeral_msg.edit(view=view)

    @bot.tree.command(name="remember", description="Add a custom memory to the bot's knowledge about you.")
    @app_commands.describe(memory="The memory you want to add")
    async def remember(interaction: discord.Interaction, memory: str):
        user_id = str(interaction.user.id)
        if user_id not in user_data:
            user_data[user_id] = {
                "token_usage": 0,
                "premium": False,
                "conversation_history": [],
                "core_memories": ""
            }

        # Add the memory to core memories
        current_memories = user_data[user_id].get("core_memories", "")
        user_data[user_id]["core_memories"] = f"{current_memories}\n- {memory}"

        await interaction.response.send_message("I'll remember that.", ephemeral=True)

    @bot.tree.command(name="status", description="Check your status with the bot.")
    async def status(interaction: discord.Interaction):
        user_id = str(interaction.user.id)
        if user_id not in user_data:
            await interaction.response.send_message("No data found for you.", ephemeral=True)
            return

        token_usage = user_data[user_id].get("token_usage", 0)
        premium_status = "Premium" if user_data[user_id].get("premium", False) else "Standard"
        conversation_length = len(user_data[user_id].get("conversation_history", []))
        newline='\n'
        await interaction.response.send_message(
            f"Status:\n"
            f"- Plan: {premium_status}\n"
            f"- Token usage: {token_usage:,} tokens\n"
            f"- Conversation length: {conversation_length} messages\n"
            f"- Core memories: {len(user_data[user_id].get('core_memories', '').split(newline))} entries",
            ephemeral=True
        )

    @bot.tree.command(name="help", description="Get help with bot commands.")
    async def help_command(interaction: discord.Interaction):
        commands_list = [
            ("forget", "Reset your entire conversation history"),
            ("forget_last", "Selectively forget recent messages"),
            ("remember", "Add a custom memory"),
            ("reroll", "Get a different response to your last message"),
            ("status", "Check your usage statistics"),
            ("help", "Show this help message")
        ]

        help_text = "Available Commands:\n" + "\n".join([f"/{cmd} - {desc}" for cmd, desc in commands_list])

        await interaction.response.send_message(help_text, ephemeral=True)

    @bot.tree.command(name="reroll", description="Reroll the last assistant response with optional additional context.")
    @app_commands.describe(context="Additional context to include (optional).")
    async def reroll(interaction: discord.Interaction, context: str = None):
        user_id = str(interaction.user.id)
        conv_history = user_data.get(user_id, {}).get("conversation_history", [])
        if not conv_history:
            await interaction.response.send_message("No conversation history available to reroll.", ephemeral=True)
            return

        # Remove the last assistant message if it exists.
        if conv_history[-1]["role"] == "assistant":
            temp_history = conv_history[:-1]
        else:
            temp_history = conv_history[:]

        if context:
            temp_history.append({
                "role": "user",
                "content": "[OOC]: " + context + "\nIf you respond to this context, please use [OOC] tags."
            })

        core_mem = user_data[user_id].get("core_memories", "")
        system_text = f"{CORE_PROMPT}\n\nCore Memories:\n{core_mem}"
        model = PREMIUM_MODEL if user_data[user_id].get("premium", False) else DEFAULT_MODEL

        temp_user_data = {
            user_id: {
                "conversation_history": temp_history,
                "core_memories": core_mem,
                "premium": user_data[user_id].get("premium", False)
            }
        }

        # Define the reroll callback for the "Redo" functionality.
        async def reroll_callback(user_id: str, system_text: str, model: str, temp_user_data: dict) -> str:
            async with interaction.channel.typing():
                new_response = await call_claude(
                    user_id=user_id,
                    user_dict=temp_user_data,
                    model=model,
                    system_prompt=system_text,
                    user_content=None,
                    temperature=1.0,
                    max_tokens=1250,
                    verbose=False
                )
            return new_response.choices[0].message["content"]

        # Before processing this reroll, disable any previous active reroll views for this user.
        disable_previous_views(user_id)

        # Defer the response and call the LLM.
        await interaction.response.defer(ephemeral=True)
        async with interaction.channel.typing():
            response = await call_claude(
                user_id=user_id,
                user_dict=temp_user_data,
                model=model,
                system_prompt=system_text,
                user_content=None,
                temperature=1.0,
                max_tokens=1250,
                verbose=False
            )
        result = response.choices[0].message["content"]

        # Send the ephemeral message and get its message object.
        ephemeral_msg = await interaction.followup.send(content=result, ephemeral=True)
        # Create a view instance with the original ephemeral message.
        new_view = RerollView(
            result=result,
            user_id=user_id,
            system_text=system_text,
            model=model,
            temp_user_data=temp_user_data,
            reroll_callback=reroll_callback,
            original_message=ephemeral_msg
        )
        # Register the new view in our active views.
        active_reroll_views.setdefault(user_id, []).append(new_view)
        # Edit the ephemeral message to add the view.
        await ephemeral_msg.edit(view=new_view)
