from __future__ import annotations

import datetime
import asyncio
import pandas as pd
from typing import Any, Dict

# Restore original audio_util import - this should not be replaced
from audio_util import CHANNELS, SAMPLE_RATE, AudioPlayerAsync

# Import from our new modules
from ui_components import SessionDisplay, AudioStatusIndicator
from logging_utils import init_logging
from realtime_handler import RealtimeHandler
from data_utils import load_dummy_dataframe

# Original imports
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Button, Static, RichLog, Input
from textual.containers import Container, Horizontal

from openai import AsyncOpenAI
import websockets.exceptions
import weave
import wandb


class RealtimeApp(App[None]):
    CSS = """
        Screen {
            background: #1a1b26;  /* Dark blue-grey background */
        }

        Container {
            border: double rgb(91, 164, 91);
        }

        Horizontal {
            width: 100%;
        }

        #input-container {
            height: 10;  /* Explicit height for input container */
            margin: 1 1;
            padding: 1 2;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
        }

        Input {
            width: 80%;
            height: 3;  /* Explicit height for input */
        }

        Button {
            width: 20%;
            height: 3;  /* Explicit height for button */
        }

        #bottom-pane {
            width: 100%;
            height: 82%;  /* Reduced to make room for input container */
            border: round rgb(205, 133, 63);
            content-align: center middle;
        }

        #status-indicator {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        #session-display {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        Static {
            color: white;
        }
    """

    def __init__(self) -> None:
        super().__init__()
        self.client = AsyncOpenAI()
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.dataframe = load_dummy_dataframe()
        self.accumulated_transcript = ""
        self.pending_function_calls = {}
        
        # Create the realtime handler
        self.realtime_handler = RealtimeHandler(
            self,
            self.client,
            self.audio_player,
            self.dataframe,
            self.connected,
            self.pending_function_calls
        )

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            yield SessionDisplay(id="session-display")
            yield AudioStatusIndicator(id="status-indicator")
            with Horizontal(id="input-container"):
                yield Input(placeholder="Type your message here...", id="text-input")
                yield Button("Send", id="send-button", variant="primary")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        # Initialize logging
        init_logging()
        
        # Start the realtime connection and audio workers
        self.run_worker(self.realtime_handler.handle_realtime_connection())
        self.run_worker(self.realtime_handler.send_mic_audio())
        
        # Display initial dataframe information
        bottom_pane = self.query_one("#bottom-pane", RichLog)
        bottom_pane.write("\nAsk questions about this data by pressing K and speaking.")

        # Start automated questions task
        self.run_worker(self._run_automated_questions())

    async def _run_automated_questions(self) -> None:
        """Runs a predefined sequence of questions automatically."""
        # Wait a moment for the connection to establish
        await asyncio.sleep(2) 
        
        questions = []
        
        input_widget = self.query_one("#text-input", Input)
        
        for question in questions:
            self.query_one("#bottom-pane", RichLog).write(f"[bold green]Automated Question:[/bold green] {question}")
            input_widget.value = question
            await self.realtime_handler.handle_text_input()
            # Wait a bit for the response before sending the next question
            await asyncio.sleep(15)  # Adjust delay as needed

    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "enter":
            # Only process text input on Enter key if the input field has focus
            input_widget = self.query_one("#text-input", Input)
            # Check if the focused element is the input widget before handling text input
            if self.focused is input_widget and input_widget.value: 
                await self.realtime_handler.handle_text_input()
                return

        if event.key == "q":
            self.exit()
            return

        if event.key == "k":
            await self.realtime_handler.toggle_recording()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "send-button":
            await self.realtime_handler.handle_text_input()


if __name__ == "__main__":
    app = RealtimeApp()
    app.run()