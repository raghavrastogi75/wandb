#!/usr/bin/env uv run
####################################################################
# Sample TUI app with a push to talk interface to the Realtime API #
# If you have `uv` installed and the `OPENAI_API_KEY`              #
# environment variable set, you can run this example with just     #
#                                                                  #
# `./examples/realtime/push_to_talk_app.py`                        #
####################################################################
#
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "textual",
#     "numpy",
#     "pyaudio",
#     "pydub",
#     "sounddevice",
#     "openai[realtime]",
#     "pandas",
#     "weave",    # Added Weave dependency
# ]
#
# [tool.uv.sources]
# openai = { path = "../../", editable = true }
# ///
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
        
        questions = [
            # --- Basic Info & Describe ---
            "Describe the 'recall' column.",
            "Give me info about the 'status' column.",
            "Summarize the 'accuracy' metric.",
            "What are the characteristics of the 'total tokens' column?",
            "Provide descriptive statistics for 'latency'.",
            "Show the data type and number of non-null values for 'model name'.",
            "Describe 'hallucination rate' using quartiles.",
            "What's the range (min and max) of the 'f1 score'?",
            "Info on 'scheduler type'.",
            "Describe 'learning rate'.",

            # --- Filtering ---
            "Find runs where 'status' is 'running'.",
            "Show runs with 'accuracy' below 0.5.",
            "List runs where 'model name' does not contain 'sft'.",
            "Which runs have 'latency' greater than 1,000,000 ms?",
            "Filter for runs where 'recall' is exactly 1.0.",
            "Show runs whose 'display name' ends with 'debug'.",
            "Find runs with 'precision' not between 0.4 and 0.6.",
            "List runs where 'epochs' is null or missing.",
            "Show runs where 'status' is 'success' AND 'accuracy' is below 0.6.",
            "Find runs where 'model name' is 'HHEM_2-1' OR 'latency' is less than 10000.",

            # --- Aggregation & Grouping ---
            "What is the total number of runs?",
            "Calculate the sum of 'total tokens' used across all runs.",
            "Find the maximum 'recall' score recorded.",
            "What is the minimum 'precision' observed?",
            "Group by 'status' and count the number of runs in each group.",
            "Calculate the average 'accuracy' for each 'scheduler type'.",
            "Find the median 'latency' for runs where 'model name' contains 'gpt'.",
            "What is the standard deviation of 'f1 score' for successful runs?",
            "For each 'model name', find the min and max 'latency'.",
            "Count the number of unique 'scheduler types'.",

            # --- Sorting & Ranking ---
            "Show the top 5 runs based on lowest 'hallucination rate'.",
            "List the bottom 3 runs according to 'f1 score'.",
            "Sort all runs by 'latency' in ascending order.",
            "Display runs ordered by 'accuracy' descending, then 'precision' descending.",
            "Show the top 10 runs sorted by 'recall' (highest first).",
            "List the runs sorted by 'display name' alphabetically.",
            "Find the 5 runs with the highest 'total tokens'.",
            "Sort by 'status' then by 'latency' ascending.",
            "Show the bottom 5 runs based on 'precision'.",
            "Order runs by 'f1 score' descending and show the top 4.",

            # --- Correlation & Relationships ---
            "What is the Pearson correlation between 'accuracy' and 'latency'?",
            "Calculate the Spearman correlation for 'hallucination rate' and 'f1 score'.",
            "Is 'warmup ratio' related to 'recall' score?",
            "Correlate 'epochs' with 'precision'.",
            "Show the correlation matrix for 'latency', 'precision', and 'recall'.",
            "What's the relationship between 'total tokens' and 'latency'?",
            "Calculate Kendall correlation between 'accuracy' and 'f1 score'.",
            "How does 'status' relate to average 'f1 score'?", # Requires grouping then comparison
            "Correlate 'learning rate' and 'accuracy'.",
            "What is the Spearman correlation between 'epochs' and 'latency'?",

            # --- Hyperparameters & Complex Queries ---
            "Find the hyperparameters ('learning rate', 'epochs') for the run with the median 'accuracy'.", # Needs median first, then filter/lookup
            "Which 'scheduler type' is associated with the overall highest average 'precision'?", # Needs group-by-agg then sort
            "Show the 'status' and 'latency' for the run with the minimum 'f1 score'.",
            "List the top 3 'model names' based on the number of runs.",
            "What is the average 'recall' for runs using the 'cosine' scheduler?",
            "Compare the minimum 'latency' of 'gpt-4o' vs 'gpt-4o-mini'.", # Requires filtering twice and comparing
            "Find runs where 'precision' is above average and 'recall' is below average.", # Needs averages first, then filter
            "Describe 'latency' for runs where 'model name' is 'SmolLM2-135M'.",
            "Which run had the highest 'f1 score' among those with 'status' running?",
            "Show the average 'precision', 'recall', and 'f1 score' grouped by 'model name', limited to the top 5 models by average F1.", # Group, aggregate, sort, limit
        ]

        
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