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
# ]
#
# [tool.uv.sources]
# openai = { path = "../../", editable = true }
# ///
from __future__ import annotations

import base64
import asyncio
import json
import uuid
import pandas as pd
from typing import Any, cast, Dict, Optional
from typing_extensions import override

from textual import events
from audio_util import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
from textual.app import App, ComposeResult
from textual.widgets import Button, Static, RichLog
from textual.reactive import reactive
from textual.containers import Container

from openai import AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

# Import data functions from the new module
from data_utils import load_dummy_dataframe, get_dataframe_info, query_dataframe


class SessionDisplay(Static):
    """A widget that shows the current session ID."""
    session_id = reactive("")

    @override
    def render(self) -> str:
        return f"Session ID: {self.session_id}" if self.session_id else "Connecting..."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio recording status."""
    is_recording = reactive(False)

    @override
    def render(self) -> str:
        status = (
            "🔴 Recording... (Press K to stop)" if self.is_recording else "⚪ Press K to start recording (Q to quit)"
        )
        return status


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
            height: 5;  /* Explicit height for input container */
            margin: 1 1;
            padding: 1 2;
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
            height: 82%;  /* Reduced to make room for session display */
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

    client: AsyncOpenAI
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    connection: AsyncRealtimeConnection | None
    session: Session | None
    connected: asyncio.Event
    dataframe: pd.DataFrame
    accumulated_transcript: str
    pending_function_calls: Dict[str, Dict[str, Any]]

    def __init__(self) -> None:
        super().__init__()
        self.connection = None
        self.session = None
        self.client = AsyncOpenAI()
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.dataframe = load_dummy_dataframe()
        self.accumulated_transcript = ""
        self.pending_function_calls = {}

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            yield SessionDisplay(id="session-display")
            yield AudioStatusIndicator(id="status-indicator")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        self.run_worker(self.handle_realtime_connection())
        self.run_worker(self.send_mic_audio())
        
        # Display initial dataframe information
        bottom_pane = self.query_one("#bottom-pane", RichLog)
        bottom_pane.write("Loaded sample DataFrame with the following columns:")
        bottom_pane.write(f"{', '.join(self.dataframe.columns)}")
        bottom_pane.write("\nSample data:")
        bottom_pane.write(str(self.dataframe.head(3)))
        bottom_pane.write("\nAsk questions about this data by pressing K and speaking.")

    async def handle_realtime_connection(self) -> None:
        async with self.client.beta.realtime.connect(model="gpt-4o-realtime-preview") as conn:
            self.connection = conn
            self.connected.set()

            # Configure session with data query function and server VAD
            await conn.session.update(session={
                "turn_detection": {"type": "server_vad"},
                "tools": [
                    {
                        "type": "function",
                        "name": "query_dataframe",
                        "description": "Query the DataFrame to retrieve information or perform operations.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query_type": {
                                    "type": "string",
                                    "enum": ["general", "filter", "aggregate"],
                                    "description": "Type of query to perform"
                                },
                                "column": {
                                    "type": "string",
                                    "description": "Column to query"
                                },
                                "value": {
                                    "type": ["string", "number", "boolean"],
                                    "description": "Value to filter by (for filter queries)"
                                },
                                "operator": {
                                    "type": "string",
                                    "enum": ["==", ">", "<", "contains"],
                                    "description": "Comparison operator (for filter queries)"
                                },
                                "function": {
                                    "type": "string",
                                    "enum": ["mean", "sum", "count"],
                                    "description": "Aggregation function (for aggregate queries)"
                                },
                                "group_by": {
                                    "type": "string",
                                    "description": "Column to group by (for aggregate queries)"
                                }
                            },
                            "required": ["query_type"]
                        }
                    }
                ],
                "tool_choice": "auto"
            })

            acc_items: dict[str, Any] = {}
            user_inputs: dict[str, str] = {}

            async for event in conn:
                bottom_pane = self.query_one("#bottom-pane", RichLog)

                if event.type == "session.created":
                    self.session = event.session
                    session_display = self.query_one(SessionDisplay)
                    assert event.session.id is not None
                    session_display.session_id = event.session.id
                    continue

                if event.type == "session.updated":
                    self.session = event.session
                    continue
                
                if event.type == "error":
                    # Fix: Access error message correctly through the error attribute
                    if hasattr(event, 'error') and hasattr(event.error, 'message'):
                        bottom_pane.write(f"\n[red]Error: {event.error.message}[/red]")
                    else:
                        bottom_pane.write(f"\n[red]Error occurred: {event}[/red]")
                    continue

                if event.type == "response.audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)
                    continue
                    
                if event.type == "input.speech_text.delta":
                    try:
                        text = user_inputs.get(event.item_id, "")
                        user_inputs[event.item_id] = text + event.delta
                    except KeyError:
                        user_inputs[event.item_id] = event.delta
                    
                    # Process completed speech when "is_final" is True
                    if event.is_final:
                        full_text = user_inputs.get(event.item_id, "")
                        bottom_pane.write(f"\nYou asked: {full_text}")
                
                # Function calling handling - fixed to work with actual event structure
                if event.type == "response.function_call_arguments.delta":
                    if not hasattr(event, 'call_id') or not event.call_id:
                        continue
                    
                    call_id = event.call_id
                    if call_id not in self.pending_function_calls:
                        self.pending_function_calls[call_id] = {
                            'arguments': event.delta,
                            # Don't try to get name from the event - we'll get it from response.done
                        }
                    else:
                        self.pending_function_calls[call_id]['arguments'] += event.delta
                
                if event.type == "response.done":
                    # Check if there's a function call in the output
                    if hasattr(event, 'response') and hasattr(event.response, 'output'):
                        for output_item in event.response.output:
                            if hasattr(output_item, 'type') and output_item.type == 'function_call':
                                call_id = output_item.call_id
                                func_name = output_item.name
                                
                                # Try to parse arguments either from the pending calls or directly from output_item
                                try:
                                    if call_id in self.pending_function_calls and 'arguments' in self.pending_function_calls[call_id]:
                                        arguments_str = self.pending_function_calls[call_id]['arguments']
                                        arguments = json.loads(arguments_str)
                                    else:
                                        arguments = json.loads(output_item.arguments)
                                    
                                    bottom_pane.write(f"\nProcessing query: {arguments}")
                                    
                                    # Execute the function
                                    if func_name == 'query_dataframe':
                                        result = query_dataframe(self.dataframe, arguments)
                                        
                                        # Send function results back to model
                                        await conn.send({
                                            "type": "conversation.item.create",
                                            "event_id": str(uuid.uuid4()),
                                            "item": {
                                                "type": "function_call_output",
                                                "call_id": call_id,
                                                "output": json.dumps(result)
                                            }
                                        })
                                        
                                        # Generate a response with the function output
                                        await conn.send({
                                            "type": "response.create",
                                            "event_id": str(uuid.uuid4())
                                        })
                                except json.JSONDecodeError:
                                    bottom_pane.write(f"\n[red]Error parsing function arguments: Invalid JSON[/red]")
                                except Exception as e:
                                    bottom_pane.write(f"\n[red]Error processing function call: {str(e)}[/red]")

                if event.type == "response.audio_transcript.delta":
                    try:
                        text = acc_items[event.item_id]
                    except KeyError:
                        acc_items[event.item_id] = event.delta
                    else:
                        acc_items[event.item_id] = text + event.delta

                    # Clear and update the entire content because RichLog otherwise treats each delta as a new line
                    bottom_pane.clear()
                    bottom_pane.write(acc_items[event.item_id])
                    continue

                if event.type == "input_audio_buffer.speech_started":
                    status_indicator = self.query_one(AudioStatusIndicator)
                    status_indicator.is_recording = True
                    continue

                if event.type == "input_audio_buffer.speech_stopped":
                    status_indicator = self.query_one(AudioStatusIndicator)
                    status_indicator.is_recording = False
                    continue

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        import sounddevice as sd  # type: ignore

        sent_audio = False

        device_info = sd.query_devices()
        print(device_info)

        read_size = int(SAMPLE_RATE * 0.02)

        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
        )
        stream.start()

        status_indicator = self.query_one(AudioStatusIndicator)

        try:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                await self.should_send_audio.wait()
                status_indicator.is_recording = True

                data, _ = stream.read(read_size)

                connection = await self._get_connection()
                if not sent_audio:
                    await connection.send({
                        "type": "response.cancel", 
                        "event_id": str(uuid.uuid4())
                    })
                    sent_audio = True

                await connection.input_audio_buffer.append(audio=base64.b64encode(cast(Any, data)).decode("utf-8"))

                await asyncio.sleep(0)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()

    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "enter":
            self.query_one(Button).press()
            return

        if event.key == "q":
            self.exit()
            return

        if event.key == "k":
            status_indicator = self.query_one(AudioStatusIndicator)
            if status_indicator.is_recording:
                self.should_send_audio.clear()
                status_indicator.is_recording = False

                if self.session and self.session.turn_detection is None:
                    # The default in the API is that the model will automatically detect when the user has
                    # stopped talking and then start responding itself.
                    #
                    # However if we're in manual `turn_detection` mode then we need to
                    # manually tell the model to commit the audio buffer and start responding.
                    conn = await self._get_connection()
                    await conn.input_audio_buffer.commit()
                    await conn.response.create()
            else:
                self.should_send_audio.set()
                status_indicator.is_recording = True
                
                # Clear any previous audio buffer when starting new recording
                conn = await self._get_connection()
                await conn.send({
                    "type": "input_audio_buffer.clear",
                    "event_id": str(uuid.uuid4())
                })


if __name__ == "__main__":
    app = RealtimeApp()
    app.run()