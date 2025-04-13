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
from textual.widgets import Button, Static, RichLog, Input
from textual.reactive import reactive
from textual.containers import Container, Horizontal

from openai import AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

# Import websockets for exception handling
import websockets.exceptions

# Import Weave for tracing
import weave
import datetime

# Import data functions from the new module
from data_utils import load_dummy_dataframe, get_dataframe_info, query_dataframe, get_tools_schema
# from data_utils_1 import load_dummy_dataframe, get_dataframe_info, query_dataframe, get_tools_schema

import wandb

@weave.op(call_display_name="log_conversation")
def log_conversation_turn(turn_type: str, content: str, input_type: str = "voice") -> None:
    """Log conversation turns for tracing with Weave.
    
    Args:
        turn_type: Type of turn ("user_query" or "assistant_response")
        content: The text content of the turn
        input_type: The type of input ("voice" or "text")
    """
    # Make sure wandb is initialized
    if wandb.run is None:
        wandb.init(project="realtime-voice-app", name=f"conversation-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
    # Log the conversation turn with the standard wandb log method
    wandb.log({
        "turn_type": turn_type,
        "input_type": input_type,
        "timestamp": datetime.datetime.now().isoformat(),
        "content_length": len(content),
        "content": content
    })
    
    # Add trace attributes for better querying
    weave.attributes({
        "turn_type": turn_type,
        "input_type": input_type,
        "content_length": len(content),
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    return content

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
            with Horizontal(id="input-container"):
                yield Input(placeholder="Type your message here...", id="text-input")
                yield Button("Send", id="send-button", variant="primary")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        # Initialize Weave with a project name
        weave.init("realtime-voice-app")

        if wandb.run is None:
            wandb.init(project="realtime-voice-app", name=f"session-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

        
        self.run_worker(self.handle_realtime_connection())
        self.run_worker(self.send_mic_audio())
        
        # Display initial dataframe information
        bottom_pane = self.query_one("#bottom-pane", RichLog)
        bottom_pane.write("Loaded sample DataFrame with the following columns:")
        bottom_pane.write(f"{', '.join(self.dataframe.columns)}")
        bottom_pane.write("\nSample data:")
        bottom_pane.write(str(self.dataframe.head(3)))
        bottom_pane.write("\nAsk questions about this data by pressing K and speaking.")

    @weave.op(call_display_name="realtime_connection")
    async def handle_realtime_connection(self) -> None:
        async with self.client.beta.realtime.connect(model="gpt-4o-realtime-preview") as conn:
            self.connection = conn
            self.connected.set()

            tools_schema = get_tools_schema()
            await conn.session.update(session={
                "turn_detection": {"type": "server_vad"},
                "tools": [tools_schema],
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
                        
                        # Log the user's query with Weave
                        log_conversation_turn("user_query", full_text, input_type="voice")
                    
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
                                        # Wrap the function call with Weave attributes for better tracing
                                        with weave.attributes({'function_call': func_name, 'arguments': arguments}):
                                            result = query_dataframe(self.dataframe, arguments)
                                            log_conversation_turn("function_result", json.dumps(result))
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

                    current_text = acc_items[event.item_id]
                    if len(current_text) > 10 and "." in current_text[-10:]:
                        log_conversation_turn("assistant_response", current_text)
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

    @weave.op(call_display_name="send_mic_audio")
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
        bottom_pane = self.query_one("#bottom-pane", RichLog)

        try:
            while True:
                try:
                    if stream.read_available < read_size:
                        await asyncio.sleep(0)
                        continue

                    await self.should_send_audio.wait()
                    status_indicator.is_recording = True

                    data, _ = stream.read(read_size)

                    try:
                        connection = await self._get_connection()
                        if not sent_audio:
                            try:
                                await connection.send({
                                    "type": "response.cancel", 
                                    "event_id": str(uuid.uuid4())
                                })
                                sent_audio = True
                            except websockets.exceptions.ConnectionClosedOK:
                                bottom_pane.write("\n[yellow]Connection closed. Attempting to reconnect...[/yellow]")
                                # Reset connection and wait for reconnection
                                self.connected.clear()
                                self.connection = None
                                self.run_worker(self.handle_realtime_connection())
                                await self.connected.wait()
                                connection = await self._get_connection()
                                # Try again after reconnection
                                await connection.send({
                                    "type": "response.cancel", 
                                    "event_id": str(uuid.uuid4())
                                })
                                sent_audio = True
                            except Exception as e:
                                bottom_pane.write(f"\n[red]Error canceling response: {str(e)}[/red]")

                        try:
                            await connection.input_audio_buffer.append(audio=base64.b64encode(cast(Any, data)).decode("utf-8"))
                        except websockets.exceptions.ConnectionClosedOK:
                            # Connection was closed - attempt to reconnect
                            bottom_pane.write("\n[yellow]Connection closed while sending audio. Reconnecting...[/yellow]")
                            self.connected.clear()
                            self.connection = None
                            self.run_worker(self.handle_realtime_connection())
                            await self.connected.wait()
                            # Don't immediately retry sending this chunk, wait for next loop iteration
                        except Exception as e:
                            bottom_pane.write(f"\n[red]Error sending audio data: {str(e)}[/red]")
                    except Exception as e:
                        bottom_pane.write(f"\n[red]Connection error: {str(e)}[/red]")
                        await asyncio.sleep(1)  # Brief pause before retry

                    await asyncio.sleep(0)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    bottom_pane.write(f"\n[red]Error in audio processing loop: {str(e)}[/red]")
                    await asyncio.sleep(1)  # Brief pause before continuing
        except KeyboardInterrupt:
            pass
        except asyncio.CancelledError:
            # Handle task cancellation gracefully
            bottom_pane.write("\n[yellow]Audio recording task cancelled[/yellow]")
        except Exception as e:
            bottom_pane.write(f"\n[red]Fatal error in audio recording: {str(e)}[/red]")
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception as e:
                bottom_pane.write(f"\n[red]Error closing audio stream: {str(e)}[/red]")

    @weave.op(call_display_name="handle_key_press")
    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "enter":
            # Only process text input on Enter key if the input field has focus
            input_widget = self.query_one("#text-input", Input)
            if self.focused is input_widget:
                await self.handle_text_input()
                return

        if event.key == "q":
            self.exit()
            return

        if event.key == "k":
            status_indicator = self.query_one(AudioStatusIndicator)
            bottom_pane = self.query_one("#bottom-pane", RichLog)
            
            if status_indicator.is_recording:
                self.should_send_audio.clear()
                status_indicator.is_recording = False

                if self.session and self.session.turn_detection is None:
                    try:
                        # The default in the API is that the model will automatically detect when the user has
                        # stopped talking and then start responding itself.
                        #
                        # However if we're in manual `turn_detection` mode then we need to
                        # manually tell the model to commit the audio buffer and start responding.
                        conn = await self._get_connection()
                        await conn.input_audio_buffer.commit()
                        await conn.response.create()
                    except websockets.exceptions.ConnectionClosedOK:
                        bottom_pane.write("\n[yellow]Connection closed. Attempting to reconnect...[/yellow]")
                        # Restart the connection worker
                        self.connected.clear()
                        self.connection = None
                        self.run_worker(self.handle_realtime_connection())
                        await self.connected.wait()
                    except Exception as e:
                        bottom_pane.write(f"\n[red]Error stopping recording: {str(e)}[/red]")
            else:
                self.should_send_audio.set()
                status_indicator.is_recording = True
                
                # Clear any previous audio buffer when starting new recording
                try:
                    conn = await self._get_connection()
                    await conn.send({
                        "type": "input_audio_buffer.clear",
                        "event_id": str(uuid.uuid4())
                    })
                except websockets.exceptions.ConnectionClosedOK:
                    bottom_pane.write("\n[yellow]Connection closed. Attempting to reconnect...[/yellow]")
                    # Restart the connection worker
                    self.connected.clear()
                    self.connection = None
                    self.run_worker(self.handle_realtime_connection())
                    await self.connected.wait()
                    
                    # Try again after reconnection
                    try:
                        conn = await self._get_connection()
                        await conn.send({
                            "type": "input_audio_buffer.clear",
                            "event_id": str(uuid.uuid4())
                        })
                    except Exception as e:
                        bottom_pane.write(f"\n[red]Error starting recording: {str(e)}[/red]")
                        status_indicator.is_recording = False
                        self.should_send_audio.clear()
                except Exception as e:
                    bottom_pane.write(f"\n[red]Error starting recording: {str(e)}[/red]")
                    status_indicator.is_recording = False
                    self.should_send_audio.clear()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "send-button":
            await self.handle_text_input()
    
    @weave.op(call_display_name="handle_text_input")
    async def handle_text_input(self) -> None:
        """Process text input from the input field."""
        input_widget = self.query_one("#text-input", Input)
        text = input_widget.value.strip()
        log_conversation_turn("user_query", text, input_type="text")
        if not text:
            return
            
        # Clear the input field
        input_widget.value = ""
        
        # Display the user's input in the bottom pane
        bottom_pane = self.query_one("#bottom-pane", RichLog)
        bottom_pane.write(f"\nYou asked (text): {text}")
        
        try:
            # Check if we're already connected before trying to log or send
            if not self.connected.is_set():
                bottom_pane.write("\n[yellow]Connection not ready. Waiting for connection to establish...[/yellow]")
                await self.connected.wait()
                
            # Log the user's text input with Weave
            log_conversation_turn("user_query", text, input_type="text")
            
            # Send the text to the model as a conversation item
            connection = await self._get_connection()
            
            # Try to cancel any ongoing response
            try:
                await connection.send({
                    "type": "response.cancel",
                    "event_id": str(uuid.uuid4())
                })
            except websockets.exceptions.ConnectionClosedOK:
                # Connection was already closed, attempt to reconnect
                bottom_pane.write("\n[yellow]Connection closed. Attempting to reconnect...[/yellow]")
                # Restart the connection worker
                self.connected.clear()
                self.connection = None
                self.run_worker(self.handle_realtime_connection())
                await self.connected.wait()
                connection = await self._get_connection()
            
            # Create a new conversation item with the text - with corrected content type
            try:
                await connection.send({
                    "type": "conversation.item.create",
                    "event_id": str(uuid.uuid4()),
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}]
                    }
                })

                log_conversation_turn("user_query", text, input_type="text")
                
                # Request a response from the model
                await connection.send({
                    "type": "response.create",
                    "event_id": str(uuid.uuid4())
                })
            except websockets.exceptions.ConnectionClosedError as e:
                bottom_pane.write(f"\n[red]Connection error: {str(e)}[/red]")
            except Exception as e:
                bottom_pane.write(f"\n[red]Error sending message: {str(e)}[/red]")
        except Exception as e:
            bottom_pane.write(f"\n[red]Error handling text input: {str(e)}[/red]")


if __name__ == "__main__":
    app = RealtimeApp()
    app.run()