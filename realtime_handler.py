"""
Handles the real-time interaction logic for a Textual application.

Manages the connection to a real-time service (like OpenAI's), processes
incoming events (audio, text, function calls, errors), handles audio input/output,
processes text input, executes function calls (specifically for querying a DataFrame),
and updates the application's UI accordingly. Integrates with Weave for logging.
"""
import asyncio
import base64
import json
import uuid
import weave
import sounddevice as sd
from typing import Any, Dict, cast
import websockets.exceptions

# Import directly from audio_util to maintain original code structure
from audio_util import CHANNELS, SAMPLE_RATE

from logging_utils import log_input_question, log_llm_queries, log_llm_query_result, log_llm_outputs
from data_utils import get_tools_schema, query_dataframe

from openai import AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

instructions = "Only answer the questions that are related to querying the dataset using the tool provided " \
"If you get any other type of question that is not related to a query, politely tell the user to ask the questions related to dataset only"


class RealtimeHandler:
    """
    Manages the real-time WebSocket connection, audio streaming, text input,
    function call execution, and UI updates for the application.
    """
    def __init__(self, app, client, audio_player, dataframe, connected, pending_function_calls):
        """
        Initializes the RealtimeHandler.

        Args:
            app: The Textual application instance.
            client: The AsyncOpenAI client instance.
            audio_player: An instance of AudioPlayerAsync for playing back audio responses.
            dataframe: The pandas DataFrame to be queried by the function call tool.
            connected: An asyncio.Event signaling the connection status.
            pending_function_calls: A dictionary to store accumulating function call arguments.
        """
        self.app = app
        self.client = client
        self.audio_player = audio_player
        self.dataframe = dataframe
        self.connected = connected
        self.pending_function_calls = pending_function_calls
        self.connection = None
        self.session = None
        self.last_audio_item_id = None
        self.should_send_audio = app.should_send_audio
        self.logged_responses = set()  # Keep track of responses we've already logged
        self.current_response_id = None  # Track current response being processed

    async def _get_connection(self) -> AsyncRealtimeConnection:
        """
        Waits for the connection to be established and returns it.

        Ensures that operations requiring a connection wait until it's ready.

        Returns:
            The active AsyncRealtimeConnection instance.

        Raises:
            AssertionError: If the connection is None after the event is set (should not happen).
        """
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def handle_realtime_connection(self) -> None:
        """
        Establishes and manages the real-time WebSocket connection.

        Connects to the real-time service, configures the session (VAD, tools, instructions),
        and enters an event loop to process incoming messages like session updates,
        errors, audio deltas, speech-to-text deltas, function call arguments,
        and response lifecycle events. Updates UI elements and triggers function calls.
        """
        async with self.client.beta.realtime.connect(model="gpt-4o-realtime-preview") as conn:
            self.connection = conn
            self.connected.set()

            tools_schema = get_tools_schema()
            await conn.session.update(session={
                "turn_detection": {"type": "server_vad"},
                "tools": [tools_schema],
                "tool_choice": "auto",
                "instructions": instructions,
                "temperature": 0.6
            })

            acc_items: dict[str, Any] = {}
            user_inputs: dict[str, str] = {}

            async for event in conn:
                bottom_pane = self.app.query_one("#bottom-pane")

                if event.type == "session.created":
                    self.session = event.session
                    session_display = self.app.query_one("#session-display")
                    assert event.session.id is not None
                    session_display.session_id = event.session.id
                    continue

                if event.type == "session.updated":
                    self.session = event.session
                    continue

                if event.type == "error":
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

                    if event.is_final:
                        full_text = user_inputs.get(event.item_id, "")
                        bottom_pane.write(f"\nYou asked: {full_text}")
                        log_input_question("user_query", full_text, input_type="voice")

                if event.type == "response.function_call_arguments.delta":
                    if not hasattr(event, 'call_id') or not event.call_id:
                        continue

                    call_id = event.call_id
                    if call_id not in self.pending_function_calls:
                        self.pending_function_calls[call_id] = {
                            'arguments': event.delta,
                        }
                    else:
                        self.pending_function_calls[call_id]['arguments'] += event.delta

                if event.type == "response.created":
                    acc_items.clear()
                    self.current_response_id = None
                    if hasattr(event, 'response') and hasattr(event.response, 'id'):
                        self.current_response_id = event.response.id
                    continue

                if event.type == "response.done":
                    if self.current_response_id and self.current_response_id not in self.logged_responses:
                        current_text = ""
                        for item_id, text in acc_items.items():
                            if len(text) > 0 and (not current_text or len(text) > len(current_text)):
                                current_text = text

                        if current_text:
                            log_llm_outputs("assistant_response", current_text)
                            self.logged_responses.add(self.current_response_id)

                    if hasattr(event, 'response') and hasattr(event.response, 'output'):
                        for output_item in event.response.output:
                            if hasattr(output_item, 'type') and output_item.type == 'function_call':
                                call_id = output_item.call_id
                                func_name = output_item.name

                                try:
                                    if call_id in self.pending_function_calls and 'arguments' in self.pending_function_calls[call_id]:
                                        arguments_str = self.pending_function_calls[call_id]['arguments']
                                        arguments = json.loads(arguments_str)
                                    else:
                                        arguments = json.loads(output_item.arguments)

                                    bottom_pane.write(f"\nProcessing query: {arguments}")

                                    if func_name == 'query_dataframe':
                                        with weave.attributes({'function_call': func_name, 'arguments': arguments}):
                                            result = query_dataframe(self.dataframe, arguments)
                                            log_llm_queries("arguments", arguments)
                                        await conn.send({
                                            "type": "conversation.item.create",
                                            "event_id": str(uuid.uuid4()),
                                            "item": {
                                                "type": "function_call_output",
                                                "call_id": call_id,
                                                "output": json.dumps(result)
                                            }
                                        })

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

                    bottom_pane.clear()
                    bottom_pane.write(acc_items[event.item_id])
                    continue

                if event.type == "input_audio_buffer.speech_started":
                    status_indicator = self.app.query_one("#status-indicator")
                    status_indicator.is_recording = True
                    continue

                if event.type == "input_audio_buffer.speech_stopped":
                    status_indicator = self.app.query_one("#status-indicator")
                    status_indicator.is_recording = False
                    continue

    @weave.op(call_display_name="send_mic_audio")
    async def send_mic_audio(self) -> None:
        """
        Captures audio from the microphone and sends it over the connection.

        Runs in a loop, reading audio chunks from the default input device using
        sounddevice. Waits for the `should_send_audio` event before capturing and
        sending. Encodes audio data in base64 and sends it via the active connection's
        `input_audio_buffer.append` method. Handles connection closures and attempts
        reconnection. Updates UI status indicator.
        """
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

        status_indicator = self.app.query_one("#status-indicator")
        bottom_pane = self.app.query_one("#bottom-pane")

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
                                self.connected.clear()
                                self.connection = None
                                self.app.run_worker(self.handle_realtime_connection())
                                await self.connected.wait()
                                connection = await self._get_connection()
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
                            bottom_pane.write("\n[yellow]Connection closed while sending audio. Reconnecting...[/yellow]")
                            self.connected.clear()
                            self.connection = None
                            self.app.run_worker(self.handle_realtime_connection())
                            await self.connected.wait()
                        except Exception as e:
                            bottom_pane.write(f"\n[red]Error sending audio data: {str(e)}[/red]")
                    except Exception as e:
                        bottom_pane.write(f"\n[red]Connection error: {str(e)}[/red]")
                        await asyncio.sleep(1)

                    await asyncio.sleep(0)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    bottom_pane.write(f"\n[red]Error in audio processing loop: {str(e)}[/red]")
                    await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        except asyncio.CancelledError:
            bottom_pane.write("\n[yellow]Audio recording task cancelled[/yellow]")
        except Exception as e:
            bottom_pane.write(f"\n[red]Fatal error in audio recording: {str(e)}[/red]")
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception as e:
                bottom_pane.write(f"\n[red]Error closing audio stream: {str(e)}[/red]")

    async def toggle_recording(self) -> None:
        """
        Handles the user action to toggle audio recording on or off.

        Updates the UI status indicator. Sets or clears the `should_send_audio` event
        to control the `send_mic_audio` loop. If stopping recording in manual
        turn detection mode, commits the audio buffer. If starting recording,
        clears any previous audio buffer on the server. Handles connection errors
        and attempts reconnection if necessary.
        """
        status_indicator = self.app.query_one("#status-indicator")
        bottom_pane = self.app.query_one("#bottom-pane")

        if status_indicator.is_recording:
            self.should_send_audio.clear()
            status_indicator.is_recording = False

            if self.session and self.session.turn_detection is None:
                try:
                    conn = await self._get_connection()
                    await conn.input_audio_buffer.commit()
                    await conn.response.create()
                except websockets.exceptions.ConnectionClosedOK:
                    bottom_pane.write("\n[yellow]Connection closed. Attempting to reconnect...[/yellow]")
                    self.connected.clear()
                    self.connection = None
                    self.app.run_worker(self.handle_realtime_connection())
                    await self.connected.wait()
                except Exception as e:
                    bottom_pane.write(f"\n[red]Error stopping recording: {str(e)}[/red]")
        else:
            self.should_send_audio.set()
            status_indicator.is_recording = True

            try:
                conn = await self._get_connection()
                await conn.send({
                    "type": "input_audio_buffer.clear",
                    "event_id": str(uuid.uuid4())
                })
            except websockets.exceptions.ConnectionClosedOK:
                bottom_pane.write("\n[yellow]Connection closed. Attempting to reconnect...[/yellow]")
                self.connected.clear()
                self.connection = None
                self.app.run_worker(self.handle_realtime_connection())
                await self.connected.wait()

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

    @weave.op(call_display_name="handle_text_input")
    async def handle_text_input(self) -> None:
        """
        Processes text entered into the application's text input field.

        Retrieves the text, clears the input field, displays the input in the UI,
        logs the input using Weave, cancels any ongoing response, sends the text
        as a user message via the active connection, and requests a new response
        from the model. Handles connection readiness and potential errors during sending.
        """
        input_widget = self.app.query_one("#text-input")
        text = input_widget.value.strip()

        if not text:
            return

        acc_items = {}

        input_widget.value = ""

        bottom_pane = self.app.query_one("#bottom-pane")
        bottom_pane.write(f"\nYou asked (text): {text}")

        try:
            if not self.connected.is_set():
                bottom_pane.write("\n[yellow]Connection not ready. Waiting for connection to establish...[/yellow]")
                await self.connected.wait()

            log_input_question("user_query", text, input_type="text")

            connection = await self._get_connection()

            try:
                await connection.send({
                    "type": "response.cancel",
                    "event_id": str(uuid.uuid4())
                })
            except websockets.exceptions.ConnectionClosedOK:
                bottom_pane.write("\n[yellow]Connection closed. Attempting to reconnect...[/yellow]")
                self.connected.clear()
                self.connection = None
                self.app.run_worker(self.handle_realtime_connection())
                await self.connected.wait()
                connection = await self._get_connection()

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
