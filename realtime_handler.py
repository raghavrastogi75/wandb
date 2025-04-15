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

from logging_utils import log_conversation_turn
from data_utils import get_tools_schema, query_dataframe

from openai import AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

class RealtimeHandler:
    def __init__(self, app, client, audio_player, dataframe, connected, pending_function_calls):
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
    
    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection
    
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
                
                # Function calling handling
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
                
                if event.type == "response.done":
                    # Check if there's a function call in the output
                    if hasattr(event, 'response') and hasattr(event.response, 'output'):
                        for output_item in event.response.output:
                            if hasattr(output_item, 'type') and output_item.type == 'function_call':
                                call_id = output_item.call_id
                                func_name = output_item.name
                                
                                # Try to parse arguments
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
                                            log_conversation_turn("arguments", arguments)
                                            log_conversation_turn("result", result)
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

                    # Clear and update the entire content
                    bottom_pane.clear()
                    bottom_pane.write(acc_items[event.item_id])

                    current_text = acc_items[event.item_id]
                    if len(current_text) > 10 and "." in current_text[-10:]:
                        log_conversation_turn("assistant_response", current_text)
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
                                # Reset connection and wait for reconnection
                                self.connected.clear()
                                self.connection = None
                                self.app.run_worker(self.handle_realtime_connection())
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
                            self.app.run_worker(self.handle_realtime_connection())
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

    async def toggle_recording(self) -> None:
        """Toggle recording on/off when user presses K."""
        status_indicator = self.app.query_one("#status-indicator")
        bottom_pane = self.app.query_one("#bottom-pane")
        
        if status_indicator.is_recording:
            self.should_send_audio.clear()
            status_indicator.is_recording = False

            if self.session and self.session.turn_detection is None:
                try:
                    # Manually commit the audio buffer in manual turn_detection mode
                    conn = await self._get_connection()
                    await conn.input_audio_buffer.commit()
                    await conn.response.create()
                except websockets.exceptions.ConnectionClosedOK:
                    bottom_pane.write("\n[yellow]Connection closed. Attempting to reconnect...[/yellow]")
                    # Restart the connection worker
                    self.connected.clear()
                    self.connection = None
                    self.app.run_worker(self.handle_realtime_connection())
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
                self.app.run_worker(self.handle_realtime_connection())
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
                
    @weave.op(call_display_name="handle_text_input")
    async def handle_text_input(self) -> None:
        """Process text input from the input field."""
        input_widget = self.app.query_one("#text-input")
        text = input_widget.value.strip()
        
        if not text:
            return
            
        # Clear the input field
        input_widget.value = ""
        
        # Display the user's input in the bottom pane
        bottom_pane = self.app.query_one("#bottom-pane")
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
                self.app.run_worker(self.handle_realtime_connection())
                await self.connected.wait()
                connection = await self._get_connection()
            
            # Create a new conversation item with the text
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
