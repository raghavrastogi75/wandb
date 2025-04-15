from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import base64
import asyncio
import json
import uuid
import pandas as pd
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import io
from pydub import AudioSegment

# Import components from the original speech app
from openai import AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

import websockets.exceptions
import weave
import wandb
import datetime

# Import data functions
from data_utils import load_dummy_dataframe, get_dataframe_info, query_dataframe, get_tools_schema

# Initialize FastAPI app
app = FastAPI(title="Voice Assistant Web App")
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set up templates and static files
templates_dir = Path(current_dir) / "templates"
static_dir = Path(current_dir) / "static"

# Create directories if they don't exist
templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory=str(templates_dir))

# Set up static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global variables
client = AsyncOpenAI()
dataframe = load_dummy_dataframe()
active_connections: Dict[str, WebSocket] = {}
connection_sessions: Dict[str, Dict[str, Any]] = {}
pending_function_calls: Dict[str, Dict[str, Any]] = {}

@weave.op(call_display_name="log_conversation")
def log_conversation_turn(turn_type: str, content: str, input_type: str = "voice") -> None:
    """Log conversation turns for tracing with Weave."""
    # Make sure wandb is initialized
    if wandb.run is None:
        wandb.init(project="realtime-voice-webapp", name=f"conversation-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
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

@app.on_event("startup")
async def startup_event():
    # Initialize Weave with a project name
    weave.init("realtime-voice-webapp")
    
    if wandb.run is None:
        wandb.init(project="realtime-voice-webapp", name=f"session-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/text-input/")
async def process_text_input(message: str = Form(...)):
    """Process text input from the user."""
    try:
        # Log the text input
        log_conversation_turn("user_query", message, input_type="text")
        
        # Connect to OpenAI
        async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as conn:
            tools_schema = get_tools_schema()
            await conn.session.update(session={
                "tools": [tools_schema],
                "tool_choice": "auto"
            })
            
            # Send the text message
            await conn.send({
                "type": "conversation.item.create",
                "event_id": str(uuid.uuid4()),
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": message}]
                }
            })
            
            # Request a response
            await conn.send({
                "type": "response.create",
                "event_id": str(uuid.uuid4())
            })
            
            # Process the response
            response_text = ""
            function_result = None
            
            async for event in conn:
                if event.type == "response.audio_transcript.delta":
                    response_text += event.delta
                
                elif event.type == "response.function_call_arguments.delta":
                    if not hasattr(event, 'call_id') or not event.call_id:
                        continue
                    
                    call_id = event.call_id
                    if call_id not in pending_function_calls:
                        pending_function_calls[call_id] = {'arguments': event.delta}
                    else:
                        pending_function_calls[call_id]['arguments'] += event.delta
                
                elif event.type == "response.done":
                    # Check if there's a function call in the output
                    if hasattr(event, 'response') and hasattr(event.response, 'output'):
                        for output_item in event.response.output:
                            if hasattr(output_item, 'type') and output_item.type == 'function_call':
                                call_id = output_item.call_id
                                func_name = output_item.name
                                
                                try:
                                    if call_id in pending_function_calls and 'arguments' in pending_function_calls[call_id]:
                                        arguments_str = pending_function_calls[call_id]['arguments']
                                        arguments = json.loads(arguments_str)
                                    else:
                                        arguments = json.loads(output_item.arguments)
                                    
                                    # Execute the function
                                    if func_name == 'query_dataframe':
                                        with weave.attributes({'function_call': func_name, 'arguments': arguments}):
                                            result = query_dataframe(dataframe, arguments)
                                            log_conversation_turn("arguments", str(arguments))
                                            log_conversation_turn("result", str(result))
                                            function_result = result
                                        
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
                                except Exception as e:
                                    return JSONResponse(content={"error": f"Function call error: {str(e)}"}, status_code=500)
                    else:
                        # If we're done and there's no function call, we can break out
                        break
            
            # Log the assistant's response
            log_conversation_turn("assistant_response", response_text)
            
            return {
                "response": response_text,
                "function_result": function_result
            }
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handle WebSocket connections for audio streaming."""
    await websocket.accept()
    active_connections[client_id] = websocket
    
    # Initialize connection data, including the audio buffer
    connection_sessions[client_id] = {
        "openai_connection": None,
        "session_id": None,
        "recording": False,
        "text_accumulator": {},
        "audio_buffer": []  # Initialize buffer for audio chunks
    }
    
    try:
        # Connect to OpenAI Realtime API
        async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as conn:
            # Store the connection
            connection_sessions[client_id]["openai_connection"] = conn
            
            # Configure the session
            tools_schema = get_tools_schema()
            await conn.session.update(session={
                "turn_detection": {"type": "server_vad"},
                "tools": [tools_schema],
                "tool_choice": "auto"
            })
            
            # Forward events from OpenAI to the web client
            openai_task = asyncio.create_task(handle_openai_events(conn, websocket, client_id))
            
            # Process audio from the web client
            client_task = asyncio.create_task(handle_client_messages(websocket, conn, client_id))
            
            # Wait for either task to finish
            await asyncio.gather(openai_task, client_task)
    
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
    except Exception as e:
        print(f"Error in WebSocket connection: {str(e)}")
    finally:
        # Clean up
        if client_id in active_connections:
            del active_connections[client_id]
        if client_id in connection_sessions:
            # Ensure buffer is cleared on disconnect if needed
            if 'audio_buffer' in connection_sessions[client_id]:
                connection_sessions[client_id]['audio_buffer'].clear()
            del connection_sessions[client_id]

async def handle_openai_events(conn, websocket, client_id):
    """Process events from OpenAI and send them to the web client."""
    acc_items = {}
    user_inputs = {}
    
    async for event in conn:
        try:
            # Session events
            if event.type == "session.created":
                session_id = event.session.id
                connection_sessions[client_id]["session_id"] = session_id
                await websocket.send_json({"type": "session", "session_id": session_id})
            
            # Audio response
            elif event.type == "response.audio.delta":
                audio_data = base64.b64decode(event.delta)
                await websocket.send_bytes(audio_data)
            
            # Text transcript of user speech
            elif event.type == "input.speech_text.delta":
                text = user_inputs.get(event.item_id, "")
                user_inputs[event.item_id] = text + event.delta
                
                if event.is_final:
                    full_text = user_inputs.get(event.item_id, "")
                    log_conversation_turn("user_query", full_text, input_type="voice")
                    await websocket.send_json({"type": "speech_text", "text": full_text})
            
            # Model's text response
            elif event.type == "response.audio_transcript.delta":
                try:
                    text = acc_items.get(event.item_id, "")
                    acc_items[event.item_id] = text + event.delta
                    
                    current_text = acc_items[event.item_id]
                    await websocket.send_json({"type": "transcript", "text": current_text})
                    
                    if len(current_text) > 10 and "." in current_text[-10:]:
                        log_conversation_turn("assistant_response", current_text)
                except Exception as e:
                    print(f"Error processing transcript: {str(e)}")
            
            # Function calling
            elif event.type == "response.function_call_arguments.delta":
                if not hasattr(event, 'call_id') or not event.call_id:
                    continue
                
                call_id = event.call_id
                if call_id not in pending_function_calls:
                    pending_function_calls[call_id] = {'arguments': event.delta}
                else:
                    pending_function_calls[call_id]['arguments'] += event.delta
                
                await websocket.send_json({"type": "function_args_progress", "data": event.delta})
            
            elif event.type == "response.done":
                # Check if there's a function call in the output
                if hasattr(event, 'response') and hasattr(event.response, 'output'):
                    for output_item in event.response.output:
                        if hasattr(output_item, 'type') and output_item.type == 'function_call':
                            call_id = output_item.call_id
                            func_name = output_item.name
                            
                            try:
                                if call_id in pending_function_calls and 'arguments' in pending_function_calls[call_id]:
                                    arguments_str = pending_function_calls[call_id]['arguments']
                                    arguments = json.loads(arguments_str)
                                else:
                                    arguments = json.loads(output_item.arguments)
                                
                                # Execute the function
                                if func_name == 'query_dataframe':
                                    with weave.attributes({'function_call': func_name, 'arguments': arguments}):
                                        result = query_dataframe(dataframe, arguments)
                                        log_conversation_turn("arguments", str(arguments))
                                        log_conversation_turn("result", str(result))
                                    
                                    # Send result to the client
                                    await websocket.send_json({"type": "function_result", "result": result})
                                    
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
                            except Exception as e:
                                await websocket.send_json({"type": "error", "message": f"Function call error: {str(e)}"})
                
                # Send done event to client
                await websocket.send_json({"type": "done"})
            
            # Speech detection events
            elif event.type == "input_audio_buffer.speech_started":
                connection_sessions[client_id]["recording"] = True
                await websocket.send_json({"type": "speech_status", "recording": True})
            
            elif event.type == "input_audio_buffer.speech_stopped":
                connection_sessions[client_id]["recording"] = False
                await websocket.send_json({"type": "speech_status", "recording": False})
            
            # Error events
            elif event.type == "error":
                error_message = getattr(event.error, 'message', str(event)) if hasattr(event, 'error') else str(event)
                await websocket.send_json({"type": "error", "message": error_message})
        
        except Exception as e:
            print(f"Error processing OpenAI event: {str(e)}")
            await websocket.send_json({"type": "error", "message": f"Server error: {str(e)}"})

async def handle_client_messages(websocket, conn, client_id):
    """Process messages from the web client and send them to OpenAI."""
    try:
        connection_active = True
        while connection_active:
            try:
                message = await websocket.receive()

                if message["type"] == "websocket.receive":
                    if "text" in message:
                        # Process text message (JSON commands or audio chunks)
                        data = json.loads(message["text"])
                        
                        # Check if it's an audio chunk message
                        if data.get("type") == "audio_chunk":
                            received_b64 = data.get("data")
                            if received_b64 and client_id in connection_sessions:
                                try:
                                    # 1. Decode Base64 received from client
                                    encoded_audio_bytes = base64.b64decode(received_b64)
                                    # 2. Append to buffer (DO NOT PROCESS YET)
                                    connection_sessions[client_id]['audio_buffer'].append(encoded_audio_bytes)
                                except KeyError:
                                    print(f"Client session {client_id} not found for buffering audio.")
                                except Exception as e:
                                    print(f"Error buffering audio chunk for {client_id}: {str(e)}")
                            continue # Skip further command processing for audio chunks

                        # Process regular commands
                        command = data.get("command")
                        
                        if command == "start_recording":
                            try:
                                # Clear previous buffer on new recording start
                                if client_id in connection_sessions:
                                    connection_sessions[client_id]['audio_buffer'].clear()
                                
                                await conn.send({
                                    "type": "input_audio_buffer.clear",
                                    "event_id": str(uuid.uuid4())
                                })
                                connection_sessions[client_id]["recording"] = True
                                await websocket.send_json({"type": "status", "message": "Recording started"})
                            except Exception as e:
                                await websocket.send_json({"type": "error", "message": f"Start recording error: {str(e)}"})
                        
                        elif command == "stop_recording":
                            try:
                                connection_sessions[client_id]["recording"] = False
                                await websocket.send_json({"type": "status", "message": "Recording stopped by user. Processing audio..."})

                                # --- Process accumulated audio ---
                                if client_id in connection_sessions and connection_sessions[client_id]['audio_buffer']:
                                    audio_buffer = connection_sessions[client_id]['audio_buffer']
                                    combined_audio_bytes = b"".join(audio_buffer)
                                    audio_buffer.clear() # Clear buffer after getting data

                                    if combined_audio_bytes:
                                        print(f"Processing {len(combined_audio_bytes)} bytes of accumulated audio for {client_id}")
                                        try:
                                            # Use pydub to load the combined audio bytes
                                            audio_segment = AudioSegment.from_file(
                                                io.BytesIO(combined_audio_bytes), 
                                                format="webm" 
                                            )
                                            
                                            # Convert to desired PCM format
                                            audio_segment = audio_segment.set_frame_rate(16000) 
                                            audio_segment = audio_segment.set_channels(1)
                                            audio_segment = audio_segment.set_sample_width(2) 

                                            raw_pcm_bytes = audio_segment.raw_data
                                            
                                            # Encode PCM to Base64
                                            pcm_b64 = base64.b64encode(raw_pcm_bytes).decode("utf-8")

                                            # Send the final PCM Base64 data to OpenAI
                                            if hasattr(conn, '_ws') and conn._ws and not conn._ws.closed:
                                                try:
                                                    await conn.input_audio_buffer.append(audio=pcm_b64)
                                                    print(f"Sent processed audio ({len(raw_pcm_bytes)} PCM bytes) to OpenAI for {client_id}")
                                                    await websocket.send_json({"type": "status", "message": "Audio processed and sent."})
                                                except websockets.exceptions.ConnectionClosedOK:
                                                    print(f"OpenAI connection closed cleanly for client {client_id} during processed audio send.")
                                                    await websocket.send_json({"type": "error", "message": "Connection closed while sending audio."})
                                                except Exception as send_err:
                                                    print(f"Error sending processed audio for {client_id}: {send_err}")
                                                    await websocket.send_json({"type": "error", "message": f"Error sending audio: {send_err}"})
                                            else:
                                                print(f"OpenAI connection closed for client {client_id} before sending processed audio.")
                                                await websocket.send_json({"type": "error", "message": "Connection timed out or closed during audio processing. Please try again."})

                                        except FileNotFoundError as fnf_error:
                                            if fnf_error.winerror == 2:
                                                print(f"FFmpeg not found error during final processing: {str(fnf_error)}. Check PATH.")
                                                await websocket.send_json({"type": "error", "message": "Server audio processing error: FFmpeg not found."})
                                            else:
                                                print(f"File system error during final processing: {str(fnf_error)}")
                                        except Exception as e:
                                            ffmpeg_output = getattr(e, 'stderr', None) or getattr(e, 'stdout', None)
                                            error_msg = f"Error processing/converting accumulated audio for OpenAI: {str(e)}"
                                            if ffmpeg_output:
                                                try:
                                                    ffmpeg_output_str = ffmpeg_output.decode('utf-8', errors='ignore')
                                                    error_msg += f"\n\nOutput from ffmpeg/avlib:\n\n{ffmpeg_output_str}"
                                                except Exception: pass
                                            print(error_msg)
                                            await websocket.send_json({"type": "error", "message": f"Audio processing error: {str(e)}"})
                                    else:
                                        print(f"No audio data in buffer for {client_id} on stop_recording.")
                                else:
                                    print(f"Client session {client_id} not found or buffer empty on stop_recording.")
                                # --- End processing ---

                            except websockets.exceptions.ConnectionClosedOK:
                                print(f"OpenAI connection closed cleanly for client {client_id} during stop recording.")
                            except Exception as e:
                                await websocket.send_json({"type": "error", "message": f"Stop recording error: {str(e)}"})
                        
                        elif command == "send_text":
                            text = data.get("text", "").strip()
                            if text:
                                try:
                                    log_conversation_turn("user_query", text, input_type="text")
                                    try:
                                        await conn.send({
                                            "type": "response.cancel",
                                            "event_id": str(uuid.uuid4())
                                        })
                                    except websockets.exceptions.ConnectionClosedOK:
                                        await websocket.send_json({
                                            "type": "error", 
                                            "message": "OpenAI connection closed. Please try sending again."
                                        })
                                        continue
                                    except Exception:
                                        pass
                                    
                                    await conn.send({
                                        "type": "conversation.item.create",
                                        "event_id": str(uuid.uuid4()),
                                        "item": {
                                            "type": "message",
                                            "role": "user",
                                            "content": [{"type": "input_text", "text": text}]
                                        }
                                    })
                                    
                                    await conn.send({
                                        "type": "response.create",
                                        "event_id": str(uuid.uuid4())
                                    })
                                    await websocket.send_json({"type": "status", "message": "Text sent, processing..."})
                                except websockets.exceptions.ConnectionClosedOK:
                                    await websocket.send_json({
                                        "type": "error", 
                                        "message": "OpenAI connection was closed while sending text."
                                    })
                                except Exception as e:
                                    await websocket.send_json({"type": "error", "message": f"Text processing error: {str(e)}"})

                elif message["type"] == "websocket.disconnect":
                    print(f"Client {client_id} disconnected")
                    connection_active = False
                    break
                    
            except WebSocketDisconnect:
                print(f"Client {client_id} disconnected")
                connection_active = False
                break
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {str(e)}")
                try:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON format"})
                except:
                    connection_active = False
                    break
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                try:
                    await websocket.send_json({"type": "error", "message": f"Message processing error: {str(e)}"})
                except:
                    connection_active = False
                    break
    
    except Exception as e:
        print(f"Error in client message handling: {str(e)}")

# Create the HTML template file at startup
@app.on_event("startup")
async def create_template_files():
    # Create the index.html template
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Assistant Web App</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1a1b26;
            color: white;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #2a2b36;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        h1, h2, h3 {
            color: #61afef;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
        }
        .main-content {
            display: flex;
            flex: 1;
            gap: 20px;
        }
        .chat-panel {
            flex: 3;
            display: flex;
            flex-direction: column;
        }
        .info-panel {
            flex: 2;
            display: flex;
            flex-direction: column;
            background-color: #343746;
            border-radius: 10px;
            padding: 15px;
        }
        .status-bar {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
            padding: 10px;
            background-color: #343746;
            border-radius: 5px;
        }
        .conversation-container {
            height: 400px;
            overflow-y: auto;
            padding: 15px;
            background-color: #343746;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .query-container {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            background-color: #2c3042;
            border-radius: 5px;
            margin-bottom: 10px;
            min-height: 200px;
        }
        .user-message, .assistant-message {
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 5px;
            max-width: 80%;
        }
        .user-message {
            background-color: #3b4252;
            align-self: flex-end;
            margin-left: auto;
        }
        .assistant-message {
            background-color: #4c566a;
            align-self: flex-start;
            margin-right: auto;
        }
        .message-container {
            display: flex;
            flex-direction: column;
        }
        .input-area {
            display: flex;
            margin-top: 20px;
        }
        .text-input {
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 5px 0 0 5px;
            background-color: #3b4252;
            color: white;
        }
        .send-button, .voice-button {
            padding: 10px 15px;
            border: none;
            color: white;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .send-button {
            border-radius: 0 5px 5px 0;
            background-color: #61afef;
        }
        .voice-button {
            border-radius: 5px;
            margin-left: 10px;
            background-color: #98c379;
        }
        .send-button:hover, .voice-button:hover {
            opacity: 0.9;
        }
        .recording {
            background-color: #e06c75 !important;
        }
        .data-display {
            margin-top: 20px;
            padding: 15px;
            background-color: #343746;
            border-radius: 5px;
        }
        .session-info {
            font-size: 0.8em;
            color: #abb2bf;
        }
        .function-result {
            margin-top: 10px;
            padding: 10px;
            background-color: #4c566a;
            border-radius: 5px;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .query-item {
            margin-bottom: 10px;
            padding: 10px;
            background-color: #3b4252;
            border-radius: 5px;
            border-left: 3px solid #98c379;
        }
        .query-result {
            margin-top: 5px;
            padding: 10px;
            background-color: #4c566a;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            color: #d8dee9;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #4c566a;
        }
        th {
            background-color: #434c5e;
            color: #eceff4;
        }
        .error-toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: rgba(224, 108, 117, 0.9);
            color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            z-index: 1000;
            display: none;
            max-width: 300px;
        }
        .debug-panel {
            position: fixed;
            bottom: 10px;
            left: 10px;
            padding: 5px 10px;
            background-color: rgba(0, 0, 0, 0.6);
            border-radius: 5px;
            font-size: 12px;
            color: #98c379;
            display: none;
            z-index: 1000;
        }
        .audio-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: #61afef;
            margin-left: 10px;
        }
        .audio-playing {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Voice Assistant Web App</h1>
        
        <div class="status-bar">
            <div id="connection-status">WebSocket: Disconnected</div>
            <div id="session-id">Session ID: None</div>
            <div id="audio-status">
                Audio Output <span id="audio-indicator" class="audio-indicator"></span>
            </div>
        </div>
        
        <div class="main-content">
            <div class="chat-panel">
                <div class="message-container conversation-container" id="conversation">
                    <!-- Messages will appear here -->
                    <div class="assistant-message">Hello! How can I help you today?</div>
                </div>
                
                <div class="input-area">
                    <input type="text" id="text-input" class="text-input" placeholder="Type your message here...">
                    <button id="send-button" class="send-button">Send</button>
                    <button id="voice-button" class="voice-button">Record</button>
                </div>
            </div>
            
            <div class="info-panel">
                <h2>Queries & Results</h2>
                <div id="query-container" class="query-container">
                    <!-- Queries and results will appear here -->
                    <div class="query-item">
                        Welcome! Query results will appear here.
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Floating error notification -->
    <div id="error-container" class="error-toast"></div>
    
    <!-- Debug panel -->
    <div id="debug-panel" class="debug-panel">
        Audio Debug: <span id="debug-info">No events yet</span>
    </div>
    
    <script>
        // WebSocket Connection
        let ws;
        let clientId = Math.random().toString(36).substring(2, 15);
        let isRecording = false;
        let mediaRecorder = null;
        let audioChunks = [];
        let audioContext = null;
        let audioProcessor = null;
        let audioPlaying = false;
        let debugMode = false; // Set to true to show debug panel
        
        // DOM Elements
        const conversationDiv = document.getElementById('conversation');
        const queryContainer = document.getElementById('query-container');
        const connectionStatus = document.getElementById('connection-status');
        const sessionIdDisplay = document.getElementById('session-id');
        const textInput = document.getElementById('text-input');
        const sendButton = document.getElementById('send-button');
        const voiceButton = document.getElementById('voice-button');
        const errorContainer = document.getElementById('error-container');
        const debugPanel = document.getElementById('debug-panel');
        const debugInfo = document.getElementById('debug-info');
        const audioIndicator = document.getElementById('audio-indicator');
        
        // Enable debug mode with URL parameter ?debug=true
        if (new URLSearchParams(window.location.search).get('debug') === 'true') {
            debugMode = true;
            debugPanel.style.display = 'block';
        }
        
        // Debug function
        function debug(info) {
            if (debugMode) {
                debugInfo.textContent = info;
                console.log('Debug:', info);
            }
        }
        
        // Connect to WebSocket
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${clientId}`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                connectionStatus.textContent = 'WebSocket: Connected';
                connectionStatus.style.color = '#98c379';
                debug('WebSocket connection established');
            };
            
            ws.onclose = function() {
                connectionStatus.textContent = 'WebSocket: Disconnected';
                connectionStatus.style.color = '#e06c75';
                debug('WebSocket connection closed');
                
                // Attempt to reconnect after a delay
                setTimeout(connectWebSocket, 3000);
            };
            
            ws.onerror = function(error) {
                showError(`WebSocket error: ${error}`);
                debug(`WebSocket error: ${error}`);
            };
            
            ws.onmessage = function(event) {
                if (event.data instanceof Blob) {
                    // Handle audio data
                    debug(`Received audio blob: ${event.data.size} bytes`);
                    playAudio(event.data);
                } else {
                    // Handle JSON messages
                    try {
                        const data = JSON.parse(event.data);
                        handleWSMessage(data);
                    } catch (e) {
                        showError(`Error parsing message: ${e.message}`);
                        debug(`Error parsing WebSocket message: ${e.message}`);
                    }
                }
            };
        }
        
        // Handle different types of WebSocket messages
        function handleWSMessage(data) {
            debug(`Received message type: ${data.type}`);
            
            switch (data.type) {
                case 'session':
                    sessionIdDisplay.textContent = `Session ID: ${data.session_id}`;
                    break;
                
                case 'transcript':
                    updateAssistantMessage(data.text);
                    break;
                
                case 'speech_text':
                    addUserMessage(data.text, 'voice'); 
                    break;
                
                case 'speech_status':
                    updateRecordingStatus(data.recording);
                    break;
                
                case 'function_result':
                    addToQueryContainer(`Result:`, data.result);
                    break;
                
                case 'function_args_progress':
                    debug(`Function args progress: ${data.data}`);
                    break;
                
                case 'error':
                    showError(data.message);
                    break;
                
                case 'status':
                    debug(`Status: ${data.message}`);
                    break;
                
                case 'done':
                    debug('Response completed');
                    const lastAssistantMsg = conversationDiv.querySelector('.assistant-message:last-child');
                    if (lastAssistantMsg) {
                        lastAssistantMsg.setAttribute('data-complete', 'true');
                    }
                    break;
                
                default:
                    debug(`Unhandled message type: ${data.type}`);
            }
        }
        
        // Add a function to update the query container
        function addToQueryContainer(label, data) {
            const queryItem = document.createElement('div');
            queryItem.className = 'query-item';
            
            const labelElement = document.createElement('div');
            labelElement.textContent = label;
            queryItem.appendChild(labelElement);
            
            if (data) {
                const queryResult = document.createElement('div');
                queryResult.className = 'query-result';
                
                if (typeof data === 'object') {
                    if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'object') {
                        const table = createTableFromData(data);
                        queryResult.appendChild(table);
                    } else {
                        queryResult.textContent = JSON.stringify(data, null, 2);
                    }
                } else {
                    queryResult.textContent = data;
                }
                
                queryItem.appendChild(queryResult);
            }
            
            queryContainer.appendChild(queryItem);
            queryContainer.scrollTop = queryContainer.scrollHeight;
        }
        
        function createTableFromData(data) {
            const table = document.createElement('table');
            
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            
            Object.keys(data[0]).forEach(key => {
                const th = document.createElement('th');
                th.textContent = key;
                headerRow.appendChild(th);
            });
            
            thead.appendChild(headerRow);
            table.appendChild(thead);
            
            const tbody = document.createElement('tbody');
            
            data.forEach(item => {
                const row = document.createElement('tr');
                
                Object.values(item).forEach(value => {
                    const td = document.createElement('td');
                    td.textContent = value;
                    row.appendChild(td);
                });
                
                tbody.appendChild(row);
            });
            
            table.appendChild(tbody);
            return table;
        }
        
        async function setupAudioRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                debug('Microphone access granted');
                
                const supportedTypes = [
                    'audio/webm;codecs=opus',
                    'audio/ogg;codecs=opus',
                    'audio/wav',
                    'audio/webm',
                    'audio/ogg',
                ];
                let mimeType = '';
                for (const type of supportedTypes) {
                    if (MediaRecorder.isTypeSupported(type)) {
                        mimeType = type;
                        break;
                    }
                }
                if (!mimeType) {
                    showError('No supported audio format found for recording.');
                    debug('No supported audio format found.');
                    return false;
                }
                debug(`Using MIME type: ${mimeType} for MediaRecorder`);

                mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });
                
                mediaRecorder.ondataavailable = function(event) {
                    if (event.data.size > 0) {
                        debug(`Audio chunk captured: ${event.data.size} bytes, type: ${event.data.type}`);
                        
                        const reader = new FileReader();
                        reader.onloadend = function() {
                            const base64Audio = reader.result.split(',')[1];
                            if (ws && ws.readyState === WebSocket.OPEN) {
                                try {
                                    ws.send(JSON.stringify({
                                        type: "audio_chunk",
                                        data: base64Audio
                                    }));
                                } catch (err) {
                                    debug(`Error sending audio chunk JSON: ${err.message}`);
                                    if (isRecording) {
                                        stopRecording();
                                        showError('Connection error sending audio. Recording stopped.');
                                    }
                                }
                            } else if (isRecording) {
                                stopRecording();
                                showError('WebSocket connection lost sending audio. Recording stopped.');
                                debug('Attempting to reconnect WebSocket...');
                                connectWebSocket();
                            }
                        };
                        reader.onerror = function(error) {
                            debug(`FileReader error: ${error}`);
                            showError('Error reading audio data.');
                        };
                        reader.readAsDataURL(event.data);
                    }
                };
                
                audioContext = new AudioContext();
                const source = audioContext.createMediaStreamSource(stream);
                
                return true;
            } catch (err) {
                showError(`Microphone access error: ${err.message}`);
                debug(`Error accessing microphone: ${err.message}`);
                return false;
            }
        }

        function startRecording() {
            if (!mediaRecorder) {
                debug('Setting up audio recording');
                setupAudioRecording().then(success => {
                    if (success) startRecording();
                });
                return;
            }
            
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                showError('WebSocket connection not available. Reconnecting...');
                connectWebSocket();
                setTimeout(() => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        startRecording();
                    } else {
                        showError('Could not establish WebSocket connection. Please try again later.');
                    }
                }, 1000);
                return;
            }
            
            audioChunks = [];
            isRecording = true;
            voiceButton.textContent = 'Stop';
            voiceButton.classList.add('recording');
            
            try {
                if (mediaRecorder.state === 'inactive') {
                    debug('Starting media recorder');
                    mediaRecorder.start(100);
                    
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            command: 'start_recording'
                        }));
                    }
                }
            } catch (err) {
                showError(`Recording error: ${err.message}`);
                debug(`Recording start error: ${err.message}`);
                isRecording = false;
                voiceButton.textContent = 'Record';
                voiceButton.classList.remove('recording');
            }
        }
        
        function stopRecording() {
            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                isRecording = false;
                voiceButton.textContent = 'Record';
                voiceButton.classList.remove('recording');
                return;
            }
            
            isRecording = false;
            voiceButton.textContent = 'Record';
            voiceButton.classList.remove('recording');
            
            try {
                debug('Stopping media recorder');
                mediaRecorder.stop();
                
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        command: 'stop_recording'
                    }));
                }
            } catch (err) {
                showError(`Error stopping recording: ${err.message}`);
                debug(`Error stopping recording: ${err.message}`);
            }
        }
        
        function updateRecordingStatus(recording) {
            if (recording !== isRecording) {
                if (recording) {
                    voiceButton.textContent = 'Stop';
                    voiceButton.classList.add('recording');
                    isRecording = true;
                } else {
                    voiceButton.textContent = 'Record';
                    voiceButton.classList.remove('recording');
                    isRecording = false;
                }
            }
        }
        
        function playAudio(audioBlob) {
            debug(`Playing audio blob: ${audioBlob.size} bytes, type: ${audioBlob.type || 'unknown'}`);
            
            try {
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio();

                audio.innerHTML = `
                    <source src="${audioUrl}" type="audio/wav">
                    <source src="${audioUrl}" type="audio/mp3">
                    <source src="${audioUrl}" type="audio/mpeg">
                    <source src="${audioUrl}" type="audio/ogg">
                    Your browser does not support the audio element.
                `;
                
                audio.onloadedmetadata = () => debug(`Audio metadata loaded: duration=${audio.duration}s`);
                audio.oncanplaythrough = () => debug('Audio can play through');

                audio.onplay = function() {
                    debug('Audio playback started');
                    audioPlaying = true;
                    audioIndicator.classList.add('audio-playing');
                };
                
                audio.onended = function() {
                    debug('Audio playback ended');
                    audioPlaying = false;
                    audioIndicator.classList.remove('audio-playing');
                    URL.revokeObjectURL(audioUrl);
                };
                
                audio.onpause = function() {
                    debug('Audio playback paused');
                };
                
                audio.onerror = function(e) {
                    const errorCode = e.target.error ? e.target.error.code : 'unknown';
                    debug(`Audio playback error: ${errorCode}`);
                    audioIndicator.classList.remove('audio-playing');
                    showError('Error playing audio response');
                    URL.revokeObjectURL(audioUrl);
                };
                
                audio.play().catch(e => {
                    debug(`Audio play() error: ${e.message}`);
                    showError(`Audio playback failed: ${e.message}. Try clicking somewhere on the page first.`);
                    URL.revokeObjectURL(audioUrl);
                });
                
            } catch (err) {
                debug(`Error setting up audio playback: ${err.message}`);
                showError(`Audio playback error: ${err.message}`);
            }
        }

        function addUserMessage(text, type = 'text') {
            conversationDiv.innerHTML = ''; 
            
            const messageDiv = document.createElement('div');
            messageDiv.className = 'user-message';
            messageDiv.textContent = text;
            conversationDiv.appendChild(messageDiv);
            
            addAssistantMessage("..."); 
            
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
            
            addToQueryContainer(`Query (${type}): ${text}`, null);
            
            return messageDiv;
        }
        
        function addAssistantMessage(text) {
            const placeholder = conversationDiv.querySelector('.assistant-message:last-child');
            if (placeholder && placeholder.textContent === "...") {
                placeholder.textContent = text;
                placeholder.removeAttribute('data-complete');
                conversationDiv.scrollTop = conversationDiv.scrollHeight;
                return placeholder;
            } else {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'assistant-message';
                messageDiv.textContent = text;
                conversationDiv.appendChild(messageDiv);
                conversationDiv.scrollTop = conversationDiv.scrollHeight;
                return messageDiv;
            }
        }
        
        function updateAssistantMessage(text) {
            const messages = conversationDiv.querySelectorAll('.assistant-message');
            if (messages.length > 0) {
                const lastMessage = messages[messages.length - 1];
                if (!lastMessage.hasAttribute('data-complete')) {
                    lastMessage.textContent = text;
                    conversationDiv.scrollTop = conversationDiv.scrollHeight;
                    return lastMessage;
                }
            }
            return addAssistantMessage(text);
        }
        
        function showError(message) {
            errorContainer.textContent = message;
            errorContainer.style.display = 'block';
            debug(`Error: ${message}`);
            
            setTimeout(() => {
                errorContainer.style.display = 'none';
            }, 5000);
        }
        
        function sendTextMessage() {
            const text = textInput.value.trim();
            if (!text) return;
            
            addUserMessage(text, 'text'); 
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    command: 'send_text',
                    text: text
                }));
                
                textInput.value = '';
            } else {
                showError('WebSocket connection not available. Please try again.');
                conversationDiv.innerHTML = '<div class="assistant-message">Connection Error. Please try again.</div>';
            }
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            connectWebSocket();
            
            voiceButton.addEventListener('click', function() {
                if (isRecording) {
                    stopRecording();
                } else {
                    startRecording();
                }
            });
            
            sendButton.addEventListener('click', sendTextMessage);
            
            textInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendTextMessage();
                }
            });
        });
    </script>
</body>
</html>
"""
    
    with open(templates_dir / "index.html", "w") as f:
        f.write(index_html)
    
    print(f"Created template file at {templates_dir / 'index.html'}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
