"""
Utilities for handling real-time audio input and output using sounddevice and pydub.

Provides functionality for:
- Converting audio bytes to base64 encoded PCM16 format at a specific sample rate.
- An asynchronous audio player (`AudioPlayerAsync`) to play back audio chunks.
- An asynchronous worker (`send_audio_worker_sounddevice`) to capture audio
  from the microphone and send it over an AsyncRealtimeConnection (presumably for
  services like OpenAI's real-time transcription/translation).
"""
from __future__ import annotations

import io
import base64
import asyncio
import threading
from typing import Callable, Awaitable

import numpy as np
import pyaudio
import sounddevice as sd
from pydub import AudioSegment

from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

CHUNK_LENGTH_S = 0.05  # Chunk length in seconds for the player callback
SAMPLE_RATE = 24000  # Target sample rate for audio processing
FORMAT = pyaudio.paInt16 # Audio format (though sounddevice uses numpy dtype)
CHANNELS = 1  # Target number of audio channels (mono)

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false


def audio_to_pcm16_base64(audio_bytes: bytes) -> bytes:
    """
    Converts raw audio bytes (from any format pydub supports) into
    raw PCM16 bytes at the specified SAMPLE_RATE and CHANNELS.

    Args:
        audio_bytes: Raw bytes representing the input audio.

    Returns:
        Raw PCM16 audio bytes (mono, 24kHz).
    """
    # load the audio file from the byte stream
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    print(f"Loaded audio: {audio.frame_rate=} {audio.channels=} {audio.sample_width=} {audio.frame_width=}")
    # resample to 24kHz mono pcm16
    pcm_audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2).raw_data
    return pcm_audio


class AudioPlayerAsync:
    """
    A simple asynchronous audio player for realtime audio streaming using sounddevice.

    Manages an output audio stream and a queue to play incoming audio chunks
    asynchronously without blocking. Uses a separate thread implicitly via the
    sounddevice callback mechanism.
    """

    def __init__(self):
        """Initializes the audio player, queue, lock, and output stream."""
        self.queue = []
        self.lock = threading.Lock() # Lock for thread-safe access to the queue
        self.stream = sd.OutputStream(
            callback=self.callback,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.int16, # Data type expected by the stream
            blocksize=int(CHUNK_LENGTH_S * SAMPLE_RATE), # Size of blocks requested in callback
        )
        self.playing = False # Flag to indicate if the stream is active
        self._frame_count = 0 # Counter for frames played

    def callback(self, outdata, frames, time, status):  # noqa
        """
        Sounddevice callback function executed in a separate thread.

        Pulls data from the queue and fills the output buffer (`outdata`).
        Handles potential underflow by filling remaining space with zeros.

        Args:
            outdata (numpy.ndarray): The buffer to fill with audio data.
            frames (int): The number of frames requested.
            time (CData): Timing information (not used here).
            status (CallbackFlags): Status flags (not used here).
        """
        with self.lock:
            data = np.empty(0, dtype=np.int16)

            # get next item from queue if there is still space in the buffer
            while len(data) < frames and len(self.queue) > 0:
                item = self.queue.pop(0)
                frames_needed = frames - len(data)
                data = np.concatenate((data, item[:frames_needed]))
                # If the item was longer than needed, put the remainder back
                if len(item) > frames_needed:
                    self.queue.insert(0, item[frames_needed:])

            self._frame_count += len(data)

            # fill the rest of the frames with zeros if there is no more data (underflow)
            if len(data) < frames:
                data = np.concatenate((data, np.zeros(frames - len(data), dtype=np.int16)))

        outdata[:] = data.reshape(-1, 1) # Reshape to (frames, channels)

    def reset_frame_count(self):
        """Resets the internal frame counter to zero."""
        self._frame_count = 0

    def get_frame_count(self):
        """Returns the total number of frames played since the last reset."""
        return self._frame_count

    def add_data(self, data: bytes):
        """
        Adds raw PCM16 audio data to the playback queue.

        Converts the bytes to a numpy array and appends it to the queue.
        Starts the stream if it's not already playing.

        Args:
            data: Raw PCM16 audio data (mono, 24kHz) as bytes.
        """
        with self.lock:
            # bytes is pcm16 single channel audio data, convert to numpy array
            np_data = np.frombuffer(data, dtype=np.int16)
            self.queue.append(np_data)
            # Start playing automatically when data is added if not already playing
            if not self.playing:
                self.start()

    def start(self):
        """Starts the audio output stream."""
        self.playing = True
        self.stream.start()

    def stop(self):
        """Stops the audio output stream and clears the playback queue."""
        self.playing = False
        self.stream.stop()
        with self.lock:
            self.queue = [] # Clear any remaining data

    def terminate(self):
        """Closes the audio output stream permanently."""
        self.stream.close()


async def send_audio_worker_sounddevice(
    connection: AsyncRealtimeConnection,
    should_send: Callable[[], bool] | None = None,
    start_send: Callable[[], Awaitable[None]] | None = None,
):
    """
    Asynchronously captures audio using sounddevice and sends it over a connection.

    Continuously reads audio chunks from the default input device, encodes them
    in base64, and sends them via the provided `AsyncRealtimeConnection`.
    Uses `should_send` to conditionally send data and `start_send` as a hook
    before the first audio chunk is sent in a sequence. Commits the buffer
    when `should_send` becomes false after having sent data.

    Args:
        connection: The asynchronous connection object to send audio data to.
        should_send: An optional callable that returns True if audio should currently
                     be sent, False otherwise. If None, always sends.
        start_send: An optional awaitable callable executed just before the first
                    audio chunk is sent after `should_send` becomes True.
    """
    sent_audio = False # Flag to track if we've sent audio since should_send was last false

    device_info = sd.query_devices()
    print(device_info) # Print available audio devices

    read_size = int(SAMPLE_RATE * 0.02) # Read audio in 20ms chunks

    stream = sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="int16", # Data type for input
    )
    stream.start()
    print("Microphone stream started.")

    try:
        while True:
            # Wait until enough data is available to read
            if stream.read_available < read_size:
                await asyncio.sleep(0.001) # Small sleep to prevent busy-waiting
                continue

            data, overflowed = stream.read(read_size)
            if overflowed:
                print("Warning: Input audio overflowed!")

            # Check if we should be sending audio
            if should_send() if should_send else True:
                # If this is the first chunk after a pause, call start_send
                if not sent_audio and start_send:
                    await start_send()
                # Send the audio data, base64 encoded
                await connection.send(
                    {"type": "input_audio_buffer.append", "audio": base64.b64encode(data).decode("utf-8")}
                )
                sent_audio = True

            # If we were sending but should stop now
            elif sent_audio:
                print("Sending stopped, triggering inference/response.")
                # Commit the buffer on the receiving end
                await connection.send({"type": "input_audio_buffer.commit"})
                # Optionally trigger a response (specific to the connection protocol)
                await connection.send({"type": "response.create", "response": {}})
                sent_audio = False # Reset the flag

            await asyncio.sleep(0.001) # Small sleep in the loop

    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping audio worker.")
    except Exception as e:
        print(f"An error occurred in send_audio_worker: {e}")
    finally:
        print("Stopping microphone stream.")
        stream.stop()
        stream.close()
        print("Microphone stream closed.")