import base64
import asyncio
from typing import Any, cast

class AudioPlayerAsync:
    """Handles audio playback in an asynchronous context."""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.frame_count = 0
    
    def add_data(self, data: bytes) -> None:
        """Add audio data to the playback queue."""
        self.frame_count += len(data) // 2  # Assuming 16-bit audio (2 bytes per sample)
        self.queue.put_nowait(data)
    
    def reset_frame_count(self) -> None:
        """Reset the frame counter."""
        self.frame_count = 0
    
    async def get_data(self):
        """Get data from the queue for playback."""
        return await self.queue.get()
