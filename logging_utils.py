import datetime
import weave
import wandb

def init_logging():
    """Initialize Weave and WandB logging."""
    # Initialize Weave with a project name
    weave.init("realtime-voice-app")

    if wandb.run is None:
        wandb.init(project="realtime-voice-app", name=f"session-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

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
