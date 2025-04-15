import datetime
import weave
import wandb
import os  # Import the os module

# Define the log file path
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "conversation_log.txt")

def init_logging():
    """Initialize Weave and WandB logging."""
    # Initialize Weave with a project name
    weave.init("realtime-voice-app")

    if wandb.run is None:
        wandb.init(project="realtime-voice-app", name=f"session-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

def _write_to_log_file(log_entry: str):
    """Helper function to append a log entry to the text file."""
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} - {log_entry}\n")
    except Exception as e:
        print(f"Error writing to log file {LOG_FILE_PATH}: {e}") # Print error to console if file logging fails

@weave.op(call_display_name="input_questions")
def log_input_question(turn_type: str, content: str, input_type: str = "voice") -> None:
    """Log conversation turns for tracing with Weave and to a text file.
    
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
    
    # Log to text file
    log_entry = f"Type: {turn_type}, Input: {input_type}, Content: {content}"
    _write_to_log_file(log_entry)
    
    return content

@weave.op(call_display_name="llm_queries")
def log_llm_queries(turn_type: str, content: str, input_type: str = "text") -> None:
    """Log conversation turns for tracing with Weave and to a text file.
    
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
    
    # Log to text file
    log_entry = f"Type: {turn_type}, Input: {input_type}, Content: {content}"
    _write_to_log_file(log_entry)
    
    return content

@weave.op(call_display_name="log_llm_query_result")
def log_llm_query_result(turn_type: str, content: str, input_type: str = "text") -> None:
    """Log conversation turns for tracing with Weave and to a text file.
    
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
    
    # Log to text file
    log_entry = f"Type: {turn_type}, Input: {input_type}, Content: {content}"
    _write_to_log_file(log_entry)
    
    return content


@weave.op(call_display_name="llm_outputs")
def log_llm_outputs(turn_type: str, content: str, input_type: str = "text") -> None:
    """Log conversation turns for tracing with Weave and to a text file.
    
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
    
    # Log to text file
    log_entry = f"Type: {turn_type}, Input: {input_type}, Content: {content}"
    _write_to_log_file(log_entry)
    
    return content

