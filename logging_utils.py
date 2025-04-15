"""
Utilities for logging conversation events using Weave, WandB, and a local text file.

Provides functions to initialize logging and log different parts of the conversation flow,
including user inputs, LLM function call arguments (queries), function call results,
and final LLM text outputs. Each logging function also adds Weave attributes for tracing.
"""
import datetime
import weave
import wandb
import os  # Import the os module

# Define the log file path
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "conversation_log.txt")

def init_logging():
    """Initialize Weave and WandB logging sessions.

    Initializes Weave with a project name and starts a new WandB run
    if one is not already active.
    """
    # Initialize Weave with a project name
    weave.init("realtime-voice-app")

    if wandb.run is None:
        wandb.init(project="realtime-voice-app", name=f"session-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

def _write_to_log_file(log_entry: str):
    """Helper function to append a log entry to the text file.

    Writes a timestamped log entry to the file specified by LOG_FILE_PATH.
    Includes basic error handling to print to console if file writing fails.

    Args:
        log_entry: The string message to log.
    """
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} - {log_entry}\n")
    except Exception as e:
        print(f"Error writing to log file {LOG_FILE_PATH}: {e}") # Print error to console if file logging fails

@weave.op(call_display_name="input_questions")
def log_input_question(turn_type: str, content: str, input_type: str = "voice") -> None:
    """Log user input (voice or text) to Weave, WandB, and a text file.

    Args:
        turn_type: Typically "user_query".
        content: The transcribed text content of the user's input.
        input_type: The source of the input ("voice" or "text").
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

    # Return content to potentially chain operations if needed, though primarily used for side effects.
    return content

@weave.op(call_display_name="llm_queries")
def log_llm_queries(turn_type: str, content: str, input_type: str = "text") -> None:
    """Log LLM function call arguments (queries) to Weave, WandB, and a text file.

    Args:
        turn_type: Typically "arguments" or similar indicating function call input.
        content: The structured arguments (usually JSON string) being sent to the tool.
        input_type: Typically "text" as arguments are processed as text/JSON.
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

    # Return content to potentially chain operations if needed.
    return content

@weave.op(call_display_name="log_llm_query_result")
def log_llm_query_result(turn_type: str, content: str, input_type: str = "text") -> None:
    """Log the result returned by the LLM function call tool to Weave, WandB, and a text file.

    Args:
        turn_type: Typically "result" or similar indicating function call output.
        content: The result returned by the executed tool (e.g., DataFrame query result as JSON string).
        input_type: Typically "text" as results are often serialized.
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

    # Return content to potentially chain operations if needed.
    return content


@weave.op(call_display_name="llm_outputs")
def log_llm_outputs(turn_type: str, content: str, input_type: str = "text") -> None:
    """Log the final LLM text response to Weave, WandB, and a text file.

    Args:
        turn_type: Typically "assistant_response".
        content: The final textual response generated by the LLM.
        input_type: Typically "text" representing the LLM's generated text.
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

    # Return content to potentially chain operations if needed.
    return content

