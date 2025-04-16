# Speech Model Evaluation Data Explorer

This project provides a utility and a simple web application to analyze and visualize speech recognition model evaluation data, focusing on metrics like performance scores (F1, precision, recall, accuracy), latency, and hallucination detection.

## Installation

1.  **Clone the repository (if applicable) or ensure you have the project files.**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### W&B Authentication

Before running the application, ensure you are authenticated with Weights & Biases. You have two primary options:

1.  **Log in via CLI:** Run `wandb login` in your terminal before running `python speech_app.py`. This will prompt you for your API key (if not already configured) and store it locally.
    ```bash
    wandb login
    ```
2.  **Set Environment Variable:** Set the `WANDB_API_KEY` environment variable to your API key. The application will automatically use this key.
    *   **PowerShell:**
        ```powershell
        $env:WANDB_API_KEY = 'YOUR_API_KEY'
        ```
    *   **Command Prompt:**
        ```cmd
        set WANDB_API_KEY=YOUR_API_KEY
        ```
    *(Replace `YOUR_API_KEY` with your actual W&B API key). You need to set this in the same terminal session where you run the script, or configure it as a system-wide environment variable.*

### OpenAI API Key

If you intend to use features that interact with OpenAI models (e.g., for natural language query processing if implemented), you'll also need to configure your OpenAI API key. The application expects this key to be available as an environment variable.

*   **Set Environment Variable:** Set the `OPENAI_API_KEY` environment variable to your API key.
    *   **PowerShell:**
        ```powershell
        $env:OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
        ```
    *   **Command Prompt:**
        ```cmd
        set OPENAI_API_KEY=YOUR_OPENAI_API_KEY
        ```
    *(Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key). Similar to the W&B key, set this in your current terminal session or system-wide.*

## Running the Application

To launch the interactive web application, run the following command from the project's root directory:

```bash
python speech_app.py
```

This will open the application in your default web browser.

## Example Queries

You can interact with the application using natural language queries. Here are some examples based on the available data columns (like f1 score, hallucination rate, model, latency, precision, status, display name, recall, accuracy, learning rate, epochs, warmup ratio, scheduler type, etc.):

### I. Basic Statistics & Information:

*   "What are the basic statistics (mean, median, std dev, min, max) for the 'latency' column?"
*   "Show summary stats for 'f1 score', including the 10th, 50th, and 90th percentiles."
*   "How many unique models are represented in this dataset?"
*   "What are the different 'status' values recorded, and how frequent is each?"
*   "Show the counts for each combination of 'model' and 'scheduler type'."
*   "What data types are the 'epochs' and 'latency' columns?"

### II. Filtering & Selecting Data:

*   "Show me all runs where the 'model' is exactly 'gpt-4o'."
*   "List all runs that did not use the 'gpt-4o' model."
*   "Find runs where the 'hallucination rate' is greater than 0.5."
*   "Which runs had a 'latency' of less than 100 milliseconds?" (Assuming latency is in ms)
*   "Show me runs where the 'display name' starts with 'SmolLM2-360M'."
*   "List runs where the 'model' name contains the substring 'gpt'."
*   "Find runs with an 'f1 score' between 0.7 and 0.8 (inclusive)."
*   "Show runs using either the 'gpt-4o' model or the 'HHEM_2-1' model."
*   "Identify runs where the feedback emoji (shortcut: 'summary.weave.feedback.wandb.reaction.1.payload.emoji') is missing or null."
*   "Filter for runs that have a 'status' of 'success' AND achieved a 'precision' score above 0.85."

### III. Aggregation & Grouping:

*   "What is the overall average 'f1 score' across all runs?"
*   "Calculate the average 'latency' for each distinct 'model'."
*   "What's the standard deviation of the 'recall' scores?"
*   "Find the median 'hallucination rate'."
*   "For each 'model' and 'status' combination, what is the average 'latency' and average 'f1 score'?"
*   "Count the number of successful runs for each 'scheduler type'."

### IV. Ranking & Sorting:

*   "Which are the top 5 runs with the highest 'accuracy'?"
*   "Show the 3 runs with the lowest 'latency'."
*   "List all runs sorted primarily by 'model' name alphabetically, and secondarily by 'f1 score' descending."

### V. Relationships & Correlations:

*   "Is there a correlation between 'latency' and 'f1 score'?"
*   "Calculate the Spearman correlation coefficient between 'precision', 'recall', and 'accuracy'."
*   "How does 'learning rate' correlate with the final 'f1 score'?"

### VI. Hyperparameter Analysis:

*   "Which combination of 'learning rate', 'epochs', 'warmup ratio', and 'scheduler type' yielded the highest 'f1 score'?"
*   "What set of hyperparameters ('learning rate', 'epochs', etc.) resulted in the minimum 'latency'?"
*   "For the 'gpt-4o' model, what was the best 'f1 score' achieved, and what were its hyperparameters?"
