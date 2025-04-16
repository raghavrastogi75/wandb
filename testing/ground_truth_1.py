import pandas as pd
import numpy as np
import io
import traceback
import sys

# --- 1. Load Your Data ---
# IMPORTANT: Replace this line with your actual data loading code
# Example: df = pd.read_csv("your_actual_data.csv")
# For demonstration, creating a dummy DataFrame. Replace this!

df = pd.read_csv("..\data\weave_export_hallucination_2025-04-11.csv")
# Convert potential integer columns read as float due to NAs
df['attributes.num_train_epochs'] = df['attributes.num_train_epochs'].astype('Int64')

print("--- Dummy DataFrame Info ---")
df.info()
print("\n--- Starting Ground Truth Generation ---")


# --- 2. Define Queries (Using Shortcuts) ---
# This list comes directly from the artifact explicit_queries_list
queries_shortcuts = [
    # Describe / Info Queries (~12)
    "Describe the precision column.",                                                                                   # 1
    "Tell me about the recall metric.",                                                                                 # 2
    "Summarize the f1 score column.",                                                                                   # 3
    "Give me basic statistics for accuracy.",                                                                           # 4
    "Describe mean model latency values.",                                                                              # 5
    "What are the characteristics of the hallucination rate column?",                                                   # 6
    "Get information on the status column.",                                                                            # 7
    "Provide info about the mean model latency data type.",                                                             # 8
    "What's the data type for the recall column?",                                                                      # 9
    "Describe the accuracy column using percentiles.",                                                                  # 10
    "Summarize the precision metric.",                                                                                  # 11
    "Basic statistics for recall.",                                                                                     # 12

    # Value Counts (Top 3) Queries (~8)
    "What are the top 3 most frequent status values and their counts?",                                                 # 13
    "Show the top 3 `summary.weave.attributes.os_name` values recorded and their counts.",                              # 14
    "What are the top 3 `summary.weave.attributes.input_dataset_name` values used most often and their counts?",        # 15
    "Value counts for status, limit to 3.",                                                                             # 16
    "Top 3 most common values in `summary.weave.attributes.os_name`?",                                                  # 17
    "Frequency count for the top 3 `summary.weave.attributes.input_dataset_name` values.",                              # 18
    "Show the 3 most frequent status values and counts.",                                                               # 19
    "What are the top 3 values and counts in the status column?",                                                       # 20

    # Aggregate + GroupBy (Easy Columns/Metrics) Queries (~20)
    "What is the average precision per status?",                                                                        # 21
    "Calculate the average recall for each status.",                                                                    # 22
    "Average f1 score grouped by status?",                                                                              # 23
    "What's the mean precision for each value in `summary.weave.attributes.os_name`?",                                  # 24
    "Calculate the average recall per `summary.weave.attributes.os_name`.",                                             # 25
    "Average f1 score grouped by `summary.weave.attributes.os_name`?",                                                  # 26
    "What is the average precision per `summary.weave.attributes.input_dataset_name`?",                                 # 27
    "Calculate the mean recall for each `summary.weave.attributes.input_dataset_name`.",                                # 28
    "Average f1 score grouped by `summary.weave.attributes.input_dataset_name`?",                                       # 29
    "Show mean precision based on the run status.",                                                                     # 30
    "Average recall by `summary.weave.attributes.os_name`?",                                                            # 31
    "Average f1 score by `summary.weave.attributes.input_dataset_name`?",                                               # 32
    "Group by status and find average precision.",                                                                      # 33
    "Group by `summary.weave.attributes.os_name` and calculate mean recall.",                                           # 34
    "Group by `summary.weave.attributes.input_dataset_name` and show average f1 score.",                                # 35
    "What's the average recall for 'success' status runs?",                                                             # 36
    "Mean precision for runs where `summary.weave.attributes.os_name` is 'Linux'?",                                     # 37
    "Average f1 score for runs using the 'RAGTruth-processed_finqa-data-processed-hallucination' dataset? (Using `summary.weave.attributes.input_dataset_name`)", # 38
    "Group by status, calculate average recall.",                                                                       # 39
    "Group by `summary.weave.attributes.os_name`, calculate average precision.",                                        # 40

    # Top N / Bottom N (Values, Max 5) Queries (~20)
    "What are the top 5 precision scores?",                                                                             # 41
    "What are the bottom 3 recall scores?",                                                                             # 42
    "What are the top 5 f1 scores?",                                                                                    # 43
    "What are the 5 lowest accuracy scores?",                                                                           # 44
    "What are the 3 lowest mean model latency values?",                                                                 # 45
    "What are the 5 highest mean model latency values?",                                                                # 46
    "What is the highest precision score?",                                                                             # 47
    "What is the lowest recall score?",                                                                                 # 48
    "What is the highest f1 score?",                                                                                    # 49
    "What is the lowest accuracy score?",                                                                               # 50
    "What is the minimum mean model latency value?",                                                                    # 51
    "What is the maximum mean model latency value?",                                                                    # 52
    "What are the top 2 recall scores?",                                                                                # 53
    "What are the bottom 4 precision scores?",                                                                          # 54
    "What are the top 5 f1 scores recorded?",                                                                           # 55
    "What are the bottom 2 mean model latency values?",                                                                 # 56
    "What is the highest accuracy score achieved?",                                                                     # 57
    "What are the 3 highest mean model latency values observed?",                                                       # 58
    "What are the top 4 recall scores?",                                                                                # 59
    "What are the bottom 5 f1 scores?",                                                                                 # 60

    # Correlation (Direct Phrasing) Queries (~10)
    "Calculate the Pearson correlation between precision and recall.",                                                  # 61
    "Calculate the Pearson correlation between f1 score and accuracy.",                                                 # 62
    "Calculate the Pearson correlation between mean model latency and precision score.",                                # 63
    "Calculate the Spearman correlation between recall and f1 score.",                                                  # 64
    "Calculate the Kendall correlation between accuracy and precision.",                                                # 65
    "Calculate the Pearson correlation for mean model latency and recall.",                                             # 66
    "Calculate the Pearson correlation between precision and f1 score.",                                                # 67
    "Calculate the Pearson correlation between accuracy and recall.",                                                   # 68
    "Calculate the Pearson correlation between mean model latency and f1 score.",                                       # 69
    "Calculate the Pearson correlation between precision and accuracy.",                                                # 70

    # Simple Filter (Counts / Specific Value / Limited IDs) Queries (~10)
    "How many runs have status 'success'?",                                                                             # 71
    "How many runs have precision greater than 0.75?",                                                                  # 72
    "List the id for the first 5 runs found with recall less than 0.2.",                                                # 73
    "What is the average mean model latency for runs with f1 score > 0.65?",                                            # 74
    "How many runs have mean model latency below 1.5 seconds?",                                                         # 75
    "What is the minimum precision found in runs with accuracy >= 0.6?",                                                # 76
    "Count runs with a hallucination rate of 0.",                                                                       # 77
    "List the id for runs with status 'running'.",                                                                      # 78
    "How many runs have precision < 0.25?",                                                                             # 79
    "List the id(s) where recall is exactly 1.0."                                                                       # 80
]


# --- 3. Define Pandas Code for Each Query ---
# Map query index (1-based) to its corresponding Pandas code string.
# IMPORTANT: Use the FULL EXPLICIT column names from your dataset here.
# You MUST fill this dictionary out completely for all 80 queries.
pandas_code = {
    # --- Describe / Info Examples ---
    1: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'].describe()",
    2: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].describe()",
    3: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.f1'].describe()",
    4: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.accuracy'].describe()",
    5: "df['output.model_latency.mean'].describe()",
    6: "df['output.HalluScorerEvaluator.is_hallucination.true_fraction'].describe()",
    7: """
buffer = io.StringIO()
df['summary.weave.status'].info(buf=buffer)
buffer.getvalue()
""",
    8: "str(df['output.model_latency.mean'].dtype)",
    9: "str(df['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].dtype)",
    10: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.accuracy'].describe(percentiles=[.25, .5, .75])", # Default percentiles
    11: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'].describe()",
    12: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].describe()",

    # --- Value Counts Examples ---
    13: "df['summary.weave.status'].value_counts().head(3)",
    14: "df['summary.weave.attributes.os_name'].value_counts().head(3)",
    15: "df['summary.weave.attributes.input_dataset_name'].value_counts().head(3)",
    16: "df['summary.weave.status'].value_counts().head(3)",
    17: "df['summary.weave.attributes.os_name'].value_counts().head(3)",
    18: "df['summary.weave.attributes.input_dataset_name'].value_counts().head(3)",
    19: "df['summary.weave.status'].value_counts().head(3)",
    20: "df['summary.weave.status'].value_counts().head(3)",

    # --- Aggregate + GroupBy Examples ---
    21: "df.groupby('summary.weave.status')['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'].mean()",
    22: "df.groupby('summary.weave.status')['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].mean()",
    23: "df.groupby('summary.weave.status')['output.HalluScorerEvaluator.scorer_evaluation_metrics.f1'].mean()",
    24: "df.groupby('summary.weave.attributes.os_name')['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'].mean()",
    25: "df.groupby('summary.weave.attributes.os_name')['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].mean()",
    26: "df.groupby('summary.weave.attributes.os_name')['output.HalluScorerEvaluator.scorer_evaluation_metrics.f1'].mean()",
    27: "df.groupby(df['summary.weave.attributes.input_dataset_name'].fillna('Unknown'))['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'].mean()",
    28: "df.groupby(df['summary.weave.attributes.input_dataset_name'].fillna('Unknown'))['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].mean()",
    29: "df.groupby(df['summary.weave.attributes.input_dataset_name'].fillna('Unknown'))['output.HalluScorerEvaluator.scorer_evaluation_metrics.f1'].mean()",
    30: "df.groupby('summary.weave.status')['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'].mean()", # Same as 21
    31: "df.groupby('summary.weave.attributes.os_name')['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].mean()", # Same as 25
    32: "df.groupby(df['summary.weave.attributes.input_dataset_name'].fillna('Unknown'))['output.HalluScorerEvaluator.scorer_evaluation_metrics.f1'].mean()", # Same as 29
    33: "df.groupby('summary.weave.status')['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'].mean()", # Same as 21
    34: "df.groupby('summary.weave.attributes.os_name')['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].mean()", # Same as 25
    35: "df.groupby(df['summary.weave.attributes.input_dataset_name'].fillna('Unknown'))['output.HalluScorerEvaluator.scorer_evaluation_metrics.f1'].mean()", # Same as 29
    36: "df[df['summary.weave.status'] == 'success']['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].mean()",
    37: "df[df['summary.weave.attributes.os_name'] == 'Linux']['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'].mean()",
    38: "df[df['summary.weave.attributes.input_dataset_name'] == 'RAGTruth-processed_finqa-data-processed-hallucination']['output.HalluScorerEvaluator.scorer_evaluation_metrics.f1'].mean()",
    39: "df.groupby('summary.weave.status')['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].mean()", # Same as 22
    40: "df.groupby('summary.weave.attributes.os_name')['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'].mean()", # Same as 24

    # --- Top N / Bottom N Examples ---
    41: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'].nlargest(5)",
    42: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].nsmallest(3)",
    43: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.f1'].nlargest(5)",
    44: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.accuracy'].nsmallest(5)",
    45: "df['output.model_latency.mean'].nsmallest(3)",
    46: "df['output.model_latency.mean'].nlargest(5)",
    47: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'].max()",
    48: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].min()",
    49: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.f1'].max()",
    50: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.accuracy'].min()",
    51: "df['output.model_latency.mean'].min()",
    52: "df['output.model_latency.mean'].max()",
    53: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].nlargest(2)",
    54: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'].nsmallest(4)",
    55: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.f1'].nlargest(5)", # Same as 43
    56: "df['output.model_latency.mean'].nsmallest(2)",
    57: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.accuracy'].max()",
    58: "df['output.model_latency.mean'].nlargest(3)",
    59: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'].nlargest(4)",
    60: "df['output.HalluScorerEvaluator.scorer_evaluation_metrics.f1'].nsmallest(5)",

    # --- Correlation Examples ---
    61: "df[['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision', 'output.HalluScorerEvaluator.scorer_evaluation_metrics.recall']].corr(method='pearson')",
    62: "df[['output.HalluScorerEvaluator.scorer_evaluation_metrics.f1', 'output.HalluScorerEvaluator.scorer_evaluation_metrics.accuracy']].corr(method='pearson')",
    63: "df[['output.model_latency.mean', 'output.HalluScorerEvaluator.scorer_evaluation_metrics.precision']].corr(method='pearson')",
    64: "df[['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall', 'output.HalluScorerEvaluator.scorer_evaluation_metrics.f1']].corr(method='spearman')",
    65: "df[['output.HalluScorerEvaluator.scorer_evaluation_metrics.accuracy', 'output.HalluScorerEvaluator.scorer_evaluation_metrics.precision']].corr(method='kendall')",
    66: "df[['output.model_latency.mean', 'output.HalluScorerEvaluator.scorer_evaluation_metrics.recall']].corr(method='pearson')",
    67: "df[['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision', 'output.HalluScorerEvaluator.scorer_evaluation_metrics.f1']].corr(method='pearson')",
    68: "df[['output.HalluScorerEvaluator.scorer_evaluation_metrics.accuracy', 'output.HalluScorerEvaluator.scorer_evaluation_metrics.recall']].corr(method='pearson')",
    69: "df[['output.model_latency.mean', 'output.HalluScorerEvaluator.scorer_evaluation_metrics.f1']].corr(method='pearson')",
    70: "df[['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision', 'output.HalluScorerEvaluator.scorer_evaluation_metrics.accuracy']].corr(method='pearson')",

    # --- Simple Filter Examples ---
    71: "len(df[df['summary.weave.status'] == 'success'])",
    72: "len(df[df['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'] > 0.75])",
    73: "df[df['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'] < 0.2]['id'].head(5).tolist()",
    74: "df[df['output.HalluScorerEvaluator.scorer_evaluation_metrics.f1'] > 0.65]['output.model_latency.mean'].mean()",
    75: "len(df[df['output.model_latency.mean'] < 1.5])",
    76: "df[df['output.HalluScorerEvaluator.scorer_evaluation_metrics.accuracy'] >= 0.6]['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'].min()",
    77: "len(df[df['output.HalluScorerEvaluator.is_hallucination.true_fraction'] == 0])",
    78: "df[df['summary.weave.status'] == 'running']['id'].tolist()",
    79: "len(df[df['output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'] < 0.25])",
    80: "df[df['output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'] == 1.0]['id'].tolist()",

    # Add entries for any missing queries here...
}

# --- 4. Execute Queries and Store Results ---
ground_truth_results = {}
output_buffer = io.StringIO() # Buffer to capture print output like .info()

# Redirect stdout to capture prints
original_stdout = sys.stdout
sys.stdout = output_buffer

for i, query in enumerate(queries_shortcuts):
    query_num = i + 1
    # print(f"\nProcessing Query {query_num}: {query}") # Print progress to original stdout if needed
    code_to_run = pandas_code.get(query_num)
    result_str = "" # Initialize result string

    if code_to_run:
        try:
            # Clear the buffer before execution
            output_buffer.seek(0)
            output_buffer.truncate(0)

            # Execute the pandas code
            # Using exec allows for multi-line strings and assignments if needed later
            # It also handles methods like .info() that print directly
            exec_scope = {'pd': pd, 'df': df, 'np': np, 'io': io, 'result': None}
            exec(f"result = {code_to_run}", exec_scope)
            result = exec_scope['result']

            # Capture printed output (like from .info())
            printed_output = output_buffer.getvalue()

            # Format the result to string
            if printed_output: # If something was printed (like .info())
                 result_str = printed_output.strip()
            elif isinstance(result, pd.DataFrame):
                 result_str = result.to_string()
            elif isinstance(result, pd.Series):
                 result_str = result.to_string()
            else: # Handle scalar values, lists, etc.
                 result_str = str(result)

            # print(f"Result:\n{result_str}") # Print result to original stdout if needed
            ground_truth_results[query] = result_str
        except Exception as e:
            # print(f"ERROR executing code for Query {query_num}: {e}", file=original_stdout)
            # print(traceback.format_exc(), file=original_stdout) # Uncomment for detailed traceback
            ground_truth_results[query] = f"ERROR: {e}"
            # Capture any error message printed during exception
            error_output = output_buffer.getvalue().strip()
            if error_output:
                 ground_truth_results[query] += f"\nOutput during error:\n{error_output}"

    else:
        # print(f"Pandas code not defined for Query {query_num}.", file=original_stdout)
        ground_truth_results[query] = "ERROR: Pandas code not implemented."

# Restore stdout
sys.stdout = original_stdout

# --- 5. Output Ground Truth ---
print("\n--- Generated Ground Truth ---")
output_file = "ground_truth_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("--- Data Analysis Results ---\n")
    # You might want to add a timestamp here
    f.write("------------------------------\n\n")
    f.write("--- Answering User Questions ---\n\n")
    for query, result in ground_truth_results.items():
        f.write(f"Q: {query}\n")
        # Ensure results don't have excessive leading/trailing whitespace
        f.write(f"{result.strip()}\n\n")
    f.write("--- Analysis Complete ---")

print(f"\nGround truth saved to {output_file}")
