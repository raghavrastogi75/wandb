import pandas as pd
import io
import numpy as np
import sys
import contextlib # Import contextlib for redirecting stdout


# Read the CSV data
df = pd.read_csv("weave_export_hallucination_2025-04-11.csv")

# --- Define column name mappings (same as before) ---
precision_col = 'output.HalluScorerEvaluator.scorer_evaluation_metrics.precision'
recall_col = 'output.HalluScorerEvaluator.scorer_evaluation_metrics.recall'
f1_col = 'output.HalluScorerEvaluator.scorer_evaluation_metrics.f1'
accuracy_col = 'output.HalluScorerEvaluator.scorer_evaluation_metrics.accuracy'
hallucination_rate_col = 'output.HalluScorerEvaluator.is_hallucination.true_fraction'
latency_col = 'output.model_latency.mekan'
total_tokens_col = 'output.HalluScorerEvaluator.total_tokens.mean'
display_name_col = 'display_name'
model_name_col = 'attributes.model_name'
status_col = 'summary.weave.status'
run_id_col = 'id'
learning_rate_col = 'attributes.learning_rate'
epochs_col = 'attributes.num_train_epochs'
warmup_ratio_col = 'attributes.warmup_ratio'
scheduler_type_col = 'attributes.lr_scheduler_type'
dataset_name_col = 'attributes.dataset_name'
emoji_col = 'summary.weave.feedback.wandb.reaction.1.payload.emoji'

# --- Data Cleaning and Type Conversion (same as before) ---
numeric_cols = [
    precision_col, recall_col, f1_col, accuracy_col,
    hallucination_rate_col, latency_col, total_tokens_col,
    learning_rate_col, epochs_col, warmup_ratio_col
]
original_stderr = sys.stderr # Keep track of original stderr for warnings
warnings_io = io.StringIO() # Capture warnings separately if needed

# Capture warnings during numeric conversion
with contextlib.redirect_stderr(warnings_io):
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # else: # Don't print warning here, let the analysis part handle it
        #     print(f"Warning: Column '{col}' not found in DataFrame.", file=sys.stderr) # Print to original stderr

    if epochs_col in df.columns:
        df[epochs_col] = df[epochs_col].astype('Int64')

# --- Define output filename ---
output_filename = "analysis_results.txt"

# --- Answering Questions and Saving to File ---
# Use contextlib.redirect_stdout to write all print output to the file
try:
    with open(output_filename, 'w', encoding='utf-8') as f, contextlib.redirect_stdout(f):
        print("--- Data Analysis Results ---")
        print(f"Output generated on: {pd.Timestamp.now()}") # Add timestamp
        print("-" * 30)

        # Print captured warnings first (optional)
        captured_warnings = warnings_io.getvalue()
        if captured_warnings:
             print("\n--- Data Conversion Warnings ---")
             print(captured_warnings)
             print("-" * 30)

        # --- Answering Questions (same code as before) ---
        print("\n--- Answering User Questions ---")

        # Q: Describe precision. (First instance)
        print("\nQ: Describe precision. (First instance)")
        if precision_col in df.columns:
            # Use to_string() for better formatting in text file
            print(df[precision_col].describe().to_string())
        else:
            print(f"Column '{precision_col}' not found.")

        # Q: Describe precision. (Second instance, after duplicate arguments) - Same as above
        print("\nQ: Describe precision. (Second instance)")
        if precision_col in df.columns:
            print(df[precision_col].describe().to_string())
        else:
            print(f"Column '{precision_col}' not found.")

        # Q: Show runs where hallucination rate is greater than 0.5 and model contains gpt.
        print("\nQ: Show runs where hallucination rate > 0.5 and display_name contains 'gpt'")
        if hallucination_rate_col in df.columns and display_name_col in df.columns:
            # Assuming 'model contains gpt' refers to the display name
            gpt_hallu_runs = df[
                (df[hallucination_rate_col] > 0.5) &
                (df[display_name_col].str.contains('gpt', case=False, na=False))
            ]
            cols_to_show_q3 = [run_id_col, display_name_col, hallucination_rate_col]
            # Use to_string() for better table format in text file
            print(gpt_hallu_runs[[col for col in cols_to_show_q3 if col in df.columns]].to_string())
        else:
            print(f"Required columns ('{hallucination_rate_col}', '{display_name_col}') not found.")

        # Q: What is the average latency per model?
        print("\nQ: What is the average latency per model (using display_name)?")
        if latency_col in df.columns and display_name_col in df.columns:
            avg_latency_per_model = df.groupby(display_name_col)[latency_col].mean()
            print(avg_latency_per_model.to_string())
        else:
            print(f"Required columns ('{latency_col}', '{display_name_col}') not found.")

        # Q: Show the top 3 runs by precision.
        print("\nQ: Show the top 3 runs by precision.")
        if precision_col in df.columns:
            top_3_precision = df.sort_values(by=precision_col, ascending=False).head(3)
            cols_to_show_q5 = [run_id_col, display_name_col, precision_col]
            print(top_3_precision[[col for col in cols_to_show_q5 if col in df.columns]].to_string())
        else:
            print(f"Column '{precision_col}' not found.")

        # Q: How many runs per model?
        print("\nQ: How many runs per model (using display_name)?")
        if display_name_col in df.columns:
            runs_per_model = df[display_name_col].value_counts()
            print(runs_per_model.to_string())
        else:
            print(f"Column '{display_name_col}' not found.")

        # Q: Give me information about the latency column.
        print("\nQ: Give me information about the latency column.")
        if latency_col in df.columns:
            print(f"Info for column: {latency_col}")
            # Capture info() output to string
            info_buffer = io.StringIO()
            df[latency_col].info(buf=info_buffer)
            print(info_buffer.getvalue())
            print("\nDescription:")
            print(df[latency_col].describe().to_string())
        else:
            print(f"Column '{latency_col}' not found.")

        # Q: Show runs where the model is gpt-4o.
        print("\nQ: Show runs where the display_name contains 'gpt-4o'.")
        if display_name_col in df.columns:
            gpt4o_runs = df[df[display_name_col].str.contains('gpt-4o', case=False, na=False)]
            # Selecting a few key columns for display
            cols_to_show_q8 = [run_id_col, display_name_col, f1_col, latency_col, status_col]
            print(gpt4o_runs[[col for col in cols_to_show_q8 if col in df.columns]].to_string())
        else:
            print(f"Column '{display_name_col}' not found.")


        # Q: What's the correlation between f1 score and latency?
        print("\nQ: What's the correlation between f1 score and latency?")
        if f1_col in df.columns and latency_col in df.columns:
            correlation_f1_latency = df[[f1_col, latency_col]].corr()
            print(correlation_f1_latency.to_string())
        else:
            print(f"Required columns ('{f1_col}', '{latency_col}') not found.")

        # Q: Show runs where the model is not gpt-4o.
        print("\nQ: Show runs where the display_name does not contain 'gpt-4o'.")
        if display_name_col in df.columns:
            not_gpt4o_runs = df[~df[display_name_col].str.contains('gpt-4o', case=False, na=False)]
            # Selecting a few key columns for display
            cols_to_show_q10 = [run_id_col, display_name_col, f1_col, latency_col, status_col]
            print(not_gpt4o_runs[[col for col in cols_to_show_q10 if col in df.columns]].to_string())
        else:
            print(f"Column '{display_name_col}' not found.")

        # Q: Find runs with latency less than 0.1 seconds.
        print("\nQ: Find runs with latency less than 0.1 seconds.")
        if latency_col in df.columns:
            low_latency_runs = df[df[latency_col] < 0.1]
            cols_to_show_q11 = [run_id_col, display_name_col, f1_col, latency_col]
            print(low_latency_runs[[col for col in cols_to_show_q11 if col in df.columns]].to_string())
        else:
             print(f"Column '{latency_col}' not found.")

        # Q: Show runs where the display name starts with SmolLM2-360M-sft.
        print("\nQ: Show runs where the display name starts with 'SmolLM2-360M-sft'.")
        if display_name_col in df.columns:
            smolm_runs = df[df[display_name_col].str.startswith('SmolLM2-360M-sft', na=False)]
            cols_to_show_q12 = [run_id_col, display_name_col, f1_col, latency_col]
            print(smolm_runs[[col for col in cols_to_show_q12 if col in df.columns]].to_string())
        else:
            print(f"Column '{display_name_col}' not found.")

        # Q: Find runs with f1 score between 0.5 and 0.6.
        print("\nQ: Find runs with f1 score between 0.5 and 0.6.")
        if f1_col in df.columns:
            f1_range_runs = df[(df[f1_col] >= 0.5) & (df[f1_col] <= 0.6)]
            cols_to_show_q13 = [run_id_col, display_name_col, f1_col, precision_col, recall_col]
            print(f1_range_runs[[col for col in cols_to_show_q13 if col in df.columns]].to_string())
        else:
            print(f"Column '{f1_col}' not found.")

        # Q: Show runs where the model is gpt-4o or HHEM_2-1. (Using display_name)
        print("\nQ: Show runs where the display_name contains 'gpt-4o' or 'HHEM_2-1'.")
        if display_name_col in df.columns:
            specific_model_runs = df[
                df[display_name_col].str.contains('gpt-4o', case=False, na=False) |
                df[display_name_col].str.contains('HHEM_2-1', case=False, na=False) # Assuming HHEM_2-1 might appear in display name
            ]
            cols_to_show_q14 = [run_id_col, display_name_col, f1_col, latency_col]
            print(specific_model_runs[[col for col in cols_to_show_q14 if col in df.columns]].to_string())
        else:
            print(f"Column '{display_name_col}' not found.")

        # Q: Find rows where the emoji feedback is null.
        print("\nQ: Find rows where the emoji feedback is null.")
        if emoji_col in df.columns:
            null_emoji_runs = df[df[emoji_col].isnull()]
            cols_to_show_q15 = [run_id_col, display_name_col, emoji_col]
            print(null_emoji_runs[[col for col in cols_to_show_q15 if col in df.columns]].to_string())
        else:
            # In the sample data, this column doesn't exist with data, checking if the column itself exists
            if 'summary.weave.feedback.wandb.reaction.1.payload.emoji' in df.columns:
                 print(f"Column '{emoji_col}' exists but might be all null or wasn't parsed correctly.")
                 print(df[df[emoji_col].isnull()][[run_id_col, display_name_col]].to_string())
            else:
                print(f"Column '{emoji_col}' not found.")


        # Q: Show successful runs with an f1 score of 0.6 or higher.
        print("\nQ: Show successful runs with an f1 score >= 0.6.")
        if status_col in df.columns and f1_col in df.columns:
            high_f1_success_runs = df[(df[status_col] == 'success') & (df[f1_col] >= 0.6)]
            cols_to_show_q16 = [run_id_col, display_name_col, status_col, f1_col]
            print(high_f1_success_runs[[col for col in cols_to_show_q16 if col in df.columns]].to_string())
        else:
            print(f"Required columns ('{status_col}', '{f1_col}') not found.")

        # Q: Calculate the mean, standard deviation, and count for latency and f1 score.
        print("\nQ: Calculate the mean, standard deviation, and count for latency and f1 score.")
        if latency_col in df.columns and f1_col in df.columns:
            stats_latency_f1 = df[[latency_col, f1_col]].agg(['mean', 'std', 'count'])
            print(stats_latency_f1.to_string())
        else:
            print(f"Required columns ('{latency_col}', '{f1_col}') not found.")

        # Q: What is the median hallucination rate?
        print("\nQ: What is the median hallucination rate?")
        if hallucination_rate_col in df.columns:
            median_hallucination = df[hallucination_rate_col].median()
            print(f"Median Hallucination Rate: {median_hallucination}")
        else:
            print(f"Column '{hallucination_rate_col}' not found.")

        # Q: How many unique models are there? (Using display_name)
        print("\nQ: How many unique models are there (using display_name)?")
        if display_name_col in df.columns:
            unique_model_count = df[display_name_col].nunique()
            print(f"Number of unique display names: {unique_model_count}")
        else:
            print(f"Column '{display_name_col}' not found.")

        # Q: Calculate the average latency grouped by model and status. (Using display_name)
        print("\nQ: Calculate the average latency grouped by display_name and status.")
        if latency_col in df.columns and display_name_col in df.columns and status_col in df.columns:
            avg_latency_grouped = df.groupby([display_name_col, status_col])[latency_col].mean()
            print(avg_latency_grouped.to_string())
        else:
            print(f"Required columns ('{latency_col}', '{display_name_col}', '{status_col}') not found.")

        # Q: Sort by model ascending then latency descending and show the top 5. (Using display_name)
        print("\nQ: Sort by display_name ascending then latency descending and show the top 5.")
        if display_name_col in df.columns and latency_col in df.columns:
            sorted_runs = df.sort_values(by=[display_name_col, latency_col], ascending=[True, False])
            cols_to_show_q21 = [run_id_col, display_name_col, latency_col, f1_col]
            print(sorted_runs[[col for col in cols_to_show_q21 if col in df.columns]].head(5).to_string())
        else:
            print(f"Required columns ('{display_name_col}', '{latency_col}') not found.")

        # Q: Give me the value counts for combinations of model and status, limit to 10. (Using display_name)
        print("\nQ: Value counts for combinations of display_name and status (limit 10).")
        if display_name_col in df.columns and status_col in df.columns:
            model_status_counts = df.groupby([display_name_col, status_col]).size().reset_index(name='count')
            print(model_status_counts.sort_values('count', ascending=False).head(10).to_string())
        else:
             print(f"Required columns ('{display_name_col}', '{status_col}') not found.")

        # Q: Calculate the Spearman correlation between precision, recall, and accuracy.
        print("\nQ: Calculate the Spearman correlation between precision, recall, and accuracy.")
        metric_cols_corr = [precision_col, recall_col, accuracy_col]
        if all(col in df.columns for col in metric_cols_corr):
            spearman_corr = df[metric_cols_corr].corr(method='spearman')
            print(spearman_corr.to_string())
        else:
            print(f"One or more required columns ({metric_cols_corr}) not found.")

        # Q: Describe latency using the 10th, 50th, and 90th percentiles.
        print("\nQ: Describe latency using the 10th, 50th, and 90th percentiles.")
        if latency_col in df.columns:
            latency_percentiles = df[latency_col].quantile([0.1, 0.5, 0.9])
            print(latency_percentiles.to_string())
        else:
            print(f"Column '{latency_col}' not found.")

        # Q: Find the hyperparameters for the highest f1 score.
        print("\nQ: Find the hyperparameters for the highest f1 score.")
        hyperparam_cols = [learning_rate_col, epochs_col, warmup_ratio_col, scheduler_type_col, display_name_col]
        if f1_col in df.columns and all(col in df.columns for col in hyperparam_cols):
            best_f1_run = df.loc[df[f1_col].idxmax()]
            print(f"Highest F1 Score: {best_f1_run[f1_col]}")
            print("Hyperparameters:")
            print(best_f1_run[hyperparam_cols].to_string())
        else:
            print(f"One or more required columns ('{f1_col}', {hyperparam_cols}) not found.")

        # Q: Find the hyperparameters for the lowest latency.
        print("\nQ: Find the hyperparameters for the lowest latency.")
        if latency_col in df.columns and all(col in df.columns for col in hyperparam_cols):
            lowest_latency_run = df.loc[df[latency_col].idxmin()]
            print(f"Lowest Latency: {lowest_latency_run[latency_col]}")
            print("Hyperparameters:")
            print(lowest_latency_run[hyperparam_cols].to_string())
        else:
            print(f"One or more required columns ('{latency_col}', {hyperparam_cols}) not found.")

        # Q: List models where status is success and latency is below 5s and f1 score is above 0.5. (Using display_name)
        print("\nQ: List display_names where status is success, latency < 5s, and f1 score > 0.5.")
        req_cols_filter = [status_col, latency_col, f1_col, display_name_col]
        if all(col in df.columns for col in req_cols_filter):
            filtered_models = df[
                (df[status_col] == 'success') &
                (df[latency_col] < 5) &
                (df[f1_col] > 0.5)
            ][display_name_col].unique()
            print(list(filtered_models))
        else:
            print(f"One or more required columns ({req_cols_filter}) not found.")

        # Q: What is the standard deviation of precision for runs where display_name contains 'SmolLM'?
        print("\nQ: Standard deviation of precision for runs where display_name contains 'SmolLM'.")
        if precision_col in df.columns and display_name_col in df.columns:
            smolm_precision_std = df[df[display_name_col].str.contains('SmolLM', case=False, na=False)][precision_col].std()
            print(f"Std Dev of Precision for SmolLM runs: {smolm_precision_std}")
        else:
            print(f"Required columns ('{precision_col}', '{display_name_col}') not found.")

        # Q: For each scheduler type, find the minimum latency and maximum f1 score.
        print("\nQ: For each scheduler type, find the minimum latency and maximum f1 score.")
        req_cols_group = [scheduler_type_col, latency_col, f1_col]
        if all(col in df.columns for col in req_cols_group):
            scheduler_stats = df.groupby(scheduler_type_col).agg(
                min_latency=(latency_col, 'min'),
                max_f1=(f1_col, 'max')
            )
            print(scheduler_stats.to_string())
        else:
            print(f"One or more required columns ({req_cols_group}) not found.")

        # Q: Show the top 2 models based on median f1 score. (Using display_name)
        print("\nQ: Top 2 display_names based on median f1 score.")
        if display_name_col in df.columns and f1_col in df.columns:
            median_f1_models = df.groupby(display_name_col)[f1_col].median()
            top_2_models = median_f1_models.sort_values(ascending=False).head(2)
            print(top_2_models.to_string())
        else:
            print(f"Required columns ('{display_name_col}', '{f1_col}') not found.")

        # Q: Compare the average hallucination rate for models starting with 'gpt' versus models starting with 'SmolLM'. (Using display_name)
        print("\nQ: Compare average hallucination rate for 'gpt' vs 'SmolLM' display_names.")
        if display_name_col in df.columns and hallucination_rate_col in df.columns:
            # Use a temporary column for grouping without modifying the original df permanently within the loop
            df_temp = df.copy()
            df_temp['model_group'] = np.nan # Create temporary column
            df_temp.loc[df_temp[display_name_col].str.startswith('gpt', na=False), 'model_group'] = 'gpt_starts'
            df_temp.loc[df_temp[display_name_col].str.startswith('SmolLM', na=False), 'model_group'] = 'SmolLM_starts'

            avg_hallu_comparison = df_temp.groupby('model_group')[hallucination_rate_col].mean()
            print(avg_hallu_comparison.to_string())
        else:
            print(f"Required columns ('{display_name_col}', '{hallucination_rate_col}') not found.")


        # Q: Which run has the lowest latency among those with accuracy greater than 0.6?
        print("\nQ: Which run (id) has the lowest latency among those with accuracy > 0.6?")
        if latency_col in df.columns and accuracy_col in df.columns and run_id_col in df.columns:
            high_acc_runs = df[df[accuracy_col] > 0.6]
            if not high_acc_runs.empty:
                lowest_latency_high_acc = high_acc_runs.loc[high_acc_runs[latency_col].idxmin()]
                print(f"Run ID: {lowest_latency_high_acc[run_id_col]}")
                print(f"Latency: {lowest_latency_high_acc[latency_col]}")
                print(f"Accuracy: {lowest_latency_high_acc[accuracy_col]}")
            else:
                print("No runs found with accuracy > 0.6.")
        else:
            print(f"Required columns ('{latency_col}', '{accuracy_col}', '{run_id_col}') not found.")


        # Q: Tell me about the runs with the highest recall.
        print("\nQ: Tell me about the runs with the highest recall.")
        if recall_col in df.columns:
            max_recall = df[recall_col].max()
            highest_recall_runs = df[df[recall_col] == max_recall]
            print(f"Highest Recall Value: {max_recall}")
            cols_to_show_q33 = [run_id_col, display_name_col, recall_col, precision_col, f1_col]
            print(highest_recall_runs[[col for col in cols_to_show_q33 if col in df.columns]].to_string())
        else:
            print(f"Column '{recall_col}' not found.")

        # Q: What are the different statuses recorded?
        print("\nQ: What are the different statuses recorded?")
        if status_col in df.columns:
            statuses = df[status_col].unique()
            print(list(statuses))
        else:
            print(f"Column '{status_col}' not found.")

        # Q: Show the learning rate and epochs for the run with the best f1 score where the model used was 'gpt-4o'. (Using display_name)
        print("\nQ: Learning rate and epochs for the best f1 score where display_name contains 'gpt-4o'.")
        req_cols_gpt4o = [f1_col, learning_rate_col, epochs_col, display_name_col]
        if all(col in df.columns for col in req_cols_gpt4o):
            gpt4o_runs_only = df[df[display_name_col].str.contains('gpt-4o', case=False, na=False)]
            if not gpt4o_runs_only.empty:
                # Handle case where f1 might be NaN for all gpt-4o runs
                if gpt4o_runs_only[f1_col].notna().any():
                    best_f1_gpt4o_run = gpt4o_runs_only.loc[gpt4o_runs_only[f1_col].idxmax()]
                    print(f"Best F1 Score for gpt-4o run: {best_f1_gpt4o_run[f1_col]}")
                    print(f"Learning Rate: {best_f1_gpt4o_run[learning_rate_col]}")
                    print(f"Epochs: {best_f1_gpt4o_run[epochs_col]}")
                else:
                    print("No non-NaN F1 scores found for 'gpt-4o' runs.")
            else:
                print("No runs found with 'gpt-4o' in display name.")
        else:
            print(f"One or more required columns ({req_cols_gpt4o}) not found.")


        # Q: Describe all numeric metric columns like precision, recall, f1 score, and accuracy.
        print("\nQ: Describe numeric metric columns (precision, recall, f1, accuracy).")
        metric_cols_desc = [precision_col, recall_col, f1_col, accuracy_col]
        if all(col in df.columns for col in metric_cols_desc):
            print(df[metric_cols_desc].describe().to_string())
        else:
            print(f"One or more required metric columns not found.")


        # Q: What's the relationship between epochs and f1 score? (Correlation)
        print("\nQ: What's the relationship between epochs and f1 score? (Correlation)")
        if epochs_col in df.columns and f1_col in df.columns:
            try:
                # Drop rows where either column is NaN before calculating correlation
                correlation_epochs_f1 = df[[epochs_col, f1_col]].dropna().corr()
                print(correlation_epochs_f1.to_string())
            except Exception as e:
                print(f"Could not calculate correlation, possibly due to data types or NaNs: {e}")
                print("Info for relevant columns:")
                info_buffer = io.StringIO()
                df[[epochs_col, f1_col]].info(buf=info_buffer)
                print(info_buffer.getvalue())
        else:
            print(f"Required columns ('{epochs_col}', '{f1_col}') not found.")


        # Q: Count the runs grouped by model and scheduler type. (Using display_name)
        print("\nQ: Count runs grouped by display_name and scheduler type.")
        if display_name_col in df.columns and scheduler_type_col in df.columns:
            model_scheduler_counts = df.groupby([display_name_col, scheduler_type_col]).size().reset_index(name='count')
            print(model_scheduler_counts.sort_values('count', ascending=False).to_string())
        else:
            print(f"Required columns ('{display_name_col}', '{scheduler_type_col}') not found.")


        # Q: Show runs where the model is gpt-4o or the status is success and latency is high. (Using display_name, defining 'high latency' as > 90th percentile)
        print("\nQ: Runs where display_name contains 'gpt-4o' OR (status is success AND latency > 90th percentile).")
        req_cols_complex_filter = [display_name_col, status_col, latency_col]
        if all(col in df.columns for col in req_cols_complex_filter):
            if df[latency_col].notna().any():
                high_latency_threshold = df[latency_col].quantile(0.9)
                print(f"(Using latency > {high_latency_threshold:.4f} as 'high latency')")
                complex_filter_runs = df[
                    df[display_name_col].str.contains('gpt-4o', case=False, na=False) |
                    ((df[status_col] == 'success') & (df[latency_col] > high_latency_threshold))
                ]
                cols_to_show_q39 = [run_id_col, display_name_col, status_col, latency_col, f1_col]
                print(complex_filter_runs[[col for col in cols_to_show_q39 if col in df.columns]].to_string())
            else:
                print("Latency column contains all NaNs, cannot calculate 90th percentile.")
        else:
            print(f"One or more required columns ({req_cols_complex_filter}) not found.")

        # Q: List the different dataset names used.
        print("\nQ: List the different dataset names used.")
        if dataset_name_col in df.columns:
            datasets = df[dataset_name_col].unique()
            print(list(datasets))
        else:
            print(f"Column '{dataset_name_col}' not found.")

        # Q: Group by model and status, then show the average latency and max f1 score for each group. (Using display_name)
        print("\nQ: Group by display_name and status: Average latency and Max F1 score.")
        req_cols_group_agg = [display_name_col, status_col, latency_col, f1_col]
        if all(col in df.columns for col in req_cols_group_agg):
            grouped_stats = df.groupby([display_name_col, status_col]).agg(
                avg_latency=(latency_col, 'mean'),
                max_f1=(f1_col, 'max')
            )
            print(grouped_stats.to_string())
        else:
            print(f"One or more required columns ({req_cols_group_agg}) not found.")


        # Q: What are the basic statistics (mean, median, std dev, min, max) for the 'latency' column?
        print("\nQ: Basic statistics for latency.")
        if latency_col in df.columns:
            print(df[latency_col].agg(['mean', 'median', 'std', 'min', 'max']).to_string())
        else:
            print(f"Column '{latency_col}' not found.")

        # Q: Show summary stats for 'f1 score', including the 10th, 50th, and 90th percentiles.
        print("\nQ: Summary stats for f1 score (including percentiles).")
        if f1_col in df.columns:
            print(df[f1_col].describe(percentiles=[0.1, 0.5, 0.9]).to_string())
        else:
            print(f"Column '{f1_col}' not found.")

        # Q: How many unique models are represented in this dataset? (Using display_name) - Repetition
        print("\nQ: How many unique models (display_name)?")
        if display_name_col in df.columns:
            print(f"Number of unique display names: {df[display_name_col].nunique()}")
        else:
            print(f"Column '{display_name_col}' not found.")

        # Q: What are the different 'status' values recorded, and how frequent is each?
        print("\nQ: Status value counts.")
        if status_col in df.columns:
            print(df[status_col].value_counts().to_string())
        else:
            print(f"Column '{status_col}' not found.")

        # Q: Show the counts for each combination of 'model' and 'scheduler type'. (Using display_name) - Repetition
        print("\nQ: Counts for display_name and scheduler type combinations.")
        if display_name_col in df.columns and scheduler_type_col in df.columns:
            print(df.groupby([display_name_col, scheduler_type_col]).size().reset_index(name='count').sort_values('count', ascending=False).to_string())
        else:
            print(f"Required columns ('{display_name_col}', '{scheduler_type_col}') not found.")


        # Q: What data types are the 'epochs' and 'latency' columns?
        print("\nQ: Data types for epochs and latency.")
        if epochs_col in df.columns and latency_col in df.columns:
            print(df[[epochs_col, latency_col]].dtypes.to_string())
        else:
            print(f"One or both columns ('{epochs_col}', '{latency_col}') not found.")

        # Q: What is the overall average 'f1 score' across all runs?
        print("\nQ: Overall average f1 score.")
        if f1_col in df.columns:
            print(f"Average F1 Score: {df[f1_col].mean()}")
        else:
            print(f"Column '{f1_col}' not found.")

        # Q: Calculate the average 'latency' for each distinct 'model'. (Using display_name) - Repetition
        print("\nQ: Average latency per display_name.")
        if latency_col in df.columns and display_name_col in df.columns:
            print(df.groupby(display_name_col)[latency_col].mean().to_string())
        else:
            print(f"Required columns ('{latency_col}', '{display_name_col}') not found.")

        # Q: What's the standard deviation of the 'recall' scores?
        print("\nQ: Standard deviation of recall scores.")
        if recall_col in df.columns:
            print(f"Std Dev of Recall: {df[recall_col].std()}")
        else:
            print(f"Column '{recall_col}' not found.")

        # Q: Find the median 'hallucination rate'. - Repetition
        print("\nQ: Median hallucination rate.")
        if hallucination_rate_col in df.columns:
            print(f"Median Hallucination Rate: {df[hallucination_rate_col].median()}")
        else:
            print(f"Column '{hallucination_rate_col}' not found.")

        # Q: For each 'model' and 'status' combination, what is the average 'latency' and average 'f1 score'? (Using display_name) - Repetition
        print("\nQ: Group by display_name and status: Average latency and Average F1 score.")
        if display_name_col in df.columns and status_col in df.columns and latency_col in df.columns and f1_col in df.columns:
            print(df.groupby([display_name_col, status_col]).agg(
                avg_latency=(latency_col, 'mean'),
                avg_f1=(f1_col, 'mean')
            ).to_string())
        else:
            print(f"One or more required columns not found.")


        # Q: Count the number of successful runs for each 'scheduler type'.
        print("\nQ: Count successful runs per scheduler type.")
        if status_col in df.columns and scheduler_type_col in df.columns:
            successful_runs_scheduler = df[df[status_col] == 'success'].groupby(scheduler_type_col).size().reset_index(name='successful_runs_count')
            print(successful_runs_scheduler.to_string())
        else:
            print(f"Required columns ('{status_col}', '{scheduler_type_col}') not found.")


        # Q: Which are the top 5 runs ids with the highest 'accuracy'?
        print("\nQ: Top 5 run IDs by accuracy.")
        if accuracy_col in df.columns and run_id_col in df.columns:
            top_5_accuracy_ids = df.sort_values(accuracy_col, ascending=False).head(5)[run_id_col].tolist()
            print(top_5_accuracy_ids)
        else:
            print(f"Required columns ('{accuracy_col}', '{run_id_col}') not found.")

        # Q: Show the 3 runs ids with the lowest 'latency'.
        print("\nQ: Bottom 3 run IDs by latency.")
        if latency_col in df.columns and run_id_col in df.columns:
            bottom_3_latency_ids = df.sort_values(latency_col, ascending=True).head(3)[run_id_col].tolist()
            print(bottom_3_latency_ids)
        else:
            print(f"Required columns ('{latency_col}', '{run_id_col}') not found.")

        # Q: List all runs ids sorted primarily by 'model' name alphabetically, and secondarily by 'f1 score' descending. (Using display_name)
        print("\nQ: Run IDs sorted by display_name (A-Z), then f1 score (desc).")
        if display_name_col in df.columns and f1_col in df.columns and run_id_col in df.columns:
            sorted_ids = df.sort_values(by=[display_name_col, f1_col], ascending=[True, False])[run_id_col].tolist()
            # Print list elements one per line for readability in text file
            for item_id in sorted_ids:
                print(item_id)
        else:
            print(f"Required columns ('{display_name_col}', '{f1_col}', '{run_id_col}') not found.")


        # Q: Is there a correlation between 'latency' and 'f1 score'? - Repetition (showing Pearson)
        print("\nQ: Correlation between latency and f1 score (Pearson).")
        if latency_col in df.columns and f1_col in df.columns:
            print(df[[latency_col, f1_col]].corr(method='pearson').to_string())
        else:
            print(f"Required columns ('{latency_col}', '{f1_col}') not found.")

        # Q: Calculate the Spearman correlation coefficient between 'precision', 'recall', and 'accuracy'. - Repetition
        print("\nQ: Spearman correlation for precision, recall, accuracy.")
        metric_cols_corr_spearman = [precision_col, recall_col, accuracy_col]
        if all(col in df.columns for col in metric_cols_corr_spearman):
            print(df[metric_cols_corr_spearman].corr(method='spearman').to_string())
        else:
            print(f"One or more required columns ({metric_cols_corr_spearman}) not found.")

        # Q: How does 'learning rate' correlate with the final 'f1 score'?
        print("\nQ: Correlation between learning rate and f1 score.")
        if learning_rate_col in df.columns and f1_col in df.columns:
            print(df[[learning_rate_col, f1_col]].corr().to_string())
        else:
            print(f"Required columns ('{learning_rate_col}', '{f1_col}') not found.")

        # Q: Which combination of 'learning rate', 'epochs', 'warmup ratio', and 'scheduler type' yielded the highest 'f1 score'? - Repetition (similar to finding hyperparameters for highest f1)
        print("\nQ: Hyperparameters (lr, epochs, warmup, scheduler) for highest f1 score.")
        hyperparam_cols_f1 = [learning_rate_col, epochs_col, warmup_ratio_col, scheduler_type_col, display_name_col] # Added display_name for context
        if f1_col in df.columns and all(col in df.columns for col in hyperparam_cols_f1):
            if df[f1_col].notna().any():
                best_f1_run_hyperparams = df.loc[df[f1_col].idxmax()]
                print(f"Highest F1 Score: {best_f1_run_hyperparams[f1_col]}")
                print("Hyperparameters:")
                print(best_f1_run_hyperparams[hyperparam_cols_f1].to_string())
            else:
                print("F1 column contains all NaNs.")
        else:
            print(f"One or more required columns ('{f1_col}', {hyperparam_cols_f1}) not found.")


        # Q: What set of hyperparameters ('learning rate', 'epochs', etc.) resulted in the minimum 'latency'? - Repetition (similar to finding hyperparameters for lowest latency)
        print("\nQ: Hyperparameters for lowest latency.")
        hyperparam_cols_latency = [learning_rate_col, epochs_col, warmup_ratio_col, scheduler_type_col, display_name_col] # Added display_name for context
        if latency_col in df.columns and all(col in df.columns for col in hyperparam_cols_latency):
            if df[latency_col].notna().any():
                min_latency_run_hyperparams = df.loc[df[latency_col].idxmin()]
                print(f"Lowest Latency: {min_latency_run_hyperparams[latency_col]}")
                print("Hyperparameters:")
                print(min_latency_run_hyperparams[hyperparam_cols_latency].to_string())
            else:
                 print("Latency column contains all NaNs.")
        else:
            print(f"One or more required columns ('{latency_col}', {hyperparam_cols_latency}) not found.")


        # Q: For the 'gpt-4o' model, what was the best 'f1 score' achieved, and what were its hyperparameters? (Using display_name) - Repetition
        print("\nQ: Best f1 score and hyperparameters for display_name containing 'gpt-4o'.")
        hyperparam_cols_gpt4o_f1 = [f1_col, learning_rate_col, epochs_col, warmup_ratio_col, scheduler_type_col, display_name_col]
        if all(col in df.columns for col in hyperparam_cols_gpt4o_f1):
            gpt4o_runs_only = df[df[display_name_col].str.contains('gpt-4o', case=False, na=False)]
            if not gpt4o_runs_only.empty:
                if gpt4o_runs_only[f1_col].notna().any():
                    best_f1_gpt4o_run_details = gpt4o_runs_only.loc[gpt4o_runs_only[f1_col].idxmax()]
                    print(f"Best F1 Score for gpt-4o run: {best_f1_gpt4o_run_details[f1_col]}")
                    print("Hyperparameters:")
                    print(best_f1_gpt4o_run_details[hyperparam_cols_gpt4o_f1].to_string()) # Show relevant hyperparams + f1 + name
                else:
                    print("No non-NaN F1 scores found for 'gpt-4o' runs.")
            else:
                print("No runs found with 'gpt-4o' in display name.")
        else:
            print(f"One or more required columns ({hyperparam_cols_gpt4o_f1}) not found.")


        print("\n--- Answering Log 2 ('Log v3') Questions ---")

        # Q: Describe the 'recall' column.
        print("\nQ: Describe recall column.")
        if recall_col in df.columns:
            print(df[recall_col].describe().to_string())
        else:
            print(f"Column '{recall_col}' not found.")

        # Q: Give me info about the 'status' column.
        print("\nQ: Info about status column.")
        if status_col in df.columns:
            info_buffer = io.StringIO()
            df[status_col].info(buf=info_buffer)
            print(info_buffer.getvalue())
            print("\nValue Counts:")
            print(df[status_col].value_counts().to_string())
        else:
            print(f"Column '{status_col}' not found.")

        # Q: Summarize the 'accuracy' metric.
        print("\nQ: Summarize accuracy metric.")
        if accuracy_col in df.columns:
            print(df[accuracy_col].describe().to_string())
        else:
            print(f"Column '{accuracy_col}' not found.")

        # Q: What are the characteristics of the 'total tokens' column?
        print("\nQ: Characteristics of total tokens column.")
        if total_tokens_col in df.columns:
            print(df[total_tokens_col].describe().to_string())
        else:
            print(f"Column '{total_tokens_col}' not found.")

        # Q: Provide descriptive statistics for 'latency'. - Repetition
        print("\nQ: Descriptive statistics for latency.")
        if latency_col in df.columns:
            print(df[latency_col].describe().to_string())
        else:
            print(f"Column '{latency_col}' not found.")

        # Q: Show the data type and number of non-null values for 'model name'. (Using attributes.model_name here)
        print("\nQ: Data type and non-null count for model_name column.")
        if model_name_col in df.columns:
            print(f"Data type: {df[model_name_col].dtype}")
            print(f"Non-null count: {df[model_name_col].count()}")
        else:
            print(f"Column '{model_name_col}' not found.")

        # Q: Describe 'hallucination rate' using quartiles.
        print("\nQ: Describe hallucination rate using quartiles.")
        if hallucination_rate_col in df.columns:
            print(df[hallucination_rate_col].describe(percentiles=[0.25, 0.5, 0.75]).to_string())
        else:
            print(f"Column '{hallucination_rate_col}' not found.")

        # Q: What's the range (min and max) of the 'f1 score'?
        print("\nQ: Range (min and max) of f1 score.")
        if f1_col in df.columns:
            f1_range = df[f1_col].agg(['min', 'max'])
            print(f1_range.to_string())
        else:
            print(f"Column '{f1_col}' not found.")

        # Q: Info on 'scheduler type'.
        print("\nQ: Info on scheduler type.")
        if scheduler_type_col in df.columns:
            info_buffer = io.StringIO()
            df[scheduler_type_col].info(buf=info_buffer)
            print(info_buffer.getvalue())
            print("\nValue Counts:")
            print(df[scheduler_type_col].value_counts().to_string())
        else:
            print(f"Column '{scheduler_type_col}' not found.")

        # Q: Describe 'learning rate'.
        print("\nQ: Describe learning rate.")
        if learning_rate_col in df.columns:
            print(df[learning_rate_col].describe().to_string())
        else:
            print(f"Column '{learning_rate_col}' not found.")

        # Q: Find runs where 'status' is 'running'.
        print("\nQ: Find runs where status is 'running'.")
        if status_col in df.columns:
            running_runs = df[df[status_col] == 'running']
            if running_runs.empty:
                print("No runs found with status 'running'.")
            else:
                cols_to_show_v3_11 = [run_id_col, display_name_col, status_col]
                print(running_runs[[col for col in cols_to_show_v3_11 if col in df.columns]].to_string())
        else:
            print(f"Column '{status_col}' not found.")


        # Q: Show runs with 'accuracy' below 0.5.
        print("\nQ: Show runs with accuracy < 0.5.")
        if accuracy_col in df.columns:
            low_accuracy_runs = df[df[accuracy_col] < 0.5]
            cols_to_show_v3_12 = [run_id_col, display_name_col, accuracy_col]
            print(low_accuracy_runs[[col for col in cols_to_show_v3_12 if col in df.columns]].to_string())
        else:
            print(f"Column '{accuracy_col}' not found.")

        # Q: List runs where 'model name' does not contain 'sft'. (Using attributes.model_name)
        print("\nQ: Runs where model_name does not contain 'sft'.")
        if model_name_col in df.columns:
            no_sft_runs = df[~df[model_name_col].str.contains('sft', case=False, na=True)] # Handle potential NaN
            cols_to_show_v3_13 = [run_id_col, model_name_col, display_name_col]
            print(no_sft_runs[[col for col in cols_to_show_v3_13 if col in df.columns]].to_string())
        else:
            print(f"Column '{model_name_col}' not found.")


        # Q: Which runs have 'latency' greater than 1000 seconds.
        print("\nQ: Runs with latency > 1000 seconds.")
        if latency_col in df.columns:
            very_high_latency_runs = df[df[latency_col] > 1000]
            if very_high_latency_runs.empty:
                print("No runs found with latency > 1000 seconds.")
            else:
                cols_to_show_v3_14 = [run_id_col, display_name_col, latency_col]
                print(very_high_latency_runs[[col for col in cols_to_show_v3_14 if col in df.columns]].to_string())
        else:
            print(f"Column '{latency_col}' not found.")

        # Q: Filter for runs where 'recall' is exactly 1.0.
        print("\nQ: Filter for runs where recall is exactly 1.0.")
        if recall_col in df.columns:
            perfect_recall_runs = df[df[recall_col] == 1.0]
            cols_to_show_v3_15 = [run_id_col, display_name_col, recall_col]
            print(perfect_recall_runs[[col for col in cols_to_show_v3_15 if col in df.columns]].to_string())
        else:
            print(f"Column '{recall_col}' not found.")


        # Q: Show runs whose 'display name' ends with 'debug'.
        print("\nQ: Runs whose display name ends with 'debug'.")
        if display_name_col in df.columns:
            debug_runs = df[df[display_name_col].str.endswith('debug', na=False)]
            if debug_runs.empty:
                print("No runs found ending with 'debug'.")
            else:
                cols_to_show_v3_16 = [run_id_col, display_name_col]
                print(debug_runs[[col for col in cols_to_show_v3_16 if col in df.columns]].to_string())
        else:
            print(f"Column '{display_name_col}' not found.")


        # Q: Find runs with 'precision' not between 0.4 and 0.6.
        print("\nQ: Runs with precision not between 0.4 and 0.6.")
        if precision_col in df.columns:
            precision_outside_runs = df[~df[precision_col].between(0.4, 0.6, inclusive='both')] # Not between 0.4 and 0.6 inclusive
            cols_to_show_v3_17 = [run_id_col, display_name_col, precision_col]
            print(precision_outside_runs[[col for col in cols_to_show_v3_17 if col in df.columns]].to_string())
        else:
             print(f"Column '{precision_col}' not found.")

        # Q: List runs where 'epochs' is null or missing.
        print("\nQ: Runs where epochs is null.")
        if epochs_col in df.columns:
            null_epochs_runs = df[df[epochs_col].isnull()]
            cols_to_show_v3_18 = [run_id_col, display_name_col, epochs_col]
            print(null_epochs_runs[[col for col in cols_to_show_v3_18 if col in df.columns]].to_string())
        else:
            print(f"Column '{epochs_col}' not found.")

        # Q: Show runs where 'status' is 'success' AND 'accuracy' is below 0.6.
        print("\nQ: Runs where status is 'success' AND accuracy < 0.6.")
        if status_col in df.columns and accuracy_col in df.columns:
            success_low_acc_runs = df[(df[status_col] == 'success') & (df[accuracy_col] < 0.6)]
            cols_to_show_v3_19 = [run_id_col, display_name_col, status_col, accuracy_col]
            print(success_low_acc_runs[[col for col in cols_to_show_v3_19 if col in df.columns]].to_string())
        else:
            print(f"Required columns ('{status_col}', '{accuracy_col}') not found.")

        # Q: Find runs where 'model name' is 'HHEM_2-1' OR latency < 0.09 seconds.
        print("\nQ: Runs where model_name is 'HHEM_2-1' OR latency < 0.09 seconds.")
        if model_name_col in df.columns and latency_col in df.columns:
            model_or_low_latency = df[
                (df[model_name_col] == 'HHEM_2-1') | (df[latency_col] < 0.09)
            ]
            cols_to_show_v3_20 = [run_id_col, model_name_col, display_name_col, latency_col]
            print(model_or_low_latency[[col for col in cols_to_show_v3_20 if col in df.columns]].to_string())
        else:
            print(f"Required columns ('{model_name_col}', '{latency_col}') not found.")


        # Q: What is the total number of runs?
        print("\nQ: Total number of runs.")
        print(f"Total runs: {len(df)}")

        # Q: Calculate the sum of 'total tokens' used across all runs.
        print("\nQ: Sum of total tokens.")
        if total_tokens_col in df.columns:
            total_tokens_sum = df[total_tokens_col].sum()
            print(f"Sum of total tokens: {total_tokens_sum}")
        else:
            print(f"Column '{total_tokens_col}' not found.")

        # Q: Find the maximum 'recall' score recorded.
        print("\nQ: Maximum recall score.")
        if recall_col in df.columns:
            max_recall = df[recall_col].max()
            print(f"Maximum Recall: {max_recall}")
        else:
            print(f"Column '{recall_col}' not found.")

        # Q: What is the minimum 'precision' observed?
        print("\nQ: Minimum precision observed.")
        if precision_col in df.columns:
            min_precision = df[precision_col].min()
            print(f"Minimum Precision: {min_precision}")
        else:
            print(f"Column '{precision_col}' not found.")

        # Q: Group by 'status' and count the number of runs in each group. - Repetition
        print("\nQ: Group by status and count runs.")
        if status_col in df.columns:
            print(df.groupby(status_col).size().reset_index(name='count').to_string())
        else:
            print(f"Column '{status_col}' not found.")

        # Q: Calculate the average 'accuracy' for each 'scheduler type'.
        print("\nQ: Average accuracy per scheduler type.")
        if accuracy_col in df.columns and scheduler_type_col in df.columns:
            avg_acc_scheduler = df.groupby(scheduler_type_col)[accuracy_col].mean()
            print(avg_acc_scheduler.to_string())
        else:
            print(f"Required columns ('{accuracy_col}', '{scheduler_type_col}') not found.")

        # Q: Find the median 'latency' for runs where 'model name' contains 'gpt'. (Using attributes.model_name)
        print("\nQ: Median latency for runs where model_name contains 'gpt'.")
        if latency_col in df.columns and model_name_col in df.columns:
            gpt_latency_runs = df[df[model_name_col].str.contains('gpt', case=False, na=False)][latency_col]
            if gpt_latency_runs.notna().any():
                median_latency_gpt = gpt_latency_runs.median()
                print(f"Median Latency for 'gpt' models: {median_latency_gpt}")
            else:
                print("No non-NaN latency values found for 'gpt' models.")
        else:
            print(f"Required columns ('{latency_col}', '{model_name_col}') not found.")


        # Q: What is the standard deviation of 'f1 score' for successful runs?
        print("\nQ: Standard deviation of f1 score for successful runs.")
        if f1_col in df.columns and status_col in df.columns:
            f1_std_success = df[df[status_col] == 'success'][f1_col].std()
            print(f"Std Dev of F1 for successful runs: {f1_std_success}")
        else:
            print(f"Required columns ('{f1_col}', '{status_col}') not found.")

        # Q: For each 'model name', find the min and max 'latency'. (Using attributes.model_name)
        print("\nQ: Min and Max latency per model_name.")
        if model_name_col in df.columns and latency_col in df.columns:
            latency_range_model = df.groupby(model_name_col)[latency_col].agg(['min', 'max'])
            print(latency_range_model.to_string())
        else:
            print(f"Required columns ('{model_name_col}', '{latency_col}') not found.")


        # Q: Count the number of unique 'scheduler types'.
        print("\nQ: Count unique scheduler types.")
        if scheduler_type_col in df.columns:
            unique_schedulers = df[scheduler_type_col].nunique()
            print(f"Number of unique scheduler types: {unique_schedulers}")
        else:
            print(f"Column '{scheduler_type_col}' not found.")

        # Q: Show the top 5 runs based on lowest 'hallucination rate'.
        print("\nQ: Top 5 runs by lowest hallucination rate.")
        if hallucination_rate_col in df.columns:
            top_5_low_hallu = df.sort_values(hallucination_rate_col, ascending=True).head(5)
            cols_to_show_v3_31 = [run_id_col, display_name_col, hallucination_rate_col, f1_col]
            print(top_5_low_hallu[[col for col in cols_to_show_v3_31 if col in df.columns]].to_string())
        else:
            print(f"Column '{hallucination_rate_col}' not found.")

        # Q: List the bottom 3 runs according to 'f1 score'.
        print("\nQ: Bottom 3 runs by f1 score.")
        if f1_col in df.columns:
            bottom_3_f1 = df.sort_values(f1_col, ascending=True).head(3)
            cols_to_show_v3_32 = [run_id_col, display_name_col, f1_col]
            print(bottom_3_f1[[col for col in cols_to_show_v3_32 if col in df.columns]].to_string())
        else:
            print(f"Column '{f1_col}' not found.")


        # Q: Sort all runs by 'latency' in ascending order.
        print("\nQ: All runs sorted by latency ascending (showing top 20).")
        if latency_col in df.columns:
            sorted_by_latency = df.sort_values(latency_col, ascending=True)
            cols_to_show_v3_33 = [run_id_col, display_name_col, latency_col, f1_col]
            print(sorted_by_latency[[col for col in cols_to_show_v3_33 if col in df.columns]].head(20).to_string()) # Show more for context
        else:
            print(f"Column '{latency_col}' not found.")

        # Q: Display runs ordered by 'accuracy' descending, then 'precision' descending.
        print("\nQ: Runs ordered by accuracy (desc), then precision (desc) (showing top 20).")
        if accuracy_col in df.columns and precision_col in df.columns:
            sorted_acc_prec = df.sort_values(by=[accuracy_col, precision_col], ascending=[False, False])
            cols_to_show_v3_34 = [run_id_col, display_name_col, accuracy_col, precision_col, f1_col]
            print(sorted_acc_prec[[col for col in cols_to_show_v3_34 if col in df.columns]].head(20).to_string()) # Show more
        else:
            print(f"Required columns ('{accuracy_col}', '{precision_col}') not found.")

        # Q: Show the top 10 runs sorted by 'recall' (highest first).
        print("\nQ: Top 10 runs sorted by recall (desc).")
        if recall_col in df.columns:
            top_10_recall = df.sort_values(recall_col, ascending=False).head(10)
            cols_to_show_v3_35 = [run_id_col, display_name_col, recall_col, f1_col]
            print(top_10_recall[[col for col in cols_to_show_v3_35 if col in df.columns]].to_string())
        else:
            print(f"Column '{recall_col}' not found.")


        # Q: List the runs sorted by 'display name' alphabetically.
        print("\nQ: Runs sorted by display name (A-Z) (showing top 20).")
        if display_name_col in df.columns:
            sorted_by_name = df.sort_values(display_name_col, ascending=True)
            cols_to_show_v3_36 = [run_id_col, display_name_col, f1_col]
            print(sorted_by_name[[col for col in cols_to_show_v3_36 if col in df.columns]].head(20).to_string()) # Show more
        else:
            print(f"Column '{display_name_col}' not found.")

        # Q: Find the 5 runs with the highest 'total tokens'.
        print("\nQ: 5 runs with highest total tokens.")
        if total_tokens_col in df.columns:
            top_5_tokens = df.sort_values(total_tokens_col, ascending=False).head(5)
            cols_to_show_v3_37 = [run_id_col, display_name_col, total_tokens_col]
            print(top_5_tokens[[col for col in cols_to_show_v3_37 if col in df.columns]].to_string())
        else:
            print(f"Column '{total_tokens_col}' not found.")

        # Q: Sort by 'status' then by 'latency' ascending.
        print("\nQ: Sort by status, then latency ascending (showing top 20).")
        if status_col in df.columns and latency_col in df.columns:
            sorted_status_latency = df.sort_values(by=[status_col, latency_col], ascending=[True, True])
            cols_to_show_v3_38 = [run_id_col, display_name_col, status_col, latency_col]
            print(sorted_status_latency[[col for col in cols_to_show_v3_38 if col in df.columns]].head(20).to_string()) # Show more
        else:
            print(f"Required columns ('{status_col}', '{latency_col}') not found.")

        # Q: Show the bottom 5 runs based on 'precision'.
        print("\nQ: Bottom 5 runs by precision.")
        if precision_col in df.columns:
            bottom_5_precision = df.sort_values(precision_col, ascending=True).head(5)
            cols_to_show_v3_39 = [run_id_col, display_name_col, precision_col, f1_col]
            print(bottom_5_precision[[col for col in cols_to_show_v3_39 if col in df.columns]].to_string())
        else:
            print(f"Column '{precision_col}' not found.")

        # Q: Order runs by 'f1 score' descending and show the top 4.
        print("\nQ: Top 4 runs by f1 score (desc).")
        if f1_col in df.columns:
            top_4_f1 = df.sort_values(f1_col, ascending=False).head(4)
            cols_to_show_v3_40 = [run_id_col, display_name_col, f1_col, precision_col, recall_col]
            print(top_4_f1[[col for col in cols_to_show_v3_40 if col in df.columns]].to_string())
        else:
            print(f"Column '{f1_col}' not found.")


        # Q: What is the Pearson correlation between 'accuracy' and 'latency'?
        print("\nQ: Pearson correlation between accuracy and latency.")
        if accuracy_col in df.columns and latency_col in df.columns:
            print(df[[accuracy_col, latency_col]].corr(method='pearson').to_string())
        else:
            print(f"Required columns ('{accuracy_col}', '{latency_col}') not found.")

        # Q: Calculate the Spearman correlation for 'hallucination rate' and 'f1 score'.
        print("\nQ: Spearman correlation for hallucination rate and f1 score.")
        if hallucination_rate_col in df.columns and f1_col in df.columns:
            print(df[[hallucination_rate_col, f1_col]].corr(method='spearman').to_string())
        else:
            print(f"Required columns ('{hallucination_rate_col}', '{f1_col}') not found.")

        # Q: Is 'warmup ratio' related to 'recall' score? (Correlation)
        print("\nQ: Correlation between warmup ratio and recall.")
        if warmup_ratio_col in df.columns and recall_col in df.columns:
            print(df[[warmup_ratio_col, recall_col]].corr().to_string())
        else:
            print(f"Required columns ('{warmup_ratio_col}', '{recall_col}') not found.")

        # Q: Correlate 'epochs' with 'precision'.
        print("\nQ: Correlation between epochs and precision.")
        if epochs_col in df.columns and precision_col in df.columns:
             try:
                print(df[[epochs_col, precision_col]].dropna().corr().to_string())
             except Exception as e:
                print(f"Could not calculate correlation: {e}")
        else:
            print(f"Required columns ('{epochs_col}', '{precision_col}') not found.")


        # Q: Show the correlation matrix for 'latency', 'precision', and 'recall'.
        print("\nQ: Correlation matrix for latency, precision, recall.")
        corr_cols_multi = [latency_col, precision_col, recall_col]
        if all(col in df.columns for col in corr_cols_multi):
            print(df[corr_cols_multi].corr().to_string())
        else:
             print(f"One or more required columns ({corr_cols_multi}) not found.")


        # Q: What's the relationship between 'total tokens' and 'latency'? (Correlation)
        print("\nQ: Correlation between total tokens and latency.")
        if total_tokens_col in df.columns and latency_col in df.columns:
            print(df[[total_tokens_col, latency_col]].corr().to_string())
        else:
             print(f"Required columns ('{total_tokens_col}', '{latency_col}') not found.")


        # Q: Calculate Kendall correlation between 'accuracy' and 'f1 score'.
        print("\nQ: Kendall correlation between accuracy and f1 score.")
        if accuracy_col in df.columns and f1_col in df.columns:
            print(df[[accuracy_col, f1_col]].corr(method='kendall').to_string())
        else:
            print(f"Required columns ('{accuracy_col}', '{f1_col}') not found.")

        # Q: How does 'status' relate to average 'f1 score'?
        print("\nQ: Average f1 score per status.")
        if status_col in df.columns and f1_col in df.columns:
            avg_f1_status = df.groupby(status_col)[f1_col].mean()
            print(avg_f1_status.to_string())
        else:
            print(f"Required columns ('{status_col}', '{f1_col}') not found.")

        # Q: Correlate 'learning rate' and 'accuracy'.
        print("\nQ: Correlation between learning rate and accuracy.")
        if learning_rate_col in df.columns and accuracy_col in df.columns:
            print(df[[learning_rate_col, accuracy_col]].corr().to_string())
        else:
            print(f"Required columns ('{learning_rate_col}', '{accuracy_col}') not found.")

        # Q: What is the Spearman correlation between 'epochs' and 'latency'?
        print("\nQ: Spearman correlation between epochs and latency.")
        if epochs_col in df.columns and latency_col in df.columns:
             try:
                print(df[[epochs_col, latency_col]].dropna().corr(method='spearman').to_string())
             except Exception as e:
                print(f"Could not calculate correlation: {e}")
        else:
            print(f"Required columns ('{epochs_col}', '{latency_col}') not found.")

        # Q: Find the hyperparameters ('learning rate', 'epochs') for the run with the median 'accuracy'.
        print("\nQ: Hyperparameters (lr, epochs) for the run with median accuracy.")
        median_acc_hyperparam_cols = [learning_rate_col, epochs_col, accuracy_col, display_name_col]
        if all(col in df.columns for col in median_acc_hyperparam_cols):
            if df[accuracy_col].notna().any():
                median_accuracy = df[accuracy_col].median()
                # Find the run closest to the median accuracy
                median_acc_run = df.iloc[(df[accuracy_col]-median_accuracy).abs().argsort()[:1]]
                if not median_acc_run.empty:
                    print(f"Run closest to median accuracy ({median_accuracy:.4f}):")
                    print(median_acc_run[[display_name_col, learning_rate_col, epochs_col, accuracy_col]].to_string())
                else:
                    print("Could not find run with median accuracy.")
            else:
                print("Accuracy column contains all NaNs.")
        else:
            print(f"One or more required columns ({median_acc_hyperparam_cols}) not found.")


        # Q: Which 'scheduler type' is associated with the overall highest average 'precision'?
        print("\nQ: Scheduler type with highest average precision.")
        if scheduler_type_col in df.columns and precision_col in df.columns:
            avg_precision_scheduler = df.groupby(scheduler_type_col)[precision_col].mean()
            if not avg_precision_scheduler.empty:
                best_scheduler = avg_precision_scheduler.idxmax()
                print(f"Scheduler type with highest average precision: {best_scheduler} ({avg_precision_scheduler.max():.4f})")
                print("\nAverage precision per scheduler:")
                print(avg_precision_scheduler.to_string())
            else:
                print("Could not group by scheduler type (perhaps only one type or all NaNs).")
        else:
            print(f"Required columns ('{scheduler_type_col}', '{precision_col}') not found.")

        # Q: Show the 'status' and 'latency' for the run with the minimum 'f1 score'.
        print("\nQ: Status and latency for the run with minimum f1 score.")
        if f1_col in df.columns and status_col in df.columns and latency_col in df.columns:
            if df[f1_col].notna().any():
                min_f1_run = df.loc[df[f1_col].idxmin()]
                print(f"Minimum F1 Score: {min_f1_run[f1_col]}")
                print(f"Status: {min_f1_run[status_col]}")
                print(f"Latency: {min_f1_run[latency_col]}")
            else:
                 print("F1 column contains all NaNs.")
        else:
            print(f"One or more required columns ('{f1_col}', '{status_col}', '{latency_col}') not found.")


        # Q: List the top 3 'model names' based on the number of runs. (Using attributes.model_name)
        print("\nQ: Top 3 model_names by run count.")
        if model_name_col in df.columns:
            top_3_models_count = df[model_name_col].value_counts().head(3)
            print(top_3_models_count.to_string())
        else:
            print(f"Column '{model_name_col}' not found.")


        # Q: What is the average 'recall' for runs using the 'cosine' scheduler?
        print("\nQ: Average recall for 'cosine' scheduler.")
        if recall_col in df.columns and scheduler_type_col in df.columns:
            avg_recall_cosine = df[df[scheduler_type_col] == 'cosine'][recall_col].mean()
            print(f"Average Recall for cosine scheduler: {avg_recall_cosine}")
        else:
            print(f"Required columns ('{recall_col}', '{scheduler_type_col}') not found.")


        # Q: Compare the minimum 'latency' of 'gpt-4o' vs 'gpt-4o-mini'. (Using display_name)
        print("\nQ: Compare minimum latency of 'gpt-4o' vs 'gpt-4o-mini' in display_name.")
        if latency_col in df.columns and display_name_col in df.columns:
            gpt4o_latency = df[df[display_name_col].str.contains('gpt-4o', case=False, na=False) & ~df[display_name_col].str.contains('mini', case=False, na=False)][latency_col]
            gpt4o_mini_latency = df[df[display_name_col].str.contains('gpt-4o-mini', case=False, na=False)][latency_col]

            min_latency_gpt4o = gpt4o_latency.min() if gpt4o_latency.notna().any() else 'N/A'
            min_latency_gpt4o_mini = gpt4o_mini_latency.min() if gpt4o_mini_latency.notna().any() else 'N/A'

            print(f"Min Latency for 'gpt-4o' runs: {min_latency_gpt4o}")
            print(f"Min Latency for 'gpt-4o-mini' runs: {min_latency_gpt4o_mini}")
        else:
            print(f"Required columns ('{latency_col}', '{display_name_col}') not found.")

        # Q: Find runs where 'precision' is above average and 'recall' is below average.
        print("\nQ: Runs where precision > avg_precision AND recall < avg_recall.")
        if precision_col in df.columns and recall_col in df.columns:
            if df[precision_col].notna().any() and df[recall_col].notna().any():
                avg_precision = df[precision_col].mean()
                avg_recall = df[recall_col].mean()
                print(f"(Using avg_precision={avg_precision:.4f}, avg_recall={avg_recall:.4f})")
                filtered_runs_prec_recall = df[
                    (df[precision_col] > avg_precision) & (df[recall_col] < avg_recall)
                ]
                cols_to_show_v3_56 = [run_id_col, display_name_col, precision_col, recall_col]
                print(filtered_runs_prec_recall[[col for col in cols_to_show_v3_56 if col in df.columns]].to_string())
            else:
                print("Precision or Recall column contains all NaNs.")
        else:
             print(f"Required columns ('{precision_col}', '{recall_col}') not found.")


        # Q: Describe 'latency' for runs where 'model name' is 'SmolLM2-135M'. (Using attributes.model_name)
        print("\nQ: Describe latency for model_name 'SmolLM2-135M'.")
        if latency_col in df.columns and model_name_col in df.columns:
            smol135m_latency = df[df[model_name_col] == 'SmolLM2-135M'][latency_col]
            if smol135m_latency.empty:
                print("No runs found for model_name 'SmolLM2-135M'.")
            else:
                print(smol135m_latency.describe().to_string())
        else:
            print(f"Required columns ('{latency_col}', '{model_name_col}') not found.")

        # Q: Which run had the highest 'f1 score' among those with 'status' running?
        print("\nQ: Highest f1 score among runs with status 'running'.")
        if f1_col in df.columns and status_col in df.columns:
            running_runs_f1 = df[df[status_col] == 'running']
            if running_runs_f1.empty:
                print("No runs found with status 'running'.")
            else:
                if running_runs_f1[f1_col].notna().any():
                    best_f1_running_run = running_runs_f1.loc[running_runs_f1[f1_col].idxmax()]
                    cols_to_show_v3_58 = [run_id_col, display_name_col, status_col, f1_col]
                    print("Run details:")
                    print(best_f1_running_run[[col for col in cols_to_show_v3_58 if col in df.columns]].to_string())
                else:
                    print("No non-NaN F1 scores found for 'running' runs.")
        else:
            print(f"Required columns ('{f1_col}', '{status_col}') not found.")


        # Q: Show the average 'precision', 'recall', and 'f1 score' grouped by 'model name', limited to the top 5 models by average F1. (Using attributes.model_name)
        print("\nQ: Average precision, recall, f1 grouped by model_name (Top 5 by avg F1).")
        group_agg_metrics = [model_name_col, precision_col, recall_col, f1_col, run_id_col] # Added run_id for count
        if all(col in df.columns for col in group_agg_metrics):
            model_avg_metrics = df.groupby(model_name_col).agg(
                avg_precision=(precision_col, 'mean'),
                avg_recall=(recall_col, 'mean'),
                avg_f1=(f1_col, 'mean'),
                count=(run_id_col, 'count') # Add count for context
            )
            top_5_models_by_f1 = model_avg_metrics.sort_values('avg_f1', ascending=False).head(5)
            print(top_5_models_by_f1.to_string())
        else:
            print(f"One or more required columns ({group_agg_metrics}) not found.")


        print("\n--- Analysis Complete ---")

    # Indicate completion after the file is closed
    print(f"Analysis results saved to '{output_filename}'")

except Exception as e:
    print(f"An error occurred during analysis or file writing: {e}", file=sys.stderr) # Print error to original stderr