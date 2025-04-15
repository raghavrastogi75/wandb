# In data_utils.py

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Union, Optional, Any, Literal

"""
Utilities for loading, processing, and querying pandas DataFrames,
often derived from WandB/Weave exports. Includes functionality for:
- Translating user-friendly shortcut column names to full names.
- Generating a JSON schema for available query operations.
- Loading data from CSV or creating a dummy DataFrame.
- Serializing various data types (including pandas objects) to JSON-friendly formats.
- Performing common DataFrame operations like info, describe, filter, aggregate,
  sort, value counts, correlation, top N values, and finding best hyperparameters.
- A central dispatcher function (`query_dataframe`) to handle different query types.
"""

# --- Shortcut Map and Translate Function (Keep as before) ---
COLUMN_SHORTCUTS = {
    "f1 score": 'output.HalluScorerEvaluator.scorer_evaluation_metrics.f1',
    "f1": 'output.HalluScorerEvaluator.scorer_evaluation_metrics.f1',
    "hallucination fraction": 'output.HalluScorerEvaluator.is_hallucination.true_fraction',
    "hallucination rate": 'output.HalluScorerEvaluator.is_hallucination.true_fraction',
    "hallucination count": 'output.HalluScorerEvaluator.is_hallucination.true_count',
    "accuracy": 'output.HalluScorerEvaluator.scorer_evaluation_metrics.accuracy',
    "precision": 'output.HalluScorerEvaluator.scorer_evaluation_metrics.precision',
    "recall": 'output.HalluScorerEvaluator.scorer_evaluation_metrics.recall',
    "scorer accuracy": 'output.HalluScorerEvaluator.scorer_accuracy.true_fraction',
    "ground truth hallucination fraction": 'output.HalluScorerEvaluator.is_hallucination_ground_truth.true_fraction',
    "model name": 'attributes.model_name',
    "model": 'attributes.model_name',
    "display name": 'display_name',
    "run name": 'display_name',
    "wandb run name": 'attributes.wandb_run_name',
    "status": 'summary.weave.status',
    "id": 'id',
    "run id": 'attributes.wandb_run_id',
    "latency": 'summary.weave.latency_ms',
    "latency ms": 'summary.weave.latency_ms',
    "mean model latency": 'output.model_latency.mean',
    "generation time": 'output.HalluScorerEvaluator.generation_time.mean',
    "total tokens mean": 'output.HalluScorerEvaluator.total_tokens.mean',
    "completion tokens mean": 'output.HalluScorerEvaluator.completion_tokens.mean',
    # Add specific hyperparameter shortcuts if desired
    "learning rate": "attributes.learning_rate",
    "epochs": "attributes.num_train_epochs",
    "warmup ratio": "attributes.warmup_ratio",
    "scheduler type": "attributes.lr_scheduler_type",
}

def translate_column_name(name: str) -> str:
    """Translates a shortcut column name to its full name if found.

    Uses the `COLUMN_SHORTCUTS` dictionary for translation. If the input
    is not a string or the shortcut is not found, the original name is returned.

    Args:
        name: The column name or shortcut to translate.

    Returns:
        The full column name if a shortcut is found, otherwise the original name.
    """
    if not isinstance(name, str): return name
    shortcut = name.lower().strip()
    return COLUMN_SHORTCUTS.get(shortcut, name)

# --- Schema Function (MODIFIED) ---
def get_tools_schema() -> Dict[str, Any]:
    """ Generate the tools schema including 'best_hyperparameters'.

    Defines the structure and parameters expected by the `query_dataframe`
    function, suitable for use with function-calling models or similar tools.
    Includes descriptions and types for various query parameters.

    Returns:
        A dictionary representing the JSON schema for the `query_dataframe` function.
    """
    schema_parameters = {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                # *** ADDED 'best_hyperparameters' to enum ***
                "enum": [
                    "general", "info", "filter", "aggregate", "sort", "group",
                    "top_n", "correlation", "describe", "compare", "time_series",
                    "value_counts", "best_hyperparameters" # <-- New type
                ],
                "description": "Type of query or analysis."
            },
            "columns": { "type": ["string", "array"], "items": {"type": "string"}, "description": "Column(s)/shortcut(s) for info, describe, aggregate, correlation." },
            "column": { "type": "string", "description": "Alias for single column/shortcut." },
            "filters": { "type": "array", "items": { "type": "object", "properties": { "column": {"type": "string", "description": "Column/shortcut."}, "operator": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "contains", "startswith", "endswith", "between", "isin", "isnull", "notnull"], "description": "Operator."}, "value": { "type": ["string", "number", "boolean", "array"], "items": { "type": ["string", "number", "boolean"] }, "description": "Value(s)." } }, "required": ["column", "operator"] }, "description": "Filters (AND)." },
            "filter_column": {"type": "string", "description": "Column/shortcut for simple filter."},
            "filter_operator": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "contains", "startswith", "endswith", "between", "isin", "isnull", "notnull"], "description": "Operator."},
            "filter_value": { "type": ["string", "number", "boolean", "array"], "items": { "type": ["string", "number", "boolean"] }, "description": "Value."},
            "function": {"type": ["string", "array"], "items": {"type": "string"}, "enum": ["mean", "median", "sum", "min", "max", "count", "std", "var", "first", "last", "nunique"], "description": "Aggregation func(s)."},
            "group_by": {"type": ["string", "array"], "items": {"type": "string"}, "description": "Column(s)/shortcut(s) to group by."},
            "sort_by": {"type": ["string", "array"], "items": {"type": "string"}, "description": "Column(s)/shortcut(s) for sorting."},
            "ascending": {"type": ["boolean", "array"], "items": {"type": "boolean"}, "default": True, "description": "Sort direction(s)."},
            "limit": {"type": "integer", "minimum": 1, "description": "Max rows/values."},
            "n": {"type": "integer", "minimum": 1, "description": "Alias for 'limit' (for top_n)."},
            "percentiles": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "description": "Percentiles for 'describe'."},
            "correlation_method": {"type": "string", "enum": ["pearson", "kendall", "spearman"], "default": "pearson", "description": "Correlation method."},
             # *** ADDED Parameters for 'best_hyperparameters' ***
            "metric_column": {
                 "type": "string",
                 "description": "Metric column name/shortcut (e.g., 'f1 score', 'latency') to find best value for."
            },
            "hyperparameter_columns": {
                 "type": "array",
                 "items": {"type": "string"},
                 "description": "List of hyperparameter column names/shortcuts to return (e.g., ['learning rate', 'epochs'])."
            },
            "find_max": {
                 "type": "boolean",
                 "default": True,
                 "description": "Set to true to find max metric value (higher is better), false for min (lower is better)."
            }
        },
        # Required parameters depend on the query_type, handled in dispatcher logic
        # 'required': ["query_type"] # Keep this simple
    }
    return {
        "type": "function",
        "name": "query_dataframe",
        "description": "Query/analyze DataFrame. Use full names OR shortcuts ('f1 score', 'latency', 'model name'). Can find best hyperparameters for a metric.",
        "parameters": schema_parameters,
    }
# --- End of get_tools_schema ---


# --- Load DF, Serialize, Info, Describe, Filter, Aggregate, Sort, Value Counts, Correlation, Top N Values (Keep as corrected before) ---
# (Including _serialize_result, load_dummy_dataframe, get_dataframe_info, describe_dataframe, filter_dataframe, aggregate_dataframe, sort_dataframe, value_counts_dataframe, correlation_dataframe, get_top_n_values)
# ... (Paste all those functions from the previous corrected version here) ...
def load_dummy_dataframe(file_path=None):
    """Load a DataFrame from a specified path or create a fallback dummy.

    Attempts to load a CSV file from the given `file_path` or a default path.
    Removes unnamed columns and attempts basic numeric conversion.
    If loading fails, it creates and returns a small dummy DataFrame.

    Args:
        file_path: Optional path to the CSV file to load.

    Returns:
        A pandas DataFrame, either loaded from the file or a dummy DataFrame.
    """
    default_path = r"weave_export_hallucination_2025-04-11.csv" # CHECK/CHANGE PATH
    load_path = file_path if file_path else default_path
    try:
        print(f"Attempting to load DataFrame from: {load_path}")
        df = pd.read_csv(load_path)
        print(f"Successfully loaded DataFrame. Shape: {df.shape}")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        for col in df.columns:
            if df[col].dtype == 'object':
                try: converted_col = pd.to_numeric(df[col], errors='coerce')
                except (ValueError, TypeError): pass
        print("\nDataFrame Info after loading:"); df.info(verbose=False)
        return df
    except FileNotFoundError: print(f"Warning: File not found: {load_path}. Using dummy."); return pd.DataFrame({'attributes.model_name': ['M1'], 'output.HalluScorerEvaluator.scorer_evaluation_metrics.f1': [0.5], 'summary.weave.latency_ms': [100]})
    except Exception as e: print(f"Error loading DF: {e}. Using dummy."); return pd.DataFrame({'attributes.model_name': ['M1'], 'output.HalluScorerEvaluator.scorer_evaluation_metrics.f1': [0.5], 'summary.weave.latency_ms': [100]})

def _serialize_result(data: Any, limit: Optional[int] = 20) -> Dict[str, Any]:
    """Serialize results, handles Series with index correctly now.

    Converts various data types (pandas DataFrame/Series, numpy types, basic Python types)
    into a JSON-serializable dictionary format. Handles potential issues like
    infinity, NaN, and limits the number of rows/items returned for DataFrames/Series.

    Args:
        data: The data to serialize (DataFrame, Series, scalar, list, dict, etc.).
        limit: The maximum number of rows (for DataFrame) or items (for Series)
               to include in the serialized output. Defaults to 20.

    Returns:
        A dictionary containing the serialized 'result', original 'count' (if applicable),
        and a 'note' indicating truncation if `limit` was applied. Returns an 'error'
        key if serialization fails.
    """
    note = ""; original_count = None
    try:
        if isinstance(data, pd.DataFrame):
            original_count = len(data)
            res_df = data.head(limit) if limit is not None and len(data) > limit else data
            if limit is not None and len(data) > limit: note = f" (showing first {limit} of {len(data)} rows)"
            res_df = res_df.infer_objects(copy=False).replace([np.inf, -np.inf], [float('inf'), float('-inf')]).fillna("NaN")
            return {"result": res_df.to_dict(orient="records"), "count": original_count, "note": note.strip()}
        elif isinstance(data, pd.Series):
            original_count = len(data)
            res_series = data.head(limit) if limit is not None and len(data) > limit else data
            if limit is not None and len(data) > limit: note = f" (showing first {limit} of {len(data)} items)"
            res_series = res_series.replace([np.inf, -np.inf], [float('inf'), float('-inf')]).fillna("NaN")
            return {"result": res_series.to_dict(), "count": original_count, "note": note.strip()} # Default to dict
        elif isinstance(data, (int, float, str, bool, dict, list, type(None))): count = len(data) if isinstance(data, (list, dict)) else None; return {"result": data, "count": count}
        elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return {"result": int(data)}
        elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
             if np.isinf(data): return {"result": float('inf') if data > 0 else float('-inf')};
             if np.isnan(data): return {"result": "NaN"}; return {"result": float(data)}
        elif isinstance(data, (np.complex_, np.complex64, np.complex128)): return {"result": {'real': float(data.real), 'imag': float(data.imag)}}
        elif isinstance(data, (np.bool_)): return {"result": bool(data)}
        elif isinstance(data, (np.void)): return {"result": "void type"}
        elif isinstance(data, pd.Timestamp): return {"result": data.isoformat()}
        else:
            try:
                return {"result": str(data)}
            except Exception as e_str:
                return {"error": f"Result type {type(data)} could not be serialized: {e_str}"}
    except Exception as e_main: return {"error": f"Serialization error: {e_main}"}


def get_dataframe_info(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
    """Get basic info or info for specific (translated) columns.

    Provides general information about the DataFrame (shape, columns, dtypes, memory usage)
    or detailed information for specified columns (dtype, non-null count, unique count,
    sample values). Handles column name translation using shortcuts.

    Args:
        df: The input DataFrame.
        columns: Optional. A single column name/shortcut or a list of column
                 names/shortcuts to get specific info for. If None, provides
                 general DataFrame info.

    Returns:
        A dictionary containing the requested information or an error message.
    """
    if df.empty: return {"error": "DataFrame is empty."}
    if columns:
        if isinstance(columns, str): columns = [columns]
        cols_translated = [translate_column_name(col) for col in columns]
        cols_to_show = [col for col in cols_translated if col in df.columns]
        if not cols_to_show: return {"error": f"Columns (translated) not found: {cols_translated}"}
        df_subset = df[cols_to_show]
        info = {"columns_info": { col: { "dtype": str(df_subset[col].dtype), "non_null_count": int(df_subset[col].count()), "unique_count": df_subset[col].nunique(), "sample_values": _serialize_result(df_subset[col].dropna().head(5), limit=5).get('result', []) } for col in cols_to_show }}
    else: info = { "shape": df.shape, "columns": df.columns.tolist(), "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}, "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB", "sample": _serialize_result(df.head(3), limit=3).get('result', []) }
    return info


def describe_dataframe(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None, percentiles: Optional[List[float]] = None) -> Dict[str, Any]:
    """Get descriptive statistics. Handles own serialization.

    Calculates descriptive statistics (count, mean, std, min, max, percentiles)
    for numeric columns in the DataFrame or for specified columns. Handles
    column name translation and serializes the output.

    Args:
        df: The input DataFrame.
        columns: Optional. A single column name/shortcut or a list of column
                 names/shortcuts to describe. If None, describes all numeric columns.
        percentiles: Optional. A list of percentiles to include in the output
                     (e.g., [0.25, 0.5, 0.75]).

    Returns:
        A dictionary containing the descriptive statistics or an error message.
        The result is nested under the "description" key.
    """
    if df.empty: return {"error": "DataFrame is empty."}
    target_df = df
    if columns:
        if isinstance(columns, str): columns = [columns]
        cols_translated = [translate_column_name(col) for col in columns]
        cols_to_describe = [col for col in cols_translated if col in df.columns]
        if not cols_to_describe: return {"error": f"Columns (translated) for describe not found: {cols_translated}"}
        target_df = df[cols_to_describe]
    try:
        description = target_df.describe(percentiles=percentiles, include=np.number)
        if description.empty and not target_df.select_dtypes(include=np.number).empty: description = target_df.describe(percentiles=percentiles, include='all')
        elif description.empty: return {"result": "No numeric columns found/specified to describe."}
        serializable_desc = description.replace([np.inf, -np.inf], [float('inf'), float('-inf')]).fillna("NaN").to_dict(orient="index")
        return {"description": serializable_desc}
    except Exception as e: return {"error": f"Error describing DataFrame: {str(e)}"}


def filter_dataframe(df: pd.DataFrame, filters: Optional[List[Dict[str, Any]]] = None, filter_column: Optional[str] = None, filter_operator: Optional[str] = None, filter_value: Any = None) -> Dict[str, Any]:
    """Filter the DataFrame. Includes pandas version compatibility for bools.

    Applies one or more filters to the DataFrame. Supports complex filters via the
    `filters` list or a simple filter via `filter_column`, `filter_operator`,
    and `filter_value`. Handles column name translation, various operators
    (==, !=, >, >=, <, <=, contains, startswith, endswith, between, isin, isnull, notnull),
    and attempts type conversion based on the column's dtype.

    Args:
        df: The input DataFrame.
        filters: Optional. A list of filter dictionaries, each with 'column'
                 (name/shortcut), 'operator', and optionally 'value'. Filters
                 are combined with AND logic.
        filter_column: Optional. Column name/shortcut for a single filter.
        filter_operator: Optional. Operator for the single filter.
        filter_value: Optional. Value for the single filter.

    Returns:
        A dictionary containing the serialized filtered DataFrame ('result', 'count', 'note')
        or an error message.
    """
    if df.empty: return {"result": [], "count": 0, "note": "Input DataFrame was empty."}
    filtered_df = df.copy(); conditions = []; active_filters = filters
    if not filters and filter_column and filter_operator:
         translated_filter_column = translate_column_name(filter_column)
         if translated_filter_column not in df.columns: return {"error": f"Filter column '{filter_column}' (->'{translated_filter_column}') not found."}
         active_filters = [{"column": translated_filter_column, "operator": filter_operator, "value": filter_value}]
    if active_filters:
        for f_orig in active_filters:
            f = f_orig.copy(); col_orig = f.get('column'); op = f.get('operator'); val = f.get('value')
            if not col_orig or not op: return {"error": f"Filter missing 'column' or 'operator': {f_orig}"}
            col = translate_column_name(col_orig)
            if col not in filtered_df.columns: return {"error": f"Filter column '{col_orig}' (->'{col}') not found."}
            try:
                series = filtered_df[col]
                if op not in ['isnull', 'notnull', 'isin'] and val is not None:
                     if isinstance(val, list) and op == 'between':
                           val = [pd.to_numeric(v, errors='coerce') if pd.api.types.is_numeric_dtype(series.dtype) else pd.to_datetime(v, errors='coerce') if pd.api.types.is_datetime64_any_dtype(series.dtype) else v for v in val]
                           if pd.isna(val[0]) or pd.isna(val[1]): raise ValueError(f"Bad 'between' vals for {col_orig}")
                     elif not isinstance(val, list):
                          if pd.api.types.is_numeric_dtype(series.dtype): val = pd.to_numeric(val, errors='coerce')
                          elif pd.api.types.is_datetime64_any_dtype(series.dtype): val = pd.to_datetime(val, errors='coerce')
                          elif series.dtype == object or str(series.dtype).startswith('string'): # Check if potentially boolean-like object/string
                              if isinstance(val, str):
                                   lv = val.lower();
                                   if lv in ['true', '1', 'yes']: val = True
                                   elif lv in ['false', '0', 'no']: val = False
                              else: 
                                  try:
                                     val = bool(val) 
                                  except: 
                                      pass
                          if pd.isna(val) and str(f_orig.get('value')).lower() != 'nan': raise ValueError(f"Bad conversion for {col_orig}")
                if op == "==": conditions.append(series == val)
                elif op == "!=": conditions.append(series != val)
                elif op == ">": conditions.append(series > val)
                elif op == ">=": conditions.append(series >= val)
                elif op == "<": conditions.append(series < val)
                elif op == "<=": conditions.append(series <= val)
                elif op == "contains": conditions.append(series.astype(str).str.contains(str(val), na=False))
                elif op == "startswith": conditions.append(series.astype(str).str.startswith(str(val), na=False))
                elif op == "endswith": conditions.append(series.astype(str).str.endswith(str(val), na=False))
                elif op == "isin": conditions.append(series.isin(val)) if isinstance(val, list) else {"error": f"'isin' needs list value."}
                elif op == "between": conditions.append(series.between(val[0], val[1])) if isinstance(val, list) and len(val)==2 else {"error": f"'between' needs list [min, max]."}
                elif op == "isnull": conditions.append(series.isnull())
                elif op == "notnull": conditions.append(series.notnull())
                else: return {"error": f"Unsupported operator: {op}"}
            except Exception as e: return {"error": f"Filter error ({col_orig} {op} {f_orig.get('value')}): {str(e)}"}
    if conditions: final_condition = conditions[0]; [final_condition := final_condition & c for c in conditions[1:]]; filtered_df = filtered_df.loc[final_condition]
    return _serialize_result(filtered_df)


def aggregate_dataframe(df: pd.DataFrame, columns: Union[str, List[str]], function: Union[str, List[str]], group_by: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
    """Aggregate data in the DataFrame, checking for valid functions.

    Performs aggregation operations (mean, median, sum, min, max, count, std, var,
    first, last, nunique) on specified columns, optionally grouping by other columns.
    Handles column name translation for both aggregation and grouping columns.
    Checks for valid aggregation function names.

    Args:
        df: The input DataFrame.
        columns: A single column name/shortcut or a list of column names/shortcuts
                 to aggregate.
        function: A single aggregation function name or a list of function names.
        group_by: Optional. A single column name/shortcut or a list of column
                  names/shortcuts to group by before aggregation.

    Returns:
        A dictionary containing the serialized aggregated DataFrame ('result', 'count', 'note')
        or an error message.
    """
    if df.empty: return {"error": "Cannot aggregate empty DataFrame."}
    if isinstance(columns, str): original_columns = columns; columns = [columns]
    else: original_columns = columns # Keep original list for warning messages
    if isinstance(function, str): function = [function]

    columns_translated = [translate_column_name(col) for col in columns]
    valid_columns = [col for col in columns_translated if col in df.columns]
    if not valid_columns: return {"error": f"Agg columns (translated) not found: {columns_translated}"}
    if len(valid_columns) < len(columns): print(f"Warning: Agg columns not found (input: {original_columns}, translated: {columns_translated})")
    columns = valid_columns; valid_group_by = None; original_group_by = group_by
    if group_by:
        if isinstance(group_by, str): group_by = [group_by]
        group_by_translated = [translate_column_name(gb_col) for gb_col in group_by]
        valid_group_by = [gb_col for gb_col in group_by_translated if gb_col in df.columns]
        if not valid_group_by: return {"error": f"Group by columns (translated) not found: {group_by_translated}"}
        if len(valid_group_by) < len(group_by): print(f"Warning: Group by columns not found (input: {original_group_by}, translated: {group_by_translated})")

    # Check for invalid function names upfront
    allowed_funcs = {"mean", "median", "sum", "min", "max", "count", "std", "var", "first", "last", "nunique"}
    invalid_funcs = [f for f in function if f not in allowed_funcs]
    if invalid_funcs:
        return {"error": f"Unsupported aggregation function(s): {', '.join(invalid_funcs)}. Allowed: {', '.join(allowed_funcs)}"}

    agg_spec = {col: function for col in columns}
    try:
        if valid_group_by:
            grouped = df.groupby(valid_group_by, dropna=False); result = grouped.agg(agg_spec)
            if isinstance(result.columns, pd.MultiIndex): result.columns = ['_'.join(map(str, col)).strip() for col in result.columns.values]
            result = result.reset_index()
        else:
             results = {};
             for func in function:
                  for col in columns:
                      col_func_name = f"{col}_{func}"; series = df[col]
                      if func == 'count': results[col_func_name] = series.count()
                      elif func == 'nunique': results[col_func_name] = series.nunique()
                      elif func in ['first', 'last']: results[col_func_name] = getattr(series.dropna(), func)() if not series.dropna().empty else None
                      elif func in ['mean', 'median', 'sum', 'std', 'var', 'min', 'max']:
                            try: numeric_series = pd.to_numeric(series, errors='coerce'); results[col_func_name] = getattr(numeric_series, func)() if numeric_series.notna().any() else None
                            except Exception as e: results[col_func_name] = f"Agg Error: {e}"
                      # else case removed due to upfront check
             result = pd.DataFrame([results])
        return _serialize_result(result)
    except Exception as e: return {"error": f"Error during aggregation execution: {str(e)}"}


def sort_dataframe(df: pd.DataFrame, sort_by: Union[str, List[str]], ascending: Union[bool, List[bool]] = True, limit: Optional[int] = None) -> Dict[str, Any]:
    """Sort the DataFrame.

    Sorts the DataFrame based on one or more columns. Handles column name
    translation and allows specifying ascending/descending order for each
    sort column. Optionally limits the number of rows returned.

    Args:
        df: The input DataFrame.
        sort_by: A single column name/shortcut or a list of column names/shortcuts
                 to sort by.
        ascending: Optional. A single boolean or a list of booleans indicating
                   the sort direction for each `sort_by` column. Defaults to True.
        limit: Optional. The maximum number of rows to return after sorting.

    Returns:
        A dictionary containing the serialized sorted DataFrame ('result', 'count', 'note')
        or an error message.
    """
    if df.empty: return {"result": [], "count": 0, "note": "Input DataFrame was empty."}
    if isinstance(sort_by, str): original_sort_by = sort_by; sort_by = [sort_by]
    else: original_sort_by = sort_by # Keep original list for warning messages
    sort_by_translated = [translate_column_name(col) for col in sort_by]
    valid_sort_by = [col for col in sort_by_translated if col in df.columns]
    if not valid_sort_by: return {"error": f"Sort columns (translated) not found: {sort_by_translated}"}
    if len(valid_sort_by) < len(sort_by):
         print(f"Warning: Sort columns not found (input: {original_sort_by}, translated: {sort_by_translated})")
         indices_to_keep = [i for i, col in enumerate(sort_by_translated) if col in valid_sort_by]
         if isinstance(ascending, list): ascending = [ascending[i] for i in indices_to_keep]
         sort_by = valid_sort_by
    else: sort_by = valid_sort_by
    if isinstance(ascending, bool): ascending = [ascending] * len(sort_by)
    if len(sort_by) != len(ascending): return {"error": "# sort_by must match # ascending."}
    try:
        sorted_df = df.sort_values(by=sort_by, ascending=ascending, na_position='last')
        return _serialize_result(sorted_df, limit=limit)
    except Exception as e: return {"error": f"Error sorting DataFrame: {str(e)}"}


def value_counts_dataframe(df: pd.DataFrame, columns: Union[str, List[str]], limit: Optional[int] = None) -> Dict[str, Any]:
     """Get value counts for specified column(s).

     Calculates the frequency of unique values in one or more specified columns.
     Handles column name translation and optionally limits the number of unique
     values returned per column.

     Args:
         df: The input DataFrame.
         columns: A single column name/shortcut or a list of column names/shortcuts
                  for which to calculate value counts.
         limit: Optional. The maximum number of unique values (and their counts)
                to return for each column.

     Returns:
         If a single column is requested, returns the serialized value counts
         (dict format) for that column. If multiple columns are requested, returns
         a dictionary where keys are the original column names and values are the
         serialized value counts for each, nested under a "value_counts" key.
         Returns an error message if issues occur.
     """
     if df.empty: return {"error": "Cannot get value counts from empty DataFrame."}
     original_columns = columns
     if isinstance(columns, str): columns = [columns]

     results = {}
     try:
         columns_translated = [translate_column_name(col) for col in columns]
         for i, col_t in enumerate(columns_translated):
             col_orig = original_columns if isinstance(original_columns, str) else original_columns[i]
             if col_t not in df.columns: results[col_orig] = {"error": f"Column '{col_orig}' (->'{col_t}') not found"}
             else: counts = df[col_t].value_counts(dropna=False); results[col_orig] = _serialize_result(counts, limit=limit)

         if isinstance(original_columns, str): return results[original_columns]
         elif isinstance(original_columns, list) and len(original_columns) == 1: return results[original_columns[0]]
         else: return {"value_counts": results}
     except Exception as e: return {"error": f"Error getting value counts: {str(e)}"}


def correlation_dataframe(df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'pearson') -> Dict[str, Any]:
     """Calculate pairwise correlation of columns.

     Computes the pairwise correlation between numeric columns in the DataFrame,
     or a specified subset of columns. Handles column name translation.

     Args:
         df: The input DataFrame.
         columns: Optional. A list of column names/shortcuts to include in the
                  correlation calculation. If None, uses all numeric columns.
         method: Optional. The correlation method ('pearson', 'kendall', 'spearman').
                 Defaults to 'pearson'.

     Returns:
         A dictionary containing the serialized correlation matrix (as a dict of dicts)
         under the "correlation_matrix" key, or an error message.
     """
     if df.empty: return {"error": "Cannot calculate correlation on empty DataFrame."}
     target_df = df
     if columns:
         original_columns = columns; cols_translated = [translate_column_name(col) for col in columns]
         cols_to_corr = [col for col in cols_translated if col in df.columns]
         if not cols_to_corr: return {"error": f"Columns (translated) not found for correlation: {cols_translated}"}
         if len(cols_to_corr) < len(columns): print(f"Warning: Columns not found for correlation (input: {original_columns}, translated: {cols_translated})")
         target_df = df[cols_to_corr]
     numeric_df = target_df.select_dtypes(include=np.number)
     if numeric_df.shape[1] < 2: return {"error": f"Correlation requires >= 2 numeric columns. Found: {numeric_df.columns.tolist()}"}
     try:
         correlation_matrix = numeric_df.corr(method=method)
         return {"correlation_matrix": _serialize_result(correlation_matrix.to_dict(orient="index"))}
     except Exception as e: return {"error": f"Error calculating correlation: {str(e)}"}


def get_top_n_values(df: pd.DataFrame, sort_by: str, ascending: bool, n: int) -> Dict[str, Any]:
    """Gets the top N values from a single column after sorting.

    Extracts the top (or bottom) N values from a single specified column after
    sorting it. Handles column name translation and attempts numeric conversion
    for object-type columns if possible.

    Args:
        df: The input DataFrame.
        sort_by: The column name/shortcut to sort and extract values from.
        ascending: Boolean indicating the sort direction (True for ascending/smallest,
                   False for descending/largest).
        n: The number of top/bottom values to return.

    Returns:
        A dictionary containing the list of top N values ('result'), the count ('count'),
        a descriptive 'note', or an error message.
    """
    if df.empty: return {"error": "Cannot get top N from empty DataFrame."}
    original_sort_by = sort_by; sort_by_col = translate_column_name(sort_by)
    if sort_by_col not in df.columns: return {"error": f"Column '{original_sort_by}' (->'{sort_by_col}') not found."}
    try:
        series = df[sort_by_col]
        if series.dtype == 'object':
            numeric_series = pd.to_numeric(series, errors='coerce')
            if not numeric_series.isnull().all(): series = numeric_series
            else: return {"error": f"Col '{sort_by_col}' (object type) could not be converted to numeric for sorting."}

        if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_string_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
             return {"error": f"Col '{sort_by_col}' has dtype {series.dtype} which may not be reliably sortable."}

        sorted_series = series.sort_values(ascending=ascending, na_position='last')
        top_n_series = sorted_series.head(n)
        result_list = top_n_series.replace([np.inf, -np.inf], [float('inf'), float('-inf')]).fillna("NaN").tolist()
        return {"result": result_list, "count": len(result_list), "note": f"Top {n} values from '{sort_by_col}'"}
    except Exception as e: return {"error": f"Error getting top N values for '{sort_by_col}': {str(e)}"}


# --- 1. New Helper Function for Best Hyperparameters ---
def find_best_hyperparameters(df: pd.DataFrame, metric_column: str, hyperparameter_columns: List[str], find_max: bool = True) -> Dict[str, Any]:
    """Finds the hyperparameter values corresponding to the best metric value.

    Identifies the row with the maximum or minimum value in the specified `metric_column`
    and returns the values from the corresponding `hyperparameter_columns` for that row.
    Handles column name translation for both metric and hyperparameter columns.

    Args:
        df: The input DataFrame.
        metric_column: The name/shortcut of the column containing the metric to optimize.
        hyperparameter_columns: A list of column names/shortcuts representing the
                                hyperparameters to retrieve.
        find_max: Optional. If True (default), finds the row with the maximum metric value.
                  If False, finds the row with the minimum metric value.

    Returns:
        A dictionary containing the best metric value and a dictionary of the
        corresponding hyperparameter values, nested under the "result" key.
        Returns an error message if issues occur.
    """
    if df.empty: return {"error": "Cannot find best hyperparameters in empty DataFrame."}
    if not hyperparameter_columns: return {"error": "No hyperparameter columns specified."}

    original_metric = metric_column
    metric_col_t = translate_column_name(metric_column)
    if metric_col_t not in df.columns:
        return {"error": f"Metric column '{original_metric}' (->'{metric_col_t}') not found."}

    valid_hyperparams_map = []
    missing_hyperparams_input = []
    for hp_orig in hyperparameter_columns:
        hp_t = translate_column_name(hp_orig)
        if hp_t in df.columns:
            valid_hyperparams_map.append((hp_orig, hp_t))
        else:
            missing_hyperparams_input.append(hp_orig)

    if not valid_hyperparams_map:
        return {"error": f"None of the specified hyperparameter columns were found (input: {hyperparameter_columns})."}
    if missing_hyperparams_input:
        print(f"Warning: Hyperparameter columns not found/translated: {missing_hyperparams_input}")

    try:
        metric_series = df[metric_col_t]
        if not pd.api.types.is_numeric_dtype(metric_series):
             numeric_metric = pd.to_numeric(metric_series, errors='coerce')
             if numeric_metric.isnull().all(): return {"error": f"Metric column '{metric_col_t}' not numeric/convertible."}
             metric_series = numeric_metric

        if metric_series.isna().all(): return {"error": f"Metric column '{metric_col_t}' has only NaN values."}

        try:
            best_index = metric_series.idxmax() if find_max else metric_series.idxmin()
        except TypeError as e:
             print(f"Warning: idxmax/idxmin failed ({e}), trying sort_values approach.")
             sorted_df = df.sort_values(by=metric_col_t, ascending=(not find_max), na_position='last')
             if sorted_df.empty: return {"error": f"No valid rows found after sorting by {metric_col_t}"}
             best_index = sorted_df.index[0]

        best_row = df.loc[best_index]
        best_metric_value = best_row[metric_col_t]

        hyperparameter_values = {}
        for hp_orig, hp_t in valid_hyperparams_map:
             hyperparameter_values[hp_orig] = best_row[hp_t]

        result_data = {
            f"best_{'max' if find_max else 'min'}_{original_metric}": best_metric_value,
            "hyperparameters": hyperparameter_values
        }
        final_result = {}
        for k, v in result_data.items():
             if k == "hyperparameters":
                  final_result[k] = {hp_name: _serialize_result(hp_val).get('result') for hp_name, hp_val in v.items()}
             else:
                  final_result[k] = _serialize_result(v).get('result')

        return {"result": final_result}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": f"Error finding best hyperparameters for metric '{original_metric}': {str(e)}"}


# --- Main Dispatcher (MODIFIED) ---
def query_dataframe(df: pd.DataFrame, query_params: Dict[str, Any]) -> Dict[str, Any]:
    """ Dispatcher function for DataFrame queries. Helpers handle translation.

    Acts as the main entry point for performing various analyses on the DataFrame.
    It takes a dictionary of query parameters, determines the `query_type`, and
    calls the appropriate helper function (e.g., `filter_dataframe`,
    `aggregate_dataframe`, `find_best_hyperparameters`). Column name translation
    is typically handled within the helper functions.

    Args:
        df: The input DataFrame to query.
        query_params: A dictionary containing the query parameters, including:
            - query_type (str): The type of operation to perform (e.g., "filter",
              "aggregate", "sort", "best_hyperparameters").
            - Other parameters specific to the query_type (e.g., "columns",
              "filters", "metric_column", "hyperparameter_columns"). See
              `get_tools_schema` for details.

    Returns:
        A dictionary containing the results of the query (usually serialized)
        or an error message.
    """
    print(f"Received query params: {query_params}")
    query_type = query_params.get("query_type")
    limit = query_params.get("limit") or query_params.get("n")
    try:
        if df is None or df.empty: return {"error": "DataFrame is not loaded or is empty."}

        if query_type == "info": return get_dataframe_info(df, columns=query_params.get("columns"))
        elif query_type == "describe": return describe_dataframe(df, columns=query_params.get("columns"), percentiles=query_params.get("percentiles"))
        elif query_type == "filter": return filter_dataframe(df, filters=query_params.get("filters"), filter_column=query_params.get("filter_column"), filter_operator=query_params.get("filter_operator"), filter_value=query_params.get("filter_value"))
        elif query_type in ["aggregate", "group"]:
             cols = query_params.get("columns"); func = query_params.get("function"); group_by = query_params.get("group_by")
             if not cols or not func: return {"error": "Aggregate requires 'columns' and 'function'."}
             return aggregate_dataframe(df, columns=cols, function=func, group_by=group_by)
        elif query_type == "sort":
            sort_by = query_params.get("sort_by"); ascending = query_params.get("ascending", True)
            if not sort_by: return {"error": "Sort requires 'sort_by'."}
            return sort_dataframe(df, sort_by=sort_by, ascending=ascending, limit=limit)
        elif query_type == "top_n":
            sort_by = query_params.get("sort_by"); ascending = query_params.get("ascending", False); n = limit
            if not sort_by or not isinstance(sort_by, str): return {"error": "Top N requires a single 'sort_by' string."}
            if not n: return {"error": "Top N requires 'limit' or 'n'."}
            return get_top_n_values(df, sort_by=sort_by, ascending=ascending, n=n)
        elif query_type == "value_counts":
             cols = query_params.get("columns")
             if not cols: return {"error": "Value Counts requires 'columns'."}
             return value_counts_dataframe(df, columns=cols, limit=limit)
        elif query_type == "correlation":
             cols = query_params.get("columns"); method = query_params.get("correlation_method", "pearson")
             return correlation_dataframe(df, columns=cols, method=method)
        elif query_type == "best_hyperparameters":
            metric_col = query_params.get("metric_column")
            hp_cols = query_params.get("hyperparameter_columns")
            find_max = query_params.get("find_max", True)
            if not metric_col or not hp_cols:
                return {"error": "'best_hyperparameters' query requires 'metric_column' and 'hyperparameter_columns'."}
            if not isinstance(hp_cols, list):
                 return {"error": "'hyperparameter_columns' must be a list of column names/shortcuts."}
            return find_best_hyperparameters(df, metric_column=metric_col, hyperparameter_columns=hp_cols, find_max=find_max)
        elif query_type == "general": return get_dataframe_info(df)
        else: print(f"Unknown query type '{query_type}'. Falling back to general info."); return get_dataframe_info(df)
    except Exception as e: import traceback; print(f"Error in query_dataframe dispatcher: {traceback.format_exc()}"); return {"error": f"Unexpected error dispatching query ({query_type}): {str(e)}"}


# --- Testing Block (MODIFIED) ---
if __name__ == "__main__":
    df_test = load_dummy_dataframe()
    if df_test is not None and not df_test.empty:
        print("\n--- Comprehensive Testing ---")
        f1_shortcut = "f1 score"; hallu_frac_shortcut = "hallucination rate"; model_shortcut = "model";
        latency_shortcut = "latency"; precision_shortcut = "precision"; status_shortcut = "status";
        display_name_shortcut = "display name"; recall_shortcut = "recall"; accuracy_shortcut="accuracy";
        lr_shortcut = "learning rate"; epochs_shortcut = "epochs"; warmup_shortcut = "warmup ratio"; scheduler_shortcut="scheduler type"

        f1_col = COLUMN_SHORTCUTS.get(f1_shortcut); model_col = COLUMN_SHORTCUTS.get(model_shortcut)
        latency_col = COLUMN_SHORTCUTS.get(latency_shortcut); precision_col = COLUMN_SHORTCUTS.get(precision_shortcut)
        status_col = COLUMN_SHORTCUTS.get(status_shortcut); display_name_col = COLUMN_SHORTCUTS.get(display_name_shortcut)
        recall_col = COLUMN_SHORTCUTS.get(recall_shortcut); accuracy_col = COLUMN_SHORTCUTS.get(accuracy_shortcut)
        lr_col = COLUMN_SHORTCUTS.get(lr_shortcut); epochs_col = COLUMN_SHORTCUTS.get(epochs_shortcut)
        warmup_col = COLUMN_SHORTCUTS.get(warmup_shortcut); scheduler_col = COLUMN_SHORTCUTS.get(scheduler_shortcut)
        full_col_name_example = "output.HalluScorerEvaluator.is_hallucination_ground_truth.true_count"
        null_check_col_shortcut = "summary.weave.feedback.wandb.reaction.1.payload.emoji"

        def check_cols(df, cols_to_check, check_numeric=False):
            """Helper function to check if required columns exist in the DataFrame for testing."""
            if cols_to_check is None: cols_to_check = []
            cols_to_check = [c for c in cols_to_check if c is not None]
            if not cols_to_check: return False
            missing = [c for c in cols_to_check if c not in df.columns]
            if missing: print(f"\n--- Check Failed: Cols not found: {missing} ---"); return False
            if check_numeric:
                 non_numeric = [c for c in cols_to_check if not pd.api.types.is_numeric_dtype(df[c])]
                 if non_numeric: print(f"\n--- Check Failed: Cols not numeric: {non_numeric} ---"); return False
            return True
        def run_test(test_num, description, params):
            """Helper function to run a query test and print results/errors."""
            print(f"\n{test_num}. Test {description}:")
            try:
                result = query_dataframe(df_test, params)
                print(json.dumps(result, indent=2, default=str))
                is_error_test = test_num > 30
                has_error = isinstance(result, dict) and "error" in result
                if is_error_test and not has_error: print(f"--- !!! TEST {test_num} FAILED: Expected error, got success. !!! ---")
                elif not is_error_test and has_error: print(f"--- !!! TEST {test_num} FAILED with UNEXPECTED error. !!! ---")
                elif is_error_test and has_error: print(f"--- TEST {test_num} CORRECTLY produced an error. ---")
            except Exception as e: print(f"[bold red]!!! TEST {test_num} PYTHON EXCEPTION: {e} !!![/bold red]"); import traceback; traceback.print_exc()

        if check_cols(df_test, [precision_col]): run_test(1, f"Describe ('{precision_shortcut}')", {"query_type": "describe", "columns": [precision_shortcut]})
        if check_cols(df_test, [COLUMN_SHORTCUTS.get(hallu_frac_shortcut), model_col]): run_test(2, f"Filter ('{hallu_frac_shortcut}' > 0.5 AND {model_shortcut} contains 'gpt')", {"query_type": "filter", "filters": [{"column": hallu_frac_shortcut, "operator": ">", "value": 0.5}, {"column": model_shortcut, "operator": "contains", "value": "gpt"}]})
        if check_cols(df_test, [latency_col, model_col]): run_test(3, f"Aggregate (Mean '{latency_shortcut}' by '{model_shortcut}')", {"query_type": "aggregate", "columns": [latency_shortcut], "function": "mean", "group_by": model_shortcut})
        if check_cols(df_test, [precision_col]): run_test(4, f"Top N Values (Top 3 by '{precision_shortcut}' desc)", {"query_type": "top_n", "sort_by": precision_shortcut, "ascending": False, "limit": 3})
        if check_cols(df_test, [model_col]): run_test(5, f"Value Counts ('{model_shortcut}')", {"query_type": "value_counts", "columns": model_shortcut})
        if check_cols(df_test, [latency_col]): run_test(6, f"Info ('{latency_shortcut}')", {"query_type": "info", "columns": [latency_shortcut]})
        if check_cols(df_test, [model_col]): run_test(7, f"Filter (Simple filter: '{model_shortcut}' == 'gpt-4o')", {"query_type": "filter", "filter_column": model_shortcut, "filter_operator": "==", "filter_value": "gpt-4o"})
        if check_cols(df_test, [f1_col, latency_col], check_numeric=True): run_test(8, f"Correlation ('{f1_shortcut}' and '{latency_shortcut}')", {"query_type": "correlation", "columns": [f1_shortcut, latency_shortcut]})
        else: print(f"\n8. Skipping Correlation test.")

        print("\n--- Additional Tests ---")
        if check_cols(df_test, [model_col]): run_test(9, f"Filter (!= '{model_shortcut}')", {"query_type": "filter", "filter_column": model_shortcut, "filter_operator": "!=", "filter_value": "gpt-4o"})
        if check_cols(df_test, [latency_col]): run_test(10, f"Filter ('{latency_shortcut}' < 100000)", {"query_type": "filter", "filter_column": latency_shortcut, "filter_operator": "<", "filter_value": 100000})
        if check_cols(df_test, [display_name_col]): run_test(11, f"Filter ('{display_name_shortcut}' startswith 'SmolLM2-360M-sft')", {"query_type": "filter", "filter_column": display_name_shortcut, "filter_operator": "startswith", "filter_value": "SmolLM2-360M-sft"})
        if check_cols(df_test, [f1_col]): run_test(12, f"Filter ('{f1_shortcut}' between 0.5 and 0.6)", {"query_type": "filter", "filter_column": f1_shortcut, "filter_operator": "between", "filter_value": [0.5, 0.6]})
        if check_cols(df_test, [model_col]): run_test(13, f"Filter ('{model_shortcut}' isin ['gpt-4o', 'HHEM_2-1'])", {"query_type": "filter", "filter_column": model_shortcut, "filter_operator": "isin", "filter_value": ["gpt-4o", "HHEM_2-1"]})
        null_col_full_name = translate_column_name(null_check_col_shortcut)
        if check_cols(df_test, [null_col_full_name]): run_test(14, f"Filter ('{null_check_col_shortcut}' isnull - using '{null_col_full_name}')", {"query_type": "filter", "filter_column": null_check_col_shortcut, "filter_operator": "isnull"})
        else: print(f"\n14. Skipping isnull test.")
        full_col_exists = full_col_name_example in df_test.columns
        if check_cols(df_test, [f1_col, status_col]) and full_col_exists: run_test(15, f"Filter (Complex Full Names: '{f1_col}' >= 0.6 AND '{status_col}' == 'success')", {"query_type": "filter", "filters": [{"column": f1_col, "operator": ">=", "value": 0.6}, {"column": status_col, "operator": "==", "value": "success"}]})
        else: print(f"\n15. Skipping complex full name filter test.")
        if check_cols(df_test, [latency_col, f1_col]): run_test(16, "Aggregate (Multiple funcs, no group)", {"query_type": "aggregate", "columns": [latency_shortcut, f1_shortcut], "function": ["mean", "std", "count"]})
        if check_cols(df_test, [COLUMN_SHORTCUTS.get(hallu_frac_shortcut)]): run_test(17, "Aggregate (Median hallucination rate)", {"query_type": "aggregate", "columns": hallu_frac_shortcut, "function": "median"})
        if check_cols(df_test, [model_col]): run_test(18, "Aggregate (nunique models)", {"query_type": "aggregate", "columns": model_shortcut, "function": "nunique"})
        if check_cols(df_test, [model_col, status_col, latency_col]): run_test(19, f"Aggregate (Group by '{model_shortcut}' and '{status_shortcut}')", {"query_type": "aggregate", "columns": latency_shortcut, "function": "mean", "group_by": [model_shortcut, status_shortcut]})
        else: print(f"\n19. Skipping multi-group test.")
        if check_cols(df_test, [model_col, latency_col]): run_test(20, f"Sort (by '{model_shortcut}' asc, '{latency_shortcut}' desc, limit 5)", {"query_type": "sort", "sort_by": [model_shortcut, latency_shortcut], "ascending": [True, False], "limit": 5})
        if check_cols(df_test, [model_col, status_col]): run_test(21, f"Value Counts (Multiple: '{model_shortcut}', '{status_shortcut}')", {"query_type": "value_counts", "columns": [model_shortcut, status_shortcut], "limit": 10})
        else: print(f"\n21. Skipping multi value_counts test.")
        cols_for_corr = [precision_col, recall_col, accuracy_col]
        if check_cols(df_test, cols_for_corr, check_numeric=True): run_test(22, "Correlation (Spearman: precision, recall, accuracy)", {"query_type": "correlation", "columns": ["precision", "recall", "accuracy"], "correlation_method": "spearman"})
        else: print(f"\n22. Skipping spearman correlation test.")
        if check_cols(df_test, [latency_col]): run_test(23, f"Describe ('{latency_shortcut}' with custom percentiles)", {"query_type": "describe", "columns": latency_shortcut, "percentiles": [0.1, 0.5, 0.9]})

        hp_cols_to_get = [lr_shortcut, epochs_shortcut, warmup_shortcut, scheduler_shortcut]
        if check_cols(df_test, [f1_col] + [COLUMN_SHORTCUTS.get(hp) for hp in hp_cols_to_get]):
            run_test(24, f"Best Hyperparameters (Max '{f1_shortcut}')", {
                "query_type": "best_hyperparameters",
                "metric_column": f1_shortcut,
                "hyperparameter_columns": hp_cols_to_get,
                "find_max": True
            })
            if check_cols(df_test, [latency_col] + [COLUMN_SHORTCUTS.get(hp) for hp in hp_cols_to_get]):
                 run_test(25, f"Best Hyperparameters (Min '{latency_shortcut}')", {
                     "query_type": "best_hyperparameters",
                     "metric_column": latency_shortcut,
                     "hyperparameter_columns": hp_cols_to_get,
                     "find_max": False
                 })
        else: print(f"\n24/25. Skipping best_hyperparameters test.")


        print("\n--- Error Handling Tests (Expect Errors Below) ---")
        run_test(31, "Error: Non-existent column filter", {"query_type": "filter", "filter_column": "non_existent_column_xyz", "filter_operator": "==", "filter_value": "test"})
        run_test(32, "Error: Invalid operator", {"query_type": "filter", "filter_column": model_shortcut, "filter_operator": "@@@", "filter_value": "test"})
        run_test(33, "Error: Type mismatch filter", {"query_type": "filter", "filter_column": f1_shortcut, "filter_operator": ">", "filter_value": "not_a_number"})
        run_test(34, "Error: Bad aggregation function", {"query_type": "aggregate", "columns": latency_shortcut, "function": "invalid_function"})
        run_test(35, "Error: Top N multiple columns", {"query_type": "top_n", "sort_by": [f1_shortcut, latency_shortcut], "ascending": False, "limit": 3})
        run_test(36, "Error: Top N non-existent column", {"query_type": "top_n", "sort_by": "non_existent_col", "ascending": False, "limit": 3})
        run_test(37, "Error: Correlation non-numeric column", {"query_type": "correlation", "columns": [model_shortcut, latency_shortcut]})
        run_test(38, "Error: Best HP missing metric", {"query_type": "best_hyperparameters", "hyperparameter_columns": [lr_shortcut]})
        run_test(39, "Error: Best HP missing hyperparams", {"query_type": "best_hyperparameters", "metric_column": f1_shortcut})
        run_test(40, "Error: Best HP bad metric col", {"query_type": "best_hyperparameters", "metric_column": "non_existent_col", "hyperparameter_columns": [lr_shortcut]})
        run_test(41, "Error: Best HP bad HP col", {"query_type": "best_hyperparameters", "metric_column": f1_shortcut, "hyperparameter_columns": ["non_existent_hp"]})

    else:
        print("DataFrame could not be loaded or is empty. Skipping tests.")