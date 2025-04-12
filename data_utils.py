# In data_utils.py

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Union, Optional, Any, Literal

# --- Shortcut Map and Translate Function (Keep as before) ---
COLUMN_SHORTCUTS = {
    # ... (keep your existing shortcuts) ...
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
}

def translate_column_name(name: str) -> str:
    """Translates a shortcut column name to its full name if found."""
    if not isinstance(name, str): return name
    shortcut = name.lower().strip()
    return COLUMN_SHORTCUTS.get(shortcut, name)

# --- Schema Function (Keep as before) ---
def get_tools_schema() -> Dict[str, Any]:
    """ Generate the tools schema... (Keep the corrected version without examples) """
    schema_parameters = {
        "type": "object",
        "properties": {
            "query_type": { "type": "string", "enum": [ "general", "info", "filter", "aggregate", "sort", "group", "top_n", "correlation", "describe", "compare", "time_series", "value_counts" ], "description": "Type of query." },
            "columns": { "type": ["string", "array"], "items": {"type": "string"}, "description": "Column(s) or shortcut(s)." },
            "column": { "type": "string", "description": "Alias for single column/shortcut." },
            "filters": { "type": "array", "items": { "type": "object", "properties": { "column": {"type": "string", "description": "Column/shortcut."}, "operator": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "contains", "startswith", "endswith", "between", "isin", "isnull", "notnull"], "description": "Operator."}, "value": { "type": ["string", "number", "boolean", "array"], "items": { "type": ["string", "number", "boolean"] }, "description": "Value(s)." } }, "required": ["column", "operator"] }, "description": "Filters (AND)." },
            "filter_column": {"type": "string", "description": "Column/shortcut for simple filter."},
            "filter_operator": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "contains", "startswith", "endswith", "between", "isin", "isnull", "notnull"], "description": "Operator."},
            "filter_value": { "type": ["string", "number", "boolean", "array"], "items": { "type": ["string", "number", "boolean"] }, "description": "Value."},
            "function": {"type": ["string", "array"], "items": {"type": "string"}, "enum": ["mean", "median", "sum", "min", "max", "count", "std", "var", "first", "last", "nunique"], "description": "Aggregation func(s)."},
            "group_by": {"type": ["string", "array"], "items": {"type": "string"}, "description": "Column(s)/shortcut(s) to group by."},
            "sort_by": {"type": ["string", "array"], "items": {"type": "string"}, "description": "Column name or shortcut for sorting (top_n expects a single column)."}, # Modified description
            "ascending": {"type": ["boolean", "array"], "items": {"type": "boolean"}, "default": True, "description": "Sort direction(s)."},
            "limit": {"type": "integer", "minimum": 1, "description": "Max rows/values."}, # Modified description
            "n": {"type": "integer", "minimum": 1, "description": "Alias for 'limit'."},
            "percentiles": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "description": "Percentiles for 'describe'."},
            "correlation_method": {"type": "string", "enum": ["pearson", "kendall", "spearman"], "default": "pearson", "description": "Correlation method."}
        },
        "required": ["query_type"]
    }
    return { "type": "function", "name": "query_dataframe", "description": "Query/analyze DataFrame. Use full names OR shortcuts like 'f1 score', 'latency', 'model name'.", "parameters": schema_parameters }


# --- Load DF, Serialize, Info, Describe, Filter, Aggregate, Sort, Value Counts, Correlation (Keep as corrected before) ---
# (Including _serialize_result, value_counts_dataframe, correlation_dataframe, filter_dataframe etc. from previous corrected version)
def load_dummy_dataframe(file_path=None):
    """Load a DataFrame from a specified path or create a fallback dummy."""
    default_path = r"C:\Users\ragha\OneDrive\Documents\wandb\weave_export_hallucination_2025-04-11.csv" # CHECK/CHANGE PATH
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
    """Serialize results, handles Series with index correctly now."""
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
            # Use to_dict to preserve index (good for value_counts, maybe less ideal for simple top_n values?)
            # Let's return list of values for top_n, dict otherwise
            # Add a flag or check name? Let's handle in get_top_n_values
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
    """Get basic info or info for specific (translated) columns."""
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
    """Get descriptive statistics. Handles own serialization."""
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
        # Serialize using index orient
        serializable_desc = description.replace([np.inf, -np.inf], [float('inf'), float('-inf')]).fillna("NaN").to_dict(orient="index")
        return {"description": serializable_desc}
    except Exception as e: return {"error": f"Error describing DataFrame: {str(e)}"}

def filter_dataframe(df: pd.DataFrame, filters: Optional[List[Dict[str, Any]]] = None, filter_column: Optional[str] = None, filter_operator: Optional[str] = None, filter_value: Any = None) -> Dict[str, Any]:
    """Filter the DataFrame. Includes pandas version compatibility for bools."""
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
                # Type conversion
                if op not in ['isnull', 'notnull', 'isin'] and val is not None:
                     if isinstance(val, list) and op == 'between':
                           val = [pd.to_numeric(v, errors='coerce') if pd.api.types.is_numeric_dtype(series.dtype) else pd.to_datetime(v, errors='coerce') if pd.api.types.is_datetime64_any_dtype(series.dtype) else v for v in val]
                           if pd.isna(val[0]) or pd.isna(val[1]): raise ValueError(f"Bad 'between' vals for {col_orig}")
                     elif not isinstance(val, list):
                          if pd.api.types.is_numeric_dtype(series.dtype): val = pd.to_numeric(val, errors='coerce')
                          elif pd.api.types.is_datetime64_any_dtype(series.dtype): val = pd.to_datetime(val, errors='coerce')
                          # Compatibility boolean check
                          elif series.dtype == object or str(series.dtype).startswith('string'): # Check if potentially boolean-like object/string
                              if isinstance(val, str):
                                   lv = val.lower()
                                   if lv in ['true', '1', 'yes']: val = True
                                   elif lv in ['false', '0', 'no']: val = False # Keep as string if not clearly boolean
                              else: 
                                  try:
                                     val = bool(val) 
                                  except: 
                                      pass # Try direct conversion
                          if pd.isna(val) and str(f_orig.get('value')).lower() != 'nan': raise ValueError(f"Bad conversion for {col_orig}")
                # Apply Operators
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

# --- aggregate_dataframe, sort_dataframe, value_counts_dataframe, correlation_dataframe (Keep as corrected before) ---
def aggregate_dataframe(df: pd.DataFrame, columns: Union[str, List[str]], function: Union[str, List[str]], group_by: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
    """Aggregate data in the DataFrame, optionally grouping first."""
    if df.empty: return {"error": "Cannot aggregate empty DataFrame."}
    if isinstance(columns, str): columns = [columns]; original_columns = columns
    if isinstance(function, str): function = [function]
    columns_translated = [translate_column_name(col) for col in columns]
    valid_columns = [col for col in columns_translated if col in df.columns]
    if not valid_columns: return {"error": f"Agg columns (translated) not found: {columns_translated}"}
    if len(valid_columns) < len(columns): print(f"Warning: Agg columns not found (input: {original_columns}, translated: {columns_translated})")
    columns = valid_columns; valid_group_by = None
    if group_by:
        if isinstance(group_by, str): group_by = [group_by]; original_group_by = group_by
        group_by_translated = [translate_column_name(gb_col) for gb_col in group_by]
        valid_group_by = [gb_col for gb_col in group_by_translated if gb_col in df.columns]
        if not valid_group_by: return {"error": f"Group by columns (translated) not found: {group_by_translated}"}
        if len(valid_group_by) < len(group_by): print(f"Warning: Group by columns not found (input: {original_group_by}, translated: {group_by_translated})")
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
                      else: results[col_func_name] = f"Unsupported func '{func}'"
             result = pd.DataFrame([results])
        return _serialize_result(result)
    except Exception as e: return {"error": f"Error during aggregation: {str(e)}"}

def sort_dataframe(df: pd.DataFrame, sort_by: Union[str, List[str]], ascending: Union[bool, List[bool]] = True, limit: Optional[int] = None) -> Dict[str, Any]:
    """Sort the DataFrame."""
    if df.empty: return {"result": [], "count": 0, "note": "Input DataFrame was empty."}
    if isinstance(sort_by, str): sort_by = [sort_by]; original_sort_by = sort_by
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
     """Get value counts for specified column(s)."""
     if df.empty: return {"error": "Cannot get value counts from empty DataFrame."}
     if isinstance(columns, str): columns = [columns]; original_columns = columns
     results = {}
     try:
         columns_translated = [translate_column_name(col) for col in columns]
         for i, col_t in enumerate(columns_translated):
             col_orig = original_columns[i]
             if col_t not in df.columns: results[col_orig] = {"error": f"Column '{col_orig}' (->'{col_t}') not found"}
             else: counts = df[col_t].value_counts(dropna=False); results[col_orig] = _serialize_result(counts, limit=limit)
         return results[original_columns[0]] if len(original_columns) == 1 else {"value_counts": results}
     except Exception as e: return {"error": f"Error getting value counts: {str(e)}"}

def correlation_dataframe(df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'pearson') -> Dict[str, Any]:
     """Calculate pairwise correlation of columns."""
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
         # Describe returns a DataFrame, serialize it appropriately
         return {"correlation_matrix": _serialize_result(correlation_matrix).get('result')} # Use default orient=records for matrix
     except Exception as e: return {"error": f"Error calculating correlation: {str(e)}"}


# --- 1. New Helper Function for Top N Values ---
def get_top_n_values(df: pd.DataFrame, sort_by: str, ascending: bool, n: int) -> Dict[str, Any]:
    """Gets the top N values from a single column after sorting."""
    if df.empty: return {"error": "Cannot get top N from empty DataFrame."}

    # Translate the column name
    original_sort_by = sort_by
    sort_by_col = translate_column_name(sort_by)

    if sort_by_col not in df.columns:
        return {"error": f"Column '{original_sort_by}' (translated to '{sort_by_col}') not found for sorting."}

    try:
        series = df[sort_by_col]
        # Ensure column is sortable (e.g., numeric or string)
        if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_string_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
             # Try converting to numeric if object type
             if series.dtype == 'object':
                  series = pd.to_numeric(series, errors='coerce')
                  if series.isnull().all(): # If conversion failed for all
                       return {"error": f"Column '{sort_by_col}' is not numeric or string type and could not be converted for sorting."}
             else:
                  return {"error": f"Column '{sort_by_col}' is not numeric or string type, cannot guarantee correct sorting order."}

        # Sort the series and get the top N
        sorted_series = series.sort_values(ascending=ascending, na_position='last')
        top_n_series = sorted_series.head(n)

        # Serialize just the values as a list
        result_list = top_n_series.replace([np.inf, -np.inf], [float('inf'), float('-inf')]).fillna("NaN").tolist()

        return {"result": result_list, "count": len(result_list), "note": f"Top {n} values from '{sort_by_col}'"}

    except Exception as e:
        return {"error": f"Error getting top N values for column '{sort_by_col}': {str(e)}"}


# --- Main Dispatcher ---
def query_dataframe(df: pd.DataFrame, query_params: Dict[str, Any]) -> Dict[str, Any]:
    """ Dispatcher function for DataFrame queries. Helpers handle translation. """
    print(f"Received query params: {query_params}") # Debugging original params
    query_type = query_params.get("query_type")
    limit = query_params.get("limit") or query_params.get("n")

    try:
        if df is None or df.empty: return {"error": "DataFrame is not loaded or is empty."}

        if query_type == "info": return get_dataframe_info(df, columns=query_params.get("columns"))
        elif query_type == "describe": return describe_dataframe(df, columns=query_params.get("columns"), percentiles=query_params.get("percentiles"))
        elif query_type == "filter": return filter_dataframe(df, filters=query_params.get("filters"), filter_column=query_params.get("filter_column"), filter_operator=query_params.get("filter_operator"), filter_value=query_params.get("filter_value"))
        elif query_type in ["aggregate", "group"]:
             cols = query_params.get("columns"); func = query_params.get("function"); group_by = query_params.get("group_by")
             if not cols or not func: return {"error": "Aggregate query requires 'columns' and 'function'."}
             return aggregate_dataframe(df, columns=cols, function=func, group_by=group_by)
        elif query_type == "sort":
            sort_by = query_params.get("sort_by"); ascending = query_params.get("ascending", True)
            if not sort_by: return {"error": "Sort query requires 'sort_by'."}
            return sort_dataframe(df, sort_by=sort_by, ascending=ascending, limit=limit)
        # --- 2. Call New Helper for top_n ---
        elif query_type == "top_n":
            sort_by = query_params.get("sort_by")
            ascending = query_params.get("ascending", False) # Default false for top_n
            n = limit
            # top_n expects a single column to sort by to return only values
            if not sort_by or not isinstance(sort_by, str): return {"error": "Top N query requires a single 'sort_by' column string."}
            if not n: return {"error": "Top N query requires 'limit' or 'n'."}
            return get_top_n_values(df, sort_by=sort_by, ascending=ascending, n=n) # Call new function
        # --- End Change ---
        elif query_type == "value_counts":
             cols = query_params.get("columns")
             if not cols: return {"error": "Value Counts query requires 'columns'."}
             return value_counts_dataframe(df, columns=cols, limit=limit)
        elif query_type == "correlation":
             cols = query_params.get("columns"); method = query_params.get("correlation_method", "pearson")
             return correlation_dataframe(df, columns=cols, method=method)
        elif query_type == "general": return get_dataframe_info(df)
        else: print(f"Unknown query type '{query_type}'. Falling back to general info."); return get_dataframe_info(df)
    except Exception as e: import traceback; print(f"Error: {traceback.format_exc()}"); return {"error": f"Unexpected error ({query_type}): {str(e)}"}


# --- Expanded Testing Block ---
if __name__ == "__main__":
    df_test = load_dummy_dataframe() # Make sure path is correct!
    if df_test is not None and not df_test.empty:
        print("\n--- Comprehensive Testing ---")

        # Define shortcuts & columns used in tests
        # Add more real columns from your data if needed for better testing
        f1_shortcut = "f1 score"; hallu_frac_shortcut = "hallucination rate"; model_shortcut = "model";
        latency_shortcut = "latency"; precision_shortcut = "precision"; status_shortcut = "status";
        display_name_shortcut = "display name"; recall_shortcut = "recall"; accuracy_shortcut="accuracy" # Added recall/accuracy

        # Get actual column names for checks - uses .get() to avoid KeyError if shortcut missing
        f1_col = COLUMN_SHORTCUTS.get(f1_shortcut)
        model_col = COLUMN_SHORTCUTS.get(model_shortcut)
        latency_col = COLUMN_SHORTCUTS.get(latency_shortcut)
        precision_col = COLUMN_SHORTCUTS.get(precision_shortcut)
        status_col = COLUMN_SHORTCUTS.get(status_shortcut)
        display_name_col = COLUMN_SHORTCUTS.get(display_name_shortcut)
        recall_col = COLUMN_SHORTCUTS.get(recall_shortcut)
        accuracy_col = COLUMN_SHORTCUTS.get(accuracy_shortcut)
        # Add a known full column name that isn't a shortcut
        # Choose one that exists in your actual data
        full_col_name_example = "output.HalluScorerEvaluator.is_hallucination_ground_truth.true_count"
        # Example column that might contain nulls
        null_check_col_shortcut = "summary.weave.feedback.wandb.reaction.1.payload.emoji" # Change if needed


        # *** ADDED check_cols definition HERE ***
        def check_cols(df, cols_to_check, check_numeric=False):
            """Checks if columns exist and optionally if they are numeric."""
            # Ensure cols_to_check is a list, even if None was passed somehow
            if cols_to_check is None: cols_to_check = []
            # Filter out None values that might result from .get() if shortcut missing
            cols_to_check = [c for c in cols_to_check if c is not None]
            if not cols_to_check: return False # No valid columns to check

            missing = [c for c in cols_to_check if c not in df.columns]
            if missing:
                 print(f"\n--- Check Failed: Test columns not found: {missing} ---")
                 return False
            if check_numeric:
                 non_numeric = [c for c in cols_to_check if not pd.api.types.is_numeric_dtype(df[c])]
                 if non_numeric:
                      print(f"\n--- Check Failed: Test columns not numeric: {non_numeric} ---")
                      return False
            return True
        # **************************************

        # Helper to run and print test
        def run_test(test_num, description, params):
            print(f"\n{test_num}. Test {description}:")
            # Use try-except to catch errors during the query itself for error tests
            try:
                result = query_dataframe(df_test, params)
                # Pretty print the result
                print(json.dumps(result, indent=2, default=str))
                # Add checks for expected outcomes
                is_error_test = test_num > 23 # Assuming tests 24+ are error tests
                has_error = isinstance(result, dict) and "error" in result

                if is_error_test and not has_error:
                    print(f"--- !!! TEST {test_num} FAILED: Expected an error, but got success. !!! ---")
                elif not is_error_test and has_error:
                    print(f"--- !!! TEST {test_num} FAILED with UNEXPECTED error. !!! ---")
                elif is_error_test and has_error:
                    print(f"--- TEST {test_num} CORRECTLY produced an error. ---")
                # else: Test passed (either success or expected error)

            except Exception as e:
                 print(f"[bold red]!!! TEST {test_num} FAILED WITH PYTHON EXCEPTION: {e} !!![/bold red]")
                 import traceback
                 traceback.print_exc()


        # --- Existing & New Tests (Using run_test and check_cols) ---

        # Test 1: Describe
        if check_cols(df_test, [precision_col]): run_test(1, f"Describe ('{precision_shortcut}')", {"query_type": "describe", "columns": [precision_shortcut]})
        # Test 2: Filter Complex
        if check_cols(df_test, [COLUMN_SHORTCUTS.get(hallu_frac_shortcut), model_col]): run_test(2, f"Filter ('{hallu_frac_shortcut}' > 0.5 AND {model_shortcut} contains 'gpt')", {"query_type": "filter", "filters": [{"column": hallu_frac_shortcut, "operator": ">", "value": 0.5}, {"column": model_shortcut, "operator": "contains", "value": "gpt"}]})
        # Test 3: Aggregate Grouped
        if check_cols(df_test, [latency_col, model_col]): run_test(3, f"Aggregate (Mean '{latency_shortcut}' by '{model_shortcut}')", {"query_type": "aggregate", "columns": [latency_shortcut], "function": "mean", "group_by": model_shortcut})
        # Test 4: Top N Values
        if check_cols(df_test, [precision_col]): run_test(4, f"Top N Values (Top 3 by '{precision_shortcut}' desc)", {"query_type": "top_n", "sort_by": precision_shortcut, "ascending": False, "limit": 3})
        # Test 5: Value Counts
        if check_cols(df_test, [model_col]): run_test(5, f"Value Counts ('{model_shortcut}')", {"query_type": "value_counts", "columns": model_shortcut})
        # Test 6: Info Specific Col
        if check_cols(df_test, [latency_col]): run_test(6, f"Info ('{latency_shortcut}')", {"query_type": "info", "columns": [latency_shortcut]})
        # Test 7: Filter Simple
        if check_cols(df_test, [model_col]): run_test(7, f"Filter (Simple filter: '{model_shortcut}' == 'gpt-4o')", {"query_type": "filter", "filter_column": model_shortcut, "filter_operator": "==", "filter_value": "gpt-4o"})
        # Test 8: Correlation
        if check_cols(df_test, [f1_col, latency_col], check_numeric=True): run_test(8, f"Correlation ('{f1_shortcut}' and '{latency_shortcut}')", {"query_type": "correlation", "columns": [f1_shortcut, latency_shortcut]})
        else: print(f"\n8. Skipping Correlation test.")

        print("\n--- Additional Tests ---")

        # Test 9: Filter !=
        if check_cols(df_test, [model_col]): run_test(9, f"Filter (!= '{model_shortcut}')", {"query_type": "filter", "filter_column": model_shortcut, "filter_operator": "!=", "filter_value": "gpt-4o"})
        # Test 10: Filter <
        if check_cols(df_test, [latency_col]): run_test(10, f"Filter ('{latency_shortcut}' < 100000)", {"query_type": "filter", "filter_column": latency_shortcut, "filter_operator": "<", "filter_value": 100000})
        # Test 11: Filter startswith
        if check_cols(df_test, [display_name_col]): run_test(11, f"Filter ('{display_name_shortcut}' startswith 'SmolLM2-360M-sft')", {"query_type": "filter", "filter_column": display_name_shortcut, "filter_operator": "startswith", "filter_value": "SmolLM2-360M-sft"})
        # Test 12: Filter between
        if check_cols(df_test, [f1_col]): run_test(12, f"Filter ('{f1_shortcut}' between 0.5 and 0.6)", {"query_type": "filter", "filter_column": f1_shortcut, "filter_operator": "between", "filter_value": [0.5, 0.6]})
        # Test 13: Filter isin
        if check_cols(df_test, [model_col]): run_test(13, f"Filter ('{model_shortcut}' isin ['gpt-4o', 'HHEM_2-1'])", {"query_type": "filter", "filter_column": model_shortcut, "filter_operator": "isin", "filter_value": ["gpt-4o", "HHEM_2-1"]})
        # Test 14: Filter isnull
        null_col_full_name = translate_column_name(null_check_col_shortcut)
        if check_cols(df_test, [null_col_full_name]): run_test(14, f"Filter ('{null_check_col_shortcut}' isnull - using '{null_col_full_name}')", {"query_type": "filter", "filter_column": null_check_col_shortcut, "filter_operator": "isnull"})
        else: print(f"\n14. Skipping isnull test: Column '{null_check_col_shortcut}' (-> {null_col_full_name}) not found.")
        # Test 15: Filter Complex Full Names
        if check_cols(df_test, [f1_col, status_col]): run_test(15, f"Filter (Complex Full Names: '{f1_col}' >= 0.6 AND '{status_col}' == 'success')", {"query_type": "filter", "filters": [{"column": f1_col, "operator": ">=", "value": 0.6}, {"column": status_col, "operator": "==", "value": "success"}]})
        else: print(f"\n15. Skipping complex full name filter test.")

        # Test 16: Aggregate Multiple Funcs No Group
        if check_cols(df_test, [latency_col, f1_col]): run_test(16, "Aggregate (Multiple funcs, no group)", {"query_type": "aggregate", "columns": [latency_shortcut, f1_shortcut], "function": ["mean", "std", "count"]})
        # Test 17: Aggregate Median
        if check_cols(df_test, [COLUMN_SHORTCUTS.get(hallu_frac_shortcut)]): run_test(17, "Aggregate (Median hallucination rate)", {"query_type": "aggregate", "columns": hallu_frac_shortcut, "function": "median"})
        # Test 18: Aggregate nunique
        if check_cols(df_test, [model_col]): run_test(18, "Aggregate (nunique models)", {"query_type": "aggregate", "columns": model_shortcut, "function": "nunique"})
        # Test 19: Aggregate Multi-Group
        if check_cols(df_test, [model_col, status_col, latency_col]): run_test(19, f"Aggregate (Group by '{model_shortcut}' and '{status_shortcut}')", {"query_type": "aggregate", "columns": latency_shortcut, "function": "mean", "group_by": [model_shortcut, status_shortcut]})
        else: print(f"\n19. Skipping multi-group test.")

        # Test 20: Sort Multiple Cols
        if check_cols(df_test, [model_col, latency_col]): run_test(20, f"Sort (by '{model_shortcut}' asc, '{latency_shortcut}' desc, limit 5)", {"query_type": "sort", "sort_by": [model_shortcut, latency_shortcut], "ascending": [True, False], "limit": 5})

        # Test 21: Value Counts Multiple Cols
        if check_cols(df_test, [model_col, status_col]): run_test(21, f"Value Counts (Multiple: '{model_shortcut}', '{status_shortcut}')", {"query_type": "value_counts", "columns": [model_shortcut, status_shortcut], "limit": 10})
        else: print(f"\n21. Skipping multi value_counts test.")

        # Test 22: Correlation Spearman
        cols_for_corr = [precision_col, recall_col, accuracy_col]
        if check_cols(df_test, cols_for_corr, check_numeric=True): run_test(22, "Correlation (Spearman: precision, recall, accuracy)", {"query_type": "correlation", "columns": ["precision", "recall", "accuracy"], "correlation_method": "spearman"})
        else: print(f"\n22. Skipping spearman correlation test.")

        # Test 23: Describe Custom Percentiles
        if check_cols(df_test, [latency_col]): run_test(23, f"Describe ('{latency_shortcut}' with custom percentiles)", {"query_type": "describe", "columns": latency_shortcut, "percentiles": [0.1, 0.5, 0.9]})

        # Error Handling Tests
        print("\n--- Error Handling Tests (Expect Errors Below) ---")
        run_test(24, "Error: Non-existent column filter", {"query_type": "filter", "filter_column": "non_existent_column_xyz", "filter_operator": "==", "filter_value": "test"})
        run_test(25, "Error: Invalid operator", {"query_type": "filter", "filter_column": model_shortcut, "filter_operator": "@@@", "filter_value": "test"})
        run_test(26, "Error: Type mismatch filter", {"query_type": "filter", "filter_column": f1_shortcut, "filter_operator": ">", "filter_value": "not_a_number"})
        run_test(27, "Error: Bad aggregation function", {"query_type": "aggregate", "columns": latency_shortcut, "function": "invalid_function"})
        run_test(28, "Error: Top N multiple columns", {"query_type": "top_n", "sort_by": [f1_shortcut, latency_shortcut], "ascending": False, "limit": 3})
        run_test(29, "Error: Top N non-existent column", {"query_type": "top_n", "sort_by": "non_existent_col", "ascending": False, "limit": 3})
        run_test(30, "Error: Correlation non-numeric column", {"query_type": "correlation", "columns": [model_shortcut, latency_shortcut]}) # Assuming model name isn't numeric

    else:
        print("DataFrame could not be loaded or is empty. Skipping tests.")