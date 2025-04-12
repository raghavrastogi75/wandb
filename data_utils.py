# In data_utils.py

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Union, Optional, Any, Literal

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
            "sort_by": {"type": ["string", "array"], "items": {"type": "string"}, "description": "Column(s)/shortcut(s) to sort by."},
            "ascending": {"type": ["boolean", "array"], "items": {"type": "boolean"}, "default": True, "description": "Sort direction(s)."},
            "limit": {"type": "integer", "minimum": 1, "description": "Max rows."},
            "n": {"type": "integer", "minimum": 1, "description": "Alias for 'limit'."},
            "percentiles": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "description": "Percentiles for 'describe'."},
            "correlation_method": {"type": "string", "enum": ["pearson", "kendall", "spearman"], "default": "pearson", "description": "Correlation method."}
        },
        "required": ["query_type"]
    }
    return { "type": "function", "name": "query_dataframe", "description": "Query/analyze DataFrame. Use full names OR shortcuts like 'f1 score', 'latency', 'model name'.", "parameters": schema_parameters }


# --- Load DF (Keep as before) ---
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
                try:
                    converted_col = pd.to_numeric(df[col], errors='coerce')
                    if not converted_col.isnull().all(): df[col] = converted_col
                except (ValueError, TypeError): pass
        print("\nDataFrame Info after loading:")
        df.info(verbose=False)
        return df
    except FileNotFoundError: print(f"Warning: File not found at {load_path}. Creating dummy data.")
    except Exception as e: print(f"Error loading DataFrame from {load_path}: {e}. Creating dummy data.")
    print("Using simple fallback dummy data.")
    return pd.DataFrame({ 'attributes.model_name': ['M1', 'M2', 'M1'], 'output.HalluScorerEvaluator.scorer_evaluation_metrics.f1': [0.5, 0.6, 0.7], 'summary.weave.latency_ms': [100, 200, 150] })


# --- 3. Corrected Serialization ---
def _serialize_result(data: Any, limit: Optional[int] = 20) -> Dict[str, Any]:
    """Safely serialize DataFrame/Series results to JSON-compatible dict, applying limit."""
    note = ""
    original_count = None
    try:
        if isinstance(data, pd.DataFrame):
            original_count = len(data)
            res_df = data.head(limit) if limit is not None and len(data) > limit else data
            if limit is not None and len(data) > limit: note = f" (showing first {limit} of {len(data)} rows)"
            # Use infer_objects to potentially avoid downcasting warnings/issues before replace
            res_df = res_df.infer_objects(copy=False).replace([np.inf, -np.inf], [float('inf'), float('-inf')]).fillna("NaN")
            # IMPORTANT: Use orient='index' or 'dict' for describe(), 'records' otherwise?
            # For simplicity, let's use 'records' as default for DataFrames here.
            # Describe function will handle its own serialization now.
            return {"result": res_df.to_dict(orient="records"), "count": original_count, "note": note.strip()}
        elif isinstance(data, pd.Series):
            original_count = len(data)
            res_series = data.head(limit) if limit is not None and len(data) > limit else data
            if limit is not None and len(data) > limit: note = f" (showing first {limit} of {len(data)} items)"
            res_series = res_series.replace([np.inf, -np.inf], [float('inf'), float('-inf')]).fillna("NaN")
            # *** FIX: Always use to_dict() for Series to preserve index labels ***
            # This works well for value_counts()
            return {"result": res_series.to_dict(), "count": original_count, "note": note.strip()}
        # (Keep remaining type handling as before)
        elif isinstance(data, (int, float, str, bool, dict, list, type(None))):
             count = len(data) if isinstance(data, (list, dict)) else None; return {"result": data, "count": count}
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


# --- Info Function (Keep as before, uses translate) ---
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

# --- 4. Corrected Describe Function ---
def describe_dataframe(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None, percentiles: Optional[List[float]] = None) -> Dict[str, Any]:
    """Get descriptive statistics. Handles own serialization for correct format."""
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
        if description.empty and not target_df.select_dtypes(include=np.number).empty:
             description = target_df.describe(percentiles=percentiles, include='all')
        elif description.empty: return {"result": "No numeric columns found/specified to describe."}

        # *** FIX: Serialize describe() output correctly using to_dict() ***
        # Use orient='index' or 'dict' - 'index' groups by statistic, 'dict' groups by column
        serializable_desc = description.replace([np.inf, -np.inf], [float('inf'), float('-inf')]).fillna("NaN").to_dict(orient="index")
        return {"description": serializable_desc}
    except Exception as e: return {"error": f"Error describing DataFrame: {str(e)}"}

# --- 5. Corrected Filter Function ---
def filter_dataframe(df: pd.DataFrame, filters: Optional[List[Dict[str, Any]]] = None,
                      filter_column: Optional[str] = None, filter_operator: Optional[str] = None, filter_value: Any = None) -> Dict[str, Any]:
    """Filter the DataFrame based on simple or complex criteria (using AND logic)."""
    if df.empty: return {"result": [], "count": 0, "note": "Input DataFrame was empty."}
    filtered_df = df.copy()
    conditions = []
    active_filters = filters
    if not filters and filter_column and filter_operator:
         translated_filter_column = translate_column_name(filter_column)
         if translated_filter_column not in df.columns: return {"error": f"Filter column '{filter_column}' (->'{translated_filter_column}') not found."}
         active_filters = [{"column": translated_filter_column, "operator": filter_operator, "value": filter_value}]

    if active_filters:
        for f_orig in active_filters:
            f = f_orig.copy()
            if 'column' not in f: return {"error": f"Filter condition missing 'column': {f_orig}"}
            original_col_name = f['column']
            f['column'] = translate_column_name(original_col_name)
            col = f['column']
            op = f.get("operator")
            val = f.get("value")
            if not op: return {"error": f"Filter condition missing 'operator': {f_orig}"}
            if col not in filtered_df.columns: return {"error": f"Filter column '{original_col_name}' (->'{col}') not found."}
            try:
                series = filtered_df[col]
                if op not in ['isnull', 'notnull', 'isin'] and val is not None:
                     if isinstance(val, list) and op == 'between':
                           new_val = []
                           for item in val:
                                if pd.api.types.is_numeric_dtype(series.dtype): item = pd.to_numeric(item, errors='coerce')
                                elif pd.api.types.is_datetime64_any_dtype(series.dtype): item = pd.to_datetime(item, errors='coerce')
                                new_val.append(item)
                           val = new_val
                           if pd.isna(val[0]) or pd.isna(val[1]): raise ValueError(f"Could not convert 'between' values '{f_orig.get('value')}' for col '{original_col_name}'")
                     elif not isinstance(val, list):
                          if pd.api.types.is_numeric_dtype(series.dtype): val = pd.to_numeric(val, errors='coerce')
                          elif pd.api.types.is_datetime64_any_dtype(series.dtype): val = pd.to_datetime(val, errors='coerce')
                          # *** FIX: Compatibility for older pandas boolean check ***
                          elif series.dtype == object or series.dtype == 'string': # Check if potentially boolean-like object/string
                              if isinstance(val, str):
                                   lower_val = val.lower()
                                   if lower_val in ['true', '1', 'yes']: val = True
                                   elif lower_val in ['false', '0', 'no']: val = False
                                   # else: keep as string if not clearly boolean
                              else: # Try direct bool conversion for non-string types
                                   try: val = bool(val)
                                   except: pass # Keep original value if bool conversion fails
                          # Check if conversion resulted in NaN when original wasn't 'nan'
                          if pd.isna(val) and str(f_orig.get('value')).lower() != 'nan':
                              raise ValueError(f"Could not convert filter value '{f_orig.get('value')}' for column '{original_col_name}'")

                # (Apply Operators logic remains the same)
                if op == "==": conditions.append(series == val)
                elif op == "!=": conditions.append(series != val)
                elif op == ">": conditions.append(series > val)
                elif op == ">=": conditions.append(series >= val)
                elif op == "<": conditions.append(series < val)
                elif op == "<=": conditions.append(series <= val)
                elif op == "contains": conditions.append(series.astype(str).str.contains(str(val), na=False))
                elif op == "startswith": conditions.append(series.astype(str).str.startswith(str(val), na=False))
                elif op == "endswith": conditions.append(series.astype(str).str.endswith(str(val), na=False))
                elif op == "isin":
                    if not isinstance(val, list): return {"error": f"'isin' needs list value. Got: {type(val)}"}
                    conditions.append(series.isin(val))
                elif op == "between":
                    if not isinstance(val, list) or len(val) != 2: return {"error": f"'between' needs list [min, max]. Got: {val}"}
                    conditions.append(series.between(val[0], val[1]))
                elif op == "isnull": conditions.append(series.isnull())
                elif op == "notnull": conditions.append(series.notnull())
                else: return {"error": f"Unsupported filter operator: {op}"}
            except Exception as e: return {"error": f"Error applying filter ({original_col_name} {op} {f_orig.get('value')} -> {col} {op} {val}): {str(e)}"}

    if conditions:
        final_condition = conditions[0]
        for i in range(1, len(conditions)): final_condition &= conditions[i]
        filtered_df = filtered_df.loc[final_condition]

    # Use _serialize_result, which should handle DataFrames correctly now
    return _serialize_result(filtered_df)

# --- Other Helper Functions (aggregate, sort, value_counts, correlation - Keep as before, they use translate internally) ---
def aggregate_dataframe(df: pd.DataFrame, columns: Union[str, List[str]], function: Union[str, List[str]], group_by: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
    """Aggregate data in the DataFrame, optionally grouping first."""
    if df.empty: return {"error": "Cannot aggregate empty DataFrame."}
    if isinstance(columns, str): columns = [columns]
    if isinstance(function, str): function = [function]
    original_columns = columns
    columns_translated = [translate_column_name(col) for col in columns]
    valid_columns = [col for col in columns_translated if col in df.columns]
    if not valid_columns: return {"error": f"Agg columns (translated) not found: {columns_translated}"}
    if len(valid_columns) < len(columns): print(f"Warning: Agg columns not found (input: {original_columns}, translated: {columns_translated})")
    columns = valid_columns
    group_by_translated = None; valid_group_by = None
    if group_by:
        original_group_by = group_by
        if isinstance(group_by, str): group_by = [group_by]
        group_by_translated = [translate_column_name(gb_col) for gb_col in group_by]
        valid_group_by = [gb_col for gb_col in group_by_translated if gb_col in df.columns]
        if not valid_group_by: return {"error": f"Group by columns (translated) not found: {group_by_translated}"}
        if len(valid_group_by) < len(group_by): print(f"Warning: Group by columns not found (input: {original_group_by}, translated: {group_by_translated})")
    agg_spec = {col: function for col in columns}
    try:
        if valid_group_by:
            grouped = df.groupby(valid_group_by, dropna=False)
            result = grouped.agg(agg_spec)
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
                            try:
                                numeric_series = pd.to_numeric(series, errors='coerce');
                                if numeric_series.notna().any():
                                     if hasattr(numeric_series, func): results[col_func_name] = getattr(numeric_series, func)()
                                     else: results[col_func_name] = f"Func '{func}' error"
                                else: results[col_func_name] = None
                            except Exception as e: results[col_func_name] = f"Agg Error: {e}"
                      else: results[col_func_name] = f"Unsupported func '{func}'"
             result = pd.DataFrame([results])
        return _serialize_result(result) # Use corrected serializer
    except Exception as e: return {"error": f"Error during aggregation: {str(e)}"}

def sort_dataframe(df: pd.DataFrame, sort_by: Union[str, List[str]], ascending: Union[bool, List[bool]] = True, limit: Optional[int] = None) -> Dict[str, Any]:
    """Sort the DataFrame."""
    if df.empty: return {"result": [], "count": 0, "note": "Input DataFrame was empty."}
    if isinstance(sort_by, str): sort_by = [sort_by]
    original_sort_by = sort_by
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
        return _serialize_result(sorted_df, limit=limit) # Use corrected serializer
    except Exception as e: return {"error": f"Error sorting DataFrame: {str(e)}"}

def value_counts_dataframe(df: pd.DataFrame, columns: Union[str, List[str]], limit: Optional[int] = None) -> Dict[str, Any]:
     """Get value counts for specified column(s)."""
     if df.empty: return {"error": "Cannot get value counts from empty DataFrame."}
     if isinstance(columns, str): columns = [columns]
     results = {}
     try:
         original_columns = columns; columns_translated = [translate_column_name(col) for col in columns]
         for i, col_t in enumerate(columns_translated):
             col_orig = original_columns[i]
             if col_t not in df.columns: results[col_orig] = {"error": f"Column '{col_orig}' (->'{col_t}') not found"}
             else: counts = df[col_t].value_counts(dropna=False); results[col_orig] = _serialize_result(counts, limit=limit) # Use corrected serializer
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
         # Use _serialize_result for describe matrix format
         return {"correlation_matrix": _serialize_result(correlation_matrix).get('result')}
     except Exception as e: return {"error": f"Error calculating correlation: {str(e)}"}


# --- Main Dispatcher (Keep as before, uses translate internally now) ---
def query_dataframe(df: pd.DataFrame, query_params: Dict[str, Any]) -> Dict[str, Any]:
    """ Dispatcher function for DataFrame queries. Parameters are translated internally by helpers. """
    # No explicit preprocessing needed here anymore, it's pushed down into helpers
    print(f"Received query params: {query_params}") # Debugging original params

    query_type = query_params.get("query_type")
    limit = query_params.get("limit") or query_params.get("n")

    try:
        if df is None or df.empty:
             return {"error": "DataFrame is not loaded or is empty. Cannot perform query."}

        # Call handlers - they will perform translation internally
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
        elif query_type == "top_n":
            sort_by = query_params.get("sort_by"); ascending = query_params.get("ascending", False); n = limit
            if not sort_by: return {"error": "Top N query requires 'sort_by'."}
            if not n: return {"error": "Top N query requires 'limit' or 'n'."}
            return sort_dataframe(df, sort_by=sort_by, ascending=ascending, limit=n)
        elif query_type == "value_counts":
             cols = query_params.get("columns")
             if not cols: return {"error": "Value Counts query requires 'columns'."}
             return value_counts_dataframe(df, columns=cols, limit=limit)
        elif query_type == "correlation":
             cols = query_params.get("columns"); method = query_params.get("correlation_method", "pearson")
             return correlation_dataframe(df, columns=cols, method=method)
        elif query_type == "general": return get_dataframe_info(df)
        else: print(f"Unknown query type '{query_type}'. Falling back to general info."); return get_dataframe_info(df)
    except Exception as e: import traceback; print(f"Error in query_dataframe: {traceback.format_exc()}"); return {"error": f"Unexpected error ({query_type}): {str(e)}"}


# --- Testing Block (Keep as before) ---
if __name__ == "__main__":
    # (Test code remains the same as your previous successful run)
    df_test = load_dummy_dataframe() # Make sure path is correct!
    if df_test is not None and not df_test.empty:
        print("\n--- Testing Queries With Shortcuts ---")
        f1_shortcut = "f1 score"; hallu_frac_shortcut = "hallucination rate"; model_shortcut = "model"; latency_shortcut = "latency"
        precision_shortcut = "precision"
        f1_col = COLUMN_SHORTCUTS[precision_shortcut]; model_col = COLUMN_SHORTCUTS[model_shortcut]; latency_col = COLUMN_SHORTCUTS[latency_shortcut] # Get actual names for checking

        print(f"\n1. Test Describe ('{precision_shortcut}'):"); print(json.dumps(query_dataframe(df_test, {"query_type": "describe", "columns": [f1_shortcut]}), indent=2, default=str))
        print(f"\n2. Test Filter ('{hallu_frac_shortcut}' > 0.5 AND {model_shortcut} contains 'gpt'):"); print(json.dumps(query_dataframe(df_test, {"query_type": "filter", "filters": [{"column": hallu_frac_shortcut, "operator": ">", "value": 0.5}, {"column": model_shortcut, "operator": "contains", "value": "gpt"}]}), indent=2, default=str))
        print(f"\n3. Test Aggregate (Mean '{latency_shortcut}' by '{model_shortcut}'):"); print(json.dumps(query_dataframe(df_test, {"query_type": "aggregate", "columns": [latency_shortcut], "function": "mean", "group_by": model_shortcut}), indent=2, default=str))
        print(f"\n4. Test Top N (Top 3 by '{precision_shortcut}' desc):"); print(json.dumps(query_dataframe(df_test, {"query_type": "top_n", "sort_by": precision_shortcut, "ascending": False, "limit": 3}), indent=2, default=str))
        print(f"\n5. Test Value Counts ('{model_shortcut}'):"); print(json.dumps(query_dataframe(df_test, {"query_type": "value_counts", "columns": model_shortcut}), indent=2, default=str))
        print(f"\n6. Test Info ('{latency_shortcut}'):"); print(json.dumps(query_dataframe(df_test, {"query_type": "info", "columns": [latency_shortcut]}), indent=2, default=str))
        print(f"\n7. Test Filter (Simple filter: '{model_shortcut}' == 'gpt-4o')"); print(json.dumps(query_dataframe(df_test, {"query_type": "filter", "filter_column": model_shortcut, "filter_operator": "==", "filter_value": "gpt-4o"}), indent=2, default=str))
        # Add correlation test back if columns exist and are numeric
        if f1_col in df_test.columns and latency_col in df_test.columns and pd.api.types.is_numeric_dtype(df_test[f1_col]) and pd.api.types.is_numeric_dtype(df_test[latency_col]):
            print(f"\n8. Test Correlation ('{precision_shortcut}' and '{latency_shortcut}'):")
            print(json.dumps(query_dataframe(df_test, {"query_type": "correlation", "columns": [precision_shortcut, latency_shortcut]}), indent=2, default=str))
        else: print(f"\n8. Skipping Correlation test: Columns '{f1_col}' or '{latency_col}' not found or not numeric.")
    else: print("DataFrame could not be loaded. Skipping tests.")