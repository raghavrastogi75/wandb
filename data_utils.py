# In data_utils.py (or data_utils_1.py)

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Union, Optional, Any, Literal

# --- Function Definition ---
def get_tools_schema() -> Dict[str, Any]:
    """
    Generate the tools schema for DataFrame operations, formatted
    specifically for the Realtime API's session.update method.
    REMOVED the 'examples' key from the return value.
    """
    # Define the parameters structure separately
    schema_parameters = {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": [
                    "general", "info", "filter", "aggregate", "sort",
                    "group", "top_n", "correlation", "describe",
                    "compare", "time_series", "value_counts"
                ],
                "description": "Type of query or analysis. Use 'info' for column names/types."
            },
            "columns": { "type": ["string", "array"], "items": {"type": "string"}, "description": "Full, exact column name(s)." },
            "column": { "type": "string", "description": "Alias for a single column." },
            "filters": {
                "type": "array", "items": {
                    "type": "object", "properties": {
                        "column": {"type": "string", "description": "Full column name."},
                        "operator": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "contains", "startswith", "endswith", "between", "isin", "isnull", "notnull"], "description": "Operator."},
                        "value": { "type": ["string", "number", "boolean", "array"], "items": { "type": ["string", "number", "boolean"] }, "description": "Value(s)." }
                    }, "required": ["column", "operator"]
                }, "description": "Filters (AND logic)."
            },
            "filter_column": {"type": "string", "description": "Column for simple filter."},
            "filter_operator": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "contains", "startswith", "endswith", "between", "isin", "isnull", "notnull"], "description": "Operator."},
            "filter_value": { "type": ["string", "number", "boolean", "array"], "items": { "type": ["string", "number", "boolean"] }, "description": "Value."},
            "function": {"type": ["string", "array"], "items": {"type": "string"}, "enum": ["mean", "median", "sum", "min", "max", "count", "std", "var", "first", "last", "nunique"], "description": "Aggregation func(s)."},
            "group_by": {"type": ["string", "array"], "items": {"type": "string"}, "description": "Column(s) to group by."},
            "sort_by": {"type": ["string", "array"], "items": {"type": "string"}, "description": "Column(s) to sort by."},
            "ascending": {"type": ["boolean", "array"], "items": {"type": "boolean"}, "default": True, "description": "Sort direction(s)."},
            "limit": {"type": "integer", "minimum": 1, "description": "Max rows."},
            "n": {"type": "integer", "minimum": 1, "description": "Alias for 'limit'."},
            "percentiles": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "description": "Percentiles for 'describe'."},
            "correlation_method": {"type": "string", "enum": ["pearson", "kendall", "spearman"], "default": "pearson", "description": "Correlation method."}
        },
        "required": ["query_type"]
    }

    # *** Return the FLAT structure WITHOUT 'examples' ***
    return {
        "type": "function",
        "name": "query_dataframe",
        "description": "Query/analyze DataFrame with model metrics. Use full column names. Group/filter by e.g., 'attributes.model_name'.",
        "parameters": schema_parameters,
        # <<< MAKE SURE there is NO 'examples' key here >>>
    }
# --- End of get_tools_schema ---

# ... (The rest of your data_utils.py file containing all the query functions) ...
# (Make sure the full file content from previous steps is present)
# --- Make sure ALL helper functions are included below ---
# load_dummy_dataframe, _serialize_result, get_dataframe_info, describe_dataframe,
# filter_dataframe, aggregate_dataframe, sort_dataframe, value_counts_dataframe,
# correlation_dataframe, query_dataframe dispatcher, and the if __name__ == "__main__": block
# (Copy these from the previous correct response)
def load_dummy_dataframe(file_path=None):
    """Load a DataFrame from a specified path or create a fallback dummy."""
    # *** IMPORTANT: Update this default_path to your actual file location ***
    default_path = r"C:\Users\ragha\OneDrive\Documents\wandb\weave_export_hallucination_2025-04-11.csv" # CHANGE THIS PATH
    load_path = file_path if file_path else default_path
    try:
        print(f"Attempting to load DataFrame from: {load_path}")
        df = pd.read_csv(load_path)
        print(f"Successfully loaded DataFrame. Shape: {df.shape}")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Attempt numeric conversion more carefully
                    converted_col = pd.to_numeric(df[col], errors='coerce')
                    # Only overwrite if the conversion didn't make everything NaN
                    if not converted_col.isnull().all():
                         df[col] = converted_col
                         # print(f"Converted column '{col}' to numeric.") # Reduce noise
                except (ValueError, TypeError):
                    pass # Ignore errors if conversion fails, keep as object
        print("\nDataFrame Info after loading and potential type conversion:")
        df.info(verbose=False) # Use verbose=False for brevity if many columns
        return df
    except FileNotFoundError:
        print(f"Warning: File not found at {load_path}. Creating dummy data.")
    except Exception as e:
        print(f"Error loading DataFrame from {load_path}: {e}. Creating dummy data.")

    # Fallback dummy data
    print("Using simple fallback dummy data.")
    return pd.DataFrame({
        'attributes.model_name': ['SmolLM2-360M', 'gpt-4o', 'gpt-4o-mini'] * 3,
        'output.HalluScorerEvaluator.scorer_evaluation_metrics.f1': [0.54, 0.67, 0.58, 0.52, 0.69, 0.61, 0.55, 0.68, 0.59],
        'summary.weave.latency_ms': [120, 550, 300, 135, 580, 310, 110, 565, 290],
        'output.HalluScorerEvaluator.is_hallucination.true_fraction': [0.8, 0.1, 0.7, 0.2, 0.15, 0.75, 0.85, 0.05, 0.9]
    })


def _serialize_result(data: Any, limit: Optional[int] = 20) -> Dict[str, Any]:
    """Safely serialize DataFrame/Series results to JSON-compatible dict, applying limit."""
    note = ""
    original_count = None

    try:
        if isinstance(data, pd.DataFrame):
            original_count = len(data)
            if limit is not None and len(data) > limit:
                 res_df = data.head(limit)
                 note = f" (showing first {limit} of {len(data)} rows)"
            else:
                 res_df = data
            res_df = res_df.replace([np.inf, -np.inf], [float('inf'), float('-inf')]).fillna("NaN")
            return {"result": res_df.to_dict(orient="records"), "count": original_count, "note": note.strip()}
        elif isinstance(data, pd.Series):
            original_count = len(data)
            if limit is not None and len(data) > limit:
                 res_series = data.head(limit)
                 note = f" (showing first {limit} of {len(data)} items)"
            else:
                 res_series = data
            res_series = res_series.replace([np.inf, -np.inf], [float('inf'), float('-inf')]).fillna("NaN")
            if isinstance(res_series.index, pd.MultiIndex):
                 # Convert multi-index keys to strings for JSON compatibility
                 return {"result": {str(k): v for k, v in res_series.items()}, "count": original_count, "note": note.strip()}
            elif pd.api.types.is_numeric_dtype(res_series.index) and (res_series.index != pd.RangeIndex(start=0, stop=len(res_series), step=1)).any():
                 # Keep index if it's meaningful (numeric but not default 0, 1, 2...)
                 return {"result": res_series.to_dict(), "count": original_count, "note": note.strip()}
            else:
                 # Use list for default RangeIndex or non-numeric indices
                 return {"result": res_series.tolist(), "count": original_count, "note": note.strip()}
        elif isinstance(data, (int, float, str, bool, dict, list, type(None))):
             count = len(data) if isinstance(data, (list, dict)) else None
             return {"result": data, "count": count}
        elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return {"result": int(data)}
        elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
             if np.isinf(data): return {"result": float('inf') if data > 0 else float('-inf')}
             if np.isnan(data): return {"result": "NaN"}
             return {"result": float(data)}
        elif isinstance(data, (np.complex_, np.complex64, np.complex128)):
            return {"result": {'real': float(data.real), 'imag': float(data.imag)}} # Ensure floats
        elif isinstance(data, (np.bool_)):
            return {"result": bool(data)}
        elif isinstance(data, (np.void)):
            return {"result": "void (structured data not directly serializable)"}
        elif isinstance(data, pd.Timestamp):
              return {"result": data.isoformat()} # Serialize Timestamps
        else:
            try:
                 return {"result": str(data)}
            except Exception as e_str:
                 print(f"Serialization Error: Could not serialize type {type(data)}. Error: {e_str}")
                 return {"error": f"Result type {type(data)} could not be serialized."}
    except Exception as e_main:
         print(f"Error during serialization: {e_main}")
         return {"error": f"An unexpected error occurred during result serialization: {e_main}"}


def get_dataframe_info(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
    """Get basic info (shape, columns, dtypes, sample) or info for specific columns."""
    if df.empty: return {"error": "DataFrame is empty."}
    if columns:
        if isinstance(columns, str): columns = [columns]
        cols_to_show = [col for col in columns if col in df.columns]
        if not cols_to_show: return {"error": f"Specified columns not found: {columns}"}
        df_subset = df[cols_to_show]
        info = {
             "columns_info": {
                 col: {
                     "dtype": str(df_subset[col].dtype),
                     "non_null_count": int(df_subset[col].count()),
                     "unique_count": df_subset[col].nunique(),
                     "sample_values": _serialize_result(df_subset[col].dropna().head(5), limit=5).get('result', [])
                 } for col in cols_to_show
             }
         }
    else:
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "sample": _serialize_result(df.head(3), limit=3).get('result', [])
        }
    return info


def describe_dataframe(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None, percentiles: Optional[List[float]] = None) -> Dict[str, Any]:
    """Get descriptive statistics for numeric or all columns."""
    if df.empty: return {"error": "DataFrame is empty."}
    cols_to_describe = df.columns.tolist()
    target_df = df

    if columns:
        if isinstance(columns, str): columns = [columns]
        cols_to_describe = [col for col in columns if col in df.columns]
        if not cols_to_describe: return {"error": f"Specified columns for describe not found: {columns}"}
        target_df = df[cols_to_describe]

    try:
        # Only include numeric types for standard describe, unless include='all' is desired
        description = target_df.describe(percentiles=percentiles, include=np.number)
        if description.empty and not target_df.select_dtypes(include=np.number).empty:
             # If describe on numeric failed, maybe try describe on all
             description = target_df.describe(percentiles=percentiles, include='all')
        elif description.empty:
              return {"result": "No numeric columns found to describe."}
        # Serialize the description DataFrame
        return {"description": _serialize_result(description).get('result')}
    except Exception as e:
        return {"error": f"Error describing DataFrame: {str(e)}"}


def filter_dataframe(df: pd.DataFrame, filters: Optional[List[Dict[str, Any]]] = None,
                      filter_column: Optional[str] = None, filter_operator: Optional[str] = None, filter_value: Any = None) -> Dict[str, Any]:
    """Filter the DataFrame based on simple or complex criteria (using AND logic)."""
    if df.empty: return {"result": [], "count": 0, "note": "Input DataFrame was empty."}
    filtered_df = df.copy()
    conditions = []
    active_filters = filters
    if not filters and filter_column and filter_operator:
         active_filters = [{"column": filter_column, "operator": filter_operator, "value": filter_value}]

    if active_filters:
        for f in active_filters:
            col = f.get("column")
            op = f.get("operator")
            val = f.get("value")
            if not col or not op: return {"error": f"Invalid filter: {f}. Requires 'column' and 'operator'."}
            if col not in filtered_df.columns: return {"error": f"Filter column '{col}' not found."}
            try:
                series = filtered_df[col]
                # Attempt type conversion for non-list value based on series dtype
                if op not in ['isnull', 'notnull', 'isin'] and val is not None:
                     if isinstance(val, list) and op == 'between':
                           # Convert elements within list for between if possible
                           new_val = []
                           for item in val:
                                if pd.api.types.is_numeric_dtype(series.dtype): item = pd.to_numeric(item, errors='coerce')
                                elif pd.api.types.is_datetime64_any_dtype(series.dtype): item = pd.to_datetime(item, errors='coerce')
                                new_val.append(item)
                           val = new_val
                           if pd.isna(val[0]) or pd.isna(val[1]): raise ValueError(f"Could not convert 'between' values '{f.get('value')}' for column '{col}'")
                     elif not isinstance(val, list):
                          # Convert single value
                          if pd.api.types.is_numeric_dtype(series.dtype): val = pd.to_numeric(val, errors='coerce')
                          elif pd.api.types.is_datetime64_any_dtype(series.dtype): val = pd.to_datetime(val, errors='coerce')
                          elif pd.api.types.is_boolean_dtype(series.dtype):
                              if isinstance(val, str):
                                   if val.lower() in ['true', '1', 'yes']: val = True
                                   elif val.lower() in ['false', '0', 'no']: val = False
                                   else: raise ValueError(f"Invalid boolean string '{val}'")
                              else: val = bool(val)
                          if pd.isna(val) and str(f.get('value')).lower() != 'nan': # Don't raise if original value was intended as NaN
                                raise ValueError(f"Could not convert filter value '{f.get('value')}' for column '{col}'")


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
                elif op == "isin":
                    if not isinstance(val, list): return {"error": f"'isin' needs list value. Got: {type(val)}"}
                    conditions.append(series.isin(val))
                elif op == "between":
                    if not isinstance(val, list) or len(val) != 2: return {"error": f"'between' needs list [min, max]. Got: {val}"}
                    conditions.append(series.between(val[0], val[1]))
                elif op == "isnull": conditions.append(series.isnull())
                elif op == "notnull": conditions.append(series.notnull())
                else: return {"error": f"Unsupported filter operator: {op}"}
            except Exception as e: return {"error": f"Error applying filter ({col} {op} {val}): {str(e)}"}

    if conditions:
        final_condition = conditions[0]
        for i in range(1, len(conditions)): final_condition &= conditions[i] # AND
        filtered_df = filtered_df.loc[final_condition]

    return _serialize_result(filtered_df)


def aggregate_dataframe(df: pd.DataFrame, columns: Union[str, List[str]], function: Union[str, List[str]], group_by: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
    """Aggregate data in the DataFrame, optionally grouping first."""
    if df.empty: return {"error": "Cannot aggregate empty DataFrame."}
    if isinstance(columns, str): columns = [columns]
    if isinstance(function, str): function = [function]
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns: return {"error": f"Aggregation columns not found: {columns}"}
    if len(valid_columns) < len(columns): print(f"Warning: Columns not found for aggregation: {set(columns) - set(valid_columns)}")
    columns = valid_columns
    agg_spec = {col: function for col in columns}
    try:
        if group_by:
            if isinstance(group_by, str): group_by = [group_by]
            valid_group_by = [gb_col for gb_col in group_by if gb_col in df.columns]
            if not valid_group_by: return {"error": f"Group by columns not found: {group_by}"}
            if len(valid_group_by) < len(group_by): print(f"Warning: Group by columns not found: {set(group_by) - set(valid_group_by)}")
            group_by = valid_group_by
            grouped = df.groupby(group_by, dropna=False)
            result = grouped.agg(agg_spec)
            if isinstance(result.columns, pd.MultiIndex): result.columns = ['_'.join(map(str, col)).strip() for col in result.columns.values]
            result = result.reset_index()
        else:
            results = {}
            for func in function:
                 for col in columns:
                     col_func_name = f"{col}_{func}"
                     series = df[col]
                     if func == 'count': results[col_func_name] = series.count()
                     elif func == 'nunique': results[col_func_name] = series.nunique()
                     elif func in ['first', 'last']: results[col_func_name] = getattr(series.dropna(), func)() if not series.dropna().empty else None # Handle empty series after dropna
                     elif func in ['mean', 'median', 'sum', 'std', 'var', 'min', 'max']:
                           # Attempt numeric conversion if needed for these funcs
                           try:
                               numeric_series = pd.to_numeric(series, errors='coerce')
                               if numeric_series.notna().any(): # Check if there are any numeric values after coercion
                                    if hasattr(numeric_series, func): results[col_func_name] = getattr(numeric_series, func)()
                                    else: results[col_func_name] = f"Func '{func}' error on col '{col}'" # Should not happen for std funcs
                               else: results[col_func_name] = None # Or 0 or NaN, depending on desired behavior for all-NaN series
                           except Exception as e: results[col_func_name] = f"Error during numeric conversion/aggregation: {e}"
                     else: results[col_func_name] = f"Unsupported function '{func}' for col '{col}'"
            result = pd.DataFrame([results])
        return _serialize_result(result)
    except Exception as e: return {"error": f"Error during aggregation: {str(e)}"}


def sort_dataframe(df: pd.DataFrame, sort_by: Union[str, List[str]], ascending: Union[bool, List[bool]] = True, limit: Optional[int] = None) -> Dict[str, Any]:
    """Sort the DataFrame."""
    if df.empty: return {"result": [], "count": 0, "note": "Input DataFrame was empty."}
    if isinstance(sort_by, str): sort_by = [sort_by]
    valid_sort_by = [col for col in sort_by if col in df.columns]
    if not valid_sort_by: return {"error": f"Sort columns not found: {sort_by}"}
    if len(valid_sort_by) < len(sort_by):
         print(f"Warning: Sort columns not found: {set(sort_by) - set(valid_sort_by)}")
         indices_to_keep = [i for i, col in enumerate(sort_by) if col in valid_sort_by]
         if isinstance(ascending, list): ascending = [ascending[i] for i in indices_to_keep]
         sort_by = valid_sort_by
    if isinstance(ascending, bool): ascending = [ascending] * len(sort_by)
    if len(sort_by) != len(ascending): return {"error": "# sort_by must match # ascending."}
    try:
        sorted_df = df.sort_values(by=sort_by, ascending=ascending, na_position='last')
        return _serialize_result(sorted_df, limit=limit)
    except Exception as e: return {"error": f"Error sorting DataFrame: {str(e)}"}


def value_counts_dataframe(df: pd.DataFrame, columns: Union[str, List[str]], limit: Optional[int] = None) -> Dict[str, Any]:
     """Get value counts for specified column(s)."""
     if df.empty: return {"error": "Cannot get value counts from empty DataFrame."}
     if isinstance(columns, str): columns = [columns]
     results = {}
     try:
         for col in columns:
             if col not in df.columns: results[col] = {"error": f"Column '{col}' not found"}
             else: counts = df[col].value_counts(dropna=False); results[col] = _serialize_result(counts, limit=limit)
         return results[columns[0]] if len(columns) == 1 else {"value_counts": results}
     except Exception as e: return {"error": f"Error getting value counts: {str(e)}"}


def correlation_dataframe(df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'pearson') -> Dict[str, Any]:
     """Calculate pairwise correlation of columns."""
     if df.empty: return {"error": "Cannot calculate correlation on empty DataFrame."}
     target_df = df
     if columns:
         cols_to_corr = [col for col in columns if col in df.columns]
         if not cols_to_corr: return {"error": f"Specified columns not found for correlation: {columns}"}
         target_df = df[cols_to_corr]
     numeric_df = target_df.select_dtypes(include=np.number)
     if numeric_df.shape[1] < 2: return {"error": f"Correlation requires >= 2 numeric columns. Found: {numeric_df.columns.tolist()}"}
     try:
         correlation_matrix = numeric_df.corr(method=method)
         return {"correlation_matrix": _serialize_result(correlation_matrix).get('result')}
     except Exception as e: return {"error": f"Error calculating correlation: {str(e)}"}


def query_dataframe(df: pd.DataFrame, query_params: Dict[str, Any]) -> Dict[str, Any]:
    """ Dispatcher function for DataFrame queries. """
    query_type = query_params.get("query_type")
    limit = query_params.get("limit") or query_params.get("n")
    if "column" in query_params and "columns" not in query_params:
        col_param = query_params.get("column")
        query_params["columns"] = [col_param] if isinstance(col_param, str) else col_param
    if "columns" in query_params and isinstance(query_params["columns"], str):
         query_params["columns"] = [query_params["columns"]]
    if "filters" not in query_params and query_params.get("filter_column") and query_params.get("filter_operator"):
         query_params["filters"] = [{"column": query_params.get("filter_column"), "operator": query_params.get("filter_operator"), "value": query_params.get("filter_value")}]

    print(f"Executing query: {query_type} with params: {query_params}")
    try:
        if df is None or df.empty: # Add check for empty DataFrame early
             return {"error": "DataFrame is not loaded or is empty. Cannot perform query."}

        if query_type == "info": return get_dataframe_info(df, columns=query_params.get("columns"))
        elif query_type == "describe": return describe_dataframe(df, columns=query_params.get("columns"), percentiles=query_params.get("percentiles"))
        elif query_type == "filter": return filter_dataframe(df, filters=query_params.get("filters"))
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
    except Exception as e: import traceback; print(f"Error: {traceback.format_exc()}"); return {"error": f"Unexpected error ({query_type}): {str(e)}"}


# --- Example Usage ---
if __name__ == "__main__":
    df_test = load_dummy_dataframe() # Make sure path is correct!
    if df_test is not None and not df_test.empty:
        print("\n--- Testing Queries ---")
        # *** Replace with ACTUAL column names from your DataFrame ***
        f1_col = "output.HalluScorerEvaluator.scorer_evaluation_metrics.f1"
        hallu_frac_col = "output.HalluScorerEvaluator.is_hallucination.true_fraction"
        model_col = "attributes.model_name"
        latency_col = "summary.weave.latency_ms" # Check if this exists / is numeric
        req_col = "summary.weave.costs.gpt-4o-2024-08-06.requests" # Example, check if numeric/exists

        def check_cols(df, cols):
             missing = [c for c in cols if c not in df.columns];
             if missing: print(f"\n*** Warning: Test columns not found: {missing} ***"); return False
             return True
        test_cols_exist = check_cols(df_test, [f1_col, hallu_frac_col, model_col]) # Add others if needed

        print("\n1. Test Info:"); print(json.dumps(query_dataframe(df_test, {"query_type": "info"}), indent=2, default=str))
        if test_cols_exist:
            print(f"\n2. Test Describe ({f1_col}):"); print(json.dumps(query_dataframe(df_test, {"query_type": "describe", "columns": [f1_col]}), indent=2, default=str))
            print(f"\n3. Test Filter ({hallu_frac_col} > 0.5 AND {model_col} contains 'gpt'):"); print(json.dumps(query_dataframe(df_test, {"query_type": "filter", "filters": [{"column": hallu_frac_col, "operator": ">", "value": 0.5}, {"column": model_col, "operator": "contains", "value": "gpt"}]}), indent=2, default=str))
            if req_col in df_test.columns and pd.api.types.is_numeric_dtype(df_test[req_col]): print(f"\n4. Test Aggregate (Mean {req_col} by {model_col}):"); print(json.dumps(query_dataframe(df_test, {"query_type": "aggregate", "columns": [req_col], "function": "mean", "group_by": model_col}), indent=2, default=str))
            else: print(f"\n4. Skipping Aggregation test: Column '{req_col}' not found or not numeric.")
            print(f"\n5. Test Top N (Top 3 by {f1_col} desc):"); print(json.dumps(query_dataframe(df_test, {"query_type": "top_n", "sort_by": f1_col, "ascending": False, "limit": 3}), indent=2, default=str))
            print(f"\n6. Test Value Counts ({model_col}):"); print(json.dumps(query_dataframe(df_test, {"query_type": "value_counts", "columns": model_col}), indent=2, default=str))
            if latency_col in df_test.columns and pd.api.types.is_numeric_dtype(df_test[latency_col]): print(f"\n7. Test Correlation ({f1_col} and {latency_col}):"); print(json.dumps(query_dataframe(df_test, {"query_type": "correlation", "columns": [f1_col, latency_col]}), indent=2, default=str))
            else: print(f"\n7. Skipping Correlation test: Column '{latency_col}' not found or not numeric.")
        else: print("\nSkipping detailed tests due to missing example columns.")
    else: print("DataFrame could not be loaded. Skipping tests.")