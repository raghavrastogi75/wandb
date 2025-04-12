import pandas as pd
from typing import Any, Dict

def load_dummy_dataframe() -> pd.DataFrame:
    """Load a dummy dataframe with sample data."""
    data = pd.read_csv("C:\\Users\\ragha\\OneDrive\\Documents\\wandb\\weave_export_hallucination_2025-04-11.csv")
    return data


def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get structured information about the DataFrame."""
    return {
        'columns': list(df.columns),
        'shape': df.shape,
        'sample_data': df.head(3).to_dict(orient='records'),
        'summary_stats': df.describe().to_dict(),
        'product_counts': df['product'].value_counts().to_dict(),
        'region_counts': df['region'].value_counts().to_dict(),
        'total_sales': df['sales'].sum(),
        'avg_satisfaction': df['customer_satisfaction'].mean()
    }


def query_dataframe(df: pd.DataFrame, query_args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute queries on the DataFrame based on the provided arguments."""
    query_type = query_args.get('query_type', 'general')
    
    if query_type == 'filter':
        column = query_args.get('column')
        value = query_args.get('value')
        operator = query_args.get('operator', '==')
        
        if column and value is not None:
            if operator == '==':
                result_df = df[df[column] == value]
            elif operator == '>':
                result_df = df[df[column] > value]
            elif operator == '<':
                result_df = df[df[column] < value]
            elif operator == 'contains':
                result_df = df[df[column].astype(str).str.contains(str(value))]
            else:
                return {"error": f"Unsupported operator: {operator}"}
                
            return {
                "filtered_data": result_df.to_dict(orient='records'),
                "count": len(result_df)
            }
    
    elif query_type == 'aggregate':
        column = query_args.get('column')
        func = query_args.get('function', 'mean')
        group_by = query_args.get('group_by')
        
        if column:
            if group_by:
                if func == 'mean':
                    result = df.groupby(group_by)[column].mean().to_dict()
                elif func == 'sum':
                    result = df.groupby(group_by)[column].sum().to_dict()
                elif func == 'count':
                    result = df.groupby(group_by)[column].count().to_dict()
                else:
                    return {"error": f"Unsupported aggregate function: {func}"}
                
                return {
                    "aggregated_data": result
                }
            else:
                if func == 'mean':
                    result = df[column].mean()
                elif func == 'sum':
                    result = df[column].sum()
                elif func == 'count':
                    result = df[column].count()
                else:
                    return {"error": f"Unsupported aggregate function: {func}"}
                
                return {
                    "result": result
                }
    
    # Default to returning general info
    return get_dataframe_info(df)
