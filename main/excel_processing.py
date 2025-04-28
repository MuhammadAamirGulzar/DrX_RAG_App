import pandas as pd
import numpy as np
from pathlib import Path
import re
import tiktoken
from typing import List, Dict, Any, Tuple, Union, Optional

# Different chunking strategies for tabular data
class ChunkingStrategy:
    """Base class for different Excel/CSV chunking strategies"""
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, filename: str, sheet_name: str = "Sheet1", 
                        max_tokens: int = 512) -> List[Dict]:
        raise NotImplementedError("Subclasses must implement this method")

class RowBasedChunking(ChunkingStrategy):
    """Chunk Excel/CSV data by rows"""
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, filename: str, sheet_name: str = "Sheet1", 
                        max_tokens: int = 512, max_rows_per_chunk: int = 50) -> List[Dict]:
        """
        Split dataframe into chunks based on rows
        
        Args:
            df: DataFrame to chunk
            filename: Source filename
            sheet_name: Sheet name (for Excel)
            max_tokens: Maximum tokens per chunk
            max_rows_per_chunk: Maximum rows per chunk
        
        Returns:
            List of chunk dictionaries
        """
        tokenizer = tiktoken.get_encoding("cl100k_base")
        all_chunks = []
        total_rows = len(df)
        
        for start_idx in range(0, total_rows, max_rows_per_chunk):
            end_idx = min(start_idx + max_rows_per_chunk, total_rows)
            
            # Get the subset of rows
            df_subset = df.iloc[start_idx:end_idx]
            
            # Convert to text
            chunk_text = f"Sheet: {sheet_name} (Rows {start_idx+1}-{end_idx})\n\n"
            chunk_text += dataframe_to_text(df_subset, include_headers=(start_idx == 0))
            
            # Check token length and split if necessary
            tokens = tokenizer.encode(chunk_text)
            
            if len(tokens) <= max_tokens:
                # Add as a single chunk
                all_chunks.append({
                    "chunk_number": len(all_chunks) + 1,
                    "text": chunk_text,
                    "meta_data": {
                        "filename": filename, 
                        "sheet_name": sheet_name,
                        "page_number": start_idx // max_rows_per_chunk + 1,  # Use page_number for compatibility
                        "row_range": f"{start_idx+1}-{end_idx}",
                        "content_type": "tabular_data"
                    }
                })
            else:
                # Further split by token limits
                for i in range(0, len(tokens), max_tokens):
                    sub_tokens = tokens[i:i+max_tokens]
                    sub_text = tokenizer.decode(sub_tokens)
                    
                    all_chunks.append({
                        "chunk_number": len(all_chunks) + 1,
                        "text": sub_text,
                        "meta_data": {
                            "filename": filename, 
                            "sheet_name": sheet_name,
                            "page_number": start_idx // max_rows_per_chunk + 1,
                            "row_range": f"{start_idx+1}-{end_idx}",
                            "chunk_part": f"{i//max_tokens + 1}",
                            "content_type": "tabular_data"
                        }
                    })
        
        return all_chunks

class ColumnBasedChunking(ChunkingStrategy):
    """Chunk Excel/CSV data by columns"""
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, filename: str, sheet_name: str = "Sheet1", 
                         max_tokens: int = 512, columns_per_chunk: int = 5) -> List[Dict]:
        """
        Split dataframe into chunks based on columns
        
        Args:
            df: DataFrame to chunk
            filename: Source filename
            sheet_name: Sheet name (for Excel)
            max_tokens: Maximum tokens per chunk
            columns_per_chunk: Maximum columns per chunk
        
        Returns:
            List of chunk dictionaries
        """
        tokenizer = tiktoken.get_encoding("cl100k_base")
        all_chunks = []
        total_cols = len(df.columns)
        
        for start_idx in range(0, total_cols, columns_per_chunk):
            end_idx = min(start_idx + columns_per_chunk, total_cols)
            
            # Get subset of columns
            column_subset = df.iloc[:, start_idx:end_idx]
            
            # Convert to text format
            chunk_text = f"Sheet: {sheet_name} (Columns {df.columns[start_idx]} to {df.columns[end_idx-1]})\n\n"
            chunk_text += dataframe_to_text(column_subset)
            
            # Check token length and split if necessary
            tokens = tokenizer.encode(chunk_text)
            
            if len(tokens) <= max_tokens:
                all_chunks.append({
                    "chunk_number": len(all_chunks) + 1,
                    "text": chunk_text,
                    "meta_data": {
                        "filename": filename,
                        "sheet_name": sheet_name, 
                        "page_number": start_idx // columns_per_chunk + 1,
                        "column_range": f"{start_idx+1}-{end_idx}",
                        "content_type": "tabular_data_columns"
                    }
                })
            else:
                # Split by rows if too large
                row_chunks = RowBasedChunking.chunk_dataframe(
                    column_subset, filename, sheet_name, max_tokens
                )
                
                # Update metadata to include column information
                for chunk in row_chunks:
                    chunk["meta_data"]["column_range"] = f"{start_idx+1}-{end_idx}"
                    chunk["meta_data"]["content_type"] = "tabular_data_mixed"
                
                all_chunks.extend(row_chunks)
        
        return all_chunks

class SemanticChunking(ChunkingStrategy):
    """Chunk Excel/CSV data by semantic groups (e.g., by date ranges or categories)"""
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, filename: str, sheet_name: str = "Sheet1", 
                        max_tokens: int = 512, group_by_column: str = None) -> List[Dict]:
        """
        Split dataframe into chunks based on semantic groups
        
        Args:
            df: DataFrame to chunk
            filename: Source filename
            sheet_name: Sheet name (for Excel)  
            max_tokens: Maximum tokens per chunk
            group_by_column: Column to group by
        
        Returns:
            List of chunk dictionaries
        """
        tokenizer = tiktoken.get_encoding("cl100k_base")
        all_chunks = []
        
        # If no grouping column specified or not in dataframe, fall back to row-based chunking
        if group_by_column is None or group_by_column not in df.columns:
            return RowBasedChunking.chunk_dataframe(df, filename, sheet_name, max_tokens)
        
        # Group the dataframe by the specified column
        grouped = df.groupby(group_by_column)
        
        for group_name, group_df in grouped:
            # Convert group to text
            chunk_text = f"Sheet: {sheet_name} ({group_by_column}: {group_name})\n\n"
            chunk_text += dataframe_to_text(group_df)
            
            # Check token length and split if necessary
            tokens = tokenizer.encode(chunk_text)
            
            if len(tokens) <= max_tokens:
                all_chunks.append({
                    "chunk_number": len(all_chunks) + 1,
                    "text": chunk_text,
                    "meta_data": {
                        "filename": filename, 
                        "sheet_name": sheet_name,
                        "page_number": len(all_chunks) + 1,
                        "group_by": group_by_column,
                        "group_value": str(group_name),
                        "content_type": "tabular_data_semantic"
                    }
                })
            else:
                # If too large, fall back to row-based chunking for this group
                row_chunks = RowBasedChunking.chunk_dataframe(
                    group_df, filename, sheet_name, max_tokens
                )
                
                # Update metadata to include grouping information
                for chunk in row_chunks:
                    chunk["meta_data"]["group_by"] = group_by_column
                    chunk["meta_data"]["group_value"] = str(group_name)
                    chunk["meta_data"]["content_type"] = "tabular_data_semantic"
                
                all_chunks.extend(row_chunks)
        
        return all_chunks

def dataframe_to_text(df: pd.DataFrame, include_headers: bool = True, max_rows: int = None) -> str:
    """
    Convert a DataFrame to a structured text representation.
    
    Args:
        df: Pandas DataFrame
        include_headers: Whether to include column headers
        max_rows: Maximum number of rows to include (None for all)
        
    Returns:
        String representation of the DataFrame
    """
    # Handle empty dataframe
    if df.empty:
        return "Empty data"
    
    # Limit rows if specified
    if max_rows is not None:
        df = df.head(max_rows)
    
    # Convert DataFrame to string representation
    buffer = []
    
    # Add headers if requested
    if include_headers:
        headers = [str(col) for col in df.columns]
        buffer.append(" | ".join(headers))
        buffer.append("-" * len(" | ".join(headers)))
    
    # Add data rows
    for _, row in df.iterrows():
        row_values = [str(val) if not pd.isna(val) else "" for val in row]
        buffer.append(" | ".join(row_values))
    
    return "\n".join(buffer)

def read_excel(file_path: str) -> List[Tuple[int, str, pd.DataFrame]]:
    """
    Read Excel file and extract data with sheet information.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        List of tuples (sheet_number, sheet_name, dataframe)
    """
    sheets = []
    # Read all sheets from the Excel file
    excel_file = pd.ExcelFile(file_path)
    
    for i, sheet_name in enumerate(excel_file.sheet_names):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        sheets.append((i+1, sheet_name, df))
    
    return sheets

def read_csv(file_path: str) -> pd.DataFrame:
    """
    Read CSV file and extract data.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing CSV data
    """
    # Try to detect encoding and delimiter
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        # Try different encodings if UTF-8 fails
        df = pd.read_csv(file_path, encoding='latin1')
    except pd.errors.ParserError:
        # Try different delimiters if comma fails
        try:
            df = pd.read_csv(file_path, sep=';')
        except:
            df = pd.read_csv(file_path, sep='\t')
    
    return df

def create_tabular_structure_summary(file_path: str, filename: str) -> Dict:
    """
    Create a summary of the tabular file structure to aid in question answering.
    
    Args:
        file_path: Path to the Excel/CSV file
        filename: Name of the source file
        
    Returns:
        Dictionary with summary text and metadata
    """
    summary_lines = [f"File: {filename}"]
    
    if filename.lower().endswith(('.xlsx', '.xls')):
        # For Excel files
        sheets = read_excel(file_path)
        summary_lines.append(f"Type: Excel Spreadsheet")
        summary_lines.append(f"Total Sheets: {len(sheets)}")
        summary_lines.append("")
        
        for sheet_num, sheet_name, df in sheets:
            # Basic statistics about the sheet
            row_count = len(df)
            col_count = len(df.columns)
            summary_lines.append(f"Sheet {sheet_num}: {sheet_name}")
            summary_lines.append(f"  Rows: {row_count}, Columns: {col_count}")
            summary_lines.append(f"  Columns: {', '.join(str(col) for col in df.columns)}")
            
            # Data types
            dtypes = df.dtypes.astype(str).to_dict()
            dtype_info = [f"{col}: {dtype}" for col, dtype in dtypes.items()]
            summary_lines.append(f"  Data Types: {', '.join(dtype_info[:5])}" + ("..." if len(dtype_info) > 5 else ""))
            
            # Data sample (first few rows)
            if not df.empty:
                summary_lines.append("  Data Sample:")
                sample_text = dataframe_to_text(df.head(3), include_headers=True)
                # Indent the sample
                sample_text = "\n".join(f"    {line}" for line in sample_text.split("\n"))
                summary_lines.append(sample_text)
            
            summary_lines.append("")
            
    else:  # CSV file
        df = read_csv(file_path)
        summary_lines.append(f"Type: CSV File")
        
        # Basic statistics
        row_count = len(df)
        col_count = len(df.columns)
        summary_lines.append(f"Rows: {row_count}, Columns: {col_count}")
        summary_lines.append(f"Columns: {', '.join(str(col) for col in df.columns)}")
        
        # Data types
        dtypes = df.dtypes.astype(str).to_dict()
        dtype_info = [f"{col}: {dtype}" for col, dtype in dtypes.items()]
        summary_lines.append(f"Data Types: {', '.join(dtype_info[:5])}" + ("..." if len(dtype_info) > 5 else ""))
        
        # Data sample
        if not df.empty:
            summary_lines.append("Data Sample:")
            sample_text = dataframe_to_text(df.head(5), include_headers=True)
            # Indent the sample
            sample_text = "\n".join(f"  {line}" for line in sample_text.split("\n"))
            summary_lines.append(sample_text)
    
    return {
        "chunk_number": 0,  # Special identifier for structure summary
        "text": "\n".join(summary_lines),
        "meta_data": {
            "filename": filename,
            "page_number": 0,
            "content_type": "tabular_structure_summary"
        }
    }

def process_tabular_file(file_path: str, filename: str, 
                           chunking_strategy: ChunkingStrategy = RowBasedChunking,
                           group_by_column: str = None,
                           max_tokens: int = 512) -> Tuple[List[Dict], Dict]:
    """
    Process an Excel/CSV file and create chunks along with a structure summary.
    
    Args:
        file_path: Path to the tabular file
        filename: Name to use in metadata
        chunking_strategy: Strategy to use for chunking
        group_by_column: Column to group by (for semantic chunking)
        max_tokens: Maximum tokens per chunk
        
    Returns:
        Tuple of (List of chunks, Structure summary chunk)
    """
    # Create structure summary first
    structure_summary = create_tabular_structure_summary(file_path, filename)
    
    if filename.lower().endswith(('.xlsx', '.xls')):
        # For Excel files, process each sheet
        sheets = read_excel(file_path)
        all_chunks = []
        
        for sheet_num, sheet_name, df in sheets:
            if chunking_strategy == SemanticChunking and group_by_column is not None:
                chunks = SemanticChunking.chunk_dataframe(
                    df, filename, sheet_name, max_tokens, group_by_column
                )
            else:
                chunks = chunking_strategy.chunk_dataframe(
                    df, filename, sheet_name, max_tokens
                )
            all_chunks.extend(chunks)
            
    else:  # CSV file
        df = read_csv(file_path)
        if chunking_strategy == SemanticChunking and group_by_column is not None:
            all_chunks = SemanticChunking.chunk_dataframe(
                df, filename, "csv_data", max_tokens, group_by_column
            )
        else:
            all_chunks = chunking_strategy.chunk_dataframe(
                df, filename, "csv_data", max_tokens
            )
    
    return all_chunks, structure_summary

# Infer most appropriate chunking strategy based on data characteristics
def infer_best_chunking_strategy(file_path: str) -> Tuple[ChunkingStrategy, Optional[str]]:
    """
    Analyze tabular data and suggest the best chunking strategy
    
    Args:
        file_path: Path to the tabular file
        
    Returns:
        Tuple of (recommended chunking strategy, group_by_column if applicable)
    """
    if file_path.lower().endswith(('.xlsx', '.xls')):
        # For Excel, check the first sheet
        sheets = read_excel(file_path)
        if not sheets:
            return RowBasedChunking, None
            
        _, _, df = sheets[0]
    else:
        # For CSV
        df = read_csv(file_path)
    
    # If dataframe is small, use row-based chunking
    if len(df) < 100 and len(df.columns) < 10:
        return RowBasedChunking, None
    
    # If many columns but few rows, column-based might be better
    if len(df.columns) > 15 and len(df) < 1000:
        return ColumnBasedChunking, None
    
    # Check for potential grouping columns (date-like or categorical with few unique values)
    candidate_columns = []
    
    for col in df.columns:
        # Skip columns with too many unique values (>20% of rows)
        if df[col].nunique() > len(df) * 0.2:
            continue
            
        # Check if column is categorical or date-like
        is_date = False
        is_categorical = False
        
        # Try to infer if it's a date column
        if df[col].dtype == 'object':
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
                r'\d{4}/\d{2}/\d{2}'   # YYYY/MM/DD
            ]
            
            for pattern in date_patterns:
                if re.search(pattern, str(sample)):
                    is_date = True
                    break
                    
        # Check if it's a good categorical column (not too many unique values)
        unique_count = df[col].nunique()
        if 2 <= unique_count <= 20:  # Good range for grouping
            is_categorical = True
            
        if is_date or is_categorical:
            candidate_columns.append((col, unique_count))
    
    # If we found good candidate columns, use semantic chunking
    if candidate_columns:
        # Sort by number of unique values (fewer is better for grouping)
        candidate_columns.sort(key=lambda x: x[1])
        return SemanticChunking, candidate_columns[0][0]
    
    # Default to row-based chunking
    return RowBasedChunking, None
