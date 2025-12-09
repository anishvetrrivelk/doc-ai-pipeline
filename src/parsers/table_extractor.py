"""
src/parsers/table_extractor.py

Table extraction using pdfplumber.
Outputs clean, structured tables as lists of rows (CSV-like)
and can optionally convert them into Pandas DataFrames.
"""

from typing import List, Dict, Any, Optional
import pdfplumber
import pandas as pd


def extract_tables(pdf_path: str, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF using pdfplumber.

    Returns list of:
    {
       "page": int,
       "table_index": int,
       "data": List[List[str]],
       "as_dataframe": pd.DataFrame
    }
    """
    results = []

    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]

        for page_num, page in enumerate(pages, start=1):
            tables = page.extract_tables()

            for idx, table in enumerate(tables):
                # Convert to DataFrame (smart header detection)
                df = pd.DataFrame(table)

                # If first row looks like header → set as header
                if df.iloc[0].isnull().sum() == 0:
                    df.columns = df.iloc[0]
                    df = df[1:].reset_index(drop=True)

                results.append({
                    "page": page_num,
                    "table_index": idx,
                    "data": table,
                    "as_dataframe": df
                })

    return results


def extract_single_table(pdf_path: str, page_number: int, table_index: int = 0):
    """
    Extract one specific table by page and index.
    """
    all_tables = extract_tables(pdf_path)
    for t in all_tables:
        if t["page"] == page_number and t["table_index"] == table_index:
            return t
    raise ValueError(f"No table found at page {page_number}, table {table_index}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.parsers.table_extractor <pdf_path>")
    else:
        pdf = sys.argv[1]
        tables = extract_tables(pdf)
        print(f"Found {len(tables)} tables")
        if tables:
            print("Preview of first table:")
            print(tables[0]["as_dataframe"].head())
