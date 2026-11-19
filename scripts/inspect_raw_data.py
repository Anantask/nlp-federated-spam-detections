import os
import glob
import pandas as pd

def inspect():
    # Check data/ or data/raw/
    paths = glob.glob("data/*.csv") + glob.glob("data/raw/*.csv")
    csv_files = sorted(list(set(paths)))
    print(f"Discovered {len(csv_files)} CSV files: {csv_files}")
    
    total_rows = 0
    records = []
    for f in csv_files:
        filename = os.path.basename(f)
        df = pd.read_csv(f)
        total_rows += len(df)
        print(f"\n--- {filename} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes}")
        
        # Check target column
        target_col = None
        for col in ["CLASS", "TAG", "class", "tag", "label"]:
            if col in df.columns:
                target_col = col
                break
        print(f"Target column detected: {target_col}")
        if target_col:
            val_counts = df[target_col].value_counts(dropna=False).to_dict()
            print(f"Class counts ({target_col}): {val_counts}")
            
        print(f"Missing values per column:\n{df.isnull().sum().to_dict()}")
        
        # Duplicates within file
        dup_all = df.duplicated().sum()
        dup_comments = df.duplicated(subset=["CONTENT"]).sum() if "CONTENT" in df.columns else 0
        dup_comment_id = df.duplicated(subset=["COMMENT_ID"]).sum() if "COMMENT_ID" in df.columns else 0
        print(f"Exact duplicates: {dup_all}, Duplicate CONTENT: {dup_comments}, Duplicate COMMENT_ID: {dup_comment_id}")
        
        # Inspect empty text
        if "CONTENT" in df.columns:
            empty_count = (df["CONTENT"].fillna("").str.strip() == "").sum()
            print(f"Empty/whitespace CONTENT: {empty_count}")

    print(f"\n==========================================")
    print(f"TOTAL ROWS ACROSS ALL FILES: {total_rows}")
    print(f"==========================================")

if __name__ == "__main__":
    inspect()
