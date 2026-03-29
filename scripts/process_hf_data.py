import os
import json
import random
import pandas as pd
from datasets import load_dataset
from PIL import Image
import argparse
import io

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            try:
                # Try to decode to string if it's text
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                # If it's pure binary, replace it so it doesn't crash the JSON
                return "<binary_data_omitted>"
        return super().default(obj)

def extract_and_split_parquets(data_folder, test_size=0.2, seed=42):
    """
    Extracts images and metadata from nested parquets, randomly partitioning 
    them into specified train and test splits.
    """
    # Set a random seed so your splits are reproducible if you run this again
    random.seed(seed)

    base_folder = os.path.join(data_folder, "data")
    output_folder = os.path.join(data_folder, "extracted_data")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the partitioned media directories
    train_media_dir = os.path.join(output_folder, "train")
    test_media_dir = os.path.join(output_folder, "test")
    os.makedirs(train_media_dir, exist_ok=True)
    os.makedirs(test_media_dir, exist_ok=True)
    
    # Define paths for the split metadata files
    train_jsonl_path = os.path.join(output_folder, "train_metadata.jsonl")
    test_jsonl_path = os.path.join(output_folder, "test_metadata.jsonl")
    train_csv_path = os.path.join(output_folder, "train_metadata.csv")
    test_csv_path = os.path.join(output_folder, "test_metadata.csv")

    global_item_counter = 0
    train_count = 0
    test_count = 0

    print(f"Scanning directory: {base_folder}")
    print(f"Splitting data... (Train: {100 - (test_size*100):.0f}%, Test: {test_size*100:.0f}%)")

    # Open both JSONL files to stream the data into
    with open(train_jsonl_path, "w", encoding="utf-8") as f_train, open(test_jsonl_path, "w", encoding="utf-8") as f_test:
        
        for root, dirs, files in os.walk(base_folder):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_path = os.path.join(root, file)
                    category_name = os.path.basename(root) 
                    subset_name = file.replace(".parquet", "") 
                    
                    print(f"Processing: [{category_name}] -> {file}")
                    
                    try:
                        ds = load_dataset("parquet", data_files=parquet_path, split="train")
                    except Exception as e:
                        print(f"  -> Skipped {file} due to error: {e}")
                        continue
                    
                    for row_idx, row in enumerate(ds):
                        # 1. Decide if this row goes to train or test
                        is_test = random.random() < test_size
                        current_split = "test" if is_test else "train"
                        current_media_dir = test_media_dir if is_test else train_media_dir
                        current_file = f_test if is_test else f_train
                        
                        if is_test:
                            test_count += 1
                        else:
                            train_count += 1

                        meta_item = {
                            "_split": current_split,
                            "_category": category_name,
                            "_subset": subset_name,
                            "_original_row": row_idx
                        }
                        
                        # 2. Extract columns and save images to the correct split folder
                        for col_name, value in row.items():
                            
                            is_saved_as_image = False
                            
                            # Case A: It's a nicely decoded PIL Image
                            if isinstance(value, Image.Image):
                                image_filename = f"{category_name}_{subset_name}_{global_item_counter}.png"
                                image_path = os.path.join(current_media_dir, image_filename)
                                
                                try:
                                    value.convert("RGB").save(image_path)
                                    meta_item[col_name] = f"extracted_media/{current_split}/{image_filename}"
                                    is_saved_as_image = True
                                except Exception as e:
                                    meta_item[col_name] = f"ERROR: {e}"
                            
                            # Case B: It's raw bytes (either directly, or hidden inside a dictionary)
                            else:
                                raw_bytes = None
                                
                                # Check if it's a dictionary containing bytes (HF standard)
                                if isinstance(value, dict) and "bytes" in value and isinstance(value["bytes"], bytes):
                                    raw_bytes = value["bytes"]
                                # Check if the value itself is just raw bytes
                                elif isinstance(value, bytes):
                                    raw_bytes = value
                                    
                                # If we found bytes, let's try to see if it's an image!
                                if raw_bytes is not None:
                                    try:
                                        img = Image.open(io.BytesIO(raw_bytes))
                                        # If we got here, it's definitely an image!
                                        image_filename = f"{category_name}_{subset_name}_{global_item_counter}.png"
                                        image_path = os.path.join(current_media_dir, image_filename)
                                        
                                        img.convert("RGB").save(image_path)
                                        meta_item[col_name] = f"extracted_media/{current_split}/{image_filename}"
                                        is_saved_as_image = True
                                    except Exception:
                                        # It wasn't an image (e.g., just binary text/hashes). 
                                        # Do nothing here; it will fall through to Case C.
                                        pass

                            # Case C: It's standard metadata (text, numbers, or non-image bytes)
                            if not is_saved_as_image:
                                meta_item[col_name] = value

                        # 3. Write to the assigned split's JSONL using the SafeEncoder
                        current_file.write(json.dumps(meta_item, cls=SafeEncoder) + "\n")
                        global_item_counter += 1

    # Convert the completed JSONL files into clean CSVs
    print("\nConverting JSONL files to CSVs...")
    
    if train_count > 0:
        pd.read_json(train_jsonl_path, lines=True).to_csv(train_csv_path, index=False)
    if test_count > 0:
        pd.read_json(test_jsonl_path, lines=True).to_csv(test_csv_path, index=False)

    print(f"\nExtraction and Splitting Complete!")
    print(f"Total rows processed: {global_item_counter}")
    print(f"  -> Train set:       {train_count} items")
    print(f"  -> Test set:        {test_count} items")
    print(f"Outputs saved to:     {output_folder}")

# --- Run the function ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="data/Hades", help="Path to the dataset folder")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    extract_and_split_parquets(args.data_folder, test_size=args.test_size, seed=args.seed)