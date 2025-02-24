import os
import pandas as pd

# Define the months and file names
device = "jetson"
months = ["march", "april", "may", "june", "september"]
file_names = [
    "timer_keypoint_extraction.csv",
    "timer_ksi_keypoint_semantic_integration.csv",
    "timer_panopitic_segmentation_yolo.csv",
    "timer_semantic_encoder.csv",
    "timer_superglue_matching.csv"
]

# Output CSV file
output_file = f"{device}_summary.csv"

data = []

for month in months:
    row = {"Month": month}
    file_data = {}
    
    for file_name in file_names:
        file_path = os.path.join(device, month, file_name)
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Store data for later calculations
            file_data[file_name] = df
            
            # Compute average of second column
            try:
                avg_value = pd.to_numeric(df.iloc[:, 1], errors='coerce').mean()
                
                # Multiply the average by 2 for specific files
                if file_name in ["timer_keypoint_extraction.csv", "timer_ksi_keypoint_semantic_integration.csv"]:
                    avg_value *= 2
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                avg_value = None
            
            # Store the average value in the row
            row[file_name.replace("timer_", "").replace(".csv", "")] = avg_value
        else:
            print(f"Warning: {file_path} not found.")
            row[file_name.replace("timer_", "").replace(".csv", "")] = None
    
    # Special calculation for timer_semantic_encoder.csv
    if "timer_semantic_encoder.csv" in file_data and "timer_superglue_matching.csv" in file_data:
        try:
            encoder_rows = len(file_data["timer_semantic_encoder.csv"])
            superglue_rows = len(file_data["timer_superglue_matching.csv"])
            if superglue_rows > 0:
                row["semantic_encoder"] = (row["semantic_encoder"] * 2 * encoder_rows) / superglue_rows
        except Exception as e:
            print(f"Error adjusting timer_semantic_encoder.csv: {e}")
    
    # Compute ksi column
    row["ksi"] = row.get("ksi_keypoint_semantic_integration", 0) + row.get("semantic_encoder", 0)
    
    # Compute total column
    row["total"] = sum(
        row.get(field, 0) for field in [
            "keypoint_extraction",
            "ksi_keypoint_semantic_integration",
            "panopitic_segmentation_yolo",
            "semantic_encoder",
            "superglue_matching"
        ]
    )
    
    # Compute FPS as inverse of the sum of timer values (assuming they are in seconds)
    numeric_values = [v for v in row.values() if isinstance(v, (int, float)) and v is not None]
    total_time = sum(numeric_values)
    row["frames_per_second"] = 1 / total_time if total_time else None
    
    data.append(row)

# Convert data to a DataFrame and save to CSV
summary_df = pd.DataFrame(data)
summary_df.to_csv(output_file, index=False)
print(f"Summary saved to {output_file}")

