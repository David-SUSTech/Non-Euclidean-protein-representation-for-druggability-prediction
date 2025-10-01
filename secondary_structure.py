import pandas as pd
import re
import numpy as np
from sklearn.model_selection import LeaveOneOut
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler

# --- 1. Read the ID list file ---
ids_file_path = 'No_redundancy_ids.txt'
try:
    with open(ids_file_path, 'r', encoding='utf-8') as file:
        valid_ids = set([line.strip() for line in file if line.strip()])
    print(f"Read {len(valid_ids)} valid IDs from '{ids_file_path}'.")
except FileNotFoundError:
    print(f"Error: File '{ids_file_path}' not found. Please check if the file path is correct.")
    exit()


# --- 2. Extract secondary structure data ---
def extract_secondary_structure(file_path, common_proteins=None):
    """
    Extracts secondary structure data from a file.
    Args:
    file_path (str): The path to the file.
    common_proteins (set, optional): If provided, only process protein IDs in this set.
    Returns:
    pd.DataFrame: A DataFrame containing secondary structure percentages.
    """
    results = []
    total_found = 0
    filtered_by_common = 0
    processed_ids = set()

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        current_line = lines[i].strip()

        # Match the protein ID line, format like "6EOR'1':(Label: 1)"
        id_match = re.match(r"([A-Za-z0-9]+\'[0-9]+\'):\(Label:\s*(\d+)\)", current_line)

        if id_match and i + 1 < len(lines):
            protein_id = id_match.group(1)
            label = int(id_match.group(2))

            # Check if this protein has already been processed
            if protein_id in processed_ids:
                i += 2  # Skip this protein record (ID line and percentage line)
                continue

            # Check if filtering by common proteins is needed
            if common_proteins is not None and protein_id not in common_proteins:
                filtered_by_common += 1
                i += 2  # Skip this protein record
                continue

            processed_ids.add(protein_id)
            total_found += 1

            # Process the percentage line, format like "Alpha helix 18.87%, Beta sheet 26.35%, Turn 7.83%, Coil 46.95%"
            next_line = lines[i + 1].strip()

            # Extract secondary structure percentages
            alpha_helix = 0.0
            beta_sheet = 0.0
            beta_turn = 0.0
            random_coil = 0.0

            # Alpha helix
            alpha_match = re.search(r'Alpha helix\s+(\d+\.\d+)%', next_line)
            if alpha_match:
                alpha_helix = float(alpha_match.group(1))

            # Beta sheet
            beta_match = re.search(r'Beta sheet\s+(\d+\.\d+)%', next_line)
            if beta_match:
                beta_sheet = float(beta_match.group(1))

            # Turn (Beta turn)
            turn_match = re.search(r'Turn\s+(\d+\.\d+)%', next_line)
            if turn_match:
                beta_turn = float(turn_match.group(1))

            # Coil (Random coil)
            coil_match = re.search(r'Coil\s+(\d+\.\d+)%', next_line)
            if coil_match:
                random_coil = float(coil_match.group(1))

            # Add data to the results list
            results.append({
                'protein_id': protein_id,
                'label': label,
                'alpha_helix': alpha_helix,
                'beta_sheet': beta_sheet,
                'beta_turn': beta_turn,
                'random_coil': random_coil
            })

            i += 2  # After processing a complete protein record, move forward two lines
        else:
            i += 1  # If not in the expected format, move forward one line

    print(f"Found a total of {total_found} protein IDs.")
    if common_proteins is not None:
        print(f"Of these, {filtered_by_common} were filtered out for not being in the common proteins list.")
    print(f"Final number of proteins with secondary structure information: {len(results)}")

    return pd.DataFrame(results)


# --- 3. Main program starts ---
# Define file path
secondary_structure_file = 'Secondary_structure_for_fusion.txt'

# Extract secondary structure data, keeping only proteins in valid_ids
df = extract_secondary_structure(secondary_structure_file, valid_ids)

# --- 4. Calculate and save the distance matrix (with Min-Max normalization) ---
# Extract features
X = df[['alpha_helix', 'beta_sheet', 'beta_turn', 'random_coil']]
protein_ids = df['protein_id'].tolist()

print("\n--- Calculating distance matrix ---")

# Calculate the distance matrix using L1 norm (Manhattan distance)
distance_vector = pdist(X.values, metric='cityblock')  # 'cityblock' corresponds to L1 norm

# Use squareform to convert the 1D array back to a symmetric distance matrix
distance_matrix = squareform(distance_vector)

# Perform Min-Max normalization on the distance matrix
# First, find the minimum and maximum values in the matrix
min_distance = np.min(distance_matrix)
max_distance = np.max(distance_matrix)

# Perform Min-Max normalization: (x - min) / (max - min)
if max_distance > min_distance:  # Avoid division by zero
    normalized_matrix = (distance_matrix - min_distance) / (max_distance - min_distance)
else:
    normalized_matrix = distance_matrix  # If all values are the same, keep them as is

print(f"Distance matrix Min-Max normalization complete, mapped from [{min_distance}, {max_distance}] to [0, 1].")

# Convert the normalized matrix to a DataFrame, using protein_id as index and columns
distance_df = pd.DataFrame(normalized_matrix, index=protein_ids, columns=protein_ids)

# Save to CSV file, using the original filename
output_filename = 'No_redundancy_sec.csv'
distance_df.to_csv(output_filename)

print(f"Normalized distance matrix successfully saved to file: '{output_filename}'")


print("\n--- Program execution finished ---")