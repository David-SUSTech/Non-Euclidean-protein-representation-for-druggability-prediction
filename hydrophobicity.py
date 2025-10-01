#This code is for computing and saving the hydrophobicity distance matrix,which hasn't been filtered yet.
#For pipeline completeness, we provide these codes.

import numpy as np
import pandas as pd
import re
from tqdm import tqdm

# Hydrophobicity index - Kyte-Doolittle hydrophobicity scale
kd_hydrophobicity = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2,
    'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5,
    'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# Read No_redundancy_ids.txt
print("Reading valid protein ID list...")
try:
    with open('No_redundancy_ids.txt', 'r', encoding='utf-8') as file:
        valid_ids = set([line.strip() for line in file if line.strip()])
    print(f"Loaded {len(valid_ids)} valid protein IDs")
except Exception as e:
    print(f"Error reading ID file: {e}")
    exit(1)

# Read protein sequences and labels
print("Reading protein sequence data...")
try:
    with open('Amino_acid_sequences_(for_fusion).txt', 'r', encoding='utf-8', errors='replace') as file:
        content = file.read()
except Exception as e:
    print(f"Error reading sequence file: {e}")
    exit(1)

# Parse protein data
filtered_proteins = []
entries = content.split('\n\n')

for entry in entries:
    lines = entry.strip().split('\n')
    if len(lines) < 2:
        continue

    # Parse ID and label
    header = lines[0]
    id_match = re.search(r'(.*?) :', header)
    label_match = re.search(r'Label: (\d)', header)

    if not (id_match and label_match):
        continue

    protein_id = id_match.group(1).strip()
    label = int(label_match.group(1))
    sequence = lines[1]

    # Keep only proteins in valid_ids
    if protein_id in valid_ids:
        filtered_proteins.append({
            'id': protein_id,
            'sequence': sequence,
            'label': label
        })

print(f"Obtained {len(filtered_proteins)} proteins after filtering")

# Extract IDs and sequences
protein_ids = [p['id'] for p in filtered_proteins]
sequences = [p['sequence'] for p in filtered_proteins]


# Compute hydrophobicity distance matrix
def protein_distance(seq1, seq2, step_size=2):
    """Compute hydrophobicity distance between two protein sequences"""
    # Ensure seq1 is the shorter sequence
    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1

    short_length = len(seq1)
    num_windows = max(1, len(seq2) - short_length + 1)

    # Store distances for all windows
    distances = []

    # Compute distance for each sliding window position
    for start_pos in range(0, num_windows, step_size):
        window = seq2[start_pos:start_pos + short_length]

        if len(window) < short_length:
            continue

        # Compute distance for the current window
        distance = 0
        for i in range(short_length):
            aa1 = seq1[i] if i < len(seq1) else '-'
            aa2 = window[i] if i < len(window) else '-'

            hydro1 = kd_hydrophobicity.get(aa1, 0)
            hydro2 = kd_hydrophobicity.get(aa2, 0)

            distance += abs(hydro1 - hydro2)

        distances.append(distance)

    # Take the minimum distance and normalize
    if not distances:
        return float('inf')

    min_distance = min(distances)
    return min_distance / short_length


# Compute the full distance matrix
print("Computing hydrophobicity distance matrix...")
n = len(filtered_proteins)
distance_matrix = np.zeros((n, n))

for i in tqdm(range(n)):
    for j in range(i + 1, n):
        dist = protein_distance(sequences[i], sequences[j])
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist  # Distance matrix is symmetric

# Min-Max normalize the matrix
print("Applying Min-Max normalization to the distance matrix...")
min_val = np.min(distance_matrix)
max_val = np.max(distance_matrix)

print(f"Distance matrix range: [{min_val:.4f}, {max_val:.4f}]")

# Normalization: (x - min) / (max - min)
if max_val > min_val:
    normalized_matrix = (distance_matrix - min_val) / (max_val - min_val)
else:
    normalized_matrix = np.zeros_like(distance_matrix)
    print("Warning: All distance values are identical; normalized matrix is all zeros")

# Create DataFrame and save to CSV

# For convenience, the distance matrices, including Hyd, dtw, density and sec, symbolizing for Hydrophobicity
#PTM_DTW, PTM site density and Secondary Structure

distance_df = pd.DataFrame(normalized_matrix, index=protein_ids, columns=protein_ids)
output_file = 'No_redundancy_hyd_matrix.csv'
distance_df.to_csv(output_file)

print(f"Successfully computed normalized distance matrix and saved to: {output_file}")