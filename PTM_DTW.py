import re
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


def load_common_protein_ids(file_path):
    """
    Maintains compatibility with the original function name, but actually calls read_protein_ids.
    """
    return read_protein_ids(file_path)


def read_file(file_path):
    """
    Reads the content of a file.

    Args:
    file_path (str): The path to the file.

    Returns:
    str: The content of the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def analyze_protein_sites(data):
    # Initialize the results dictionary
    results = {}

    # Current protein name
    current_protein = None
    current_label = None

    # Define the list of features to extract and their corresponding simplified key names
    site_types_full = [
        'cAMP- and cGMP-dependent protein kinase phosphorylation site.',
        'Protein kinase C phosphorylation site.',
        'Casein kinase II phosphorylation site.',
        'Tyrosine kinase phosphorylation site 1.',
        'N-myristoylation site.',
        'N-glycosylation site.',
        'Ubiquitin specific protease (USP) domain signature 1.'
    ]

    # Corresponding simplified key names
    site_keys = [
        'camp_cgmp',
        'protein_kinase_c',
        'casein_kinase_ii',
        'tyrosine_kinase',
        'n_myristoylation',
        'n_glycosylation',
        'ubiquitin'
    ]

    # Create a mapping dictionary
    site_type_to_key = dict(zip(site_types_full, site_keys))

    # Dictionary to store data for each site type
    site_data = {key: [] for key in site_keys}

    # The site type currently being processed
    current_site_type = None

    # Maximum site position for a protein (to determine its length)
    max_site = 0

    # Split the file content by lines
    lines = data.split('\n')

    for i, line in enumerate(lines):
        # New protein identification logic: handle two different header formats
        if re.match(r'[\w\d\']+:\s*\(Label:\s*\d\)', line):
            # If a protein has already been processed, save its results
            if current_protein:
                # Save all site information for the current protein
                protein_result = {
                    'length': max_site,  # Maintain the original length calculation logic
                    'label': current_label
                }

                # Calculate features for each site type
                for site_key in site_keys:
                    sites = site_data[site_key]
                    relative_positions = [site / max_site for site in sites] if sites else [2.0]
                    site_count = len(sites)

                    protein_result[f'{site_key}_sites'] = sites
                    protein_result[f'{site_key}_relative_positions'] = relative_positions
                    protein_result[f'{site_key}_site_count'] = site_count

                results[current_protein] = protein_result

                # Reset the site_data dictionary
                site_data = {key: [] for key in site_keys}

            # Enhanced logic for extracting protein name and label
            match = re.match(r'([\w\d\']+):\s*\(Label:\s*(\d)\)', line)
            if match:
                current_protein = match.group(1)
                current_label = int(match.group(2))

            max_site = 0
            current_site_type = None

        # Check if entering a new modification site section
        for site_type in site_types_full:
            if site_type in line:
                current_site_type = site_type
                break

        # If a new separator is encountered, exit the current site section
        if line.startswith('-' * 16) and current_site_type:
            current_site_type = None

        # In any part of the current protein, update the maximum site position
        elif current_protein and 'Site :' in line:
            # Extract the site range
            match = re.search(r'Site\s*:\s*(\d+)\s*to\s*(\d+)', line)
            if match:
                start = int(match.group(1))
                end = int(match.group(2))
                # Update the maximum site position
                max_site = max(max_site, end)

                # If within a specific site type section, record the site
                if current_site_type:
                    site_key = site_type_to_key[current_site_type]
                    site_data[site_key].append(start)

    # Process the last protein
    if current_protein:
        protein_result = {
            'length': max_site,
            'label': current_label
        }

        for site_key in site_keys:
            sites = site_data[site_key]
            relative_positions = [site / max_site for site in sites] if sites else [2.0]
            site_count = len(sites)

            protein_result[f'{site_key}_sites'] = sites
            protein_result[f'{site_key}_relative_positions'] = relative_positions
            protein_result[f'{site_key}_site_count'] = site_count

        results[current_protein] = protein_result

    print(f"Total results found: {len(results)}")
    # print(results) # Optional: uncomment to see the full raw results dictionary
    return results


def read_protein_ids(file_path):
    """
    Reads protein chain IDs from a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        list: A list of protein chain IDs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read and keep all lines directly
            protein_ids = [line.strip() for line in f if line.strip()]
        print(f"Read {len(protein_ids)} protein IDs from the file.")
        return protein_ids
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []


def filter_protein_results(all_results, protein_ids):
    """
    Filters the analysis results to keep only data for the specified protein chain IDs.

    Args:
        all_results (dict): The complete protein analysis results.
        protein_ids (list): A list of protein chain IDs to keep.

    Returns:
        dict: The filtered protein analysis results.
    """
    # The most direct way: exact key matching from protein_ids
    filtered_results = {
        pid: all_results[pid]
        for pid in protein_ids
        if pid in all_results
    }
    print(f"Original number of results: {len(all_results)}")
    print(f"Number of results after filtering: {len(filtered_results)}")
    return filtered_results


def dtw_distance(seq1, seq2):
    """
    Calculates the DTW distance between two sequences.
    Handles the special case where there are no modification sites.

    Args:
    seq1 (list): List of relative positions for the first sequence.
    seq2 (list): List of relative positions for the second sequence.

    Returns:
    float: The DTW distance.
    """
    # Handle the special case of no modification sites
    if len(seq1) == 1 and seq1[0] == 2.0 and len(seq2) == 1 and seq2[0] == 2.0:
        return 0.0  # Both sequences have no modification sites

    # A sequence with no modification sites is considered to have maximum distance
    if len(seq1) == 1 and seq1[0] == 2.0:
        return 2.0  # The first sequence has no modification sites

    if len(seq2) == 1 and seq2[0] == 2.0:
        return 2.0  # The second sequence has no modification sites

    # Standard DTW distance calculation
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # Insertion
                dtw_matrix[i, j - 1],  # Deletion
                dtw_matrix[i - 1, j - 1]  # Match
            )

    return dtw_matrix[n, m] / max(n + m, 1)


def calculate_protein_distances(filtered_results):
    """
    Calculates the overall DTW distance between proteins.

    Args:
    filtered_results (dict): The filtered protein results.

    Returns:
    dict: A dictionary of distances between proteins.
    """
    site_keys = [
        'camp_cgmp', 'protein_kinase_c', 'casein_kinase_ii',
        'tyrosine_kinase', 'n_myristoylation',
        'n_glycosylation', 'ubiquitin'
    ]
    proteins = list(filtered_results.keys())
    distance_matrix = {}

    for i in range(len(proteins)):
        for j in range(i + 1, len(proteins)):
            protein1 = proteins[i]
            protein2 = proteins[j]

            total_dtw_distance = 0
            site_distances = {}

            for site_key in site_keys:
                relative_positions_key = f'{site_key}_relative_positions'
                pos1 = filtered_results[protein1].get(relative_positions_key, [2.0])
                pos2 = filtered_results[protein2].get(relative_positions_key, [2.0])

                site_dtw_distance = dtw_distance(pos1, pos2)
                total_dtw_distance += site_dtw_distance
                site_distances[site_key] = site_dtw_distance

            distance_key = f"{protein1}_vs_{protein2}"
            distance_matrix[distance_key] = {
                'total_distance': total_dtw_distance,
                'site_distances': site_distances
            }
    return distance_matrix


def construct_distance_matrix_with_index(filtered_results):
    """
    Constructs a square distance matrix with lexicographically sorted indices.

    Args:
    filtered_results (dict): The filtered protein results.

    Returns:
    tuple: (distance_matrix, index_list, labels_list)
    """
    # Sort protein names lexicographically
    sorted_proteins = sorted(filtered_results.keys())

    # Initialize the distance matrix
    distance_matrix = np.zeros((len(sorted_proteins), len(sorted_proteins)))

    # Initialize the index and labels lists
    matrix_index = []
    matrix_labels = []

    # Calculate distances and build the matrix
    for i in range(len(sorted_proteins)):
        for j in range(len(sorted_proteins)):
            protein1 = sorted_proteins[i]
            protein2 = sorted_proteins[j]

            # If it's the same protein, the distance is 0
            if i == j:
                distance_matrix[i, j] = 0
                continue

            # Calculate the overall DTW distance between proteins
            total_dtw_distance = 0
            site_keys = [
                'camp_cgmp', 'protein_kinase_c', 'casein_kinase_ii',
                'tyrosine_kinase', 'n_myristoylation',
                'n_glycosylation', 'ubiquitin'
            ]
            for site_key in site_keys:
                relative_positions_key = f'{site_key}_relative_positions'
                pos1 = filtered_results[protein1].get(relative_positions_key, [2.0])
                pos2 = filtered_results[protein2].get(relative_positions_key, [2.0])

                site_dtw_distance = dtw_distance(pos1, pos2)
                total_dtw_distance += site_dtw_distance

            distance_matrix[i, j] = total_dtw_distance

        # Construct the index (protein name + label) and label list
        protein = sorted_proteins[i]
        label = filtered_results[protein]['label']
        matrix_index.append(f"{protein} (Label: {label})")
        matrix_labels.append(label)

    return distance_matrix, matrix_index, matrix_labels


def save_distance_matrix(distance_matrix, matrix_index, output_file='No_redundancy_dtw.csv'):
    """Saves the distance matrix to a CSV file."""
    # Create DataFrame
    df = pd.DataFrame(distance_matrix, index=matrix_index, columns=matrix_index)

    # Save to CSV
    df.to_csv(output_file, index=True)

    print(f"Distance matrix saved to {output_file}")
    print(f"Matrix size: {distance_matrix.shape}")


def minmax_normalize_distance_matrix(input_file='No_redundancy_dtw.csv', output_file='No_redundancy_dtw.csv'):
    """
    Normalizes the distance matrix using MinMaxScaler.

    Args:
    input_file (str): Input distance matrix file.
    output_file (str): Output normalized distance matrix file.
    """
    # Read the original distance matrix
    df = pd.read_csv(input_file, index_col=0)

    # Extract numerical matrix values
    matrix_values = df.values

    # MinMaxScaler normalization
    scaler = MinMaxScaler()
    normalized_matrix = scaler.fit_transform(matrix_values)

    # Create a new DataFrame, keeping the original index
    normalized_df = pd.DataFrame(
        normalized_matrix,
        index=df.index,
        columns=df.columns
    )

    # Overwrite the original file
    normalized_df.to_csv(output_file, index=True)

    print("Distance matrix normalization complete.")
    print(f"Normalization range: [{normalized_matrix.min()}, {normalized_matrix.max()}]")


def build_distance_matrix(protein_distances, proteins):
    """
    Builds the distance matrix.

    Args:
    protein_distances (dict): Dictionary of distances between proteins.
    proteins (list): List of protein names.

    Returns:
    numpy.ndarray: A symmetric distance matrix.
    """
    n = len(proteins)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            key1 = f"{proteins[i]}_vs_{proteins[j]}"
            key2 = f"{proteins[j]}_vs_{proteins[i]}"

            if key1 in protein_distances:
                distance = protein_distances[key1]['total_distance']
            elif key2 in protein_distances:
                distance = protein_distances[key2]['total_distance']
            else:
                distance = 0.0

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def extract_labels(filtered_results):
    """
    Extracts labels from the filtered results.

    Args:
    filtered_results (dict): The filtered protein results.

    Returns:
    list: A list of protein labels.
    """
    # Sort by protein ID to ensure label order matches the sorted matrix
    sorted_proteins = sorted(filtered_results.keys())
    return [filtered_results[protein]['label'] for protein in sorted_proteins]



