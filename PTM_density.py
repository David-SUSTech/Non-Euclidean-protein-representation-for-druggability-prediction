import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


# The following two lines for Chinese font display have been removed as they are no longer necessary.
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False


# This elaborately-designed function is pipeline-friendly because it directly reads the raw data derived from
# the online platform NPS@, which has potential usage for constructing a pipeline in the future.
def extract_4hrl_improved(file_path, common_proteins=None):
    """Extracts protein modification site information, with an option to keep only common proteins."""
    # The types of sites we care about
    modification_sites = [
        'cAMP- and cGMP-dependent protein kinase phosphorylation site.',
        'Protein kinase C phosphorylation site.',
        'Casein kinase II phosphorylation site.',
        'Tyrosine kinase phosphorylation site 1.',
        'N-myristoylation site.',
        'N-glycosylation site.',
        'Ubiquitin specific protease (USP) domain signature 1.'
    ]

    # Simplify column names by mapping
    column_mapping = {
        'cAMP- and cGMP-dependent protein kinase phosphorylation site.': 'cAMP_cGMP_count',
        'Protein kinase C phosphorylation site.': 'PKC_count',
        'Casein kinase II phosphorylation site.': 'CKII_count',
        'Tyrosine kinase phosphorylation site 1.': 'TK_count',
        'N-myristoylation site.': 'n_myristoylation_count',
        'N-glycosylation site.': 'N_glycosylation_count',
        'Ubiquitin specific protease (USP) domain signature 1.': 'USP_count'
    }

    results = []
    current_protein = None
    current_counts = {site: 0 for site in modification_sites}
    current_site_type = None
    current_label = None
    total_found = 0
    filtered_by_common = 0
    processed_ids = set()

    # Adding a variable for detecting the terminal PTM site of the proteins
    current_max_position = 0

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Matching protein ids
            id_match = re.match(r'(\d+[A-Za-z0-9]{3}\'(\d+)\')', line)

            if id_match or line.startswith('#') or line.startswith('*'):
                # Once another ID is found, save the information obtained for the previous protein
                if current_protein:
                    protein_info = {
                        'protein_id': current_protein,
                        'label': current_label,
                        'chain_length': current_max_position
                    }
                    # Site counting
                    for site in modification_sites:
                        if site in column_mapping:
                            # Save site counting
                            protein_info[column_mapping[site]] = current_counts[site]

                    results.append(protein_info)

                # Obtain new protein ID
                if id_match:
                    current_protein = id_match.group(1)

                    # Check if skipping is needed
                    if current_protein in processed_ids:
                        current_protein = None
                        continue

                    if common_proteins is not None and current_protein not in common_proteins:
                        filtered_by_common += 1
                        current_protein = None
                        continue

                    processed_ids.add(current_protein)
                    total_found += 1

                    # Obtain labels
                    label_match = re.search(r'Label:\s*(\d+)', line)
                    if label_match:
                        current_label = int(label_match.group(1))
                    else:
                        current_label = 0
                else:
                    # Deal with manual signs '*' & '#'. Here '#' means label 0, while '*' means 1
                    current_protein = line
                    current_label = 0 if line.startswith('#') else 1

                # Reset the counter and the current modification type
                current_counts = {site: 0 for site in modification_sites}
                current_site_type = None
                current_max_position = 0  # Reset length
                continue

            # Check if we are currently at modification information
            found_site = False
            for site in modification_sites:
                if site in line:
                    current_site_type = site
                    found_site = True
                    break

            if line.startswith('---'):
                found_site = False

            # If it is a modification site, continue processing information
            if found_site:
                continue

            # Renew site count
            if current_site_type and line.startswith('Site'):
                current_counts[current_site_type] += 1

                # Obtain site information to get effective length
                pos_match = re.search(r'Site\s*:\s*\d+\s*to\s*(\d+)', line)
                if pos_match:
                    end_position = int(pos_match.group(1))
                    if end_position > current_max_position:
                        current_max_position = end_position

        # Save the final protein's information
        if current_protein:
            protein_info = {
                'protein_id': current_protein,
                'label': current_label,
                'chain_length': current_max_position  # Add length information
            }
            # Add counts for different types of sites
            for site in modification_sites:
                if site in column_mapping:
                    protein_info[column_mapping[site]] = current_counts[site]

            results.append(protein_info)

    print(f"Total proteins found: {total_found}")
    if common_proteins is not None:
        print(f"{filtered_by_common} of them were filtered out as they are not in the common proteins list.")
    print(f"Finally kept {len(results)} unique proteins.")

    df = pd.DataFrame(results)

    # Ensure all columns exist
    for col in column_mapping.values():
        if col not in df.columns:
            df[col] = 0

    # Calculate the density for each site type (count / chain length)
    for col in column_mapping.values():
        density_col = col.replace('_count', '_density')
        # Prevent division by zero
        df[density_col] = df.apply(
            lambda row: row[col] / row['chain_length'] if row['chain_length'] > 0 else 0,
            axis=1
        )

    return df


def load_common_protein_ids(file_path):
    """Loads common protein IDs from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Warning: Common proteins file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"Error reading common proteins: {e}")
        return None


def calculate_distance_matrix(df, feature_columns, metric='cityblock'):
    """
    Calculates the distance matrix between proteins.

    Args:
    df (DataFrame): DataFrame containing protein feature data.
    feature_columns (list): List of feature column names for distance calculation.
    metric (str): Distance metric method, e.g., 'euclidean', 'cityblock', 'cosine'.

    Returns:
    distance_matrix (DataFrame): Distance matrix between proteins.
    """
    # Extract feature data
    X = df[feature_columns].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate distance matrix
    distances = pdist(X_scaled, metric=metric)
    distance_matrix = squareform(distances)

    # Create DataFrame with protein IDs as index and columns
    protein_ids = df['protein_id'].values
    distance_df = pd.DataFrame(distance_matrix, index=protein_ids, columns=protein_ids)

    return distance_df


def normalize_distance_matrix(distance_matrix):
    """Normalizes the distance matrix using MinMaxScaler."""
    # Create MinMaxScaler
    scaler = MinMaxScaler()

    # Get index and columns to restore them after normalization
    indices = distance_matrix.index
    columns = distance_matrix.columns

    # Convert distance matrix to a numpy array for normalization
    distance_array = distance_matrix.values

    # Reshape to fit the scaler's requirements (n_samples, n_features)
    n_samples = distance_array.shape[0]
    flattened = distance_array.reshape(-1, 1)

    # Apply normalization
    normalized_flat = scaler.fit_transform(flattened)

    # Reshape back to matrix form
    normalized_matrix = normalized_flat.reshape(n_samples, n_samples)

    # Convert back to a DataFrame, preserving the original index and columns
    normalized_df = pd.DataFrame(normalized_matrix, index=indices, columns=columns)

    return normalized_df


def replace_matrix_indices(distance_matrix, labels_file):
    """
    Replaces the row/column indices of the distance matrix with a combination of ID and label from protein_labels.csv.

    Args:
    distance_matrix (DataFrame): The distance matrix.
    labels_file (str): Path to the CSV file containing protein IDs and labels.

    Returns:
    updated_matrix (DataFrame): The distance matrix with updated indices.
    """
    try:
        # Read the CSV file using pandas, explicitly specifying space as a separator and no header
        labels_data = pd.read_csv(labels_file, sep=' ', header=None)
        print(f"Shape of the loaded labels data: {labels_data.shape}")

        # If the file has at least two columns, use the first two for ID and label
        if labels_data.shape[1] >= 2:
            # Create an ID-to-label map
            id_to_label = {}
            for i in range(len(labels_data)):
                protein_id = str(labels_data.iloc[i, 0])  # Ensure ID is a string
                label = str(labels_data.iloc[i, 1])  # Ensure label is a string
                id_to_label[protein_id] = label

            print(f"Successfully created {len(id_to_label)} ID-to-label mappings.")
            # Print some examples for verification
            items = list(id_to_label.items())
            if items:
                print(f"First few examples of ID-to-label mapping: {items[:5]}")

            # Get the current matrix indices and convert to string type
            current_indices = [str(idx) for idx in distance_matrix.index]
            matrix_indices_set = set(current_indices)

            # Check the match between ID map and matrix indices
            matched_ids = matrix_indices_set.intersection(set(id_to_label.keys()))
            print(f"{len(matched_ids)}/{len(current_indices)} matrix indices were found in the labels file.")

            # Create new indices
            new_indices = []
            for idx in distance_matrix.index:
                idx_str = str(idx)
                if idx_str in id_to_label:
                    new_indices.append(f"{idx_str} (Label: {id_to_label[idx_str]})")
                else:
                    new_indices.append(f"{idx_str} (Label: Unknown)")

            # Create a new matrix and set the new index and column names
            updated_matrix = distance_matrix.copy()
            updated_matrix.index = new_indices
            updated_matrix.columns = new_indices

            print(f"Index replacement complete. New matrix size: {updated_matrix.shape}")
            return updated_matrix
        else:
            return distance_matrix

    except Exception as e:
        print(f"Error during index replacement: {e}")
        import traceback
        traceback.print_exc()
        # Return the original matrix if an error occurs
        return distance_matrix


def perform_knn_cross_validation(X, y, n_neighbors_range=range(1, 21)):
    """
    Performs Leave-One-Out cross-validation for a KNN classifier to find the optimal K.

    Args:
    X: Feature data.
    y: Label data.
    n_neighbors_range: Range of K values to test.

    Returns:
    best_k: The optimal K value.
    best_score: The average accuracy for the optimal K.
    cv_results: Cross-validation results for all K values.
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Store cross-validation scores for each K
    cv_scores = []

    # Use Leave-One-Out Cross-Validation
    loo = LeaveOneOut()

    # Perform LOOCV for each K value
    for k in n_neighbors_range:
        # Create KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)

        # Perform LOOCV
        scores = cross_val_score(knn, X_scaled, y, cv=loo, scoring='accuracy')

        # Store the mean score
        cv_scores.append(np.mean(scores))
        print(f"K = {k}, Mean accuracy with LOOCV: {np.mean(scores):.4f}")

    # Find the optimal K
    best_k = n_neighbors_range[np.argmax(cv_scores)]
    best_score = np.max(cv_scores)

    print(f"\nOptimal K: {best_k}, with mean accuracy: {best_score:.4f}")

    # Return the results
    cv_results = {
        'k_values': list(n_neighbors_range),
        'cv_scores': cv_scores,
        'best_k': best_k,
        'best_score': best_score
    }

    return best_k, best_score, cv_results


# Main program
try:
    # Load common protein IDs
    common_protein_file = 'No_redundancy_ids.txt'
    common_protein_ids = load_common_protein_ids(common_protein_file)

    if common_protein_ids:
        print(f"Loaded {len(common_protein_ids)} No_redundancy_ids as a filter criterion.")
    else:
        print("No_redundancy_ids not provided, will process all proteins.")

    # Extract data (filtering with common_protein_ids), using the improved extraction function
    df = extract_4hrl_improved('Modification_sites_for_fusion.txt', common_protein_ids)

    # Check for duplicate IDs
    if df['protein_id'].duplicated().any():
        print("Warning: Duplicate IDs still exist in the data, performing final deduplication.")
        df = df.drop_duplicates(subset=['protein_id'])

    print("Final dataset size:", df.shape)
    print("Data preview:")
    print(df.head())

    # Display non-zero counts for each column to confirm extraction is normal
    non_zero_counts = {col: (df[col] > 0).sum() for col in df.columns if
                       col not in ['protein_id', 'label', 'chain_length'] and not col.endswith('_density')}
    print("\nNon-zero counts for each modification site:")
    for site, count in non_zero_counts.items():
        print(f"{site}: {count}")

    # Output all extracted protein IDs
    all_protein_ids = df['protein_id'].tolist()
    print(f"\nExtracted a total of {len(all_protein_ids)} unique protein IDs.")

    # New line: Print all protein IDs
    print("\nList of all protein IDs:")
    print(all_protein_ids)  # This is the new line to print all protein IDs

    # Display a sample of IDs
    display_count = min(10, len(all_protein_ids))
    print(f"\nFirst {display_count} protein ID examples:")
    for i in range(display_count):
        print(all_protein_ids[i])
    print("..." if len(all_protein_ids) > 10 else "")

    # Terminate if the dataset is empty
    if df.empty:
        print("Warning: The dataset is empty after filtering, cannot continue analysis.")
        exit()

    # Prepare features and target variable - modified to use density instead of count
    # Create density feature column names
    feature_columns = [
        'cAMP_cGMP_density', 'PKC_density', 'CKII_density', 'TK_density',
        'n_myristoylation_density', 'N_glycosylation_density', 'USP_density'
    ]
    X = df[feature_columns]
    y = df['label']

    # Calculate distance matrix (using cityblock/Manhattan distance)
    print("\nCalculating distance matrix between proteins (using Manhattan distance)...")
    distance_matrix = calculate_distance_matrix(df, feature_columns, metric='cityblock')
    print("Distance matrix size:", distance_matrix.shape)

    # Save the original distance matrix to a CSV file
    distance_matrix.to_csv('site_density_matrix.csv')  # Changed filename to reflect density usage
    print("Distance matrix saved to 'site_density_matrix.csv'")

    # Normalize the distance matrix with MinMaxScaler
    print("\nNormalizing the distance matrix with MinMaxScaler...")
    normalized_matrix = normalize_distance_matrix(distance_matrix)
    print("Normalized distance matrix size:", normalized_matrix.shape)

    # Replace matrix indices with ID info from protein_labels.csv
    print("\nReplacing matrix indices with ID information from protein_labels.csv...")
    labeled_matrix = replace_matrix_indices(normalized_matrix, 'protein_labels.csv')
    print("Distance matrix size after updating indices:", labeled_matrix.shape)

    # Save the normalized and index-updated distance matrix
    labeled_matrix.to_csv('No_redundancy_density_matrix.csv')
    print("Normalized matrix with updated indices has been saved to 'No_redundancy_density_matrix.csv'")

except Exception as e:
    print(f"An error occurred during execution: {e}")
    import traceback

    traceback.print_exc()