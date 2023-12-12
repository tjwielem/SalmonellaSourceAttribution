from sklearn.utils import resample

from sklearn.utils import resample
import numpy as np

def balance_dataset(X, y, upsample=True):
    """
    Balance the dataset by either upsampling or downsampling.

    Parameters:
    - X: Features
    - y: Labels
    - upsample: If True, perform upsampling; if False, perform downsampling

    Returns:
    - X_balanced: Balanced features
    - y_balanced: Balanced labels
    """
    # Combine features and labels
    data = list(zip(X, y))

    # Find the class with the maximum and minimum number of examples
    unique_labels, class_counts = np.unique(y, return_counts=True)
    max_class_count = max(class_counts)
    min_class_count = min(class_counts)

    # Separate data by class
    class_data = {label: [X_i for X_i, label_i in data if label_i == label] for label in unique_labels}

    # Upsample or downsample each class
    balanced_data = []
    for label in class_data:
        if upsample:
            # Upsample by repeating examples
            class_data[label] = resample(class_data[label], n_samples=max_class_count, replace=True)
        else:
            # Downsample by randomly removing examples
            class_data[label] = resample(class_data[label], n_samples=min_class_count, replace=False)

        # Add balanced data to the final list
        balanced_data.extend([(X_i, label) for X_i in class_data[label]])

    # Shuffle the balanced data
    balanced_data = resample(balanced_data, n_samples=len(balanced_data), replace=False)

    # Separate features and labels again
    X_balanced, y_balanced = zip(*balanced_data)

    return np.array(X_balanced), np.array(y_balanced)