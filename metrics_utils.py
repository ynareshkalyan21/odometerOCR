# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 19/12/24
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
import os
def perform_metrics(my_model, dataset, char_to_index):
    log_file = os.path.join(my_model.version_dir, "metrics.log")

    f = open(log_file, 'a')
    all_true_labels = []
    all_predicted_labels = []

    # Testing
    n = len(dataset.image_paths)
    testing = n
    print(f"Total dataset images: {n}, Testing images: {testing}")
    f.write(f"Total dataset images: {n}, Testing images: {testing}")
    success, failed = 0, []

    # for i in random.sample(range(0, n), testing):
    for i in range(n):
        test_image_path = dataset.image_paths[i]
        actual_value = dataset.labels[i]
        prediction = my_model.predict(test_image_path)

        # Convert actual and predicted values to index lists
        true_indices = [char_to_index[char] for char in actual_value]
        predicted_indices = [char_to_index[char] for char in prediction]

        # Align lengths (truncate or pad predicted indices to match true indices)
        min_len = min(len(true_indices), len(predicted_indices))
        true_indices = true_indices[:min_len]
        predicted_indices = predicted_indices[:min_len]

        all_true_labels.extend(true_indices)
        all_predicted_labels.extend(predicted_indices)

        # Check if the overall string prediction matches the actual value
        try:
            is_correct = float(prediction) == float(actual_value)
        except ValueError:
            is_correct = prediction == actual_value

        if is_correct:
            success += 1
        else:
            print(f"Predicted: {prediction} :: True value: {actual_value}, {test_image_path}")
            failed.append(i)

    # Calculate precision, recall, and F1 score for each character
    labels = list(char_to_index.values())  # List of all possible label indices
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true_labels, all_predicted_labels, labels=labels, zero_division=0
    )

    # Map index-based metrics back to characters
    index_to_char = {v: k for k, v in char_to_index.items()}
    classwise_metrics = {
        index_to_char[label]: {
            "Precision": precision[idx],
            "Recall": recall[idx],
            "F1 Score": f1[idx],
        }
        for idx, label in enumerate(labels)
    }

    # Sort by F1 Score in descending order
    sorted_metrics = sorted(classwise_metrics.items(), key=lambda x: x[1]['F1 Score'], reverse=True)

    # Prepare data for tabulation
    table_data = [
        [char, metrics['F1 Score'], metrics['Precision'], metrics['Recall']]
        for char, metrics in sorted_metrics
    ]

    # Define table headers
    headers = ["Character", "F1 Score", "Precision", "Recall"]

    # Display the table with two decimal places

    # Print overall success rate
    accuracy = (success / testing) * 100 if testing > 0 else 0
    print("\nCharacter-wise Metrics sorted by F1 Score:")
    print(tabulate(table_data, headers=headers, floatfmt=".2f"))
    print(f"\nSuccess: {success}, Failed: {testing - success}, Total testing: {testing}")
    print(f"Accuracy: {accuracy:.2f}%")
    f.write("\nCharacter-wise Metrics sorted by F1 Score:\n")
    f.write(tabulate(table_data, headers=headers, floatfmt=".2f", tablefmt='plain'))
    f.write(f"\nSuccess: {success}, Failed: {testing - success}, Total testing: {testing}")
    f.write(f"\nAccuracy: {accuracy:.2f}%")