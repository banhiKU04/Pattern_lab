import csv
import random
import math
import os

# ----------------- Load Dataset -----------------
filename = os.path.join(os.path.dirname(__file__), "irish.csv")

dataset = []
with open(filename, 'r', newline='', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if len(row) < 5:
            continue

        # Convert first 4 values to floats manually
        features = []
        valid_row = True
        for i in range(4):
            try:
                features.append(float(row[i]))
            except:
                valid_row = False
                break
        if not valid_row:
            continue

        label = row[4].strip()
        dataset.append(features + [label])

# ----------------- Shuffle Dataset (manual Fisher–Yates) -----------------
n = len(dataset)
for i in range(n - 1, 0, -1):
    j = int(random.random() * (i + 1))
    temp = dataset[i]
    dataset[i] = dataset[j]
    dataset[j] = temp

# ----------------- Split Dataset -----------------
train_ratio = float(input("Enter training set ratio (e.g., 0.7): "))
train_size = int(n * train_ratio)

train_set = []
test_set = []
for i in range(n):
    if i < train_size:
        train_set.append(dataset[i])
    else:
        test_set.append(dataset[i])

# ----------------- Separate by Class -----------------
separated = {}
for row in train_set:
    label = row[-1]
    if label not in separated:
        separated[label] = []
    separated[label].append(row[:-1])

# ----------------- Calculate Mean & StdDev manually -----------------
summaries = {}
for class_value in separated:
    rows = separated[class_value]
    num_features = len(rows[0])
    feature_summaries = []

    for col_index in range(num_features):
        # Mean
        total = 0.0
        count = 0
        for row in rows:
            total += row[col_index]
            count += 1
        mean_val = total / count

        # StdDev
        variance_total = 0.0
        for row in rows:
            diff = row[col_index] - mean_val
            variance_total += diff * diff
        variance = variance_total / count

        # sqrt without math.sqrt (Newton's method)
        x = variance
        if x == 0:
            stdev_val = 0
        else:
            guess = x
            for _ in range(10):
                guess = 0.5 * (guess + x / guess)
            stdev_val = guess

        feature_summaries.append((mean_val, stdev_val))
    summaries[class_value] = feature_summaries

# ----------------- Test Model -----------------
predictions = []
for row in test_set:
    probabilities = {}
    for class_value in summaries:
        prob = 1.0
        for i in range(len(summaries[class_value])):
            mean_val, stdev_val = summaries[class_value][i]
            x = row[i]
            if stdev_val == 0:
                p = 1.0 if x == mean_val else 0.0
            else:
                exponent = math.exp(-((x - mean_val) ** 2) / (2 * stdev_val ** 2))
                p = (1.0 / ((2 * 3.141592653589793) ** 0.5 * stdev_val)) * exponent
            prob *= p
        probabilities[class_value] = prob

    # Normalize
    total_prob = 0.0
    for label in probabilities:
        total_prob += probabilities[label]
    for label in probabilities:
        if total_prob != 0:
            probabilities[label] = probabilities[label] / total_prob
        else:
            probabilities[label] = 0

    # Find max probability manually
    best_label = None
    best_prob = -1
    for label in probabilities:
        if probabilities[label] > best_prob:
            best_prob = probabilities[label]
            best_label = label

    correct = "Correct" if best_label == row[-1] else "Wrong"
    predictions.append((row[:-1], best_label, probabilities, correct, row[-1]))

# ----------------- Output -----------------
print(f"{'Features':40} {'Predicted':15} {'Iris-setosa':15} {'Iris-versicolor':15} {'Iris-virginica':15} {'Result':10} {'Actual'}")
print("-" * 120)

correct_count = 0
for features, predicted_label, probs, correct, actual in predictions:
    print(f"{str(features):40} {predicted_label:15} "
          f"{probs.get('Iris-setosa', 0):<15.4f} "
          f"{probs.get('Iris-versicolor', 0):<15.4f} "
          f"{probs.get('Iris-virginica', 0):<15.4f} "
          f"{correct:10} {actual}")
    if correct == "Correct":
        correct_count += 1

accuracy = (correct_count / len(predictions)) * 100
print("\nOverall Accuracy: {:.2f}%".format(accuracy))
