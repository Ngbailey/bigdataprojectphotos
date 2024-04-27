true_labels = [5, 0, 4, 2, 5, 1, 4, 3, 3, 2, 2, 3, 0, 4, 1, 5, 3, 5, 0, 3, 1, 5, 3, 3, 1, 5, 1, 2, 0, 3, 5, 3, 5, 0, 2, 2, 4, 4, 0, 3, 0, 5, 2, 0, 0, 1, 1, 0, 4, 5, 4, 1, 3, 2, 3, 5, 1, 3, 3, 1, 0, 4, 5, 3, 2, 0, 3, 4, 4, 3, 3, 0, 0, 3, 3, 3, 5, 2, 1, 4, 0, 5, 4, 3, 0, 3, 3, 2, 3, 5, 5, 3, 3, 1, 4, 2]

# Duplicate each number in the list
true_labels = [label for label in true_labels for _ in range(2)]

print(true_labels)
