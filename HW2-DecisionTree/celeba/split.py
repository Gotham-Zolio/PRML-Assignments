partition_file = "partition.txt"
dataset_file = "celeba_filtered.txt"
train_file = "train.txt"
test_file = "test.txt"

image_labels = {}
with open(partition_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            filename = parts[0]
            label = int(parts[1])
            image_labels[filename] = label
        else:
            print(f"Ignoring invalid line in {partition_file}: {line}")
print(f"Loaded {len(image_labels)} labels from {partition_file}")

with open(dataset_file, 'r') as f, open(train_file, 'w') as train_f, open(test_file, 'w') as test_f:
    header = f.readline()  # Assuming the first line is header, ignore it
    for line in f:
        parts = line.strip().split()
        if len(parts) > 1:
            filename = parts[1]  # Assuming the filename is in the second column
            if filename in image_labels:
                label = image_labels[filename]
                if label == 1:
                    modified_line = ','.join(parts[2:]) + ',' + parts[0] + '\n'
                    test_f.write(modified_line)
                else:
                    modified_line = ','.join(parts[2:]) + ',' + parts[0] + '\n'
                    train_f.write(modified_line)
            else:
                print(f"Image label not found for {filename}, skipping")

print(f"Train data saved to {train_file}")
print(f"Test data saved to {test_file}")