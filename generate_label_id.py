import csv
import pandas as pd
df = pd.read_csv("training_labels.csv")
label = df['label']
label_set = set(label)
class_count = len(label_set)
label_list = list(label_set)
label_id = {}
for i in range(class_count):
    label_id[label_list[i]] = i

with open("label_dict.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for i in range(class_count):
        writer.writerow([label_list[i], label_id[label_list[i]]])