import os
import pandas as pd
import re

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9.,!?;:\'"\s]', '', text)

directory = './texts/'
classes = ['Fabrication', 'Advertising', 'Manipulation', 'Propaganda', 'Satire', 'Parody']

data = []

for label in classes:
    one_hot = [1.0 if label == class_name else 0.0 for class_name in classes]
    matching_files = [os.path.join(directory, f) for f in os.listdir(directory) if label in f]

    if not matching_files:
        print(f"No files found for class: {label}")
        continue

    for filepath in matching_files:
        print(f"Processing file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                line = clean_text(line.strip())
                if 50 < len(line) < 512:
                    data.append({'Text': line, **dict(zip(classes, one_hot))})

df = pd.DataFrame(data)

output_csv = f'./labeled_data_{len(data)}.csv'
df.to_csv(output_csv, index=False, encoding='utf-8')
print(f"Labeled CSV file saved to: {output_csv}")
