import csv
import re
import os

topic_pattern = r'^(Story)*( )*\d+( )*[.:)]'
empty_pattern = r'^\n'
remove_num_pattern = r'^(\d)+(\*)*\.( )*(\*)*( )*'

directory = './texts/'
styles = ['Fabrication', 'Advertising', 'Manipulation', 'Propaganda', 'Satire', 'Parody']

def clean_text(text):
    cleaned_text1 = re.sub(topic_pattern, '', text).strip()
    cleaned_text2 = re.sub(r'^(-)+', '', cleaned_text1).strip()
    cleaned_text3 = re.sub(r'\n', '', cleaned_text2).strip()
    cleaned_text = re.sub(r'[^\w\s,:.!\-?]', '', cleaned_text3).strip()
    return cleaned_text.strip()

csv_file = f'./csv_files/data_set_big.csv'

# Открытие файла для записи данных
with open(csv_file, 'w', newline='') as output_file:
    writer = csv.writer(output_file, delimiter=';')

    # Запись заголовков CSV (если нужно)
    writer.writerow(['Text'] + styles)

    # Обработка файлов для каждого стиля
    for label in styles:
        one_hot = [1.0 if label == class_name else 0.0 for class_name in styles]
        matching_files = [os.path.join(directory, f) for f in os.listdir(directory) if label in f]

        if not matching_files:
            print(f"No files found for class: {label}")
            continue

        # Обработка каждого файла
        for filepath in matching_files:
            with open(filepath, 'r') as file:
                lines = file.readlines()
                line2 = ''
                assistant = False
                empty_line = False
                for line in lines:
                    if line.startswith('Assistant'):
                        assistant = True
                        continue
                    if assistant:
                        assistant = False
                        continue

                    if re.match(r'^\n', line) or re.match(r'^--', line) or len(line.split()) < 50:
                        if 50 < len(line2.split()) < 1000:
                            text = clean_text(line2)
                            if 50 <= len(text.split()) <= 1000:
                                row = [text] + one_hot
                                writer.writerow(row)
                        line2 = ''
                        continue

                    if re.match(topic_pattern, line) or re.match(remove_num_pattern, line) or re.match(r'^[A-Z]{3, }', line):
                        if 50 <= len(line2.split()) <= 1000:
                            text = clean_text(line2)
                            if 50 <= len(text.split()) <= 1000:
                                row = [text] + one_hot
                                writer.writerow(row)
                        line2 = ''
                    else:
                        line2 += ' ' + line

                # После окончания чтения файла добавляем последнюю строку, если она есть
                if 50 <= len(line2.split()) <= 1000:
                    text = clean_text(line2)
                    if 50 <= len(text.split()) <= 1000:
                        row = [text] + one_hot
                        writer.writerow(row)
