import torch
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Пути к сохранённым файлам
train_tokens_path = 'prepare_data/csv_files/roberta_base_246403_train.pt'
test_tokens_path = 'prepare_data/csv_files/roberta_base_61668_test.pt'

# Загрузка токенов
train_tokens = torch.load(train_tokens_path)
test_tokens = torch.load(test_tokens_path)


labels = train_tokens.tensors[2]

# Преобразуем one-hot метки в индексы классов
labels = torch.argmax(labels, dim=1).tolist()

# Подсчет распределения меток
label_counts = Counter(labels)

# Вывод статистики по меткам
for label, count in label_counts.items():
    print(f"Label {label}: {count}")

labels = test_tokens.tensors[2]

# Преобразуем one-hot метки в индексы классов
labels = torch.argmax(labels, dim=1).tolist()

# Подсчет распределения меток
label_counts = Counter(labels)

# Вывод статистики по меткам
for label, count in label_counts.items():
    print(f"Label {label}: {count}")


# # Функция для отображения статистики
# def display_statistics(length_counts, dataset_name="Dataset"):
#     print(f"--- Statistics for {dataset_name} ---")
#     total_texts = sum(length_counts.values())
#     print(f"Total samples: {total_texts}")
#     print(f"Unique lengths: {len(length_counts)}")
#     print("Length distribution (top 10 most frequent):")
#     for length, count in length_counts.most_common(10):
#         print(f"Length {length}: {count} samples ({count / total_texts * 100:.2f}%)")
#     print()
#
#     # Построение графика
#     plt.figure(figsize=(10, 6))
#     lengths, counts = zip(*sorted(length_counts.items()))
#     plt.bar(lengths, counts, color='blue', alpha=0.7)
#     plt.xlabel("Text Length (Number of Tokens)")
#     plt.ylabel("Number of Samples")
#     plt.title(f"{dataset_name} Text Length Distribution")
#     plt.show()
#
# # Вывод статистики
# display_statistics(train_length_counts, dataset_name="Train Dataset")
# display_statistics(test_length_counts, dataset_name="Test Dataset")
