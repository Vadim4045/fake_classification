import os
import pandas as pd
from transformers import RobertaTokenizer
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.utils import shuffle

# Параметры
classes = ['Fabrication', 'Advertising', 'Manipulation', 'Propaganda', 'Satire', 'Parody']
directory = './csv_files/'
csv_file = 'data_set_big.csv'
result_file_name = 'roberta_large_6_classes'
window_size = 100  # Количество соседей для сравнения

# Инициализация токенизатора
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

# Загружаем и подготавливаем данные
df = pd.read_csv(f'{directory}{csv_file}', delimiter=';')
df.iloc[:, 0] = df.iloc[:, 0].str.replace('"', '', regex=False).str.lower()

# Проверка на наличие нужных колонок
available_columns = [col for col in df.columns if col in classes]
missing_columns = [col for col in classes if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required label columns in the dataset: {missing_columns}")

# Фильтрация текстов по длине
filtered_df = df[df.iloc[:, 0].apply(lambda text: 50 <= len(text.split()) <= 1000)]

# Создание сбалансированного датасета
min_class_size = int(min(filtered_df[classes].sum()))  # Преобразуем минимальный размер в целое число

# Отбор равного количества записей для каждого класса
balanced_dfs = [
    filtered_df[filtered_df[class_name] == 1].sample(min_class_size, random_state=42)
    for class_name in classes
]

# Объединение всех классов в один датасет
final_df = pd.concat(balanced_dfs, ignore_index=True)

# Перемешивание итогового датасета
final_df = shuffle(final_df, random_state=42)

# Извлечение текстов и меток
X_filtered = final_df.iloc[:, 0].values  # Текстовые данные
y_filtered = final_df[available_columns].reindex(columns=classes, fill_value=0).values

# Токенизация
tokens = tokenizer(list(X_filtered), padding=True, truncation=True, return_tensors='pt', max_length=512)
labels_tensor = torch.tensor(y_filtered, dtype=torch.float32)

# Объединение токенов, масок и меток в контейнеры
data_containers = list(zip(
    tokens['input_ids'],
    tokens['attention_mask'],
    labels_tensor
))

# Сортировка по лексикографическому порядку `input_ids`
print("Sorting data...")
data_containers.sort(key=lambda x: x[0].tolist())  # Сортируем по токенам

# Вычисление косинусного сходства между соседними векторами
def cosine_similarity(a, b):
    """Вычисление косинусного сходства между двумя векторами."""
    dot_product = torch.sum(a * b)
    norm_a = torch.sqrt(torch.sum(a * a))
    norm_b = torch.sqrt(torch.sum(b * b))
    return dot_product / (norm_a * norm_b)

# Удаление дубликатов
print("Removing duplicates...")
unique_data = []
for i in tqdm(range(len(data_containers))):
    current_input_ids = data_containers[i][0].float()
    is_duplicate = False  # Флаг, указывающий, является ли текущий элемент дубликатом

    # Сравнение с ближайшими соседями в пределах окна
    for j in range(1, window_size + 1):
        if i + j < len(data_containers):  # Проверяем, чтобы индекс не выходил за пределы
            neighbor_input_ids = data_containers[i + j][0].float()
            similarity = cosine_similarity(current_input_ids, neighbor_input_ids)
            if similarity > 0.95:  # Если сходство выше порога, помечаем как дубликат
                is_duplicate = True
                break

    if not is_duplicate:  # Добавляем только уникальные элементы
        unique_data.append(data_containers[i])

print(f"Number of unique samples after filtering: {len(unique_data)}")

# Перемешивание данных
print("Shuffling data...")
unique_data = shuffle(unique_data, random_state=42)

# Разделение на обучающую и тестовую выборки
train_size = int(0.8 * len(unique_data))
train_containers = unique_data[:train_size]
test_containers = unique_data[train_size:]

# Преобразование в тензоры для датасетов
train_input_ids, train_attention_mask, train_labels = zip(*train_containers)
test_input_ids, test_attention_mask, test_labels = zip(*test_containers)

train_dataset = TensorDataset(
    torch.stack(train_input_ids),
    torch.stack(train_attention_mask),
    torch.stack(train_labels)
)

test_dataset = TensorDataset(
    torch.stack(test_input_ids),
    torch.stack(test_attention_mask),
    torch.stack(test_labels)
)

# Сохранение датасетов
train_file_path = f'{directory}/{result_file_name}_{len(train_dataset)}_train.pt'
test_file_path = f'{directory}/{result_file_name}_{len(test_dataset)}_test.pt'

torch.save(train_dataset, train_file_path)
torch.save(test_dataset, test_file_path)

print(f"Train dataset saved to: {train_file_path}")
print(f"Test dataset saved to: {test_file_path}")
