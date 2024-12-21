import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем данные
df = pd.read_csv('csv_files/data_set_big.csv', delimiter=';')

# Добавляем новый столбец для количества слов в тексте
df['Text_Length'] = df['Text'].apply(lambda x: len(x.split()))  # считаем количество слов

# Выводим общую статистику по количеству слов
print("General statistics on text length:")
print(df['Text_Length'].describe())

categories = ['Propaganda', 'Parody', 'Advertising', 'Manipulation', 'Fabrication', 'Satire']
category_stats = {}

# Собираем статистику для каждой категории
for category in categories:
    category_texts = df[df[category] == 1.0]
    category_lengths = category_texts['Text_Length']
    category_stats[category] = category_lengths.describe()
    print(f"\nStatistics for the category {category}:")
    print(category_stats[category])

# Визуализация
plt.figure(figsize=(14, 8))

# Гистограмма количества слов в текстах
plt.subplot(2, 1, 1)
sns.histplot(df['Text_Length'], bins=30, kde=True)
plt.title('Distribution of text length')
plt.xlabel('Text length (number of words)')
plt.ylabel('Amount')

# Гистограммы количества слов в текстах для каждой категории
plt.subplot(2, 1, 2)
for category in categories:
    category_texts = df[df[category] == 1.0]['Text_Length']
    sns.kdeplot(category_texts, label=category, fill=True)
plt.title('Distribution of text length by categories')
plt.xlabel('Text length (number of words)')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()
