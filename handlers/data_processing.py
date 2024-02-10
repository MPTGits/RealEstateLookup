from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from bulstem.stem import BulStemmer
from nltk import word_tokenize
import string
from stop_words import STOP_WORDS
import re

# nltk.download()


stemmer = BulStemmer.from_file('stem-context-1.txt', min_freq=2, left_context=1)

def extract_numbers(s):
    return int(re.findall("\d+", s)[0])
def tokenize_text(text):
    return word_tokenize(text)

def remove_stop_words(text):
    return [stemmer.stem(word) for word in text if (word and stemmer.stem(word) not in STOP_WORDS)]
def remove_punctuation(text):
    return "".join([char for char in text if char not in string.punctuation and char != '' and not char.isnumeric()])
def remove_english_words(text):
    return [word for word in text if not re.match(r'^[a-zA-Z]+$', word)]


def display_clusters(clusters_dbscan, top_words):
    # Инициализиране на речник за съхранение на думите по кластери
    clusters_words = defaultdict(list)

    # Групиране на думите по техните кластери
    for word, label in zip(top_words, clusters_dbscan):
        clusters_words[label].append(word)

    # Предполагаме, че всеки кластер има асоцииран цвят или номер за идентификация
    cluster_colors = {0: 'Red', 1: 'Blue', 2: 'Green', -1: 'Black'}  # Примерни цветове/идентификатори за кластерите

    for label, words in clusters_words.items():
        # Пропускане на шума с етикет -1, ако е необходимо
        if label == -1:
            continue
        color = cluster_colors.get(label, 'Unknown')  # Получаване на цвета или идентификатора на кластера
        print(f"Кластер {label} (Цвят: {color}):")
        print(words, "\n")  # Показване на думите в кластера


df = pd.read_csv('all_real_estates.csv')

df = df.dropna(subset=['Oписание', 'Цена'])

df['Цена'] = df['Цена'].str.replace(',', '').astype(float)
df['Oписание'] = df['Oписание'].str.lower()
df['Oписание'] = df['Oписание'].apply(tokenize_text)
df['Oписание'] = df['Oписание'].apply(lambda x: list(filter(lambda x: x != '', [remove_punctuation(word) for word in x])))
df['Oписание'] = df['Oписание'].apply(remove_stop_words)
df['Oписание'] = df['Oписание'].apply(remove_english_words)

mean_price = df['Цена'].mean()

# # Обучение на Word2Vec модел
model = Word2Vec(sentences=df['Oписание'], vector_size=100, window=5, min_count=1, workers=4)

# Извличане на топ 400 думи въз основа на тяхната честота (Word2Vec автоматично ги подрежда)
top_words = model.wv.index_to_key[:400]
top_vectors = [model.wv[word] for word in top_words]


# Преобразуване на векторите в NumPy масив и прилагане на t-SNE
vectors_np = np.array(top_vectors)
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
top_vectors_tsne = tsne.fit_transform(vectors_np)

# Прилагане на DBSCAN кластеризация
# Може да се наложи да експериментирате с различни стойности за eps и min_samples, за да намерите най-добрата конфигурация
dbscan = DBSCAN(eps=7, min_samples=5).fit(top_vectors_tsne)

# Етикетите, върнати от DBSCAN, където -1 означава шум
clusters_dbscan = dbscan.labels_

# Намиране на броя на кластерите (изключвайки шума)
n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)

# Предполагаме, че имате данни във формат на NumPy масив `data`
data = np.array(top_vectors)  # Ако използвате top_vectors от Word2Vec

# Изчисляване на разстоянията до k-тия най-близък съсед (например k = 4)
neighbors = NearestNeighbors(n_neighbors=4)
neighbors_fit = neighbors.fit(data)
distances, indices = neighbors_fit.kneighbors(data)

# Сортиране на разстоянията
sorted_distances = np.sort(distances[:, 3], axis=0)

# Визуализация на разстоянията
plt.figure(figsize=(10, 6))
plt.plot(sorted_distances)
plt.xlabel('Индекс')
plt.ylabel('Разстояние до 4-тия най-близък съсед')
plt.title('Определяне на EPS стойност за DBSCAN')
plt.show()

# Визуализация на кластерите
plt.figure(figsize=(50, 30))
unique_labels = set(clusters_dbscan)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Черен цвят, използван за шум.
        col = 'k'

    class_member_mask = (clusters_dbscan == k)

    xy = top_vectors_tsne[class_member_mask]
    for i, (x, y) in enumerate(xy):
        plt.plot(x, y, 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        # Намиране на индекса на точката в оригиналния масив, за да се получи съответстващата дума
        word_index = np.where(class_member_mask)[0][i]  # Индексът на думата във филтрирания масив
        word = top_words[word_index]  # Съответната дума
        plt.annotate(word, (x, y), textcoords="offset points", xytext=(5, 2), ha='right', fontsize=12)

plt.title(f'DBSCAN визуализация на топ 400 думи с {n_clusters_dbscan} кластера', fontsize=20)
plt.legend()
plt.show()

display_clusters(clusters_dbscan, top_words)


## Метод за откриване на потенциално фалшиви обяви на недвижими имоти от тук надолу

# median_price = df['Цена'].median()
#
# print(median_price)
#
# cheaper_apartments_df = df[df['Цена'] <= median_price]
# more_expensive_apartments_df = df[df['Цена'] > median_price]
#
# # Define a function to analyze the most common words in descriptions
# def analyze_common_words(df):
#     all_words = []
#     for description in df['Oписание']:
#             all_words.extend(description)
#     return Counter(all_words).most_common(30)
#
# # Analyze the most common words for cheaper and more expensive apartments
# most_common_words_cheaper = analyze_common_words(cheaper_apartments_df)
# most_common_words_more_expensive = analyze_common_words(more_expensive_apartments_df)
#
#
# def find_unique_words(common_words_set1, common_words_set2):
#     # Extract words from each set
#     words_set1 = {word[0] for word in common_words_set1}
#     words_set2 = {word[0] for word in common_words_set2}
#
#     # Find unique words for each set
#     unique_to_set1 = words_set1 - words_set2
#     unique_to_set2 = words_set2 - words_set1
#
#     return unique_to_set1, unique_to_set2
#
#
# # Find unique words for cheaper and more expensive apartments
# unique_words_cheaper, unique_words_more_expensive = find_unique_words(most_common_words_cheaper,
#                                                                       most_common_words_more_expensive)
#
#
#
# print("Cheap",unique_words_cheaper)
# print("Exp", unique_words_more_expensive)
#
#
#
# # Изчисляване на броя думи в скъпите и евтините обяви
# total_expensive_words = sum(1 for _ in unique_words_more_expensive)
# total_cheap_words = sum(1 for _ in unique_words_cheaper)
# threshold_price = median_price * 0.85
#
# # Определяне на прага за минимален брой срещания, който е 20% от броя думи в скъпите обяви
# threshold_expensive = total_expensive_words * 0.6
#
# # Намерете обявите, които съдържат поне 20% от най-често срещаните думи в скъпите обяви и са сред евтините
# cheap_ads_with_expensive_words = cheaper_apartments_df[
#     (cheaper_apartments_df['Oписание'].apply(lambda desc: sum(1 for word in desc if word in unique_words_more_expensive) >= threshold_expensive)) &
#     (cheaper_apartments_df['Oписание'].apply(lambda desc: sum(1 for word in desc if word in unique_words_cheaper) < total_cheap_words * 0.6)) &
#     (cheaper_apartments_df['Цена'] <= threshold_price)]
#
# # Изведете тези обяви
# # Изведете думите и цената за всяка от тези обяви
# for index, row in cheap_ads_with_expensive_words.iterrows():
#     print(f"Обява {index}: {row['Oписание']}, Цена: {row['Цена']}")
# print(len(cheap_ads_with_expensive_words))