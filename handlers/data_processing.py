import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score
from nltk.tokenize import word_tokenize
import string
from stop_words import STOP_WORDS
import re
from bulstem.stem import BulStemmer


from nltk.util import ngrams
def get_n_grams(real_estates):
    all_real_estates = []
    for real_estate_desc in real_estates:
        all_real_estates.extend(real_estate_desc)
    # extracting n-grams
    unigrams = ngrams(all_real_estates, 1)
    bigrams = ngrams(all_real_estates, 2)
    trigrams = ngrams(all_real_estates, 3)
    # getting dictionary of all n-grams for corpus
    gram_dict = {
        "Unigram": unigrams,
        "Bigram": bigrams,
        "Trigram":trigrams}
    return gram_dict

stemmer = BulStemmer.from_file('stem-context-1.txt', min_freq=2, left_context=1)
def extract_numbers(s):
    return int(re.findall("\d+", s)[0])
def tokenize_text(text):
    return word_tokenize(text)

stop_words = STOP_WORDS
def remove_stop_words(text):
    return [stemmer.stem(word) for word in text if (word and stemmer.stem(word) not in stop_words)]
def remove_punctuation(text):
    return "".join([char for char in text if char not in string.punctuation and char != '' and not char.isnumeric()])


# Load the dataset
df = pd.read_csv('all_real_estates.csv')

# Preprocess the property descriptions
df = df.dropna(subset=['Oписание', 'Цена'])
# print(keywords(df['Oписание'][1]))
df["Размер"] = df["Размер"].apply(lambda x: extract_numbers(x))
df["Размер"] = df["Размер"].astype(int)
#df = df[(df['Размер'] >= 60) & (df['Размер'] <= 100)]
df['Цена'] = df['Цена'].str.replace(',', '').astype(float)

#df = df[(df['Цена'] >= 260000)]

df['Oписание'] = df['Oписание'].str.lower()
df['Oписание'] = df['Oписание'].apply(tokenize_text)
df['Oписание'] = df['Oписание'].apply(lambda x: list(filter(lambda x: x != '', [remove_punctuation(word) for word in x])))
df['Oписание'] = df['Oписание'].apply(remove_stop_words)

n_grams = get_n_grams(df['Oписание'])

unigram_counts = {}
bigram_counts = {}
trigram_counts = {}


for unigram in n_grams.get('Unigram'):
        unigram_counts[unigram] = unigram_counts.get(unigram, 0) +1
for bigram in n_grams.get('Bigram'):
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) +1
for trigram in n_grams.get('Trigram'):
        trigram_counts[trigram] = trigram_counts.get(trigram, 0) +1

sorted_unigrams = sorted([(key,value) for key,value in unigram_counts.items()],key=lambda x:-x[1])
sorted_bigrams = sorted([(key,value) for key,value in bigram_counts.items()],key=lambda x:-x[1])
sorted_trigrams = sorted([(key,value) for key,value in trigram_counts.items()],key=lambda x:-x[1])

print(sorted_unigrams[1:15])
print(sorted_bigrams[1:15])

df['Oписание'] = df['Oписание'].apply(lambda x: ' '.join(x))
df['Oписание'] = df['Oписание'].str.strip()

vectorizer = TfidfVectorizer(max_features=800, min_df=0.03, max_df=0.8)
descriptions = vectorizer.fit_transform(df['Oписание'])

import numpy as np

non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

from sklearn.preprocessing import LabelEncoder

# # Handle non-numeric columns
if non_numeric_columns.shape[0] > 0:
    # Use label encoding for ordinal variables
    le = LabelEncoder()
    for column in non_numeric_columns:
        if column != 'Oписание':
            df[column] = le.fit_transform(df[column])

df.dropna(axis=1, inplace=True)

new_df = pd.DataFrame({'Oписание': descriptions, 'Размер': df['Размер'], 'Етаж': df['Етаж'] ,
                            'Обзавеждане': df['Обзавеждане'], 'Цена': df['Цена'], 'Строителство': df['Строителство']})

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split the dataset into training and test sets
X = np.hstack((descriptions.toarray(), new_df.drop(columns=['Цена', 'Oписание'], axis=1)))

#
# weights = np.array([1, 2, 3, 4 , 5])
#
# # Calculate the weighted sum of the source columns
# weighted_X = np.dot(X, weights)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(descriptions, new_df['Цена'], test_size=0.2)

clf = MultinomialNB()
clf.fit(X_train, y_train)

# create predictions
y_pred = clf.predict(X_test)

# find f-1 score
score = f1_score(y_test, y_pred, average='micro')
print('F-1 score : {}'.format(np.round(score,4)))

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

#cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# Train the linear regression model
reg = LinearRegression().fit(X_train, y_train)
#score = cross_val_score(LinearRegression(), X_train, y_train, cv=cv)

# Evaluate the performance of the model on the test data
score = reg.score(X_test, y_test)
print('R^2 score:', score)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, reg.predict(X_test)))

# Make predictions on new property descriptions
new_descriptions = ["Апартамент състоящ се от голяма дневна с трапезария - 50 кв.м, обор"
                    "удвана изцяло. Има работеща камина облицована с камък. В имота има"
                    " три спални - родителска спалня с баня и тоалетна, и хидромасажна вана"
                    ", две детски стаи, втора баня, тоалетна за гости към дневната. Имота има"
                    " три тераси - към дневната, към кухнята, и към едната детска стая. Отопле"
                    "нието е на ТЕЦ, има и конвекторни радиатори. В едната стая има климатик. Към имота има паркомясто във вътрешен двор и мазе. Сграда е с осем апартамента, по един на етаж!!! Достъпът е контролиран с чип и видеонаблюдение. В цената е включен наема на подземен гараж в сградата за 1 година с опция за продалжаване при желание! Обади се сега и цитирай този код 563832",
                    "Представяме на Вашето внимание невероятен южен мезонет тип пентхаус с три спални на последен етаж с гледки към Витоша планина и вътрешен озеленен двор в нов луксозен затворен комплекс. Имотът е със следното разпределение: ниво 1: огромна южна всекидневна с кухненски бокс и трапезария цели 55кв.м., тоалетна за гости и голяма тераса 17кв.м. с гледки към Витоша планина; На ниво 2 са разположени: три големи спални /20м2, 17м2 и 14м2/ и три бани и тоалетни към тях, както и два балкона.  Комплексът е разположен на 200метра от Симеоновско шосе в близост до спирки на градския транспорт и на 400метра от магазин Фантастико, където през 2024 година ще е готова новата метростанция и се отличава с красива фасада, дорийски колони и перголи. Разполага с частен фитнес и СПА, богато озеленен двор и озеленен покрив за разходки с панорамни прекрасни гледки към Витоша и София, зони за почивка около сградите и детски площадки, френски прозорци, вградено подово отопление на газ и възможност за газово готвене в кухните, противошумови мебрани на подове, стени и тавани, безшумни тръбопроводи и инсталации. За вашият комфорт, комплексът ще разполага с жива oхрана и 24/7 видеонаблюдение. Акт 16 април 2023г. За огледи и продробна информация се свържете с водещ брокер: Жаклин Банчева, +359877164800, имейл: bancheva@primoplus.bg "]
for i in [0, 1]:
    new_descriptions[i] = new_descriptions[i].lower()
    new_descriptions[i] = tokenize_text(new_descriptions[i])
    new_descriptions[i] = remove_stop_words(new_descriptions[i])
    new_descriptions[i] = ' '.join([remove_punctuation(word) for word in new_descriptions[i]]).strip()


new_descriptions = vectorizer.transform(new_descriptions)
predictions = reg.predict(new_descriptions)
print('Predictions:', predictions)


