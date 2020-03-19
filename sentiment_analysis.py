import re
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize as word_tokenize

nltk.download('stopwords')

# Step 1
print('Step 1\n')
df_small = pd.read_csv('data_raw/bb/bb_2011_2013.csv')
df_full = pd.read_csv('data_raw/bb/bb_1996_2013.csv', index_col=0)
print('df_small head: \n' + str(df_small.head()))
print()
print('df_small shape: \n' + str(df_small.shape))
print()
print('Report in first row of df_small: \n' + str(df_small['text'][0]))
print()
print('df_full head: \n' + str(df_full.head()))
print()
print('df_full shape: \n' + str(df_full.shape))
print()

# Step 2
print('Step 2\n')
positive_lexicon_files = ["data_raw/lexicons/lexicon.generic.positive.HuLiu.csv",
                          "data_raw/lexicons/lexicon.finance.positive.csv",
                          "data_raw/lexicons/lexicon.finance.positive.LoughranMcDonald.csv"]
negative_lexicon_files = ["data_raw/lexicons/lexicon.generic.negative.HuLiu.csv",
                          "data_raw/lexicons/lexicon.finance.negative.LoughranMcDonald.csv",
                          "data_raw/lexicons/lexicon.finance.negative.csv"]

pos_lexicon_df = pd.DataFrame()
neg_lexicon_df = pd.DataFrame()

for file in positive_lexicon_files:
    df_file = pd.read_csv(file, names=['positive_lexicon'])
    pos_lexicon_df = pd.concat([pos_lexicon_df, df_file])
print('pos_lexicon_df shape: \n' + str(pos_lexicon_df.shape))
print()
print('pos_lexicon_df tail: \n' + str(pos_lexicon_df.tail()))
print()
print('pos_lexicon_df head: \n' + str(pos_lexicon_df.head()))
print()

for file in negative_lexicon_files:
    df_file = pd.read_csv(file, encoding='latin_1', names=['negative_lexicon'])
    neg_lexicon_df = pd.concat([neg_lexicon_df, df_file])
print('neg_lexicon_df shape: \n' + str(neg_lexicon_df.shape))
print()
print('neg_lexicon_df tail: \n' + str(neg_lexicon_df.tail()))
print()
print('neg_lexicon_df head: \n' + str(neg_lexicon_df.head()))
print()

# Step 3
print('Step 3\n')
pos_lexicon_df = pos_lexicon_df.apply(lambda x: x.str.lower())
pos_lexicon_df = pos_lexicon_df.apply(lambda x: x.str.replace('\\t', '').replace('\\r', '').replace('`', '').replace('\\n', ''))
print('pos_lexicon_df cleaned shape: \n' + str(pos_lexicon_df.shape))
print()
print('pos_lexicon_df cleaned tail: \n' + str(pos_lexicon_df.tail()))
print()

neg_lexicon_df = neg_lexicon_df.apply(lambda x: x.str.lower())
neg_lexicon_df = neg_lexicon_df.apply(lambda x: x.str.replace('\\t', '').replace('\\r', '').replace('`', '').replace('\\n', ''))
print('neg_lexicon_df cleaned shape: \n' + str(neg_lexicon_df.shape))
print()
print('neg_lexicon_df cleaned tail: \n' + str(neg_lexicon_df.tail()))
print()

# Step 4
print('Step 4\n')


def scoreSentiment(data_df, pos_lexicon_df, neg_lexicon_df):
    pos_list = set(pos_lexicon_df['positive_lexicon'])
    neg_list = set(neg_lexicon_df['negative_lexicon'])
    scores = []
    row_num = data_df.index[0]
    for row in data_df.iterrows():
        try:
            text = data_df['text'][row_num]
        except KeyError:
            continue
        row_num += 1
        pos = 0
        neg = 0
        words = []
        text = re.sub(r'[^a-zA-Z\s\'-]', '', text)
        for word in text.split():
            words.append(word)
        # print(str(len(words)) + ' words in the report')
        for word in words:
            if word in pos_list:
                # print('POSITIVE Match:\t' + word)
                pos += 1
            elif word in neg_list:
                # print('NEGATIVE Match:\t' + word)
                neg += 1
        score = pos - neg
        # print('pos:\t' + str(pos) + '\tneg:\t' + str(neg) + '\ttotal:\t' + str(score))
        scores.append(score)
    return scores


small_scores = scoreSentiment(df_small, pos_lexicon_df, neg_lexicon_df)
print('Small Scores: \n' + str(small_scores))
print()

plt.hist(small_scores, bins=[-20, -10, 0, 10, 20, 30, 40])
plt.show()

full_scores = scoreSentiment(df_full, pos_lexicon_df, neg_lexicon_df)
print('Full Scores: \n' + str(full_scores))
print()

plt.hist(full_scores, bins=range(-60, 50, 10))
plt.show()

# Step 5
print('Step 5\n')
whole_text = ""
row_num = df_full.index[0]
for row in df_full.iterrows():
    try:
        row_text = df_small['text'][row_num]
    except KeyError:
        continue
    whole_text += row_text
    row_num += 1
print('Length of whole text: \n' + str(len(whole_text)))
print()

stopword_files = ["data_raw/stopwords/stopwords.finance.txt",
                  "data_raw/stopwords/stopwords.geographic.txt",
                  "data_raw/stopwords/stopwords.names.txt",
                  "data_raw/stopwords/stopwords.dates.numbers.txt"]

df_stopwords = pd.DataFrame()
for file in stopword_files:
    df_file = pd.read_csv(file, encoding='windows-1252', names=['stopword'])
    df_stopwords = pd.concat([df_stopwords, df_file])
df_stopwords = df_stopwords.dropna()
df_stopwords = df_stopwords.apply(lambda x: x.str.lower())
# print(df_stopwords)


stopwords = list((df_stopwords['stopword']))
stopwords = [s.strip() for s in stopwords]
print('Length of stopwords: \n' + str(len(stopwords)))
print()
print('First 10 stopwords: \n' + str(stopwords[0:10]))
print()

stopwords.append("said")
nltk_stopwords = list(set(nltk.corpus.stopwords.words('english')))
print('Length of nltk_stopwords: \n' + str(len(nltk_stopwords)))
print()
stopwords += nltk_stopwords
stopwords = set(stopwords)
print('Length of stopwords containing only unique words: \n' + str(len(stopwords)))
print()

stemmer = SnowballStemmer("english")
words = whole_text.split()
stemmed_words = [stemmer.stem(word) for word in words]
stemmed_words = ' '.join(stemmed_words)
print('Length of whole text: \n' + str(len(whole_text)))
print()
print('Length of cleaned text: \n' + str(len(stemmed_words)))

wc = WordCloud(background_color='white', max_words=2000, stopwords=stopwords)
wc.generate(str(stemmed_words))

plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
