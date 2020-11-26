import pandas as pd
import nltk
from nltk import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
import re
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import numpy
import numpy as np
import seaborn as sns
import collections
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


# book_id = data['guten_genre']
# print(book_id)
#
# for i in range(data.shape[0]):
#   print(book_id[i])


# data[data.guten_genre.isin(['Allegories'])]

# Now moving to main part...


def preprocessing(data):
    print ("Starting to fetch Sentence")
    book_id = data['book_id']
    book_content = []
    for i in range(data.shape[0]):
        parts = book_id[i].split('.')
        main_file = parts[0] + '-content.html'
        File_object = open(
            '/Users/kafi/Downloads/Gutenberg_English_Fiction_1k/Gutenberg_19th_century_English_Fiction/{}'.format(
                main_file), "r+")
        str = File_object.readlines()
        without_p = []
        for i in str:
            j = re.sub("<.*?>", "", i)
            k = re.sub('[,\.!?]', ' ', j)
            l = re.sub("\n", "", k)
            m = re.sub('[^a-zA-Z\s]', '', l, re.I | re.A)
            m = m.lower()
            without_p.append(m)
        book_content.append(without_p)

    data['book_sent'] = book_content



    print('Sentence Tokenizer')

    # Sentence Tokenize

    full_sent_list = []
    for i in range(data.shape[0]):
        sentence = data['book_sent'].iloc[i]
        sentence = sentence[1:]
        sent_tokenized_list = []
        for j in sentence:
            sent_tokenized_list.append(sent_tokenize(''.join(j)))

        full_sent_list.append(sent_tokenized_list)
    data['sent_tokenized'] = full_sent_list

    print ("Starting Word Tokenizer")

    # Word Tokenized ....

    all_word_lists = []

    from nltk.tokenize import word_tokenize

    for i in range(data.shape[0]):
        word_list = data['sent_tokenized'].iloc[i]
        # word_list = word_list[1:]

        word_tokenized_list = []
        for j in word_list:
            word_tokenized_list.append(word_tokenize(' '.join(j)))

        all_word_lists.append(word_tokenized_list)

    data['word_tokenized'] = all_word_lists


    print("Starting Stopwords")

    # StopWords....


    stop_words = list(stopwords.words("english"))

    all_stop_remove = []
    for i in range(data.shape[0]):
        word_list = data['word_tokenized'].iloc[i]
        word_tokenized_list = []
        for i in word_list:
            for j in i:
                if j not in stop_words:
                    word_tokenized_list.append(j)

        all_stop_remove.append(word_tokenized_list)
    data['stopwords_removed'] = all_stop_remove

    print ("Strating Stemming")

    # Stemming...

    from nltk.stem import PorterStemmer

    all_stopwords_stemmed = []
    porter_stemmer = PorterStemmer()

    for i in range(data.shape[0]):
        word_list = data["stopwords_removed"].iloc[i]
        stemmed = []
        for j in word_list:
            stemmed.append(porter_stemmer.stem(j))

        all_stopwords_stemmed.append(stemmed)
    data['all_stopwords_stemmed'] = all_stopwords_stemmed

    print ("starting POS Tag")

    # pos Tag

    all_sentence_pos = []
    pos_tag = []
    for i in range(data.shape[0]):
        word_list = data['all_stopwords_stemmed'].iloc[i]
        # print(word_list)
        pos_tag.append(nltk.pos_tag(word_list))

    data['pos_tag'] = pos_tag

    print("Starting Sentence Lenght")

    #Sentence Length

    all_sentence_length = []

    for i in range(data.shape[0]):
      sentence = data['sent_tokenized'].iloc[i]
      sentence_length = 0
      for item in sentence:
        sentence_length = sentence_length + len(item)
      all_sentence_length.append(sentence_length)
    data['sentence_length'] = all_sentence_length

    return data

# Reading Data
# data = pd.read_csv('/Users/kafi/Downloads/Gutenberg_English_Fiction_1k/master996.csv', delimiter=';')
# print(data.head(5))

# print(data[['Book_Name', 'book_id']])

# print ("Starting to fetch Sentence")
# book_id = data['book_id']
# book_content = []
# for i in range(data.shape[0]):
#     parts = book_id[i].split('.')
#     main_file = parts[0] + '-content.html'
#     File_object = open(
#             '/Users/kafi/Downloads/Gutenberg_English_Fiction_1k/Gutenberg_19th_century_English_Fiction/{}'.format(
#                 main_file), "r+")
#     str = File_object.readlines()
#     without_p = []
#     for i in str:
#         soup = re.sub("<.*?>", "", i)
#         soup = soup.lower()
#         without_p.append(soup)
#     book_content.append(without_p)
#
# data['book_sent'] = book_content



# print('Sentence Tokenizer')
#
# # Sentence Tokenize
#
# full_sent_list = []
# for i in range(data.shape[0]):
#     sentence = data['book_sent'].iloc[i]
#     sent_tokenized_list = []
#     for j in sentence:
#         sent_tokenized_list.append(sent_tokenize(''.join(j)))
#
#     full_sent_list.append(sent_tokenized_list)
# data['sent_tokenized'] = full_sent_list









# Creating a data Sample........

# df = data[data.guten_genre.isin(['Literary', 'Detective and Mystery', 'Western Stories', 'Ghost and Horror', 'Christmas Stories', 'Love and Romance', 'Sea and Adventure', 'Allegories', 'Humorous and Wit and Satire'])].sample(200)

# df = data[(data['guten_genre'] == 'Detective and Mystery') | (data['guten_genre'] == 'Literary') | (data['guten_genre'] == 'Western Stories') | (data['guten_genre'] == 'Ghost and Horror') | (data['guten_genre'] == 'Christmas Stories') | (data['guten_genre'] == 'Love and Romance') | (data['guten_genre'] == 'Sea and Adventure') | (data['guten_genre'] == 'Allegories') | (data['guten_genre'] == 'Humorous and Wit and Satire')].sample(100)

# data = df
# data.shape
# data = df
# data.reset_index(drop=True, inplace=True)

# new_data = preprocessing(data)

# new_data.to_csv(r'/Users/kafi/Downloads/Gutenberg_English_Fiction_1k/new_master200.csv', index = False, header = True, encoding = 'utf-8')

pre_data = pd.read_csv('/Users/kafi/Downloads/Gutenberg_English_Fiction_1k/new_master200.csv')

# print(pre_data.head(10))

# Model Using Sentence Length

import ast
def Counter_Pos(data):

    nnp_counted = []
    cc_counted = []

    for i in range(data.shape[0]):
        pos_tag = data['pos_tag'].iloc[i]
        count_tokenize = []
        # tagged = [('the', 'DT'), ('dog', 'NN'), ('sees', 'VB'), ('the', 'DT'), ('cat', 'NN')]
        # print(tagged)
        # print(type(pos_tag))
        pos_tagged_list = ast.literal_eval(pos_tag)
        # print (type(pos_tagged_list))
        # counts = Counter(tag for word, tag in tagged)
        counted = Counter(tag for word, tag in pos_tagged_list)
        noun_count = [s for s in pos_tagged_list if s[1] == 'NNP']
        total_noun = len(noun_count)
        conjunction = [s for s in pos_tagged_list if s[1] == 'CC']
        total_cc = len(conjunction)
        # print(counted)
        # break
        # for item in pos_tag:
        #     print(item)
        #     counts = Counter(item)
        #     print(counts)
        #     break
        nnp_counted.append(total_noun)
        cc_counted.append(total_cc)

    return nnp_counted, cc_counted

nnp, cc = Counter_Pos(pre_data)
pre_data['Singular Noun'] = nnp
pre_data['Conjunction'] = cc

print ("one done")
def tf_idf(data):
    vocab = ['said', 'mr', 'mrs', 'rose', 'man', 'little', 'george', 'like', 'oh', 'cat', 'lorry', 'hand', 'business', 'did', 'head', 'face', 'eyes', 'long', 'looked',
              'white', 'belding', 'nedd', 'ladd', 'time', 'house', 'mind', 'ben', 'henly', 'answered',
             'ghost', 'spirit', 'christmas', 'good', 'cried', 'scrooge', 'paul', 'lady', 'captain', 'ship', 'island', 'round', 'doctor',
              'went', 'christian', 'rhoda', 'heart', 'away', 'philip', 'sea']
    array_all = []
    for i in range(data.shape[0]):
        data_sentence = data['book_sent']
        string_all = ''
        arr = []

        for item in data_sentence:
            # string_all += str(item)
            item_list = ast.literal_eval(item)


            pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)),
                             ('tfid', TfidfTransformer())]).fit(item_list)
            pipe['count'].transform(item_list).toarray()

            # print (pipe['tfid'].idf_)
            arr.append(pipe['tfid'].idf_)
        array_all.append(arr)
    return array_all

array_all = tf_idf(pre_data)
pre_data['tfidf'] = array_all

pre_data.to_csv(r'/Users/kafi/Downloads/Gutenberg_English_Fiction_1k/new_master200.csv', index = False, header = True, encoding = 'utf-8')

def LinearClassifier(data, name):
    data_testing = data[[name, 'guten_genre']]
    y = data_testing.pop('guten_genre')
    X = data_testing[name].astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(X.index, y, test_size=0.3)
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    return y_pred, score

print("First Classification with Sentence Length")
# prediction, score = LinearClassifier(pre_data)
# print("LogisticRegression: {}".format(prediction))
# print ("Score of LogisticRegression: {}".format(score))



# print(pre_data.head(5))




def GNBClassifier(data, name):
    data_testing = data[[name, 'guten_genre']]
    y = data_testing.pop('guten_genre')
    X = data_testing[name].astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(X.index, y, test_size=0.35)
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    return y_pred, score

# pre_data.to_csv(r'/Users/kafi/Downloads/Gutenberg_English_Fiction_1k/new_master.csv', index = False, header = True, encoding = 'utf-8')

def SGD_Classifier(data, name):
    data_testing = data[[name, 'guten_genre']]
    y = data_testing.pop('guten_genre')
    X = data_testing[name].astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(X.index, y, test_size=0.3)
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    clf = make_pipeline(StandardScaler(),
                        SGDClassifier(max_iter=1000, tol=1e-3))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    return y_pred, score


def Random_Forest(data, name):
    data_testing = data[[name, 'guten_genre']]
    y = data_testing.pop('guten_genre')
    X = data_testing[name].astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(X.index, y, test_size=0.3)
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)

    # X_train, y_train = make_classification(n_samples=1000, n_features=1,
    #                            n_informative=2, n_redundant=0,
    #                            random_state=0, shuffle=False)
    clf = RandomForestClassifier(max_depth=2, random_state=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    return y_pred, score


def Plotting_all(data, name):
    clf1 = linear_model.LogisticRegression(C=1e5)
    clf2 = GaussianNB()
    clf3 = make_pipeline(StandardScaler(),
                        SGDClassifier(max_iter=100, tol=1e-3))
    clf4 = RandomForestClassifier(max_depth=2, random_state=10)

    data_testing = data[[name, 'guten_genre']]
    y = data_testing.pop('guten_genre')
    X = data_testing[name]
    X_train, X_test, y_train, y_test = train_test_split(X.index, y, test_size=0.50)
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)



    probas = [c.fit(X_train, y_train).score(X_test, y_test) for c in (clf1, clf2, clf3, clf4)]
    print(probas)

    bars = ('Logistic', 'Gaussian', 'SDG', 'Random')
    y_pos = np.arange(len(bars))
    width = [0.3, 0.3, 0.3, 0.3]

    # N = 4  # number of groups
    # ind = np.arange(N)  # group positions
    # width = 0.25  # bar width
    #
    # fig, ax = plt.subplots()
    #
    # # bars for classifier 1-3
    # p1 = ax.bar(ind, probas, width,
    #             color='green', edgecolor='k')
    #
    # p2 = ax.bar(ind, probas[1], width, color='green')
    #
    #
    # p3 = ax.bar(ind, probas[2], width, color='green')
    #
    #
    # # bars for VotingClassifier
    # p4 = ax.bar(ind, probas, width,
    #             color='green', edgecolor='k')
    #
    #
    # # plot annotations
    # plt.axvline(2.8, color='k', linestyle='dashed')
    # ax.set_xticks(ind + width)
    # ax.set_xticklabels(['LogisticRegression',
    #                     'GaussianNB',
    #                     # 'SGDClassifier'
    #                     'RandomForestClassifier',
    #                     'VotingClassifier'],
    #                    rotation=40,
    #                    ha='right')
    # plt.ylim([0, 1])
    # plt.title('Accuracy')
    # plt.legend([p1[0]], ['Accuracy'], loc='upper left')
    # plt.tight_layout()
    # plt.show()
    plt.ylim([0, 1])
    plt.bar(y_pos, probas, color=['cyan', 'red', 'green', 'blue'], width=width)
    plt.xticks(y_pos, bars)

    plt.show()

# Plotting_all(pre_data, 'tfidf')

# prediction, score = LinearClassifier(pre_data, 'tfidf')
# print("LogisticRegression: {}".format(prediction))
# print ("Score of LogisticRegression: {}".format(score))






