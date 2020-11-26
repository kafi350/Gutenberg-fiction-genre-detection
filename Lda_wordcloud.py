import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
from wordcloud import WordCloud
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import seaborn as sns

data = pd.read_csv('/Users/kafi/Downloads/Gutenberg_English_Fiction_1k/dataset_new.csv')
print (data.shape[0])
print(data.head(5))

sent_list_each_book = []
for i in range(996):
    book_name = data['Book_Name'].iloc[i]
    text_list = data['text'].iloc[i]
    # removing the 7n from the sentences first
    text_list = str(text_list)
    sent_list = nltk.tokenize.sent_tokenize(text_list)
    # sent_list_each_book.append(sent_list)
    LDA_ready = []
    for u in sent_list:
        u = str(u)
        u = re.sub(r'[,\.!?]', ' ', u)
        u = re.sub(r'[^a-zA-Z\s]', '', u, re.I | re.A)
        u = u.lower()
        if "\n" in u:
            u = u.rsplit('\n')
            for x in u:
                LDA_ready.append(x)
        else:
            LDA_ready.append(u)
        # print(i)

    sent_list_each_book.append(LDA_ready)
    u = i

# print(u)
# print(sent_list_each_book[0])
data['sentence'] = sent_list_each_book

def sentence_wordCloud(data):
    stop_words = list(stopwords.words("english"))
    list_genre = data['guten_genre'].unique()

    for genre in list_genre:
        for i in range(data.shape[0]):
            long_string = ""
            if data['guten_genre'].iloc[i] == genre:
                word_list = data["sentence"].iloc[i]
                long_string += ','.join(map(str, word_list))
                # Display the generated image:
                # the matplotlib way:
                title = 'The genre is ' + str(genre)
                # Generate a word cloud image
                wordcloud = WordCloud(stopwords=stop_words, background_color="white", max_words=20, contour_width=5,
                                      contour_color='firebrick').generate(long_string)

        plt.title(title)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

def LDA_extraction(data):
    sns.set_style('whitegrid')
    top_10 = []

    # Helper function
    def most_common_words(count_data, count_vectorizer, name):
        words = count_vectorizer.get_feature_names()
        total_counts = np.zeros(len(words))
        for t in count_data:
            total_counts += t.toarray()[0]

        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:15]
        words = [w[0] for w in count_dict]

        # removing stop words from list of WORDS

        top_10.append(words)

        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words))

        plt.figure(2, figsize=(15, 15 / 1.6180))
        title = 'most common words from the genre ' + name
        title = str(title)
        plt.subplot(title=title)
        sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
        sns.barplot(x_pos, counts, palette='husl')
        plt.xticks(x_pos, words, rotation=90)
        plt.xlabel('words')
        plt.ylabel('counts')
        plt.show()

    # getting the unique names of guten_genre column
    list_genre = data['guten_genre'].unique()
    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words='english')
    # Fit and transform the processed titles
    for genre in list_genre:
        for i in range(data.shape[0]):
            long_string = []
            if data['guten_genre'].iloc[i] == genre:
                word_list = data["sentence"].iloc[i]
                long_string += word_list
                count_data = count_vectorizer.fit_transform(long_string)
                # Display the generated image:
                # the matplotlib way:

        # Visualise the 10 most common words
        most_common_words(count_data, count_vectorizer, genre)

# sentence_wordCloud(data)
LDA_extraction(data)


### Reference Code
# weblink:https://www.datacamp.com/community/tutorials/wordcloud-python