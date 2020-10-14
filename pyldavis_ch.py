import pyLDAvis.gensim
from gensim import corpora
from gensim.models import LdaModel
import jieba
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import csv
import pymongo
 

def text_rank(text):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, lower=True, window=2)
    # tr4w.analyze(text)
    
    keywords = []
    for item in tr4w.get_keywords(5, word_min_len=2):

        keywords.append(item.word)
    print(keywords)
    return keywords


def get_text():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["weibo_one_day"]
    mycol = mydb["comments_data"]
    l = []
    for x in mycol.find({},{'text':1}):
        l.extend(x['text'])
    # print('l:',l)
    keywords_text = []
    for text in l:
        keywords = text_rank(text)
        keywords_text.append(keywords)
    return keywords_text

def get_corpus_dictionary():

    texts = get_text()
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    print(frequency)
    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]
    print(texts)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    return corpus, dictionary

def test_lda():
    corpus, dictionary = get_corpus_dictionary()
    print(corpus)
    print(dictionary)
    lda = LdaModel(corpus=corpus,num_topics=10)
    data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.show(data,open_browser=False)

if __name__ == "__main__":
    test_lda()
