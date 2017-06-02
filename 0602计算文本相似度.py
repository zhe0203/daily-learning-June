# -*- coding: utf-8 -*-
'''
步骤：
分词，去停止词
词袋模型向量化文本
TF-IDF模型向量化文本
LSI模型向量化文本
计算相似度

两篇中文文本，如何计算相似度？相似度是数学上的概念，自然语言肯定无法完成，所有要把文本转化为向量。两个向量计算相似度就很简单了，欧式距离、余弦相似度等等各种方法，只需要中学水平的数学知识。
'''

# 那么如何将文本表示成向量呢？
## 词袋模型
## 最简单的是表示方法是词袋模型，他一篇文本想象成一个个词构成的，所有词放入一个袋子里，没有先后顺序，没有语义
'''
John likes to watch movies. Mary likes too.
John also likes to watch football games.

这两个句子，可以构建出一个词典，key为上文出现过的词，value为这个词的索引序号
{"John": 1, "likes": 2,"to": 3, "watch": 4, "movies": 5,"also": 6, "football": 7, "games": 8,"Mary": 9, "too": 10}

那么，上面两个句子用词袋模型表示成向量就是：[1, 2, 1, 1, 1, 0, 0, 0, 1, 1]  [1, 1,1, 1, 0, 1, 1, 1, 0, 0]
# 其中,1,2，0分别表示上面建立的词典的，对应的词在相应句子中出现的次数

相对于英文，中文更复杂一些，涉及到分词。准确地分词是所有中文文本分析的基础，本文使用结巴分词，完全开源而且分词准确率相对有保障。
'''
# TF-IDF模型
'''
词袋模型简单易容，但存在问题，中文文本里最常见是词是的，是，有,这样的的没有实际含义的词，所以要对文本中出现的词赋予权重。
一个词的权重由TF*IDF表示，其中TF表示词频，即一个词在这篇文本中出现的频率；
IDF表示逆文档频率，即一个词在 所有 文本中出现的频率的倒数，因此，一个词在某文本中出现的越多，在其他文本中出现的越少，
则这个词能很好的反映这篇文本的内容，权重值就越大。
回过头看词袋模型，只考虑了文本的词频，而TF-IDF模型则包含了词的权重，更加准确。本文向量与词袋模型中的维数相同，只是每个词的对应
分量值换成了该词的TF-IDF值

词频（TF) = 某个词在文章中出现的次数/文中的总词数
逆文档频率（IDF) = log(语料库的文档总数/(包含该词的文档数+1))
'''

# LSI模型

'''
 TF-IDF模型足够胜任普通的文本分析任务，用TF-IDF模型计算文本相似度已经比较靠谱了，但是细究的话还存在不足之处。
 实际的中文文本，用TF-IDF表示的向量维数可能是几百、几千，不易分析计算。此外，一些文本的主题或者说中心思想，
 并不能很好地通过文本中的词来表示，能真正概括这篇文本内容的词可能没有直接出现在文本中。

因此，这里引入LSI从文本潜在的主题进行分析，LSI是概率主题模型一种，另一种常见的是LDA，核心思想是：
每篇文本中有多个概率分布不同的主题，每个主题中有包含所有的已知词，但是这些词在不同主题中的概率分布不同
LSI通过奇异值分解的方法计算出文本中各个主题的概率分布，严格的数学证明需要看相关论文。假设5个主题，那么通过LSI模型
文本向量将可以降到5维，每个分量表示对应主题的权重
'''

'''
分词上使用了结巴分词https://github.com/fxsjy/jieba，
词袋模型、TF-IDF模型、LSI模型的实现使用了gensim库  https://github.com/RaRe-Technologies/gensim
'''
import jieba.posseg as pseg
import codecs
from gensim import corpora,models,similarities
# 构建停止词
stop_words = 'stop_words.txt'
stopwords = codecs.open(stop_words,'r',encoding='utf-8').readlines()
stopwords = [w.strip() for w in stop_words]
# 结巴分词后的停用词性 [标点符号、连词、助词、副词、介词、时语素、‘的’、数词、方位词、代词]
stop_flag = ['x','c','u','d','p','t','uj','m','f','r']
# 对一篇文章分词，去停止词
def tokenization(filename):
    result = []
    with open(filename,'r') as f:
        text = f.read()
        words = pseg.cut(text)
    for word,flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result

# 选取三篇文章，前两篇是高血压主题的，第三篇是iOS主题的
filenames = ['13件小事帮您稳血压.txt','高血压患者宜喝低脂奶.txt','ios.txt']
corpus = []
for each in filenames:                # 计算得到这三篇文章的词
    corpus.append(tokenization(each))
print(len(corpus))

# 建立词袋模型
dictionary = corpora.Dictionary(corpus)
print(dictionary)
doc_vectors = [dictionary.doc2bow(text) for text in corpus]
print(len(doc_vectors))
print(doc_vectors)

# 建立TF-IDF模型
tfidf = models.TfidfModel(doc_vectors)
tfidf_vectors = tfidf[doc_vectors]
print(len(tfidf_vectors))
print(len(tfidf_vectors[0]))

# 构建一个query文本，是高血压主题的，利用词袋模型的字典将其映射到向量空间
query = tokenization('关于降压药的五个问题.txt')
query_bow = dictionary.doc2bow(query)
print(len(query_bow))
print(query_bow)
index = similarities.MatrixSimilarity(tfidf_vectors)
'''
用TF-IDF模型计算相似度，相对于前两篇高血压主题的文本，iOS主题文本与query的相似度很低。
可见TF-IDF模型是有效的，然而在语料较少的情况下，与同是高血压主题的文本相似度也不高。
'''
sims = index[query_bow]
print(list(enumerate(sims)))
# [(0, 0.28532028), (1, 0.28572506), (2, 0.023022989)]

# 构建LSI主题，设置主题数为2
lsi = models.LsiModel(tfidf_vectors,id2word=dictionary,num_topics=2)
lsi.print_topics(2)
lsi_vector = lsi[tfidf_vectors]
for vec in lsi_vector:
    print(vec)

# 在LSI向量空间中，所有文本的向量都是二维的
query = tokenization('关于降压药的五个问题.txt')
query_bow = dictionary.doc2bow(query)
print(query_bow)
query_lsi = lsi[query_bow]
print(query_lsi)

index = similarities.MatrixSimilarity(lsi_vector)
sims = index[query_lsi]
print(list(enumerate(sims)))
