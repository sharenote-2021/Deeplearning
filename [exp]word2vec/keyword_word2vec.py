import pandas as pd


import re

def test(sentence):
    # s='韓子는 싫고, 한글은 nice하다. English 쵝오 -_-ㅋㅑㅋㅑ ./?!'
    s = sentence
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
    # hangul = re.compile('[^ \u3131-\u3163\uac00-\ud7a3]+')  # 위와 동일
    result = hangul.sub('', s) # 한글과 띄어쓰기를 제외한 모든 부분을 제거
    # print (result)
    return result



df = pd.read_csv("./category.csv")
news_title = df["title"]
subject = df['subject']


# # --------------------------데이터 전처리--------------------


data = []
for sent in news_title:
    preprocess = test(sent)
    data.append(preprocess)


# # --------------------------중요 단어 추출--------------------


# 텍스트랭크 통한 중요 키워드 추출 -> 중요 키워드 별 주요 문장 추출하기 위함


# # --------------------------TDM 행렬 구축--------------------



from sklearn.feature_extraction.text import CountVectorizer


corpus = data
vector = CountVectorizer()
tdm = vector.fit_transform(corpus).toarray()
print(tdm.shape)
# (문장개수, 단어 개수)
# 코퍼스로부터 각 단어의 빈도 수를 기록한다.

tdm_vocab = vector.vocabulary_





# --------------------------토크나이저 로드--------------------

import numpy as np 

from soynlp.word import WordExtractor
from soynlp.utils import DoublespaceLineCorpus
from soynlp.tokenizer import LTokenizer


word_extractor = WordExtractor(
    min_frequency=100, # example
    min_cohesion_forward=0.05,
    min_right_branching_entropy=0.0
)

word_extractor.train(news_title)
words = word_extractor.extract()

cohesion_score = {word:score.cohesion_forward for word, score in words.items()}
tokenizer = LTokenizer(scores=cohesion_score)

cluster_data = []
bert_null_list = []

for title in news_title:
    title = test(title)
    sent = tokenizer.tokenize(title, flatten=False)
    sentence = []
    for i in sent:
        if i[0] in tdm_vocab:
            sentence.append(i[0])

    
    cluster_data.append(sentence)


# # --------------------------Word2Vec embedding--------------------

from gensim.models import Word2Vec


model = Word2Vec(cluster_data, size=100, window = 10, min_count=5, workers=5, iter=500, sg=1)
# check embedding result
word_vector = model.wv


print(word_vector.vocab)
# vocab 인덱스 초기화
vocabulary = dict((t, i) for i, t in enumerate(word_vector.vocab))


# word2vec 단어사전
# 12434문장 -> 2568개 단어 사전
print(len(vocabulary)) 
print(word_vector.vocab['코로나'])

#  코로나 : 72
print(vocabulary["코로나"])


# # --------------------------TDM 행렬 구축--------------------
# word2vec 단어사전에 따른 tdm 단어 구축하여 문서단어행렬 만듬

print("tdm 행렬 구축 \n")


from sklearn.feature_extraction.text import CountVectorizer


corpus = data
vector = CountVectorizer(vocabulary = vocabulary)
tdm = vector.fit_transform(corpus).toarray()
print(tdm.shape)
# (문장개수, 단어 개수)
# 코퍼스로부터 각 단어의 빈도 수를 기록한다.

tdm_vocab = vector.vocabulary_





vocabs = vocabulary

########################### 키워드 입력 부분 ######################
keyword_index = vocabs["코로나"]
###############################################################




# (word2vec 단어, word2vec 임베딩 차원)
# (411, 100)
word_vector_list = [word_vector[v] for v in vocabs]

print(len(vocabs), len(word_vector_list))
print(len(tdm_vocab))

print(model.most_similar(positive=["고객"], topn=10))


# --------------------------가중치 행렬 구하기--------------------


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# 가중치 행렬 : 코사인 유사도
# from sklearn.metrics.pairwise import cosine_similarity
# distance_matrix = cosine_similarity(word_vector_list, word_vector_list)
# weight_matrix = np.exp(-(distance_matrix ** 2) / (2 * np.var(distance_matrix)))

# 가중치 행렬 : 유클리디안 거리
distance_matrix = euclidean_distances(word_vector_list, word_vector_list)
weight_matrix = np.exp(-(distance_matrix ** 2) / (2 * np.var(distance_matrix)))


# 유클리드안 거리에 따른 거리 행렬을 정규분포와 비슷한 그래프로 가중치 행렬을 만들어 줌
print(weight_matrix.shape)
print(weight_matrix[0].shape, tdm[0].shape)


# 키워드에 해당하는 인덱스 번호를 통해 가중치 행렬과 tdm 행렬을 lookup table로 활용하여 내적을 진행
result = []
for index, sent in enumerate(news_title):
    k = np.dot(weight_matrix[keyword_index] , tdm[index])
    result.append((k, index))

# 내적한 결과를 내림차순으로 정렬을 하여 상위 5개 문장을 출력
result = sorted(result)
print(result[-5:0])

for q in result[-5:]:
    print(news_title[q[1]], q)