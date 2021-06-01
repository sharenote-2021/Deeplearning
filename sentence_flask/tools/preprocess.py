import re 
import numpy as np 

from soynlp.word import WordExtractor
from soynlp.utils import DoublespaceLineCorpus
from soynlp.tokenizer import LTokenizer

from sklearn.feature_extraction.text import CountVectorizer


def test(sentence):
    # s='韓子는 싫고, 한글은 nice하다. English 쵝오 -_-ㅋㅑㅋㅑ ./?!'
    s = sentence
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
    # hangul = re.compile('[^ \u3131-\u3163\uac00-\ud7a3]+')  # 위와 동일
    result = hangul.sub('', s) # 한글과 띄어쓰기를 제외한 모든 부분을 제거
    # print (result)
    return result


# # --------------------------TDM 행렬 구축--------------------


def tdm_array(data, vocabulary = None):

    corpus = data
    vector = CountVectorizer(vocabulary = vocabulary)
    tdm = vector.fit_transform(corpus).toarray()
    print("TDM Array shape is")
    print(tdm.shape)
    # (문장개수, 단어 개수)
    # 코퍼스로부터 각 단어의 빈도 수를 기록한다.

    tdm_vocab = vector.vocabulary_

    return tdm_vocab, tdm



# --------------------------토크나이저 로드--------------------

def data_tokenize(news_title):

    word_extractor = WordExtractor(
        min_frequency=100, # example
        min_cohesion_forward=0.05,
        min_right_branching_entropy=0.0
    )

    word_extractor.train(news_title)
    words = word_extractor.extract()

    cohesion_score = {word:score.cohesion_forward for word, score in words.items()}
    tokenizer = LTokenizer(scores=cohesion_score)

    return tokenizer






# --------------------------가중치 행렬 구하기--------------------


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def calc_weight_matrix(word_vector_list, news_title, tdm, keyword_index):
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