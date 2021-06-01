from gensim.models import Word2Vec


def word2vec(cluster_data):
    model = Word2Vec(cluster_data, size=100, window = 10, min_count=5, workers=5, iter=500, sg=1)
    # check embedding result
    word_vector = model.wv


    sentence_vectors = []
    for sent in cluster_data:
        sent_vector = [0]*100
        for p in sent:
            # word2vec의 min_count 조건으로 인해 없는 단어가 있을 수 있음
            try: 
                x = word_vector.get_vector(p)
            except:
                continue
            sent_vector += x
        
        sentence_vectors.append(sent_vector)


    print("comlpete vetorization")

    return sentence_vectors