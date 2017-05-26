# coding: utf-8
import sys
import json
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adagrad
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.models import load_model


def read_data(path,training):
    print ('Reading data from ',path)
    with open(path, 'r', encoding = 'utf8') as f:

        tags = []
        articles = []
        tags_list = []

        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]

                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)

                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]

            articles.append(article)

        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def f1_score(y_true,y_pred):
    thresh = 0.38
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)

    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

### read test data
(_, X_test,_) = read_data(sys.argv[1],False)

### read word_index
with open('word_index.json', 'r') as f:
    word_index = json.load(f)

### read tag_list
with open('tag_list.json', 'r') as t:
    tag_list = json.load(t)

### tokenizer for all data
tokenizer = Tokenizer()
tokenizer.word_index = word_index

### convert word sequences to index sequence
test_sequences = tokenizer.texts_to_sequences(X_test)

### padding to equal length
max_article_length = 306
test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)

### set parameters
num_words = len(word_index) + 1
embedding_dim = 100

### build model
print ('Building model.')
model = Sequential()
model.add(Embedding(num_words,
                    embedding_dim,
                    input_length=max_article_length,
                    trainable=False))

model.add(GRU(256,activation='tanh',dropout=0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(192,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(38,activation='sigmoid'))
model.summary()

adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
                optimizer=adagrad,
                metrics=[f1_score])

### output
model.load_weights('best.hdf5')
output_path = sys.argv[2]
Y_pred = model.predict(test_sequences)
thresh = 0.38
with open(output_path,'w') as output:
    print ('\"id\",\"tags\"',file=output)
    Y_pred_thresh = (Y_pred > thresh).astype('int')
    for index,labels in enumerate(Y_pred_thresh):
        labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
