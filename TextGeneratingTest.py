'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Embedding
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
import numpy as np
import keras
import random
import sys
import io
import re

def on_epoch_end(epoch):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(word_text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = []
        sentence = word_text[start_index: start_index + maxlen]
        generated.append(sentence)
        print('----- Generating with seed: "' + ' '.join(sentence) + '"')
        sys.stdout.write(' '.join(sentence))

        for i in range(30):
            x_pred = np.zeros((1, maxlen), dtype=np.int)
            for t, word in enumerate(sentence):
                x_pred[0, t] = words_dict.index(word)

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = words_dict[next_index]

            generated += next_word
            sentence = sentence[1:] + [next_word]

            sys.stdout.write(' '+next_word)
            sys.stdout.flush()
        print()


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 20
batch_size=64

dict = eval(open("dictionary.txt").read())
words_dict=list(dict.keys())

print('Build model...')
model = Sequential()
model.add(Embedding(len(words_dict), 50, input_length=maxlen))
# model.add(LSTM(256, return_sequences=True))
# model.add(Dropout(0.1))
model.add(LSTM(1000))
model.add(Dropout(0.20))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(words_dict), activation='softmax'))

optimizer = keras.optimizers.adam(0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
i=0
with io.open('.\document.txt','r', encoding='utf-8') as f:
    for line in f:
        print(i)
        i+=1     
        word_text = re.split(r'([\W])+', line)
        word_text = [x for x in word_text if x not in [' ', '\n']]
        sentences = []
        next_words = []
        for i in range(0, len(word_text) - maxlen, 1):
            sentences.append(word_text[i: i + maxlen])
            next_words.append(word_text[i + maxlen])
        print('nb sequences:', len(sentences))
        x = np.zeros((len(sentences), maxlen), dtype=np.int)
        y = np.zeros((len(sentences)), dtype=np.int)
        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence):
                x[i, t] = words_dict.index(word)
            y[i] = words_dict.index(next_words[i])
        loss=[]
        for i in range(0, len(x), batch_size):
            loss.append(model.train_on_batch(x[i:i+batch_size], y[i:i+batch_size])[0])
        loss.append(model.train_on_batch(x[i:len(x)], y[i:len(x)])[0])
        print(sum(loss)/len(loss))
    model.save('model.h5')
    on_epoch_end(0)


