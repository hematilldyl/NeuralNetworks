import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
import numpy as np
from matplotlib import style

style.use('ggplot')


def generate(model,tokenizer,length,seed,N_words):
    input_text = seed
    for _ in range(N_words):
        text_to_seq = tokenizer.texts_to_sequences([input_text])[0]
        encoded = pad_sequences([text_to_seq],maxlen=length,padding='pre')
        estimator = model.predict_classes(encoded,verbose=0)
        output_text = ''
        for word, index in tokenizer.word_index.items():
            if index == estimator:
                output_text=word
                break
        input_text += ' ' +output_text
    return input_text

#nltk.download('gutenberg')
macbeth= nltk.corpus.gutenberg.raw('shakespeare-macbeth.txt')
hamlet = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')
caesar = nltk.corpus.gutenberg.raw('shakespeare-caesar.txt')
Shakespeare=macbeth+hamlet+caesar
print(hamlet[:500])

tokenizer = Tokenizer()
tokenizer.fit_on_texts([Shakespeare])
text_to_seq = tokenizer.texts_to_sequences([Shakespeare])[0]
size = len(tokenizer.word_index)+1
print('Vocab Size: %d' % size)
sequence=[]
for i in range(2,len(text_to_seq)):
    seq=text_to_seq[i-2:i+1]
    sequence.append(seq)
print('Total Sequences: %d' % len(sequence))

max_len = max([len(seq) for seq in sequence])
sequence = pad_sequences(sequence,maxlen=max_len,padding='pre')

sequences=np.array(sequence)
X,y=sequences[:,:-1],sequences[:,-1]

#one hot encoding
y = to_categorical(y,num_classes=size)

#model
model = Sequential()
model.add(Embedding(size,10,input_length=max_len-1))
model.add(LSTM(150))
model.add(Dense(size,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.fit(X,y,epochs=3,verbose=2)

print(generate(model,tokenizer,max_len-1,'Haue you', 100))
