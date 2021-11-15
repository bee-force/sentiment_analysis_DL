#SelfSeqAttention - auf Spielzeugrezensionen trainiert & auf Werkzeugrezensionen validiert & evaluiert
#imports
import numpy as np
import pandas as pd
import re

import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score)

#visualization
import matplotlib
import matplotlib.pyplot as plt

from keras_self_attention import SeqSelfAttention

# liest die Rezensionen über Spielzeuge ein
data = pd.read_csv("C:/Users/Administrator/PycharmProjects/attention/amazon_reviews_us_Toys_v1_00.tsv", sep="\t", header=0,
                   encoding="utf-8",
                   error_bad_lines=False, nrows=4000)

data = data[['star_rating', 'review_body']]
# Rezensionen mit der Bewertung 3 werden ausgeworfen, da diese als neutral gelten
data = data[data.star_rating != 3]

# ersetzt Bewertung 4,5 mit positiv und 1,2 mit negativ
data['star_rating'] = data['star_rating'].replace([4, 5], 'positive')
data['star_rating'] = data['star_rating'].replace([1, 2], 'negative')

print(data['star_rating'].value_counts())
# hier werden alle Buchstaben in Kleinbuchstaben umgewandelt und überflüssige Zeichen werden entfernt
data['review_body'] = data['review_body'].apply(lambda x: x.lower())
data['review_body'] = data['review_body'].apply((lambda x: re.sub('[(\()|(\))|(\.)|(\\\\)]', '', x)))
data['review_body'] = data['review_body'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

y = data['star_rating']

# wandelt positiv und negativ wieder in numerische Werte
y = np.array(list(map(lambda x: 1 if x == "positive" else 0, y)))

# Aufteilung in 80% Trainingsdaten und 20% Testdaten
X_train, X_test, y_train, y_test = train_test_split(data['review_body'], y, test_size=0.20,
                                                    random_state=42)
# maximale Anzahl an Wörtern/Eigenschaften
max_features = 20000
# maximale Länge der Inputsequenzen
max_len = 200

# Initilaisierung des Tokenizers mit einer begrenzten Anzahl von Eigenschaften
tokenizer = Tokenizer(num_words=max_features)
# erstellt das Wörterbuch mit den Trainingsdaten der Rezensionen
tokenizer.fit_on_texts(X_train)
# Vektorisiert Training und Testdaten bzw. wandelt den Text eine Abfolge von Zahlen bzw. Integern
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index)) #Anzahl der einzigartigen Worte

vocab_size = len(tokenizer.word_index) + 1 #Anzahl der einzigartigen Worte + 1  = Größe des Vokabulars

# Die Längen der Sequenzen werden an max_len angepasst
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

print('Shape of data training tensor:', X_train.shape)

#embedding_dim = 100 # the dimension of the word dictinory, i.e. this will be 100-dimensional word vector
# you can think of a book that each page has embedding_dim words.
model = Sequential()

# embedding dictionary = unique words * 100 = unique words parameters
# we have a 33673 x 100 word vector, Embedding accepts 2D input and returns 3D output as shown in the summary
# input_length = the length of input sequences (i.e. reviews)
model.add(Embedding(vocab_size, 128, input_length=max_len)) #vorher 100
model.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
#model.add(SeqSelfAttention(attention_activation='sigmoid',return_attention=True, attention_width=15))
model.add(SeqSelfAttention(attention_activation='sigmoid',attention_width=15))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

#what about dropout, globalmaxpooling and so on...

#tf.keras.utils.plot_model(model, 'this_model.png', show_shapes=True)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=50, epochs=5, validation_data=(X_test, y_test))

# Evaluierung/Bewertung des Modells - Verlust und Genauigkeit
results = model.evaluate(X_test1, y_test1)
print('loss & accuracy: ', results)

#prediction = model.predict(X_test1, verbose=1)
prediction = model.predict(X_test1)
y_pred = (prediction > 0.5)

cm = sklearn.metrics.confusion_matrix(y_test1, y_pred)
cm = np.flip(cm)
print(cm)

acc2 = sklearn.metrics.accuracy_score(y_test1, y_pred)
print('accuracy: ', acc2)

precision = sklearn.metrics.precision_score(y_test1, y_pred, pos_label=1)
print('precion: ', precision)

recall = sklearn.metrics.recall_score(y_test1, y_pred, pos_label=1)
print('recall: ', recall)

print(sklearn.metrics.classification_report(y_test1, y_pred))

#Visualisierung von Accuracy & Loss der Trainings- & Testdaten
plt.figure(1)
# Zusammenfassung von history für accuracy
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

# Zusammenfassung von history für loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.tight_layout()

plt.show()





