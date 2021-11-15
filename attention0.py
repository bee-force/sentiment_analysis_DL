#no attention
import numpy as np
import pandas as pd
import re

import nltk

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
import sklearn.metrics

import matplotlib
import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Flatten, Activation, RepeatVector, Permute, Multiply, Lambda


#Einlesen der Spielzeugrezensionen
data = pd.read_csv("C:/Users/Administrator/PycharmProjects/attention/amazon_reviews_us_Toys_v1_00.tsv", sep="\t",
                   header=0,
                   encoding="utf-8",
                   error_bad_lines=False, nrows=4000)

# Einlesen der Werkzeugrezensionen
data1 = pd.read_csv("C:/Users/Administrator/PycharmProjects/attention/amazon_reviews_us_Tools_v1_00.tsv", sep="\t", header=0,
                   encoding="utf-8",
                   error_bad_lines=False, nrows=4000)

# relevante Spalten
data = data[['star_rating', 'review_body']]
data1 = data1[['star_rating', 'review_body']]

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

# gleiches Prozedere nochmal mit Werkzeugrezensionen
# Rezensionen mit der Bewertung 3 werden ausgeworfen, da diese als neutral gelten
data1 = data1[data1.star_rating != 3]

# ersetzt Bewertung 4,5 mit positiv und 1,2 mit negativ
data1['star_rating'] = data1['star_rating'].replace([4, 5], 'positive')
data1['star_rating'] = data1['star_rating'].replace([1, 2], 'negative')

print(data1['star_rating'].value_counts())
# hier werden alle Buchstaben in Kleinbuchstaben umgewandelt und überflüssige Zeichen werden entfernt
data1['review_body'] = data1['review_body'].apply(lambda x: x.lower())
data1['review_body'] = data1['review_body'].apply((lambda x: re.sub('[(\()|(\))|(\.)|(\\\\)]', '', x)))
data1['review_body'] = data1['review_body'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

y1 = data1['star_rating']
# wandelt positiv und negativ wieder in numerische Werte
y1 = np.array(list(map(lambda x: 1 if x == "positive" else 0, y1)))

# Aufteilung in 80% Trainingsdaten und 20% Testdaten
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1['review_body'], y1, test_size=0.20,
                                                    random_state=42)


# maximale Anzahl an Wörtern/Eigenschaften
max_features = 20000
# maximale Länge einer Inputsequenz
max_len = 200

# Initilaisierung des Tokenizers mit einer begrenzten Anzahl an Eigenschaften bzw. Worten - max_features
tokenizer = Tokenizer(num_words=max_features)
tokenizer1 = Tokenizer(num_words=max_features)

# erstellt das Wörterbuch mit den Trainingsdaten der Rezensionen
tokenizer.fit_on_texts(X_train)
tokenizer1.fit_on_texts(X_train1)

# Vektorisiert Training und Testdaten bzw. wandelt den Text eine Abfolge von Zahlen bzw. Integern
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_test1 = tokenizer1.texts_to_sequences(X_test1)

# Indexierung der Wörter
word_index = tokenizer.word_index
# Anzahl der einzigartigen Worte
print('Found %s unique tokens.' % len(word_index))

# Anzahl einzigartiger Worte + 1 --> Wörterbuchgröße
vocab_size = len(tokenizer.word_index) + 1

# Die Längen der Sequenzen werden an max_len angepasst bzw. Umwandlung der Sequenzen/Abfolge von Zahlen in 2D NumPy Arrays
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)
X_test1 = pad_sequences(X_test1, padding='post', maxlen=max_len)

#print('Shape of data training tensor:', X_train.shape)

# Modellierung des neuronalen Netzwerks
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_len))
model.add(Bidirectional(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
# Zusammenfassung des Modells
model.summary()
# Kompilierung des Modells
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Training und Validierung des Modells
history = model.fit(X_train, y_train, batch_size=50, epochs=5, validation_data=(X_test1, y_test1))

# Evaluierung/Bewertung des Modells - Verlust und Genauigkeit
results = model.evaluate(X_test1, y_test1)
print(results)
#loss, accuracy = model.evaluate(X_test, y_test)
#print('Evaluation accuracy: {0}'.format(accuracy))
#prediction = model.predict(X_test, verbose=1)
prediction = model.predict(X_test1)
#print(prediction)
y_pred = (prediction > 0.5)
# confusion matrix - (True Positive, False Positve, False Negative, True Negative)
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


# Visualisierung von Accuracy & Loss der Trainings- & Testdaten
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
