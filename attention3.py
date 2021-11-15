#Custom Attention Layer auf Spielzeugrezensionen trainiert & validiert
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

class Attention(tf.keras.layers.Layer):

    def __init__(self, return_sequences=True):
        super(Attention, self).__init__()

    def build(self, input_shape):
        # Shape der Gewichte und dem Bias festlegen
        # dieser Layer hat nur ein Neuron, daher steht 1
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-2], 1),
                                 initializer="zeros")

        super(Attention, self).build(input_shape)

    def call(self, x):
        # x ist der Input Tensor mit der Größe 256 (aus dem Bi-LSTM)
        # K ist der Keras Backend import

        # Prozess während des Training
        # W ist das Gewicht des Layer
        e = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        # a sind die Attention Gewichte
        a = K.softmax(e)
        a = K.expand_dims(a, axis=-1)
        output = x * a

        # gibt die Outputs zurück - a sind die Menge von Attention Gewichten
        # die zweite Variable ist ein Attention angepasster Output
        return K.sum(output, axis=1) #from version 1


#Einlesen der Spielzeugrezensionen
data = pd.read_csv("C:/Users/Administrator/PycharmProjects/attention/amazon_reviews_us_Toys_v1_00.tsv", sep="\t",
                   header=0,
                   encoding="utf-8",
                   error_bad_lines=False, nrows=4000)

# relevante Spalten
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
# maximale Länge einer Inputsequenz
max_len = 200

# Initilaisierung des Tokenizers mit einer begrenzten Anzahl an Eigenschaften bzw. Worten - max_features
tokenizer = Tokenizer(num_words=max_features)

# erstellt das Wörterbuch mit den Trainingsdaten der Rezensionen
tokenizer.fit_on_texts(X_train)

# Vektorisiert Training und Testdaten bzw. wandelt den Text eine Abfolge von Zahlen bzw. Integern
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Indexierung der Wörter
word_index = tokenizer.word_index
# Anzahl der einzigartigen Worte
print('Found %s unique tokens.' % len(word_index))

# Anzahl einzigartiger Worte + 1 --> Wörterbuchgröße
vocab_size = len(tokenizer.word_index) + 1

# Die Längen der Sequenzen werden an max_len angepasst bzw. Umwandlung der Sequenzen/Abfolge von Zahlen in 2D NumPy Arrays
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

#print('Shape of data training tensor:', X_train.shape)

# Modellierung des neuronalen Netzwerks
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_len))
model.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Attention())
#model.add(Attention(return_sequences=False))  # receive 3D and output 2D
#model.add(Attention(return_sequences=True)) # receive 3D ouput
#model.add(LSTM(32)) 
model.add(Dense(1, activation='sigmoid'))
# Zusammenfassung des Modells
model.summary()
# Kompilierung des Modells
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Training und Validierung des Modells
history = model.fit(X_train, y_train, batch_size=50, epochs=5, validation_data=(X_test, y_test))

# Evaluierung/Bewertung des Modells - Verlust und Genauigkeit
results = model.evaluate(X_test, y_test)
print(results)
#loss, accuracy = model.evaluate(X_test, y_test)
#print('Evaluation accuracy: {0}'.format(accuracy))
#prediction = model.predict(X_test, verbose=1)
prediction = model.predict(X_test)
#print(prediction)
y_pred = (prediction > 0.5)
# confusion matrix - (True Positive, False Positve, False Negative, True Negative)
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
cm = np.flip(cm)
print(cm)

acc2 = sklearn.metrics.accuracy_score(y_test, y_pred)
print('accuracy: ', acc2)

precision = sklearn.metrics.precision_score(y_test, y_pred, pos_label=1)
print('precion: ', precision)

recall = sklearn.metrics.recall_score(y_test, y_pred, pos_label=1)
print('recall: ', recall)

print(sklearn.metrics.classification_report(y_test, y_pred))


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


