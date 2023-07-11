# import libraries
import tensorflow as tf
import pandas as pd
from tensorflow import keras
'''
!pip install tensorflow-datasets
'''
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import string
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

'''
# get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv
'''

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

#decodifying:
word_map = {'ham':0, 'spam':1}


train_data = pd.read_csv(
  train_file_path, sep='\t', names=['label', 'text']
)
train_data['label'] = train_data['label'].map(word_map)

valid_data = pd.read_csv(
  test_file_path, sep='\t', names=['label', 'text']
)
valid_data['label'] = valid_data['label'].map(word_map)

print(f"train:\n")
print(train_data.head())
print(f"valid:\n")
print(valid_data.head())



#preprocessing:
def preprocess_text(message):
    no_punctuations = [char if char not in string.punctuation else ' ' for char in message]
    no_punctuations = "".join(no_punctuations)
    no_punctuations = ' '.join(no_punctuations.split())
    no_punctuations = no_punctuations.lower()

    no_stopwords = [
      word
      for word in no_punctuations.split()
      if word.isalpha()
    ]

    return no_stopwords

train_data['text'] = train_data['text'].apply(preprocess_text)
valid_data['text'] = valid_data['text'].apply(preprocess_text)

print(f"train:\n")
print(train_data.head())
print(f"valid:\n")
print(valid_data.head())

train_labels = train_data.pop('label')
valid_labels = valid_data.pop('label')

MAX_LEN = 30
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['text'].values) #se le manda una lista de strings o una lista de listas de strings, como en mi caso.

encode_train = tokenizer.texts_to_sequences(train_data['text'].values)
encode_valid = tokenizer.texts_to_sequences(valid_data['text'].values)

padded_train = pad_sequences(encode_train, maxlen=MAX_LEN, padding='post') #padding='post' hace que se agregue padding al final.
padded_valid = pad_sequences(encode_valid, maxlen=MAX_LEN, padding='post')

VOCAB_SIZE = len(tokenizer.word_index) + 1 #word_index es un diccionario con palabra: indice. Cuento esa cantidad de palabras.
#le sumo 1 ya que el primer índice es 1, no 0. Sino no se podría referenciar el índice más alto.

pretrained = True #change to False if you want to train the model.
if not pretrained:
  model = keras.Sequential([
    keras.layers.Embedding(VOCAB_SIZE, 32),
    keras.layers.LSTM(32),
    keras.layers.Dense(1, activation="sigmoid")
  ])

  model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
  history = model.fit(padded_train, train_labels.values, epochs=10, validation_data=[padded_valid, valid_labels])
  model.save("text_classifier.h5")
else:
  model = tf.keras.models.load_model("text_classifier.h5")

#result:
model.evaluate(padded_valid, valid_labels)

# function to predict messages based on model
def predict_message(pred_text):
  encoded_pred = tokenizer.texts_to_sequences([pred_text])
  padded_pred = pad_sequences(encoded_pred, maxlen=MAX_LEN, padding='post')

  pred = model.predict(padded_pred)
  pred = pred[0]

  prediction = [pred, '']

  if pred >= 0.5:
    prediction[1] = 'spam'
  else:
    prediction[1] = 'ham'


  return (prediction)

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)
