
#%% 
# Import Module
import os
import re
import datetime
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, plot_model
from keras import Sequential
from keras.layers import LSTM, Input, Dense, Bidirectional, Embedding, Dropout
from keras.callbacks import TensorBoard, EarlyStopping



#%% 
# Data Loading
df = pd.read_csv("https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv")



#%% 
# Data Inspection
df.info()
df.isna().sum() #No null values
df.duplicated().sum()
df.describe()
df.head()

plt.figure()
plt.hist(df['category'])
plt.show()



#%%
# Data Cleaning
# Remove duplicated data
# df = df.drop_duplicates() 
# df.duplicated().sum()
category = df['category']
text = df['text'] 



#Use NLTK to remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
text = text.apply(lambda x: [item for item in str(x).split() if item not in stop_words])


#Convert to string for lemmatization
text = text.apply(str)

#use lemmatizing to convert words into its base word
nltk.download('wordnet')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w,'v') for w in w_tokenizer.tokenize(text)]

text = text.apply(lemmatize_text)

#Convert to string for regex
text = text.apply(str)

#Regex to remove special characters
for index,i in enumerate(text):
    text[index] = re.sub('([^a-zA-Z])', ' ', i)





#%%
# Data Pre-processing

#Text Pre-processing
#Tokenizer
words= 10000
tokenizer = Tokenizer(
    num_words=words, 
    oov_token='<OOV>'
    )

tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index

x_train = tokenizer.texts_to_sequences(text)

#Padding
x_train = pad_sequences(x_train, 
    maxlen=150, 
    padding='post', 
    truncating='post'
    )

#Expand feature dimensions
x_train = np.expand_dims(x_train,-1)

#Category preprocessing
#One hot encoder
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(category[::, None])

#train test split
X_train, X_test, y_train, y_test = train_test_split(x_train,y_train)



#%%
# Model Development
nodes = 64

model = Sequential()
model.add(Embedding(words,nodes))
model.add(Bidirectional(LSTM(nodes, return_sequences=True)))
model.add(LSTM(nodes))
model.add(Dense(nodes, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.summary()

#Model architecture
plot_model(model, show_shapes=True)

#Compile Model
model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics='accuracy')

#Callbacks
logdir = os.path.join(os.getcwd(), 
    'logs', 
    datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    )

tb = TensorBoard(log_dir=logdir)
es = EarlyStopping(monitor='val_loss',
    patience=5
    )

history = model.fit(X_train,y_train, 
    validation_data=(X_test,y_test), 
    epochs=25, 
    callbacks=[tb,es]
    )



#%%
# Model Evaluation
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print('Classification Report \n', classification_report(y_true,y_pred))



#%%
# Model Saving

with open('saved_models.json','w') as f:
    json.dump(tokenizer.to_json(),f)

with open('onehot.pkl','wb') as f:
    pickle.dump(ohe,f)

model.save('saved_models.h5')

