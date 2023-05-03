import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, callbacks
from tensorflow.keras import Model, Sequential

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
import string


import nltk
nltk.download('stopwords')

df = pd.read_csv('Emotion_final.csv')
#df =  pd.read_csv('train.csv', encoding='latin-1')

df.info()
#classifying classes  based on emotion
import seaborn as sns
sns.countplot(df.Emotion)
plt.xlabel('Label')

# Encoded Sentiment columns
encoder = LabelEncoder()
df['Label'] = encoder.fit_transform(df['Emotion'])
df.head(10)

num_classes = df.Label.nunique()
print(num_classes)
