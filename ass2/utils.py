import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer


# reads the csv data and vectorizes the text field
def read_data(filename):
    dataset = []
    df = pd.read_csv('spam_ham_dataset.csv')
    text = df['text']
    vectorizer = TfidfVectorizer(stop_words=('english'))
    text = vectorizer.fit_transform(text)
    text = text.toarray()
    print("Size of vectorized text matrix")
    print(text.shape)
    label = df.label_num
    for i in range(len(text)):
        dataset.append({
            'text'  : text[i],
            'label' : label[i]
        })
    return dataset


# finds the cosine_similarity using functions provided by numpy for speed optimization
def cosine_similarity(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1-cos_sim  # returning 1-cos_sim because larger similarity means vectors are closer


# finds the euclidean distance
def euclidean_dist(a, b):
    c = a-b
    val = np.linalg.norm(c)
    return val


# finds the manhattan distance
def manhattan_dist(a, b):
    return np.abs(a - b).sum()


# randomly splits the data 80:20 for train and test set
def split_data(data):
    random.shuffle(data)
    x_train, x_test = data[0: int(0.8*len(data))], data[int(0.8*len(data)):]
    return x_train, x_test

