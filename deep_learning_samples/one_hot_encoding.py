import numpy as np
import string
from keras.preprocessing.text import Tokenizer


samples = ["The cat sat on the mat.", "The dog ate my homework"]

# vacabulary level tokenize
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1
            
print(token_index)
            
max_length = 10

results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1

print(results)

# character level tokenize
characters = string.printable
token_index2 = dict(zip(range(1, len(characters) + 1), characters))
print(token_index2)

max_length2 = 50
results2 = np.zeros((len(samples), max_length2, max(token_index2.keys()) + 1))

for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results2[i, j, index] = 1.

print(results2)

# keras tokenize
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode="binary")
word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))
