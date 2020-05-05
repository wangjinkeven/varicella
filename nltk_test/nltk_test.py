from __future__ import division
from nltk.book import *


# print(text1.concordance("monstrous"))

# print(text1.similar("monstrous"))

# print(text2.common_contexts(["monstrous", "very"]))

# print(text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"]))

# print(text5.generate())

# print(sorted(set(text3)))

fdist1 = FreqDist(text1)
vocabulary1 = fdist1.keys()

fdist1.plot(50, cumulative=True)

V = set(text1)
long_words = [w for w in V if len(w) > 15]
print(sorted(long_words))

# print(len(text3)/len(set(text3)))

# print(text2.concordance("affection"))

# print(text3.concordance("lived"))

# print(text5.concordance("lol"))