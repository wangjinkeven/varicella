import nltk
from nltk.corpus import gutenberg

print(gutenberg.fileids())

emma = nltk.Text(gutenberg.words('austen-emma.txt'))
print(emma.concordance("surprize"))

for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
    print("Avg Word Len: %s" % int(num_chars/num_words), "Avg Sent Len: %s" % int(num_words/num_sents), "Avg Word Freq: %s" % int(num_words/num_vocab), fileid)