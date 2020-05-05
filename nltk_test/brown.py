import nltk
from nltk.corpus import brown

print(brown.categories())

print(brown.words(categories='news'))
print(brown.words(fileids='cg22'))
print(brown.sents(categories=['news', 'editorial', 'reviews']))

modals = ['can', 'could', 'may', 'might', 'must', 'will']

news_text = brown.words(categories='news')
fdist_news = nltk.FreqDist([w.lower() for w in news_text])
for m in modals:
    print('news: ' + m + ':', fdist_news[m])
    
government_text = brown.words(categories='government')
fdist_government = nltk.FreqDist([w.lower() for w in government_text])
for m in modals:
    print('government: ' + m + ':', fdist_government[m])
    
cfd = nltk.ConditionalFreqDist((genre, word)
                               for genre in brown.categories() 
                               for word in brown.words(categories=genre))

cfd.tabulate(conditions=[g for g in brown.categories()], samples=[m for m in modals])