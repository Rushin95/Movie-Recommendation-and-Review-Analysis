#Python 3
import re
from os import listdir
import numpy as np

# Full sentiment analysis of a single review
file1 = 'positive_reviews/6_10_tt0100680.txt' #Check other
f = open(file1,'r')

vocab = {}
for line in f:
    line = line.strip().lower()
    words = re.split(' |, |: |!|\.|"|\(|\)|\?|/|;|>|<',line)
    for word in words:
        if word == '':
            continue
        vocab[word]=vocab.get(word,0)+1
f.close()

# Load polarity words, adapted from http://sentiwordnet.isti.cnr.it/
f = open('polarity_words_uniq.csv','r')
i = 0
#Dictionary of polarity words: {pol_word1:polarity1, pol_word2:polarity2,...}
pol_words = {}
next(f)
for line in f:
    line = line.strip()
    line = line.split(',')
    pol_words[line[0]] = np.sign(float(line[1]))

vote = 0
for word in vocab.keys():
    polarity = pol_words.get(word,0)
    vote += polarity
if vote>0:
    print("Positive review, vote =",vote)
elif vote == 0:
    print("Neutral review")
else:
    print("Negative review, vote =",vote)

# Using regular expressions for tokenization
fold = 'positive_reviews/'
n_pos = 0
#Total number of files
n_files = 0
#Get the name of each file in fold
for file in listdir(fold):
    n_files += 1
    if n_files>5:
        break
    file1 = fold + file
    #total vote for each review
    vote = 0
    try:
        f = open(file1,'r')
        for line in f:
            print(line)
            line = line.strip().lower()
            words = re.split(' |, |: |!|\.|"|\'|\(|\)|\?|/|;|>|<',line)
            for word in words:
                polarity = pol_words.get(word,0)
                vote += polarity
        print('\n')
        if vote>0:
            print("Positive review, vote =",vote)
            n_pos += 1
        elif vote == 0:
            print("Neutral review")
            pass
        else:
            print("Negative review, vote =",vote)
            pass
        print('\n')
    except:
        continue

#Check the 4th review with the vote -14. It is indeed difficult to say from the text that this is a positive review.
print("Classifier accuracy is",n_pos/n_files)
