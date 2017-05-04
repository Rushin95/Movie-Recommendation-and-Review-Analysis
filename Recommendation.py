from os import listdir
import re

#Form a dictionary for each movie with all reviews
fold = 'positive_reviews/'
#Dictionary: {movie_id1:[rev1,rev2,...], movie_id2:[rev1,rev2,...]}
all_reviews = {}
for file in listdir(fold):
    #file format: [revID]_[score]_[movie_id].txt
    movie_id = re.split('_|\.',file)[2]
    #list of all previously seen reviews for this movie_id
    l = all_reviews.get(movie_id,[])
    try:
        f = open(fold+file,'r')
        review = ''
        for line in f:
            #Join all lines of a review, separate them by "."
            review = review+'.'+line
        #Append each review to a list of reviews
        l.append(review)
        #Update list of reviews for this movie_id
        all_reviews[movie_id] = l
        f.close()
    except:
        continue

tot = 0
#IDs of movies with more than 10 reviews
top_rev_ids = []
top_rev_words_collection = []
for movie_id,reviews in all_reviews.items():
    #Use only movies with more than 10 reviews for recommendation
    if len(reviews)>10:
        top_rev_ids.append(movie_id)
        words_collection = {}
        for review in reviews:
            review = review.strip().lower()
            words = re.split(' |, |: |!|\.|"|\'|\(|\)|\?|/|;|>|<',review)
            for word in words:
                words_collection[word] = words_collection.get(word,0)+1
                #Word's count in all previously seen reviews for this movie_id
        top_rev_words_collection.append(words_collection)

#Vocabulary of all words in positive
full_vocab = {}
for words_collection in top_rev_words_collection:
    for word,count in words_collection.items():
        full_vocab[word] = full_vocab.get(word,0)+count

high_freq_words = set(sorted(full_vocab, key=full_vocab.get,reverse=True)[:88]) #High frequency words

# Polarity words
#Used from http://sentiwordnet.isti.cnr.it/
f = open('polarity_words_uniq.csv','r')
pol_words = {}
#To skip header row
next(f)
for line in f:
    line = line.strip()
    line = line.split(',')
    polarity = float(line[1])
    pol_words[line[0]] = polarity
f.close()

high_pol_words = set([ k for k,v in pol_words.items() if v>=0.5 or v<=-0.5 ])

#Vocabulary for use in recomendation system
vocab = set(full_vocab.keys())-high_pol_words-high_freq_words

#Dictionary of vocabulary words and their positions in vocab_vec: {word1:pos1, word2:pos2,...}
vocab_pos = dict(zip(vocab,range(len(vocab))))

vocab_vec = sorted(vocab_pos, key=vocab_pos.get,reverse=False) #List of vocabulary words in order

import numpy as np
#Number of top reviewed positive movies
n_top_rev_mov = len(top_rev_ids)
#Number of all words in vocabulary
n_vocab = len(vocab_pos)
#Initially mov_vocab_matr is set to 0s
#Rows correspond to each movie in top_rev_words_collection
#columns correspond to each word in vocab_vec
mov_vocab_matr = np.zeros((n_top_rev_mov,n_vocab), dtype=np.int)

for movInd,words_collection in enumerate(top_rev_words_collection):
    for word in words_collection.keys():
        try:
            #If word has been seen in a review, set to 1
            mov_vocab_matr[movInd,vocab_pos[word]] = 1
        except:
            continue

f = open('negative_reviews/4_4_tt0047200.txt','r') #Open negative review of an unsatisfied customer
vocab_vec_neg_mov = np.zeros(n_vocab,dtype=np.int) #vector of 0s for each word in vocabulary
for line in f:
    line = line.strip().lower()
    words = re.split(' |, |: |!|\.|"|\'|\(|\)|\?|/|;|>|<',line)
    for word in words:
        try:
            vocab_vec_neg_mov[vocab_pos[word]] = 1
            #if word has been seen in this review, set to 1
        except:
            continue
f.close()

print([top_rev_ids[i] for i in np.argsort(-np.dot(mov_vocab_matr,vocab_vec_neg_mov))][:5])
#Recommended movies (sorted from the most similar to least similar)
