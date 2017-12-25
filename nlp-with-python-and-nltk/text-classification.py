# text classification - used to classify a body of text (i.e. inbox mail and spam mail)
# often gives a positive or negative connotation

import nltk
import random
from nltk.corpus import movie_reviews

document = [(list(movie_reviews.words(fileid)), category) 
			for category in movie_reviews.categories()
			for fileid in movie_reviews.fileids(category)]

# train and test with no bais
random.shuffle(document)

# print(document[1])

all_words = []
for word in movie_reviews.words():
	all_words.append(word.lower())

# perform a frequency distribution
all_words = nltk.FreqDist(all_words)

# print(all_words.most_common(15))

# count how many times negative appears in the movie reviews
print(all_words["negative"]) # 44