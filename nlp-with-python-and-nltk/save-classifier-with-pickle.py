# Naive Bayes classifier - https://en.wikipedia.org/wiki/Naive_Bayes_classifier

import nltk
import random
from nltk.corpus import movie_reviews
import pickle

document = [(list(movie_reviews.words(fileid)), category) 
			for category in movie_reviews.categories()
			for fileid in movie_reviews.fileids(category)]

# train and test with no bais
random.shuffle(document)

all_words = []
for word in movie_reviews.words():
	all_words.append(word.lower())

# perform a frequency distribution
all_words = nltk.FreqDist(all_words)

# set a limit for the number of commonly used words to 3000
word_features = list(all_words.keys())[:3000]

# find the features that we're using
def find_features(document):
	words = set(document)
	features = {}
	for word in word_features:
		features[word] = (word in words)

	return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [ (find_features(rev), category) for (rev, category) in document]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Naive Bayes algorithm: 
# posterior = prior occurenes * liklihood / evidence

# classifer = nltk.NaiveBayesClassifier.train(training_set)

classifer_f = open("naivebayes.pickle", "rb")
classifer = pickle.load(classifer_f)
classifer_f.close()

print("Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifer, testing_set)) * 100)
classifer.show_most_informative_features(15)

# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifer, save_classifier)
# save_classifier.close()