# Naive Bayes classifier - https://en.wikipedia.org/wiki/Naive_Bayes_classifier

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

# import Scikit-LEarn classifiers

# Note: need to 'pip install' scikit-learn, matplotlib, numpy, and scipy
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

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

print("Orignal Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifer, testing_set))*100)
classifer.show_most_informative_features(15)

# MultinomialNB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# # GaussianNB
# GNB_classifier = SklearnClassifier(GaussianNB())
# GNB_classifier.train(training_set)
# print("GNB_classifier accuracy percent:", (nltk.classify.accuracy(GNB_classifier, testing_set))*100)

# BernoulliNB
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_classifier, testing_set))*100) 

# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)