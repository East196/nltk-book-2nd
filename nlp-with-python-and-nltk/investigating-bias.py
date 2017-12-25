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

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
		
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

document = [(list(movie_reviews.words(fileid)), category) 
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

# train and test with no bais
# random.shuffle(document)

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

# positive data example:
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# negative data example:
training_set = featuresets[100:]
testing_set = featuresets[:100]

# Naive Bayes algorithm: 
# posterior = prior occurenes * liklihood / evidence

# classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Orignal Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

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

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

voted_classifier = VoteClassifier(classifier, MNB_classifier, NuSVC_classifier, BNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
