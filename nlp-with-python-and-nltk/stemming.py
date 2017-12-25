# stemming - a form of prepocessing or normalization that 
# finds the stem of a word which finds the relation betwwen words
# "rid" for riding and rides --> riding and rides mean the same
# thing because of the "rid" stem

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]

# for word in example_words:
# 	print(ps.stem(word))

example_text = "Pythonistas, and pythoners a like, enjoy pythoning in Python especially when they've pythonly pythoned."

words = word_tokenize(example_text)

for  word in words:
	print(ps.stem(word))