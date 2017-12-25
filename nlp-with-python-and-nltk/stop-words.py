# stop words - words like "a", "an", and "the" which aid the
# the construction of the English language, but don't add any
# value to liguistic analysis

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "Python is a widely used high-level programming language for general-purpose programming, created by Guido van Rossum and first released in 1991."
stop_words = set(stopwords.words("english"))

# print()
# print(stop_words)
print()

words = word_tokenize(example_sentence)

# first implementation of stop-word filtered sentence:

# filtered_sentence = []

# for word in words:
# 	if word not in stop_words:
# 		filtered_sentence.append(word)

# second implementation of stop-word filtered sentence:

filtered_sentence = [word for word in words if not word in stop_words]

print(filtered_sentence)