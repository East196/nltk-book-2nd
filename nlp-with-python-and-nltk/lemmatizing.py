# Lemmatizing is basically a form of stemming in 
# which the stem is derived from a word but the stem 
# itself is actual word.

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# pos - part-of-speech - by default is set to noun
print(lemmatizer.lemmatize("cats")) # cat
print(lemmatizer.lemmatize("cacti")) # cactus
print(lemmatizer.lemmatize("geese")) # goose
print(lemmatizer.lemmatize("rocks")) # rock
print(lemmatizer.lemmatize("pythons")) # python
print(lemmatizer.lemmatize("better")) # better

# set pos is adjective so NLTK can lemmatize correctly
print(lemmatizer.lemmatize("better", pos="a")) # good
print(lemmatizer.lemmatize("best", pos="a")) # best
print(lemmatizer.lemmatize("run")) # run (as in "a coffee run")
print(lemmatizer.lemmatize("ran", pos="v")) # run