from nltk.corpus import wordnet

syns = wordnet.synsets("program")

# # a set of synonyms for the word program
# print(syns)

# # just the name of the list of synonoyms
# print(syns[0].lemmas()[0].name())

# # definitions
# print(syns[0].definition())

# # examples
# print(syns[0].examples())

synonoyms = []
antonyms = []

for syn in wordnet.synsets("good"):
	for lemma in syn.lemmas():
		# print("lemma: ", lemma)
		synonoyms.append(lemma.name())
		if lemma.antonyms():
			antonyms.append(lemma.antonyms()[0].name())

# print(set(synonoyms))
# print(set(antonyms))

word1 = wordnet.synset("ship.n.01")
word2 = wordnet.synset("boat.n.01")

# comparisons - yields a percentage of how accurate words
# are with each other

# compare word1 and word2
print(word1.wup_similarity(word2)) # about 91%

word2 = wordnet.synset("car.n.01")

print(word1.wup_similarity(word2)) # about 70%

word2 = wordnet.synset("cat.n.01")

print(word1.wup_similarity(word2)) # 32%
