# Chinking is the removal of words and characters from
# a chunk.

# For this segment of NLP we need to know quantifiers
# to create regular expressions:

# "*"	0 or more occurences of a word
# "?"	0 or 1 occurence of a word
# "."	represents any character in a language
# "+"	1 or more occurences of a word

# Part-of-Speech (POS) Tag List:

# CC	coordinating conjunction
# CD	cardinal digit
# DT	determiner
# EX	existential there (like: "there is" ... think of it like "there exists")
# FW	foreign word
# IN	preposition/subordinating conjunction
# JJ	adjective	'big'
# JJR	adjective, comparative	'bigger'
# JJS	adjective, superlative	'biggest'
# LS	list marker	1)
# MD	modal	could, will
# NN	noun, singular 'desk'
# NNS	noun plural	'desks'
# NNP	proper noun, singular	'Harrison'
# NNPS	proper noun, plural	'Americans'
# PDT	predeterminer	'all the kids'
# POS	possessive ending	parent's
# PRP	personal pronoun	I, he, she
# PRP$	possessive pronoun	my, his, hers
# RB	adverb	very, silently,
# RBR	adverb, comparative	better
# RBS	adverb, superlative	best
# RP	particle	give up
# TO	to	go 'to' the store.
# UH	interjection	errrrrrrrm
# VB	verb, base form	take
# VBD	verb, past tense	took
# VBG	verb, gerund/present participle	taking
# VBN	verb, past participle	taken
# VBP	verb, sing. present, non-3d	take
# VBZ	verb, 3rd person sing. present	takes
# WDT	wh-determiner	which
# WP	wh-pronoun	who, what
# WP$	possessive wh-pronoun	whose
# WRB	wh-abverb	where, when

import nltk

# Import State of the Union speeches as a body of text.
from nltk.corpus import state_union
# Import an unsupervised machine learning sentence tokenizer.
# The Punkt. is already trained but we'll re-train it.
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
	for word in tokenized[5:]:
		words = nltk.word_tokenize(word)
		tagged = nltk.pos_tag(words)
		
		# Remove verbs, prepostions, determiners, or the
		# word "to" from the chunks.
		chunkGram = r"""Chunk: {<.*>+}
								}<VB.?|IN|DT|TO>+{"""

		chunkParser = nltk.RegexpParser(chunkGram)
		chunked = chunkParser.parse(tagged)

		#print(chunked)
		chunked.draw()

process_content()