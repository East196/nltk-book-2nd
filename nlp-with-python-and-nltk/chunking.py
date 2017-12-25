# "Chunking is also called shallow parsing and it's basically
# the identification of parts of speech and short phrases 
# (like noun phrases). Part of speech tagging tells you 
# whether words are nouns, verbs, adjectives, etc, but it doesn't 
# give you any clue about the structure of the sentence or 
# phrases in the sentence. Sometimes it's useful to have more information 
# than just the parts of speech of words, but you don't need 
# the full parse tree that you would get from parsing."
# (https://stackoverflow.com/questions/1598940/in-natural-language-processing-what-is-the-purpose-of-chunking).

# For this segment of NLP we need to know quantifiers
# to create regular expressions:

# "*"	0 or more occurences of a word
# "?"	0 or 1 occurence of a word
# "."	represents any character in a language

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
	for word in tokenized:
		words = nltk.word_tokenize(word)
		tagged = nltk.pos_tag(words)
		
		# Use regular expression for chunking
		# "Include an adverb followed by a verb if there are any.
		# Then, require a proper noun (i.e. "Steve") followed by a
		# noun (i.e. "desk") if there is one. 
		chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP><NN>?}"""

		chunkParser = nltk.RegexpParser(chunkGram)
		chunked = chunkParser.parse(tagged)

		#print(chunked)
		chunked.draw()

process_content()