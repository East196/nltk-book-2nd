# tokenize - to parse a body of text into a list of smaller elements
# sentence tokenizer - to parse a body of text into a list of sentences
# word tokenizer... etc.

from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "Hello, Mr. Smith, how are you today? It's pretty nice out today."

# Notice that the period after "Mr" doesn't signal to the tokenizer that
# the characters before it and after a beginning delimiter are together a sentence.
print(sent_tokenize(example_text))

print()

# Notice that "Mr." appeared in the list as oppose to ... "Mr", ".", etc.
# (Note: normally "." is it's own word)
print(word_tokenize(example_text))

print()

# Example of who to process and an example text
for word in word_tokenize(example_text):
	print(word)