
# Natural Language Processing

## 1. [Tokenizing Words and Sentences with NLTK](https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15350826/)


```python
from nltk.tokenize import sent_tokenize, word_tokenize

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

print(sent_tokenize(EXAMPLE_TEXT,"czech"))
```

    ['Hello Mr.', 'Smith, how are you doing today?', 'The weather is great, and Python is awesome.', 'The sky is pinkish-blue.', "You shouldn't eat cardboard."]
    


```python
print(word_tokenize(EXAMPLE_TEXT))
```

    ['Hello', 'Mr.', 'Smith', ',', 'how', 'are', 'you', 'doing', 'today', '?', 'The', 'weather', 'is', 'great', ',', 'and', 'Python', 'is', 'awesome', '.', 'The', 'sky', 'is', 'pinkish-blue', '.', 'You', 'should', "n't", 'eat', 'cardboard', '.']
    

## 2. [Stop words with NLTK](https://pythonprogramming.net/stop-words-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15350868/)


```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))
print "stop_words", stop_words

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)
```

    stop_words set([u'all', u'just', u'being', u'over', u'both', u'through', u'yourselves', u'its', u'before', u'o', u'hadn', u'herself', u'll', u'had', u'should', u'to', u'only', u'won', u'under', u'ours', u'has', u'do', u'them', u'his', u'very', u'they', u'not', u'during', u'now', u'him', u'nor', u'd', u'did', u'didn', u'this', u'she', u'each', u'further', u'where', u'few', u'because', u'doing', u'some', u'hasn', u'are', u'our', u'ourselves', u'out', u'what', u'for', u'while', u're', u'does', u'above', u'between', u'mustn', u't', u'be', u'we', u'who', u'were', u'here', u'shouldn', u'hers', u'by', u'on', u'about', u'couldn', u'of', u'against', u's', u'isn', u'or', u'own', u'into', u'yourself', u'down', u'mightn', u'wasn', u'your', u'from', u'her', u'their', u'aren', u'there', u'been', u'whom', u'too', u'wouldn', u'themselves', u'weren', u'was', u'until', u'more', u'himself', u'that', u'but', u'don', u'with', u'than', u'those', u'he', u'me', u'myself', u'ma', u'these', u'up', u'will', u'below', u'ain', u'can', u'theirs', u'my', u'and', u've', u'then', u'is', u'am', u'it', u'doesn', u'an', u'as', u'itself', u'at', u'have', u'in', u'any', u'if', u'again', u'no', u'when', u'same', u'how', u'other', u'which', u'you', u'shan', u'needn', u'haven', u'after', u'most', u'such', u'why', u'a', u'off', u'i', u'm', u'yours', u'so', u'y', u'the', u'having', u'once'])
    ['This', 'is', 'a', 'sample', 'sentence', ',', 'showing', 'off', 'the', 'stop', 'words', 'filtration', '.']
    ['This', 'sample', 'sentence', ',', 'showing', 'stop', 'words', 'filtration', '.']
    

## 3. [Stemming words with NLTK](https://pythonprogramming.net/stemming-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15350897/)


```python
from nltk.stem import PorterStemmer


ps = PorterStemmer()
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
for w in example_words:
    print(ps.stem(w))
```

    python
    python
    python
    python
    pythonli
    


```python
from nltk.tokenize import word_tokenize

new_text = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))
```

    It
    is
    import
    to
    by
    veri
    pythonli
    while
    you
    are
    python
    with
    python
    .
    all
    python
    have
    python
    poorli
    at
    least
    onc
    .
    

## 4. [Part of Speech Tagging with NLTK](https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15350929/)


```python
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:5]:
            print i
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))


process_content()
```

    0.0402097902098 0.0735785953177 0.0383693045564 5720 299 230 22
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.00437062937063 0.00334448160535 0.00442722744881 5720 299 25 1
    0.000699300699301 0.00334448160535 0.000553403431101 5720 299 4 1
    0.0013986013986 0.00334448160535 0.00129127467257 5720 299 8 1
    0.0171328671329 0.0401337792642 0.0158642316916 5720 299 98 12
    0.00122377622378 0.0066889632107 0.000922339051835 5720 299 7 2
    0.00384615384615 0.0066889632107 0.00368935620734 5720 299 22 2
    0.00157342657343 0.00334448160535 0.00147574248294 5720 299 9 1
    0.000874125874126 0.0066889632107 0.000553403431101 5720 299 5 2
    0.00314685314685 0.0133779264214 0.00258254934514 5720 299 18 4
    0.00244755244755 0.0066889632107 0.00221361372441 5720 299 14 2
    0.00192307692308 0.00334448160535 0.00184467810367 5720 299 11 1
    0.00157342657343 0.00334448160535 0.00147574248294 5720 299 9 1
    0.0138111888112 0.0200668896321 0.0134661501568 5720 299 79 6
    0.00034965034965 0.00334448160535 0.000184467810367 5720 299 2 1
    0.00227272727273 0.00334448160535 0.00221361372441 5720 299 13 1
    0.00122377622378 0.0167224080268 0.000368935620734 5720 299 7 5
    0.00244755244755 0.0167224080268 0.0016602102933 5720 299 14 5
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.000874125874126 0.0066889632107 0.000553403431101 5720 299 5 2
    0.00996503496503 0.0066889632107 0.0101457295702 5720 299 57 2
    0.0013986013986 0.00334448160535 0.00129127467257 5720 299 8 1
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.000874125874126 0.0066889632107 0.000553403431101 5720 299 5 2
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.000524475524476 0.00334448160535 0.000368935620734 5720 299 3 1
    0.00034965034965 0.0066889632107 0.0 5720 299 2 2
    0.0295454545455 0.0066889632107 0.0308061243313 5720 299 169 2
    0.0131118881119 0.0100334448161 0.0132816823464 5720 299 75 3
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.00437062937063 0.00334448160535 0.00442722744881 5720 299 25 1
    0.0162587412587 0.0802675585284 0.0127282789153 5720 299 93 24
    0.00034965034965 0.00334448160535 0.000184467810367 5720 299 2 1
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.00244755244755 0.00334448160535 0.00239808153477 5720 299 14 1
    0.00157342657343 0.0066889632107 0.00129127467257 5720 299 9 2
    0.0013986013986 0.00334448160535 0.00129127467257 5720 299 8 1
    0.00314685314685 0.00334448160535 0.00313595277624 5720 299 18 1
    0.000524475524476 0.0066889632107 0.000184467810367 5720 299 3 2
    0.0034965034965 0.00334448160535 0.00350488839697 5720 299 20 1
    0.00611888111888 0.00334448160535 0.00627190555248 5720 299 35 1
    0.00524475524476 0.0066889632107 0.00516509869028 5720 299 30 2
    0.00104895104895 0.00334448160535 0.000922339051835 5720 299 6 1
    0.0155594405594 0.00334448160535 0.0162331673123 5720 299 89 1
    0.00227272727273 0.00334448160535 0.00221361372441 5720 299 13 1
    0.00297202797203 0.00334448160535 0.00295148496587 5720 299 17 1
    0.000699300699301 0.00334448160535 0.000553403431101 5720 299 4 1
    0.00594405594406 0.0267558528428 0.00479616306954 5720 299 34 8
    0.00034965034965 0.00334448160535 0.000184467810367 5720 299 2 1
    0.00262237762238 0.0100334448161 0.00221361372441 5720 299 15 3
    0.00104895104895 0.0133779264214 0.000368935620734 5720 299 6 4
    0.00314685314685 0.0066889632107 0.00295148496587 5720 299 18 2
    0.00104895104895 0.00334448160535 0.000922339051835 5720 299 6 1
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.0449300699301 0.0434782608696 0.0450101457296 5720 299 257 13
    PRESIDENT GEORGE W. BUSH'S ADDRESS BEFORE A JOINT SESSION OF THE CONGRESS ON THE STATE OF THE UNION
     
    January 31, 2006
    
    THE PRESIDENT: Thank you all.
    [(u'PRESIDENT', 'NNP'), (u'GEORGE', 'NNP'), (u'W.', 'NNP'), (u'BUSH', 'NNP'), (u"'S", 'POS'), (u'ADDRESS', 'NNP'), (u'BEFORE', 'IN'), (u'A', 'NNP'), (u'JOINT', 'NNP'), (u'SESSION', 'NNP'), (u'OF', 'IN'), (u'THE', 'NNP'), (u'CONGRESS', 'NNP'), (u'ON', 'NNP'), (u'THE', 'NNP'), (u'STATE', 'NNP'), (u'OF', 'IN'), (u'THE', 'NNP'), (u'UNION', 'NNP'), (u'January', 'NNP'), (u'31', 'CD'), (u',', ','), (u'2006', 'CD'), (u'THE', 'NNP'), (u'PRESIDENT', 'NNP'), (u':', ':'), (u'Thank', 'NNP'), (u'you', 'PRP'), (u'all', 'DT'), (u'.', '.')]
    Mr. Speaker, Vice President Cheney, members of Congress, members of the Supreme Court and diplomatic corps, distinguished guests, and fellow citizens: Today our nation lost a beloved, graceful, courageous woman who called America to its founding ideals and carried on a noble dream.
    [(u'Mr.', 'NNP'), (u'Speaker', 'NNP'), (u',', ','), (u'Vice', 'NNP'), (u'President', 'NNP'), (u'Cheney', 'NNP'), (u',', ','), (u'members', 'NNS'), (u'of', 'IN'), (u'Congress', 'NNP'), (u',', ','), (u'members', 'NNS'), (u'of', 'IN'), (u'the', 'DT'), (u'Supreme', 'NNP'), (u'Court', 'NNP'), (u'and', 'CC'), (u'diplomatic', 'JJ'), (u'corps', 'NN'), (u',', ','), (u'distinguished', 'JJ'), (u'guests', 'NNS'), (u',', ','), (u'and', 'CC'), (u'fellow', 'JJ'), (u'citizens', 'NNS'), (u':', ':'), (u'Today', 'VB'), (u'our', 'PRP$'), (u'nation', 'NN'), (u'lost', 'VBD'), (u'a', 'DT'), (u'beloved', 'VBN'), (u',', ','), (u'graceful', 'JJ'), (u',', ','), (u'courageous', 'JJ'), (u'woman', 'NN'), (u'who', 'WP'), (u'called', 'VBD'), (u'America', 'NNP'), (u'to', 'TO'), (u'its', 'PRP$'), (u'founding', 'NN'), (u'ideals', 'NNS'), (u'and', 'CC'), (u'carried', 'VBD'), (u'on', 'IN'), (u'a', 'DT'), (u'noble', 'JJ'), (u'dream', 'NN'), (u'.', '.')]
    Tonight we are comforted by the hope of a glad reunion with the husband who was taken so long ago, and we are grateful for the good life of Coretta Scott King.
    [(u'Tonight', 'NN'), (u'we', 'PRP'), (u'are', 'VBP'), (u'comforted', 'VBN'), (u'by', 'IN'), (u'the', 'DT'), (u'hope', 'NN'), (u'of', 'IN'), (u'a', 'DT'), (u'glad', 'JJ'), (u'reunion', 'NN'), (u'with', 'IN'), (u'the', 'DT'), (u'husband', 'NN'), (u'who', 'WP'), (u'was', 'VBD'), (u'taken', 'VBN'), (u'so', 'RB'), (u'long', 'RB'), (u'ago', 'RB'), (u',', ','), (u'and', 'CC'), (u'we', 'PRP'), (u'are', 'VBP'), (u'grateful', 'JJ'), (u'for', 'IN'), (u'the', 'DT'), (u'good', 'JJ'), (u'life', 'NN'), (u'of', 'IN'), (u'Coretta', 'NNP'), (u'Scott', 'NNP'), (u'King', 'NNP'), (u'.', '.')]
    (Applause.)
    [(u'(', '('), (u'Applause', 'NNP'), (u'.', '.'), (u')', ')')]
    President George W. Bush reacts to applause during his State of the Union Address at the Capitol, Tuesday, Jan.
    [(u'President', 'NNP'), (u'George', 'NNP'), (u'W.', 'NNP'), (u'Bush', 'NNP'), (u'reacts', 'VBZ'), (u'to', 'TO'), (u'applause', 'VB'), (u'during', 'IN'), (u'his', 'PRP$'), (u'State', 'NNP'), (u'of', 'IN'), (u'the', 'DT'), (u'Union', 'NNP'), (u'Address', 'NNP'), (u'at', 'IN'), (u'the', 'DT'), (u'Capitol', 'NNP'), (u',', ','), (u'Tuesday', 'NNP'), (u',', ','), (u'Jan', 'NNP'), (u'.', '.')]
    

## 5. [Chunking with NLTK](https://pythonprogramming.net/chunking-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15353114/)


```python
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:3]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)

            chunked.draw()

    except Exception as e:
        print(str(e))

process_content()
```

    0.0402097902098 0.0735785953177 0.0383693045564 5720 299 230 22
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.00437062937063 0.00334448160535 0.00442722744881 5720 299 25 1
    0.000699300699301 0.00334448160535 0.000553403431101 5720 299 4 1
    0.0013986013986 0.00334448160535 0.00129127467257 5720 299 8 1
    0.0171328671329 0.0401337792642 0.0158642316916 5720 299 98 12
    0.00122377622378 0.0066889632107 0.000922339051835 5720 299 7 2
    0.00384615384615 0.0066889632107 0.00368935620734 5720 299 22 2
    0.00157342657343 0.00334448160535 0.00147574248294 5720 299 9 1
    0.000874125874126 0.0066889632107 0.000553403431101 5720 299 5 2
    0.00314685314685 0.0133779264214 0.00258254934514 5720 299 18 4
    0.00244755244755 0.0066889632107 0.00221361372441 5720 299 14 2
    0.00192307692308 0.00334448160535 0.00184467810367 5720 299 11 1
    0.00157342657343 0.00334448160535 0.00147574248294 5720 299 9 1
    0.0138111888112 0.0200668896321 0.0134661501568 5720 299 79 6
    0.00034965034965 0.00334448160535 0.000184467810367 5720 299 2 1
    0.00227272727273 0.00334448160535 0.00221361372441 5720 299 13 1
    0.00122377622378 0.0167224080268 0.000368935620734 5720 299 7 5
    0.00244755244755 0.0167224080268 0.0016602102933 5720 299 14 5
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.000874125874126 0.0066889632107 0.000553403431101 5720 299 5 2
    0.00996503496503 0.0066889632107 0.0101457295702 5720 299 57 2
    0.0013986013986 0.00334448160535 0.00129127467257 5720 299 8 1
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.000874125874126 0.0066889632107 0.000553403431101 5720 299 5 2
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.000524475524476 0.00334448160535 0.000368935620734 5720 299 3 1
    0.00034965034965 0.0066889632107 0.0 5720 299 2 2
    0.0295454545455 0.0066889632107 0.0308061243313 5720 299 169 2
    0.0131118881119 0.0100334448161 0.0132816823464 5720 299 75 3
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.00437062937063 0.00334448160535 0.00442722744881 5720 299 25 1
    0.0162587412587 0.0802675585284 0.0127282789153 5720 299 93 24
    0.00034965034965 0.00334448160535 0.000184467810367 5720 299 2 1
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.00244755244755 0.00334448160535 0.00239808153477 5720 299 14 1
    0.00157342657343 0.0066889632107 0.00129127467257 5720 299 9 2
    0.0013986013986 0.00334448160535 0.00129127467257 5720 299 8 1
    0.00314685314685 0.00334448160535 0.00313595277624 5720 299 18 1
    0.000524475524476 0.0066889632107 0.000184467810367 5720 299 3 2
    0.0034965034965 0.00334448160535 0.00350488839697 5720 299 20 1
    0.00611888111888 0.00334448160535 0.00627190555248 5720 299 35 1
    0.00524475524476 0.0066889632107 0.00516509869028 5720 299 30 2
    0.00104895104895 0.00334448160535 0.000922339051835 5720 299 6 1
    0.0155594405594 0.00334448160535 0.0162331673123 5720 299 89 1
    0.00227272727273 0.00334448160535 0.00221361372441 5720 299 13 1
    0.00297202797203 0.00334448160535 0.00295148496587 5720 299 17 1
    0.000699300699301 0.00334448160535 0.000553403431101 5720 299 4 1
    0.00594405594406 0.0267558528428 0.00479616306954 5720 299 34 8
    0.00034965034965 0.00334448160535 0.000184467810367 5720 299 2 1
    0.00262237762238 0.0100334448161 0.00221361372441 5720 299 15 3
    0.00104895104895 0.0133779264214 0.000368935620734 5720 299 6 4
    0.00314685314685 0.0066889632107 0.00295148496587 5720 299 18 2
    0.00104895104895 0.00334448160535 0.000922339051835 5720 299 6 1
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.0449300699301 0.0434782608696 0.0450101457296 5720 299 257 13
    (S
      (Chunk PRESIDENT/NNP GEORGE/NNP W./NNP BUSH/NNP)
      'S/POS
      (Chunk ADDRESS/NNP)
      BEFORE/IN
      (Chunk A/NNP JOINT/NNP SESSION/NNP)
      OF/IN
      (Chunk THE/NNP CONGRESS/NNP ON/NNP THE/NNP STATE/NNP)
      OF/IN
      (Chunk THE/NNP UNION/NNP January/NNP)
      31/CD
      ,/,
      2006/CD
      (Chunk THE/NNP PRESIDENT/NNP)
      :/:
      (Chunk Thank/NNP)
      you/PRP
      all/DT
      ./.)
    (Chunk PRESIDENT/NNP GEORGE/NNP W./NNP BUSH/NNP)
    (Chunk ADDRESS/NNP)
    (Chunk A/NNP JOINT/NNP SESSION/NNP)
    (Chunk THE/NNP CONGRESS/NNP ON/NNP THE/NNP STATE/NNP)
    (Chunk THE/NNP UNION/NNP January/NNP)
    (Chunk THE/NNP PRESIDENT/NNP)
    (Chunk Thank/NNP)
    (S
      (Chunk Mr./NNP Speaker/NNP)
      ,/,
      (Chunk Vice/NNP President/NNP Cheney/NNP)
      ,/,
      members/NNS
      of/IN
      (Chunk Congress/NNP)
      ,/,
      members/NNS
      of/IN
      the/DT
      (Chunk Supreme/NNP Court/NNP)
      and/CC
      diplomatic/JJ
      corps/NN
      ,/,
      distinguished/JJ
      guests/NNS
      ,/,
      and/CC
      fellow/JJ
      citizens/NNS
      :/:
      Today/VB
      our/PRP$
      nation/NN
      lost/VBD
      a/DT
      beloved/VBN
      ,/,
      graceful/JJ
      ,/,
      courageous/JJ
      woman/NN
      who/WP
      (Chunk called/VBD America/NNP)
      to/TO
      its/PRP$
      founding/NN
      ideals/NNS
      and/CC
      carried/VBD
      on/IN
      a/DT
      noble/JJ
      dream/NN
      ./.)
    (Chunk Mr./NNP Speaker/NNP)
    (Chunk Vice/NNP President/NNP Cheney/NNP)
    (Chunk Congress/NNP)
    (Chunk Supreme/NNP Court/NNP)
    (Chunk called/VBD America/NNP)
    (S
      Tonight/NN
      we/PRP
      are/VBP
      comforted/VBN
      by/IN
      the/DT
      hope/NN
      of/IN
      a/DT
      glad/JJ
      reunion/NN
      with/IN
      the/DT
      husband/NN
      who/WP
      was/VBD
      taken/VBN
      so/RB
      long/RB
      ago/RB
      ,/,
      and/CC
      we/PRP
      are/VBP
      grateful/JJ
      for/IN
      the/DT
      good/JJ
      life/NN
      of/IN
      (Chunk Coretta/NNP Scott/NNP King/NNP)
      ./.)
    (Chunk Coretta/NNP Scott/NNP King/NNP)
    

## 6. [Chinking with NLTK](https://pythonprogramming.net/chinking-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15353145/)


```python
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:3]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            chunked.draw()

    except Exception as e:
        print(str(e))

process_content()
```

    0.0402097902098 0.0735785953177 0.0383693045564 5720 299 230 22
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.00437062937063 0.00334448160535 0.00442722744881 5720 299 25 1
    0.000699300699301 0.00334448160535 0.000553403431101 5720 299 4 1
    0.0013986013986 0.00334448160535 0.00129127467257 5720 299 8 1
    0.0171328671329 0.0401337792642 0.0158642316916 5720 299 98 12
    0.00122377622378 0.0066889632107 0.000922339051835 5720 299 7 2
    0.00384615384615 0.0066889632107 0.00368935620734 5720 299 22 2
    0.00157342657343 0.00334448160535 0.00147574248294 5720 299 9 1
    0.000874125874126 0.0066889632107 0.000553403431101 5720 299 5 2
    0.00314685314685 0.0133779264214 0.00258254934514 5720 299 18 4
    0.00244755244755 0.0066889632107 0.00221361372441 5720 299 14 2
    0.00192307692308 0.00334448160535 0.00184467810367 5720 299 11 1
    0.00157342657343 0.00334448160535 0.00147574248294 5720 299 9 1
    0.0138111888112 0.0200668896321 0.0134661501568 5720 299 79 6
    0.00034965034965 0.00334448160535 0.000184467810367 5720 299 2 1
    0.00227272727273 0.00334448160535 0.00221361372441 5720 299 13 1
    0.00122377622378 0.0167224080268 0.000368935620734 5720 299 7 5
    0.00244755244755 0.0167224080268 0.0016602102933 5720 299 14 5
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.000874125874126 0.0066889632107 0.000553403431101 5720 299 5 2
    0.00996503496503 0.0066889632107 0.0101457295702 5720 299 57 2
    0.0013986013986 0.00334448160535 0.00129127467257 5720 299 8 1
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.000874125874126 0.0066889632107 0.000553403431101 5720 299 5 2
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.000524475524476 0.00334448160535 0.000368935620734 5720 299 3 1
    0.00034965034965 0.0066889632107 0.0 5720 299 2 2
    0.0295454545455 0.0066889632107 0.0308061243313 5720 299 169 2
    0.0131118881119 0.0100334448161 0.0132816823464 5720 299 75 3
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.00437062937063 0.00334448160535 0.00442722744881 5720 299 25 1
    0.0162587412587 0.0802675585284 0.0127282789153 5720 299 93 24
    0.00034965034965 0.00334448160535 0.000184467810367 5720 299 2 1
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.00244755244755 0.00334448160535 0.00239808153477 5720 299 14 1
    0.00157342657343 0.0066889632107 0.00129127467257 5720 299 9 2
    0.0013986013986 0.00334448160535 0.00129127467257 5720 299 8 1
    0.00314685314685 0.00334448160535 0.00313595277624 5720 299 18 1
    0.000524475524476 0.0066889632107 0.000184467810367 5720 299 3 2
    0.0034965034965 0.00334448160535 0.00350488839697 5720 299 20 1
    0.00611888111888 0.00334448160535 0.00627190555248 5720 299 35 1
    0.00524475524476 0.0066889632107 0.00516509869028 5720 299 30 2
    0.00104895104895 0.00334448160535 0.000922339051835 5720 299 6 1
    0.0155594405594 0.00334448160535 0.0162331673123 5720 299 89 1
    0.00227272727273 0.00334448160535 0.00221361372441 5720 299 13 1
    0.00297202797203 0.00334448160535 0.00295148496587 5720 299 17 1
    0.000699300699301 0.00334448160535 0.000553403431101 5720 299 4 1
    0.00594405594406 0.0267558528428 0.00479616306954 5720 299 34 8
    0.00034965034965 0.00334448160535 0.000184467810367 5720 299 2 1
    0.00262237762238 0.0100334448161 0.00221361372441 5720 299 15 3
    0.00104895104895 0.0133779264214 0.000368935620734 5720 299 6 4
    0.00314685314685 0.0066889632107 0.00295148496587 5720 299 18 2
    0.00104895104895 0.00334448160535 0.000922339051835 5720 299 6 1
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.0449300699301 0.0434782608696 0.0450101457296 5720 299 257 13
    

## 7. [Named Entity Recognition with NLTK](https://pythonprogramming.net/named-entity-recognition-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15353198/)


```python
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:3]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            namedEnt.draw()
    except Exception as e:
        print(str(e))


process_content()
```

    0.0402097902098 0.0735785953177 0.0383693045564 5720 299 230 22
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.00437062937063 0.00334448160535 0.00442722744881 5720 299 25 1
    0.000699300699301 0.00334448160535 0.000553403431101 5720 299 4 1
    0.0013986013986 0.00334448160535 0.00129127467257 5720 299 8 1
    0.0171328671329 0.0401337792642 0.0158642316916 5720 299 98 12
    0.00122377622378 0.0066889632107 0.000922339051835 5720 299 7 2
    0.00384615384615 0.0066889632107 0.00368935620734 5720 299 22 2
    0.00157342657343 0.00334448160535 0.00147574248294 5720 299 9 1
    0.000874125874126 0.0066889632107 0.000553403431101 5720 299 5 2
    0.00314685314685 0.0133779264214 0.00258254934514 5720 299 18 4
    0.00244755244755 0.0066889632107 0.00221361372441 5720 299 14 2
    0.00192307692308 0.00334448160535 0.00184467810367 5720 299 11 1
    0.00157342657343 0.00334448160535 0.00147574248294 5720 299 9 1
    0.0138111888112 0.0200668896321 0.0134661501568 5720 299 79 6
    0.00034965034965 0.00334448160535 0.000184467810367 5720 299 2 1
    0.00227272727273 0.00334448160535 0.00221361372441 5720 299 13 1
    0.00122377622378 0.0167224080268 0.000368935620734 5720 299 7 5
    0.00244755244755 0.0167224080268 0.0016602102933 5720 299 14 5
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.000874125874126 0.0066889632107 0.000553403431101 5720 299 5 2
    0.00996503496503 0.0066889632107 0.0101457295702 5720 299 57 2
    0.0013986013986 0.00334448160535 0.00129127467257 5720 299 8 1
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.000874125874126 0.0066889632107 0.000553403431101 5720 299 5 2
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.000524475524476 0.00334448160535 0.000368935620734 5720 299 3 1
    0.00034965034965 0.0066889632107 0.0 5720 299 2 2
    0.0295454545455 0.0066889632107 0.0308061243313 5720 299 169 2
    0.0131118881119 0.0100334448161 0.0132816823464 5720 299 75 3
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.00437062937063 0.00334448160535 0.00442722744881 5720 299 25 1
    0.0162587412587 0.0802675585284 0.0127282789153 5720 299 93 24
    0.00034965034965 0.00334448160535 0.000184467810367 5720 299 2 1
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.00244755244755 0.00334448160535 0.00239808153477 5720 299 14 1
    0.00157342657343 0.0066889632107 0.00129127467257 5720 299 9 2
    0.0013986013986 0.00334448160535 0.00129127467257 5720 299 8 1
    0.00314685314685 0.00334448160535 0.00313595277624 5720 299 18 1
    0.000524475524476 0.0066889632107 0.000184467810367 5720 299 3 2
    0.0034965034965 0.00334448160535 0.00350488839697 5720 299 20 1
    0.00611888111888 0.00334448160535 0.00627190555248 5720 299 35 1
    0.00524475524476 0.0066889632107 0.00516509869028 5720 299 30 2
    0.00104895104895 0.00334448160535 0.000922339051835 5720 299 6 1
    0.0155594405594 0.00334448160535 0.0162331673123 5720 299 89 1
    0.00227272727273 0.00334448160535 0.00221361372441 5720 299 13 1
    0.00297202797203 0.00334448160535 0.00295148496587 5720 299 17 1
    0.000699300699301 0.00334448160535 0.000553403431101 5720 299 4 1
    0.00594405594406 0.0267558528428 0.00479616306954 5720 299 34 8
    0.00034965034965 0.00334448160535 0.000184467810367 5720 299 2 1
    0.00262237762238 0.0100334448161 0.00221361372441 5720 299 15 3
    0.00104895104895 0.0133779264214 0.000368935620734 5720 299 6 4
    0.00314685314685 0.0066889632107 0.00295148496587 5720 299 18 2
    0.00104895104895 0.00334448160535 0.000922339051835 5720 299 6 1
    0.000174825174825 0.00334448160535 0.0 5720 299 1 1
    0.0449300699301 0.0434782608696 0.0450101457296 5720 299 257 13
    

## 8. [Lemmatizing with NLTK](https://pythonprogramming.net/lemmatizing-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15353273/)


```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("dog"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))
```

    cat
    cactus
    goose
    rock
    python
    good
    best
    dog
    run
    run
    

## 9. [The corpora with NLTK](https://pythonprogramming.net/nltk-corpus-corpora-tutorial/) | [video](https://www.bilibili.com/video/av15353335/)


```python
import nltk
print(nltk.__file__)

from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg

# sample text
sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

for x in range(5):
    print(tok[x])
```

    D:\Anaconda2\lib\site-packages\nltk\__init__.pyc
    [The King James Bible]
    
    The Old Testament of the King James Bible
    
    The First Book of Moses:  Called Genesis
    
    
    1:1 In the beginning God created the heaven and the earth.
    1:2 And the earth was without form, and void; and darkness was upon
    the face of the deep.
    And the Spirit of God moved upon the face of the
    waters.
    1:3 And God said, Let there be light: and there was light.
    1:4 And God saw the light, that it was good: and God divided the light
    from the darkness.
    

## 10. [Wordnet with NLTK](https://pythonprogramming.net/wordnet-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15353391/)


```python
from nltk.corpus import wordnet
syns = wordnet.synsets("program")
print(syns[0].name())
print(syns[0].lemmas()[0].name())
print(syns[0].definition())
print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))
```

    plan.n.01
    plan
    a series of steps to be carried out or goals to be accomplished
    [u'they drew up a six-step plan', u'they discussed plans for a new bond issue']
    set([u'beneficial', u'right', u'secure', u'just', u'unspoilt', u'respectable', u'good', u'goodness', u'dear', u'salutary', u'ripe', u'expert', u'skillful', u'in_force', u'proficient', u'unspoiled', u'dependable', u'soundly', u'honorable', u'full', u'undecomposed', u'safe', u'adept', u'upright', u'trade_good', u'sound', u'in_effect', u'practiced', u'effective', u'commodity', u'estimable', u'well', u'honest', u'near', u'skilful', u'thoroughly', u'serious'])
    set([u'bad', u'badness', u'ill', u'evil', u'evilness'])
    0.909090909091
    


```python
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cat.n.01')
print(w1.wup_similarity(w2))
```

    0.909090909091
    0.695652173913
    0.32
    

## 11. [Text Classification with NLTK](https://pythonprogramming.net/text-classification-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15355245/)



```python
import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print(all_words["stupid"])
```

    ([u'here', u'is', u'a', u'film', u'that', u'is', u'so', u'unexpected', u',', u'so', u'scary', u',', u'and', u'so', u'original', u'that', u'it', u'caught', u'me', u'off', u'guard', u'and', u'threw', u'me', u'for', u'a', u'loop', u'.', u'okay', u',', u'it', u'isn', u"'", u't', u'quite', u'original', u',', u'considering', u'it', u'is', u'a', u'sequel', u'to', u'the', u'box', u'office', u'hit', u'species', u',', u'but', u'it', u'certainly', u'is', u'smart', u'.', u'most', u'films', u'of', u'this', u'genre', u'are', u'reminiscent', u'of', u'those', u'cheesy', u'b', u'-', u'horror', u'films', u'from', u'the', u'50s', u'and', u'60s', u',', u'and', u'some', u'even', u'become', u'them', u'.', u'however', u',', u'as', u'we', u'learned', u'with', u'the', u'1995', u'small', u'-', u'budget', u'horror', u'/', u'sci', u'-', u'fi', u'film', u',', u'sometimes', u'expectations', u'can', u'be', u'shattered', u'.', u'a', u'lot', u'of', u'criticism', u'has', u'gone', u'against', u'this', u'film', u'(', u'from', u'what', u'i', u'have', u'read', u'so', u'far', u',', u'anyway', u'--', u'yep', u',', u'all', u'two', u'reviews', u')', u',', u'and', u'it', u'makes', u'me', u'wonder', u'why', u'these', u'types', u'of', u'films', u'are', u'automatically', u'dismissed', u'as', u'gory', u',', u'laughable', u'pieces', u'of', u'trash', u'.', u'but', u',', u'the', u'thing', u'is', u',', u'it', u'isn', u"'", u't', u'.', u'it', u"'", u's', u'well', u'made', u',', u'well', u'acted', u',', u'and', u'quite', u'intelligent', u'.', u'i', u'can', u'see', u'most', u'of', u'the', u'critics', u'now', u'complaining', u'about', u'the', u'level', u'of', u'gore', u'or', u'the', u'level', u'of', u'sexuality', u'in', u'the', u'film', u'.', u'but', u'the', u'species', u'series', u'isn', u"'", u't', u'about', u'the', u'lack', u'of', u'these', u'elements', u'.', u'it', u"'", u's', u'about', u'how', u'much', u'it', u'can', u'get', u'into', u'one', u'film', u'.', u'and', u'yet', u',', u'behind', u'it', u'all', u',', u'it', u'has', u'this', u'basic', u'premise', u'that', u'allows', u'it', u'to', u'get', u'away', u'with', u'doing', u'so', u'.', u'species', u'ii', u'begins', u'in', u'the', u'present', u'day', u',', u'though', u'it', u'seems', u'to', u'be', u'an', u'alternate', u'universe', u'.', u'many', u'films', u'(', u'especially', u'sci', u'-', u'fi', u'ones', u')', u'create', u'similar', u'timelines', u'as', u'our', u'realistic', u'one', u',', u'but', u'change', u'it', u'to', u'fit', u'the', u'film', u"'", u's', u'needs', u'.', u'species', u'ii', u'begins', u'with', u'the', u'arrival', u'of', u'an', u'american', u'spacecraft', u',', u'the', u'excursion', u',', u'landing', u'on', u'the', u'surface', u'of', u'mars', u'.', u'aboard', u'is', u'patrick', u'ross', u'(', u'justin', u'lazard', u')', u',', u'a', u'very', u'bright', u'and', u'very', u'handsome', u'astronaut', u'.', u'patrick', u'is', u'the', u'son', u'of', u'senator', u'ross', u'(', u'james', u'cromwell', u')', u',', u'who', u'just', u'wants', u'patrick', u'to', u'succeed', u'.', u'well', u',', u'it', u'would', u'seem', u'that', u'he', u'has', u'succeeded', u'.', u'landing', u'on', u'the', u'surface', u'of', u'mars', u',', u'he', u'is', u'the', u'first', u'human', u'being', u'to', u'ever', u'do', u'so', u'.', u'of', u'course', u',', u'he', u'isn', u"'", u't', u'the', u'first', u'ever', u'.', u'about', u'a', u'billion', u'years', u'ago', u',', u'an', u'alien', u'species', u'supposedly', u'landed', u'on', u'mars', u'and', u'destroyed', u'the', u'perfect', u'living', u'conditions', u'that', u'were', u'able', u'to', u'sustain', u'life', u'.', u'now', u',', u'of', u'course', u',', u'the', u'red', u'planet', u'is', u'cold', u'and', u'rocky', u'.', u'no', u'life', u'lives', u'on', u'it', u'.', u'that', u'is', u',', u'no', u'visible', u'life', u'.', u'patrick', u',', u'upon', u'leaving', u'the', u'spacecraft', u'and', u'landing', u'on', u'the', u'red', u'soil', u',', u'collects', u'samples', u'from', u'the', u'ground', u'.', u'he', u'takes', u'them', u'aboard', u',', u'and', u'puts', u'them', u'in', u'storage', u'.', u'unfortunately', u',', u'one', u'of', u'the', u'samples', u'contains', u'a', u'form', u'of', u'life', u',', u'and', u'it', u'gets', u'loose', u'when', u'it', u'is', u'heated', u'aboard', u'the', u'ship', u'.', u'just', u'prior', u'to', u'heading', u'back', u'to', u'earth', u',', u'this', u'life', u'form', u'creeps', u'along', u'the', u'floor', u'and', u'inhabits', u'the', u'earthlings', u'.', u'they', u'pass', u'out', u'for', u'approximately', u'seven', u'minutes', u',', u'and', u'then', u'shrug', u'it', u'off', u'as', u'nothing', u',', u'because', u'they', u'can', u"'", u't', u'even', u'remember', u'.', u'they', u'blame', u'it', u'on', u'a', u'technical', u'malfunction', u'.', u'back', u'on', u'earth', u',', u'patrick', u'begins', u'to', u'have', u'strong', u'urges', u'to', u'mate', u'with', u'as', u'many', u'women', u'as', u'possible', u'.', u'as', u'we', u'know', u'from', u'the', u'original', u',', u'this', u'is', u'because', u'the', u'alien', u'wants', u'to', u'breed', u'and', u'take', u'over', u'the', u'planet', u'.', u'however', u',', u'the', u'children', u'that', u'are', u'bred', u'are', u'half', u'-', u'human', u',', u'as', u'their', u'father', u'is', u'.', u'patrick', u'is', u'really', u'looking', u'for', u'another', u'alien', u'to', u'breed', u'with', u',', u'and', u'he', u'finds', u'it', u'in', u'eve', u'(', u'natasha', u'henstridge', u')', u'.', u'eve', u'was', u'cloned', u'from', u'dna', u'taken', u'from', u'sil', u',', u'the', u'original', u'alien', u'.', u'however', u',', u'this', u'time', u'around', u',', u'most', u'of', u'her', u'"', u'alien', u'"', u'urges', u'have', u'been', u'either', u'decreased', u'dramatically', u',', u'or', u'lie', u'dormant', u'.', u'the', u'project', u'is', u'led', u'by', u'dr', u'.', u'laura', u'baker', u'(', u'marg', u'helgenberger', u',', u'reprising', u'her', u'role', u'from', u'species', u')', u',', u'and', u'her', u'motives', u'seem', u'respectable', u'.', u'since', u'she', u'was', u'involved', u'with', u'the', u'original', u'alien', u'attack', u',', u'she', u'wants', u'to', u'learn', u'how', u'to', u'stop', u'the', u'alien', u'should', u'it', u'come', u'again', u'.', u'and', u'it', u'has', u'.', u'story', u'-', u'wise', u',', u'species', u'ii', u'is', u'much', u'stronger', u'than', u'its', u'predecessor', u',', u'but', u'it', u'is', u'also', u'much', u'stronger', u'than', u',', u'say', u',', u'aliens', u'(', u'hey', u',', u'i', u'love', u'the', u'film', u',', u'but', u'you', u'can', u"'", u't', u'tell', u'me', u'it', u'was', u'strong', u'on', u'story', u')', u'.', u'what', u'surprised', u'me', u'the', u'most', u'with', u'this', u'film', u'was', u'the', u'incorporation', u'of', u'historical', u'facts', u'into', u'the', u'screenplay', u'.', u'in', u'my', u'search', u'for', u'extraterrestrial', u'intelligence', u'course', u'in', u'college', u',', u'we', u'learned', u'about', u'a', u'piece', u'of', u'rock', u'from', u'mars', u'which', u'landed', u'in', u'one', u'of', u'the', u'poles', u'.', u'this', u'piece', u'of', u'rock', u'contained', u'fossils', u'which', u'may', u'have', u'been', u'proof', u'of', u'life', u'on', u'mars', u'(', u'later', u',', u'it', u'was', u'proven', u'that', u'it', u'was', u'not', u'a', u'living', u'creature', u'that', u'created', u'it', u')', u'.', u'the', u'script', u'uses', u'this', u'effectively', u',', u'but', u'also', u'manages', u'to', u'provide', u'a', u'well', u'-', u'balanced', u'plot', u'.', u'beginning', u'with', u'the', u'first', u'man', u'on', u'mars', u'(', u'something', u'i', u'have', u'always', u'dreamed', u'of', u'seeing', u')', u',', u'i', u'was', u'hoping', u'that', u'the', u'film', u'would', u'turn', u'this', u'element', u'into', u'a', u'useful', u'starting', u'point', u'for', u'the', u'movie', u'.', u'and', u'it', u'does', u'it', u'quite', u'well', u'.', u'the', u'characters', u'are', u'all', u'smart', u',', u'and', u'they', u'know', u'what', u'to', u'do', u'and', u'what', u'not', u'to', u'do', u'.', u'the', u'only', u'character', u'that', u'seems', u'a', u'little', u'cliched', u'is', u'the', u'general', u'(', u'george', u'dzundza', u')', u',', u'and', u'yet', u',', u'he', u'remains', u'logical', u'in', u'everything', u'he', u'does', u'.', u'there', u'are', u'the', u'obvious', u'flaws', u'of', u'course', u',', u'mostly', u'lying', u'in', u'the', u'technical', u'aspects', u'.', u'the', u'special', u'effects', u'are', u'only', u'mediocre', u',', u'and', u'some', u'are', u'just', u'plain', u'bad', u'.', u'but', u'for', u'the', u'most', u'part', u',', u'they', u'remain', u'believable', u'(', u'i', u'even', u'noticed', u'a', u'homage', u'to', u'the', u'alien', u'series', u'when', u'the', u'mothers', u'gave', u'birth', u'to', u'alien', u'children', u')', u'.', u'also', u',', u'the', u'most', u'realistic', u'ones', u'are', u'usually', u'the', u'goriest', u',', u'ranging', u'from', u'people', u'being', u'torn', u'open', u',', u'or', u'someone', u"'", u's', u'head', u'being', u'blown', u'off', u'.', u'however', u',', u'some', u'plot', u'elements', u'also', u'may', u'elicit', u'laughs', u'from', u'the', u'audience', u',', u'including', u'a', u'menage', u'a', u'troi', u'that', u'is', u'all', u'but', u'necessary', u'.', u'many', u'people', u'dislike', u'the', u'species', u'series', u'because', u'all', u'it', u'is', u'is', u'an', u'excuse', u'for', u'sex', u',', u'nudity', u',', u'and', u'gory', u'violence', u'.', u'however', u',', u'i', u'tend', u'to', u'disagree', u'.', u'what', u'were', u'the', u'alien', u'films', u'about', u'?', u'and', u',', u'if', u'an', u'alien', u'species', u'ever', u'did', u'come', u'to', u'earth', u',', u'and', u'their', u'sole', u'purpose', u'was', u'to', u'destroy', u'us', u',', u'wouldn', u"'", u't', u'you', u'mate', u'as', u'quickly', u'as', u'possible', u'with', u'as', u'many', u'people', u'as', u'possible', u'?', u'my', u'only', u'gripe', u'with', u'this', u'is', u'during', u'the', u'scene', u'where', u'patrick', u'goes', u'searching', u'for', u'a', u'mate', u'in', u'a', u'grocery', u'store', u'.', u'i', u'didn', u"'", u't', u'realize', u'that', u'aliens', u'were', u'that', u'picky', u'on', u'choosing', u'women', u'to', u'mate', u'with', u'(', u'i', u'just', u'assume', u'it', u'is', u'his', u'part', u'-', u'human', u'side', u'looking', u'for', u'the', u'most', u'beautiful', u'one', u')', u'.', u'the', u'acting', u'is', u'quite', u'good', u'for', u'this', u'kind', u'of', u'film', u'.', u'it', u'is', u'a', u'vast', u'improvement', u'over', u'the', u'first', u'film', u',', u'at', u'least', u'.', u'the', u'acting', u'is', u'the', u'key', u'element', u'to', u'this', u'film', u':', u'if', u'it', u'was', u'bad', u',', u'it', u'would', u'have', u'lowered', u'itself', u'into', u'camp', u';', u'if', u'it', u'was', u'good', u',', u'it', u'would', u'have', u'asked', u'for', u'comparison', u'with', u'films', u'like', u'aliens', u'.', u'okay', u',', u'so', u'it', u'isn', u"'", u't', u'that', u'good', u'.', u'george', u'dzundza', u'is', u'probably', u'the', u'most', u'obvious', u'mistake', u'on', u'casting', u',', u'as', u'his', u'over', u'-', u'the', u'-', u'top', u'impersonation', u'of', u'a', u'general', u'makes', u'him', u'annoying', u'and', u'distracting', u'.', u'natasha', u'henstridge', u'is', u'limited', u'this', u'time', u'around', u',', u'as', u'she', u'is', u'usually', u'enclosed', u'in', u'a', u'cage', u'.', u'however', u',', u'she', u'does', u'manage', u'a', u'very', u'impressive', u'performance', u'with', u'this', u'aspect', u'hindering', u'any', u'of', u'her', u'talent', u'.', u'oh', u'yeah', u',', u'and', u'she', u"'", u's', u'quite', u'fun', u'to', u'just', u'plain', u'watch', u'.', u'marg', u'helgenberger', u'is', u'immensely', u'better', u'this', u'time', u'around', u',', u'and', u'her', u'performance', u'is', u'probably', u'the', u'best', u'in', u'this', u'film', u'.', u'michael', u'madsen', u'is', u'so', u'-', u'so', u',', u'but', u'he', u'isn', u"'", u't', u'annoying', u',', u'and', u'he', u'soon', u'becomes', u'rather', u'appealing', u'(', u'with', u'his', u'nice', u'cynic', u'personality', u')', u'.', u'james', u'cromwell', u'has', u'a', u'small', u'part', u',', u'but', u'he', u'makes', u'it', u'much', u'better', u'than', u'what', u'it', u'could', u'have', u'been', u'with', u'a', u'more', u'incapable', u'actor', u'.', u'as', u'i', u'say', u',', u'any', u'film', u'with', u'james', u'cromwell', u'dramatically', u'increases', u'in', u'likeability', u'.', u'mykelti', u'williamson', u'gives', u'an', u'enjoyable', u'performance', u',', u'and', u'he', u'gives', u'the', u'film', u'a', u'more', u'down', u'-', u'to', u'-', u'earth', u'feel', u'.', u'and', u',', u'of', u'course', u',', u'justin', u'lazard', u'.', u'lazard', u'has', u'so', u'far', u'been', u'ridiculed', u'for', u'his', u'performance', u',', u'but', u'i', u'think', u'he', u'is', u'effective', u'.', u'sure', u',', u'he', u'is', u'wooden', u',', u'but', u'isn', u"'", u't', u'that', u'what', u'his', u'character', u'is', u'like', u'?', u'the', u'moment', u'when', u'he', u'touches', u'the', u'glass', u'separating', u'henstridge', u'from', u'him', u'was', u'extremely', u'well', u'done', u',', u'due', u'to', u'the', u'couple', u"'", u's', u'acting', u'.', u'species', u'ii', u'is', u'rated', u'r', u'for', u'strong', u'sexuality', u',', u'sci', u'-', u'fi', u'violence', u'/', u'gore', u'and', u'language', u'.', u'this', u'is', u'definitely', u'an', u'r', u'rated', u'film', u'that', u'young', u'kids', u'should', u'not', u'see', u'.', u'more', u'than', u'likely', u',', u'they', u'would', u'probably', u'have', u'nightmares', u'and', u'never', u'have', u'sex', u'for', u'the', u'rest', u'of', u'their', u'lives', u'.', u'hell', u',', u'i', u'don', u"'", u't', u'even', u'know', u'if', u'i', u'will', u'.', u'what', u'is', u'sure', u'to', u'be', u'a', u'critically', u'lambasted', u'film', u'turns', u'out', u'to', u'be', u'the', u'surprise', u'film', u'of', u'the', u'year', u'.', u'i', u'probably', u'won', u"'", u't', u'see', u'another', u'film', u'where', u'i', u'was', u'expecting', u'so', u'little', u'and', u'got', u'so', u'much', u'for', u'quite', u'a', u'while', u'.', u'director', u'peter', u'medak', u'has', u'crafted', u'a', u'very', u'suspenseful', u',', u'and', u'sometimes', u'very', u'scary', u'movie', u'out', u'of', u'a', u'mediocre', u'series', u'.', u'medak', u'has', u'also', u'mastered', u'the', u'wonderful', u'"', u'jump', u'!', u'"', u'moments', u',', u'and', u'has', u'probably', u'the', u'second', u'scariest', u'moment', u'i', u'have', u'ever', u'seen', u'on', u'film', u'(', u'scream', u'still', u'has', u'the', u'first', u')', u'.', u'strong', u'acting', u',', u'smart', u'dialogue', u',', u'intelligent', u'plotting', u',', u'and', u'a', u'sure', u'-', u'handed', u'director', u',', u'species', u'ii', u'is', u'exactly', u'what', u'these', u'films', u'should', u'be', u':', u'entertaining', u'.'], u'pos')
    [(u',', 77717), (u'the', 76529), (u'.', 65876), (u'a', 38106), (u'and', 35576), (u'of', 34123), (u'to', 31937), (u"'", 30585), (u'is', 25195), (u'in', 21822), (u's', 18513), (u'"', 17612), (u'it', 16107), (u'that', 15924), (u'-', 15595)]
    253
    

## 12. [Converting words to Features with NLTK](https://pythonprogramming.net/words-as-features-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15355355/)


```python
import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

word_features
```




    [u'sonja',
     u'askew',
     u'woods',
     u'spiders',
     u'bazooms',
     u'hanging',
     u'francesca',
     u'comically',
     u'localized',
     u'disobeying',
     u'hennings',
     u'canet',
     u'scold',
     u'originality',
     u'caned',
     u'rickman',
     u'slothful',
     u'wracked',
     u'stipulate',
     u'capoeira',
     u'rawhide',
     u'taj',
     u'bringing',
     u'unsworth',
     u'liaisons',
     u'grueling',
     u'sommerset',
     u'wooden',
     u'wednesday',
     u'broiled',
     u'circuitry',
     u'crotch',
     u'elgar',
     u'stereotypical',
     u'shows',
     u'gavan',
     u'rebuilding',
     u'snuggles',
     u'francesco',
     u'feasibility',
     u'miniatures',
     u'gorman',
     u'woody',
     u'consenting',
     u'scraped',
     u'inanimate',
     u'errors',
     u'reopens',
     u'cooking',
     u'fonzie',
     u'opportunists',
     u'islamic',
     u'joely',
     u'designing',
     u'numeral',
     u'succumb',
     u'shocks',
     u'chins',
     u'crooned',
     u'jubilantly',
     u'rocque',
     u'ching',
     u'china',
     u'shandling',
     u'confronts',
     u'wiseguy',
     u'natured',
     u'existentialist',
     u'kids',
     u'uplifting',
     u'controversy',
     u'crowdpleasing',
     u'neurologist',
     u'spotty',
     u'climber',
     u'appropriately',
     u'cobblers',
     u'projection',
     u'outraging',
     u'lengthen',
     u'emerich',
     u'unsinkable',
     u'stern',
     u'kethcum',
     u'dna',
     u'catchy',
     u'insecurity',
     u'cannibal',
     u'sidebars',
     u'music',
     u'therefore',
     u'mystic',
     u'mutinies',
     u'magyuver',
     u'deloreans',
     u'mesmerize',
     u'yahoo',
     u'faceted',
     u'exuberantly',
     u'======',
     u'eggar',
     u'boorman',
     u'foregrounds',
     u'primeval',
     u'voicework',
     u'circumstances',
     u'reingold',
     u'morally',
     u'locked',
     u'daqughter',
     u'locker',
     u'locket',
     u'soundbite',
     u'gershon',
     u'tomahawk',
     u'matilda',
     u'wang',
     u'wane',
     u'unjust',
     u'pooper',
     u'dishearteningly',
     u'want',
     u'pinto',
     u'absolute',
     u'vicent',
     u'beyer',
     u'travel',
     u'copious',
     u'playback',
     u'dangerfield',
     u'dared',
     u'prostitues',
     u'cadence',
     u'thivisol',
     u'sonorra',
     u'dinosaurs',
     u'wrong',
     u'cerebrally',
     u'sentencing',
     u'domed',
     u'colorfully',
     u'glenne',
     u'recombination',
     u'subplots',
     u'sickening',
     u'tulip',
     u'18th',
     u'perpetrator',
     u'crackin',
     u'nonsensical',
     u'romper',
     u'disengaging',
     u'snugly',
     u'kuei',
     u'welcomed',
     u'concurrence',
     u'stoicism',
     u'whizzing',
     u'dethroned',
     u'sidekicks',
     u'rewarded',
     u'welcomes',
     u'hypnotist',
     u'wickedly',
     u'fit',
     u'lifeline',
     u'screaming',
     u'fix',
     u'_i_know_what_you_did_last_summer_',
     u'fig',
     u'wales',
     u'bunker',
     u'fin',
     u'zucker',
     u'songwriter',
     u'municipality',
     u'recollections',
     u'guiler',
     u'effects',
     u'multidimensional',
     u'sixteen',
     u'undeveloped',
     u'saddened',
     u'whacking',
     u'bartok',
     u'barton',
     u'foregin',
     u'sugarplums',
     u'frewer',
     u'arrow',
     u'ingrid',
     u'_eve',
     u'windmill',
     u'telescope',
     u'allah',
     u'allan',
     u'parasites',
     u'touts',
     u'oprah',
     u'smirk',
     u'scrumptiously',
     u'indiscretion',
     u'nordoff',
     u'mason',
     u'encourage',
     u'adapt',
     u'zellwegger',
     u'outburst',
     u'abbott',
     u'stamping',
     u'abbots',
     u'anonymously',
     u'beristain',
     u'pumpkins',
     u'corrects',
     u'estimate',
     u'universally',
     u'chlorine',
     u'jugs',
     u'reiser',
     u'sickeningly',
     u'mackey',
     u'disturbed',
     u'competed',
     u'dentures',
     u'loudness',
     u'wiseguys',
     u'juergen',
     u'disfigured',
     u'stylistics',
     u'kfc',
     u'megabytes',
     u'sooty',
     u'davidovitch',
     u'olds',
     u'renovated',
     u'service',
     u'forrester',
     u'corsucant',
     u'reuben',
     u'needed',
     u'master',
     u'_2001_',
     u'critter',
     u'genesis',
     u'weendigo',
     u'caitlyn',
     u'rewards',
     u'enthrall',
     u'oingo',
     u'doreen',
     u'mutilated',
     u'lyndon',
     u'positively',
     u'ahmed',
     u'bannister',
     u'handcuffs',
     u'meditative',
     u'idly',
     u'idle',
     u'exclaimed',
     u'friend',
     u'feeling',
     u'longs',
     u'sustaining',
     u'spectrum',
     u'longo',
     u'coachmen',
     u'arousal',
     u'urinate',
     u'dozen',
     u'affairs',
     u'wholesome',
     u'courier',
     u'portillo',
     u'uncouth',
     u'racers',
     u'toothed',
     u'workmates',
     u'shipments',
     u'committing',
     u'limitless',
     u'diminishing',
     u'vexing',
     u'cinematic',
     u'resonates',
     u'disjointed',
     u'mouth',
     u'reverence',
     u'resonated',
     u'expound',
     u'singer',
     u'multiracial',
     u'tech',
     u'fugitives',
     u'keeble',
     u'rayden',
     u'scream',
     u'saying',
     u'teresa',
     u'loitered',
     u'padded',
     u'ulcer',
     u'tempted',
     u'cheaply',
     u'thai',
     u'hounded',
     u'orleans',
     u'clicked',
     u'rico',
     u'bliss',
     u'rick',
     u'rich',
     u'rice',
     u'rica',
     u'plate',
     u'remaning',
     u'videodrome',
     u'outfielders',
     u'plath',
     u'platt',
     u'clumsiness',
     u'altogether',
     u'chyron',
     u'droning',
     u'stoically',
     u'nicely',
     u'boarder',
     u'pretzel',
     u'patch',
     u'eyelids',
     u'dodie',
     u'boarded',
     u'jovivich',
     u'heirloom',
     u'clarified',
     u'sensitivity',
     u'pinon',
     u'slashfest',
     u'48th',
     u'playfulness',
     u'deadpan',
     u'irs',
     u'droves',
     u'bandaras',
     u'ira',
     u'ire',
     u'wage',
     u'dethrones',
     u'extend',
     u'nature',
     u'lapping',
     u'extent',
     u'reacquaints',
     u'tyranny',
     u'benigness',
     u'veer',
     u'voyeuristic',
     u'himalayas',
     u'incense',
     u'fruity',
     u'lookin',
     u'melinda',
     u'fearlessly',
     u'eradicate',
     u'zigged',
     u'rehash',
     u'mortified',
     u'maclaine',
     u'gopher',
     u'gypsies',
     u'fondled',
     u'charnel',
     u'affiliated',
     u'surname',
     u'blonde',
     u'underdone',
     u'milquetoast',
     u'marshmallows',
     u'union',
     u'fro',
     u'studious',
     u'.',
     u'muck',
     u'much',
     u'wyman',
     u'tonino',
     u'fry',
     u'toning',
     u'ocious',
     u'obese',
     u'premier',
     u'retrospect',
     u'spit',
     u'arkin',
     u'freehold',
     u'almasy',
     u'boardroom',
     u'dave',
     u'yugoslavians',
     u'doubts',
     u'spin',
     u'professionally',
     u'paraglider',
     u'employ',
     u'nfeatured',
     u'misconstrued',
     u'prostrate',
     u'k',
     u'canoeing',
     u'ditching',
     u'verges',
     u'lackies',
     u'mirabella',
     u'eighteen',
     u'haplessly',
     u'oxymoron',
     u'breakfast',
     u'hone',
     u'protovision',
     u'hong',
     u'emmylou',
     u'inventively',
     u'portobello',
     u'remand',
     u'mummified',
     u'honk',
     u'spews',
     u'split',
     u'codename',
     u'principals',
     u'cavanaugh',
     u'boiled',
     u'effortlessly',
     u'issac',
     u'frenchmen',
     u'vivien',
     u'torpedoes',
     u'marched',
     u'buliwyf',
     u'boiler',
     u'rulebook',
     u'featherweight',
     u'wcw',
     u'noblewoman',
     u'mentors',
     u'academic',
     u'stillness',
     u'academia',
     u'goofing',
     u'odile',
     u'waitering',
     u'corporate',
     u'massaging',
     u'falstaff',
     u'gigolo',
     u'belloq',
     u'absurdities',
     u'golden',
     u'bacri',
     u'_would_',
     u'homogeneity',
     u'snickered',
     u'boondocks',
     u'portrayed',
     u'lasso',
     u'hai',
     u'hal',
     u'ham',
     u'han',
     u'hab',
     u'espouses',
     u'had',
     u'insubordination',
     u'hag',
     u'hay',
     u'mcnamara',
     u'beloved',
     u'hap',
     u'har',
     u'has',
     u'hat',
     u'preciously',
     u'hav',
     u'haw',
     u'elders',
     u'survival',
     u'unequivocally',
     u'otherworldly',
     u'indicative',
     u'shadow',
     u'flotsam',
     u'ballhaus',
     u'sleuthing',
     u'delectably',
     u'alice',
     u'noteables',
     u'festivities',
     u'misdemeanors',
     u'unabashedly',
     u'attorney',
     u'crowd',
     u'crowe',
     u'czech',
     u'mosques',
     u'crown',
     u'topping',
     u'deflection',
     u'captive',
     u'beatng',
     u'billboard',
     u'namuth',
     u'pesimism',
     u'bottom',
     u'chabert',
     u'inhuman',
     u'plucked',
     u'crookier',
     u'monogamy',
     u'seagrave',
     u'subkoff',
     u'unequipped',
     u'rooker',
     u'barcode',
     u'eduard',
     u'starring',
     u'mediocrity',
     u'disdains',
     u'bamboo',
     u'stoker',
     u'caraciture',
     u'restlessness',
     u'benches',
     u'filmcritic',
     u'bicentennial',
     u'oneness',
     u'mussenden',
     u'stoked',
     u'whoaaaaaa',
     u'kilgore',
     u'dahlings',
     u'maxwell',
     u'marshall',
     u'honeymoon',
     u'mba',
     u'liebes',
     u'beings',
     u'marshals',
     u'hallucinogenic',
     u'shoots',
     u'aggressivelly',
     u'despised',
     u'fabric',
     u'_people_',
     u'suffice',
     u'raped',
     u'grasping',
     u'despises',
     u'greatness',
     u'rapes',
     u'grooms',
     u'spurting',
     u'ballisitic',
     u'congratulations',
     u'hypsy',
     u'humbled',
     u'mat',
     u'masquerading',
     u'wacked',
     u'smashes',
     u'1600s',
     u'humbler',
     u'complications',
     u'smashed',
     u'duet',
     u'dues',
     u'passenger',
     u'disgrace',
     u'barrymore',
     u'minah',
     u'unnerve',
     u'yankovich',
     u'decapitation',
     u'paperwork',
     u'triangles',
     u'slurring',
     u'spacemusic',
     u'biederman',
     u'dowling',
     u'cambodia',
     u'rioters',
     u'pasadena',
     u'role',
     u'obliges',
     u'rolf',
     u'wreaked',
     u'vegetative',
     u'wordlessly',
     u'roll',
     u'spielbergization',
     u'intend',
     u'palms',
     u'slaver',
     u'transported',
     u'palme',
     u'comely',
     u'intent',
     u'smelling',
     u'variable',
     u'batmans',
     u'hawkes',
     u'explosions',
     u'loren',
     u'meteorologist',
     u'shootout',
     u'innuendos',
     u'overturned',
     u'gown',
     u'childs',
     u'cincinnati',
     u'chain',
     u'whoever',
     u'diggler',
     u'bandits',
     u'chair',
     u'macht',
     u'ballet',
     u'malintentioned',
     u'grapples',
     u'pell',
     u'afi',
     u'freelance',
     u'crates',
     u'crater',
     u'silencers',
     u'underlining',
     u'overpopulated',
     u'obssessed',
     u'macho',
     u'oversight',
     u'tenacious',
     u'downloading',
     u'paychecks',
     u'jerk',
     u'tastefully',
     u'jere',
     u'prancer',
     u'prances',
     u'choice',
     u'metamorphoses',
     u'embark',
     u'gloomy',
     u'ghostbusters',
     u'stays',
     u'exact',
     u'minute',
     u'cooks',
     u'masturbates',
     u'minnie',
     u'skewed',
     u'skewer',
     u'xenophobe',
     u'dialogueless',
     u'trails',
     u'copyrighted',
     u'heavyweight',
     u'chopping',
     u'shirts',
     u'ogled',
     u'headset',
     u'lavishness',
     u'massironi',
     u'antwerp',
     u'celebrated',
     u'wayward',
     u'geography',
     u'celebrates',
     u'unintentionally',
     u'drafted',
     u'oldies',
     u'climbs',
     u'blunted',
     u'topicality',
     u'gladys',
     u'address',
     u'reclining',
     u'dwindling',
     u'benson',
     u'mafioso',
     u'plunges',
     u'accomplishes',
     u'dusty',
     u'impacted',
     u'cusack',
     u'accomplished',
     u'sprouted',
     u'expressively',
     u'enrols',
     u'influx',
     u'kasinsky',
     u'houseman',
     u'betraying',
     u'fakery',
     u'red',
     u'darnell',
     u'undergone',
     u'working',
     u'goregeous',
     u'oldham',
     u'opposed',
     u'portorican',
     u'familar',
     u'perishes',
     u'ooooooo',
     u'israel',
     u'assimilation',
     u'sierra',
     u'consoles',
     u'riders',
     u'rebounding',
     u'titanium',
     u'originally',
     u'abortion',
     u'americanised',
     u'harmonious',
     u'goody',
     u'following',
     u'zippers',
     u'admired',
     u'mirrors',
     u'stetson',
     u'parachute',
     u'locks',
     u'sextette',
     u'admires',
     u'admirer',
     u'listens',
     u'septic',
     u'vainly',
     u'thanking',
     u'edouard',
     u'maude',
     u'rewatched',
     u'mintues',
     u'casualness',
     u'mythos',
     u'convincingly',
     u'fueled',
     u'meddled',
     u'commensurately',
     u'brainless',
     u'egotistical',
     u'surfing',
     u'jonnie',
     u'conscious',
     u'regressive',
     u'nebbish',
     u'skirmish',
     u'wolves',
     u'pulled',
     u'manga',
     u'impactful',
     u'years',
     u'professors',
     u'structuring',
     u'episodes',
     u'kyzynski',
     u'professory',
     u'overlord',
     u'disconnect',
     u'slimeball',
     u'jia',
     u'milked',
     u'jim',
     u'troubles',
     u'rudnick',
     u'wahlberg',
     u'jip',
     u'suspension',
     u'troubled',
     u'modestly',
     u'recipients',
     u'civilian',
     u'courageously',
     u'indigenous',
     u'overpowering',
     u'drilling',
     u'workmanlike',
     u'henpecked',
     u'sorted',
     u'\\',
     u'materialized',
     u'didn',
     u'didi',
     u'dispite',
     u'fisherman',
     u'battleships',
     u'instability',
     u'quarter',
     u'greenwald',
     u'quartet',
     u'materializes',
     u'retrieve',
     u'bursting',
     u'receipt',
     u'remembrance',
     u'sponsor',
     u'entering',
     u'salads',
     u'disasters',
     u'bouyant',
     u'interned',
     u'yojimbo',
     u'disaster_',
     u'seriously',
     u'trauma',
     u'firorina',
     u'internet',
     u'merpeople',
     u'henreid',
     u'igniting',
     u'realisation',
     u'complicates',
     u'disintegrated',
     u'hairdresser',
     u'complicated',
     u'grandma',
     u'marla',
     u'modest',
     u'marlo',
     u'initiate',
     u'aboard',
     u'socking',
     u'neglect',
     u'emotion',
     u'gunshot',
     u'tingles',
     u'saving',
     u'symmetry',
     u'spoken',
     u'velda',
     u'savini',
     u'westlake',
     u'reprisal',
     u'one',
     u'ony',
     u'punishable',
     u'periodical',
     u'haviland',
     u'tamara',
     u'onw',
     u'plotless',
     u'exaggerations',
     u'stifler',
     u'stifles',
     u'formulates',
     u'limburgher',
     u'stifled',
     u'hurricaine',
     u'oversimplified',
     u'lingering',
     u'featherbrained',
     u'beesley',
     u'cherbourg',
     u'shawn',
     u'surges',
     u'snatch',
     u'devito',
     u'anthesis',
     u'absorbs',
     u'rza',
     u'hoyle',
     u'gisbourne',
     u'farsical',
     u'crossroads',
     u'admitedly',
     u'rehab',
     u'wandering',
     u'disasterous',
     u'dilemnas',
     u'bulow',
     u'illness',
     u'aaaaaaaahhhh',
     u'stylings',
     u'sumptuous',
     u'turned',
     u'locations',
     u'jewels',
     u'balsan',
     u'uninterrupted',
     u'turner',
     u'politicos',
     u'invite',
     u'pimply',
     u'zoe',
     u'cigarettes',
     u'warriors',
     u'zoo',
     u'goodman',
     u'portents',
     u'martineau',
     u'_titus_andronicus_',
     u'mayer',
     u'lick',
     u'pimple',
     u'murphy',
     u'opposite',
     u'discerning',
     u'spewing',
     u'buffet',
     u'printed',
     u'knowingly',
     u'buffed',
     u'huns',
     u'captivatingly',
     u'touchy',
     u'phil',
     u'toucha',
     u'jitters',
     u'messier',
     u'jittery',
     u'lydia',
     u'feuds',
     u'delroy',
     u'wynn',
     u'fakeouts',
     u'imagines',
     u'friction',
     u'fecal',
     u'oderkerk',
     u'inconsistent',
     u'soviets',
     u'imagined',
     u'wynt',
     u'seminal',
     u'zahn',
     u'reconciling',
     u'coaxing',
     u'remastered',
     u'guarded',
     u'rejoiced',
     u'suitcases',
     u'revolutionized',
     u'tilting',
     u'simplistic',
     u'awaiting',
     u'matsumoto',
     u'pimp',
     u'trys',
     u'carrion',
     u'spam',
     u'recoiling',
     u'choudhury',
     u'vision',
     u'morose',
     u'attenuated',
     u'underbids',
     u'audaciously',
     u'impressions',
     u'intoxicating',
     u'aboslutely',
     u'defensively',
     u'retells',
     u'masturbatory',
     u'alarming',
     u'sponsorship',
     u'moons',
     u'enjoys',
     u'playhouse',
     u'caan',
     u'tsui',
     u'lyricized',
     u'tamahori',
     u'braggarts',
     u'punts',
     u'awards',
     u'menacing',
     u'uncharacteristically',
     u'concentrated',
     u'busting',
     u'majestically',
     u'rhodes',
     u'matheson',
     u'millionaire',
     u'flipped',
     u'policed',
     u's',
     u'workplace',
     u'concentrates',
     u'flipper',
     u'doctoring',
     u'loveliest',
     u'beowolf',
     u'anette',
     u'comparitive',
     u'collides',
     u'west',
     u'deuteronomy',
     u'collided',
     u'motives',
     ...]




```python
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
```

    {u'clamoring': False, u'madsen': False, u'sonja': False, u'unsworth': False, u'woods': False, u'spiders': False, u'gavan': False, u'francesco': False, u'francesca': False, u'fedoore': False, u'comically': False, u'negg': False, u'localized': False, u'stinks': False, u'disobeying': False, u'hennings': False, u'porno': False, u'canet': False, u'giacomo': False, u'stinky': False, u'scold': False, u'originality': False, u'neighbours': False, u'caned': False, u'rickman': False, u'worth': False, u'porns': False, u'alternating': False, u'amorous': False, u'copasetic': False, u'slothful': False, u'wracked': False, u'dougnac': False, u'aurora': False, u'stipulate': False, u'kissed_': False, u'helgenberger': False, u'capoeira': False, u'rosalba': False, u'crackin': False, u'rawhide': False, u'summarized': False, u'waterlogged': False, u'screaming': False, u'yikes': False, u'recollections': False, u'liaisons': False, u'grueling': False, u'sommerset': False, u'investigator': False, u'wooden': False, u'wednesday': False, u'broiled': False, u'samurai': False, u'circuitry': False, u'notifying': False, u'crotch': False, u'elgar': False, u'errol': False, u'stereotypical': False, u'monologue': False, u'shows': True, u'roldan': False, u'jamaica': False, u'bazooms': False, u'sabbato': False, u'snuggles': False, u'hanging': False, u'pescara': False, u'feasibility': False, u'miniatures': False, u'nerdiest': False, u'mesmerising': False, u'gorman': False, u'woody': False, u'consenting': False, u'scraped': False, u'gazon': False, u'machines': False, u'inanimate': False, u'errors': False, u'euclidean': False, u'rekindle': False, u'offshoots': False, u'cooking': False, u'fonzie': False, u'opportunists': False, u'petri': False, u'videodrome': False, u'outfielders': False, u'numeral': False, u'succumb': False, u'shocks': False, u'personifies': False, u'viewings': False, u'chins': False, u'crooned': False, u'jubilantly': False, u'rocque': False, u'spunky': False, u'dilapidating': False, u'equals': False, u'metaphorically': False, u'boyum': False, u'ching': False, u'protection': False, u'china': False, u'personified': False, u'dobie': False, u'shandling': False, u'wiseguy': False, u'natured': False, u'watermelons': False, u'kids': True, u'uplifting': False, u'k': False, u'controversy': False, u'rebhorn': False, u'crowdpleasing': False, u'stressed': False, u'neurologist': False, u'bunker': False, u'spotty': False, u'climber': False, u'appropriately': False, u'cobblers': False, u'projection': False, u'urbaniak': False, u'outraging': False, u'brs': False, u'lengthen': False, u'emerich': False, u'flotsam': False, u'bro': False, u'lavatory': False, u'archaeological': False, u'unsinkable': False, u'stern': False, u'compulsively': False, u'namuth': False, u'kethcum': False, u'sarah': False, u'plot': True, u'plow': False, u'dna': False, u'plop': False, u'catchy': False, u'insecurity': False, u'sweater': False, u'coins': False, u'ploy': False, u'cannibal': False, u'sidebars': False, u'_people_': False, u'music': True, u'therefore': False, u'superweapons': False, u'mutinies': False, u'administering': False, u'magyuver': False, u'separated': False, u'deloreans': False, u'bombast': False, u'paperwork': False, u'kohn': False, u'mesmerize': False, u'ascribe': False, u'yahoo': False, u'exuberantly': False, u'======': False, u'championships': False, u'boorman': False, u'diggler': False, u'provide': False, u'foregrounds': False, u'primeval': False, u'conditioned': False, u'voicework': False, u'blocking': False, u'circumstances': False, u'reingold': False, u'tastefully': False, u'1993': False, u'morally': False, u'locked': False, u'1994': False, u'daqughter': False, u'1996': False, u'1999': False, u'1998': False, u'overpowers': False, u'cuddly': False, u'divorces': False, u'locker': False, u'tissue': False, u'locket': False, u'era': False, u'soundbite': False, u'gershon': False, u'elbow': False, u'erm': False, u'ern': False, u'plunges': False, u'recipes': False, u'scripting': False, u'matilda': False, u'wang': False, u'indicated': False, u'wane': False, u'portorican': False, u'flung': False, u'winnfield': False, u'heartless': False, u'strangelove': False, u'titanium': False, u'dishearteningly': False, u'want': True, u'repressiveness': False, u'pinto': False, u'absolute': False, u'impassive': False, u'skyler': False, u'vicent': False, u'beyer': False, u'travel': False, u'nuts': False, u'copious': False, u'kyzynski': False, u'recovers': False, u'playback': False, u'dangerfield': False, u'conn': False, u'moron': False, u'titillate': False, u'prostitues': False, u'truthful': False, u'cadence': False, u'thivisol': False, u'sonorra': False, u'sordidness': False, u'henreid': False, u'memorial': False, u'customs': False, u'millimeter': False, u'dinosaurs': False, u'wrong': False, u'aurore': False, u'ladden': False, u'cerebrally': False, u'sentencing': False, u'dumbo': False, u'greediness': False, u'arch': False, u'foundering': False, u'hurricaine': False, u'complacent': False, u'colorfully': False, u'elusive': False, u'glenne': False, u'alienate': False, u'recombination': False, u'schneider': False, u'appreciate': False, u'americanization': False, u'subplots': False, u'purging': False, u'sickening': False, u'tulip': False, u'18th': False, u'davies': False, u'moroder': False, u'nonsensical': False, u'romper': False, u'disengaging': False, u'droagon': False, u'_american_psycho_': False, u'leeanne': False, u'snugly': False, u'gumption': False, u'kuei': False, u'scalding': False, u'welcomed': False, u'matewan': False, u'concurrence': False, u'stoicism': False, u'whizzing': False, u'wachowskis': False, u'matsumoto': False, u'sidekicks': False, u'vito': False, u'innovative': False, u'rewarded': False, u'welcomes': False, u'understates': False, u'wickedly': False, u'fit': False, u'lifeline': False, u'bringing': False, u'fix': False, u'christie': False, u'_i_know_what_you_did_last_summer_': False, u'trager': False, u'production': True, u'understated': False, u'fig': False, u'valor': False, u'wales': False, u'fin': False, u'travellers': False, u'vamires': False, u'soundbites': False, u'songwriter': False, u'municipality': False, u'locusts': False, u'safe': False, u'inhereit': False, u'collide': False, u'laconically': False, u'expressionist': False, u'roommate': False, u'guiler': False, u'effects': False, u'expressionism': False, u'cohagen': False, u'multidimensional': False, u'sixteen': False, u'undeveloped': False, u'saddened': False, u'aneeka': False, u'petrovsky': False, u'defeaningly': False, u'progression': False, u'dingo': False, u'whacking': False, u'bartok': False, u'reasonably': False, u'routines': False, u'barton': False, u'l': False, u'foregin': False, u'dingy': False, u'sugarplums': False, u'nighthawks': False, u'frewer': False, u'averse': False, u'ingrid': False, u'fearsomely': False, u'feeds': False, u'_eve': False, u'overpopulated': False, u'telescope': False, u'dumping': False, u'masseuse': False, u'allah': False, u'allan': False, u'sled': False, u'parasites': False, u'maniacs': False, u'tragicomic': False, u'slew': False, u'roadblock': False, u'inadvertently': False, u'touts': False, u'oprah': False, u'smirk': False, u'scrumptiously': False, u'indiscretion': False, u'maduro': False, u'nordoff': False, u'danforth': False, u'mason': False, u'encourage': False, u'daniel': False, u'adapt': False, u'uuuhhmmm': False, u'confections': False, u'zellwegger': False, u'judith': False, u'outburst': False, u'mullen': False, u'abbott': False, u'stamping': False, u'meatier': False, u'abbots': False, u'barrier': False, u'colorless': False, u'leftovers': False, u'beristain': False, u'pumpkins': False, u'corrects': False, u'forcibly': False, u'crowned': False, u'estimate': False, u'universally': False, u'chlorine': False, u'renee': False, u'chortled': False, u'dammit': False, u'birdie': False, u'r2': False, u'sickeningly': False, u'refugee': False, u'flawless': False, u'renew': False, u'disturbed': False, u'competed': False, u'dentures': False, u'loudness': False, u'footsteps': False, u'juergen': False, u'haunted': False, u'render': False, u'elmo': False, u'disfigured': False, u'railroads': False, u'immolation': False, u'procreate': False, u'dog_': False, u'stylistics': False, u'kfc': False, u'megabytes': False, u'antarctic': False, u'electronic': False, u'mojo': False, u'sooty': False, u'timothy': False, u'olds': False, u'renovated': False, u'service': False, u'forrester': False, u'2058': False, u'corsucant': False, u'ingen': False, u'reuben': False, u'approximately': False, u'needed': False, u'blurts': False, u'master': False, u'_2001_': False, u'cassavetes': False, u'mistic': False, u'critter': False, u'john': False, u'genesis': False, u'weendigo': False, u'caitlyn': False, u'rewards': False, u'paltry': False, u'enthrall': False, u'offhand': False, u'oingo': False, u'lunges': False, u'waster': False, u'wastes': False, u'scorpions': False, u'mutilated': False, u'enmeshed': False, u'cereal': False, u'wasted': False, u'downtime': False, u'anachronisms': False, u'positively': False, u'ahmed': False, u'reclining': False, u'guile': False, u'bannister': False, u'roelfs': False, u'handcuffs': False, u'idly': False, u'project': False, u'idle': False, u'exclaimed': False, u'guilt': False, u'friend': False, u'historical': False, u'apparantly': False, u'feeling': True, u'seminal': False, u'humble': False, u'rouen': False, u'tautness': False, u'longs': False, u'portraits': False, u'sustaining': False, u'flavorful': False, u'spectrum': False, u'enchanted': False, u'longo': False, u'refreshingly': False, u'arousal': False, u'tenuous': False, u'urinate': False, u'contents': False, u'dozen': False, u'affairs': False, u'wholesome': False, u'courier': False, u'scumbagginess': False, u'cronenberg': False, u'convenient': False, u'uncouth': False, u'gripe': False, u'saunder': False, u'racers': False, u'toothed': False, u'subjects': False, u'nuez': False, u'thundering': False, u'pilgrimage': False, u'workmates': False, u'enervation': False, u'shipments': False, u'gravy': False, u'germann': False, u'committing': False, u'bruce': False, u'limitless': False, u'diminishing': False, u'vexing': False, u'cinematic': False, u'resonates': False, u'ramblings': False, u'disjointed': False, u'stardom': False, u'mouth': False, u'culminates': False, u'reverence': False, u'scripts': False, u'resonated': False, u'parting': False, u'expound': False, u'singer': False, u'macfadyen': False, u'bracken': False, u'dubarry': False, u'musical': False, u'multiracial': False, u'swamp': False, u'bracket': False, u'tech': False, u'fugitives': False, u'keeble': False, u'rayden': False, u'rhythmless': False, u'brinkford': False, u'germany': False, u'scream': False, u'crowbar': False, u'saying': False, u'rockies': False, u'overdoes': False, u'stephens': False, u'lewis': False, u'teresa': False, u'loitered': False, u'rosselini': False, u'padded': False, u'bellow': False, u'disturbs': False, u'ulcer': False, u'alferd': False, u'tempted': False, u'cheaply': False, u'councilmembers': False, u'hounded': False, u'capitol': False, u'orleans': False, u'geyser': False, u'clicked': False, u'quaint': False, u'nullifies': False, u'grosbard': False, u'rico': False, u'bliss': False, u'rick': False, u'rich': False, u'rice': False, u'nullified': False, u'rectangle': False, u'rica': False, u'plate': False, u'cappuccino': False, u'joely': False, u'tramell': False, u'belaboured': False, u'chaney': False, u'uncountable': False, u'designing': False, u'plath': False, u'psychiatric': False, u'wisecracks': False, u'platt': False, u'clumsiness': False, u'roundabout': False, u'altogether': False, u'chyron': False, u'vividly': False, u'beleive': False, u'droning': False, u'vicariously': False, u'runs': False, u'stoically': False, u'plotholia': False, u'spilling': False, u'nicely': False, u'boarder': False, u'pretzel': False, u'patch': False, u'eyelids': False, u'rahad': False, u'rune': False, u'gears': False, u'rung': False, u'krupa': False, u'boarded': False, u'scrubbed': False, u'secretary': False, u'jovivich': False, u'heirloom': False, u'clarified': False, u'claymation': False, u'sensitivity': False, u'pinon': False, u'slashfest': False, u'horrendous': False, u'discussions': False, u'optimum': False, u'hairbrush': False, u'techniques': False, u'pastel': False, u'48th': False, u'playfulness': False, u'pressured': False, u'deadpan': False, u'pasted': False, u'away': True, u'irs': False, u'droves': False, u'bandaras': False, u'collette': False, u'bracing': False, u'arcane': False, u'arcand': False, u'meditative': False, u'ira': False, u'drawl': False, u'encounters': False, u'ire': False, u'huison': False, u'extend': False, u'nature': False, u'handful': False, u'lapping': False, u'transylvanians': False, u'diesl': False, u'taraji': False, u'gays': False, u'succumbs': False, u'extent': False, u'beelzebub': False, u'reacquaints': False, u'kitchen': False, u'tyranny': False, u'climate': False, u'benigness': False, u'psychologists': False, u'dorff': False, u'veer': False, u'disdain': False, u'voyeuristic': False, u'himalayas': False, u'compton': False, u'askew': False, u'hollywoodization': False, u'lookin': False, u'disappears': False, u'fearlessly': False, u'eradicate': False, u'zigged': False, u'rehash': False, u'mortified': False, u'tone': False, u'maclaine': False, u'murtaugh': False, u'upbeats': False, u'sobchak': False, u'gopher': False, u'gypsies': False, u'fondled': False, u'charnel': False, u'lick': False, u'affiliated': False, u'tony': False, u'surname': False, u'blonde': False, u'telecommunications': False, u'priscilla': False, u'underdone': False, u'milquetoast': False, u'wright': False, u'union': False, u'fro': False, u'resemblances': False, u'.': True, u'muck': False, u'polemic': False, u'much': False, u'wyman': False, u'noel': False, u'tonino': False, u'kanoby': False, u'fry': False, u'toning': False, u'ocious': False, u'obese': False, u'superpowers': False, u'retrospect': False, u'spit': False, u'attacked': False, u'arkin': False, u'excite': False, u'freehold': False, u'almasy': False, u'psychically': False, u'comprehensible': False, u'dave': False, u'yugoslavians': False, u'doubts': False, u'clairvoyant': False, u'spin': False, u'takaaki': False, u'diverted': False, u'righteous': False, u'espoused': False, u'professionally': False, u'paraglider': False, u'employ': False, u'nfeatured': False, u'misconstrued': False, u'thrash': False, u'prostrate': False, u'35th': False, u'characterizing': False, u'blackly': False, u'cont': False, u'krays': False, u'canoeing': False, u'beetles': False, u'ditching': False, u'verges': False, u'lackies': False, u'separatist': False, u'tylenol': False, u'mirabella': False, u'eighteen': False, u'cong': False, u'haplessly': False, u'voges': False, u'oxymoron': False, u'turner': False, u'sever': False, u'hone': False, u'protovision': False, u'hong': False, u'inventively': False, u'portobello': False, u'remand': False, u'mummified': False, u'amount': False, u'honk': False, u'writerly': False, u'spews': False, u'alevey': False, u'split': False, u'synch': False, u'mindfuck': False, u'codename': False, u'principals': False, u'cavanaugh': False, u'wheel': False, u'boiled': False, u'effortlessly': False, u'fuss': False, u'issac': False, u'frenchmen': False, u'hana': False, u'vivien': False, u'torpedoes': False, u'lyndon': False, u'bening': False, u'liberties': False, u'marched': False, u'buliwyf': False, u'boiler': False, u'dashing': False, u'rulebook': False, u'hans': False, u'selfless': False, u'brainers': False, u'featherweight': False, u'postponed': False, u'whereby': False, u'noblewoman': False, u'1600': False, u'fashionable': False, u'mentors': False, u'academic': False, u'stillness': False, u'academia': False, u'goofing': False, u'humbly': False, u'sullenly': False, u'waitering': False, u'corporate': False, u'massaging': False, u'pronouncements': False, u'gigolo': False, u'solaris': False, u'belloq': False, u'absurdities': False, u'golden': False, u'vying': False, u'newton': False, u'_would_': False, u'homogeneity': False, u'snickered': False, u'sabbatical': False, u'ol': False, u'portrayed': False, u'electronically': False, u'lasso': False, u'hai': False, u'denton': False, u'designers': False, u'hal': False, u'ham': False, u'han': False, u'cornell': False, u'similarities': False, u'hab': False, u'espouses': False, u'had': False, u'insubordination': False, u'hag': False, u'jost': False, u'hay': False, u'mcnamara': False, u'cognac': False, u'beloved': False, u'joss': False, u'hap': False, u'har': False, u'has': True, u'hat': False, u'preciously': False, u'hav': False, u'haw': False, u'packin': False, u'insensitive': False, u'elders': False, u'survival': False, u'tricking': False, u'inflicting': False, u'unequivocally': False, u'otherworldly': False, u'indicative': False, u'everton': False, u'shadow': False, u'vapors': False, u'unfounded': False, u'ballhaus': False, u'hairless': False, u'sleuthing': False, u'eroded': False, u'arcs': False, u'deviance': False, u'cooler': False, u'huns': False, u'alice': False, u'noteables': False, u'festivities': False, u'sorvino': False, u'homing': False, u'night': False, u'revisiting': False, u'grotesquely': False, u'cooled': False, u'misdemeanors': False, u'unabashedly': False, u'attorney': False, u'dimitri': False, u'crowd': False, u'crowe': False, u'czech': False, u'flatter': False, u'mosques': False, u'crown': False, u'hypsy': False, u'deflection': False, u'changwei': False, u'captive': False, u'couture': False, u'stardust': False, u'flatten': False, u'kieslowski': False, u'billboard': False, u'bore': False, u'confusing': True, u'adorably': False, u'congratulate': False, u'born': False, u'wiseacre': False, u'bottom': True, u'chabert': False, u'inhuman': False, u'plucked': False, u'asking': False, u'absolution': False, u'lahore': False, u'melange': False, u'monogamy': False, u'seagrave': False, u'participation': False, u'subkoff': False, u'unequipped': False, u'peek': False, u'rooker': False, u'peel': False, u'sadie': False, u'elucidate': False, u'barcode': False, u'shogun': False, u'eduard': False, u'starring': False, u'tribes': False, u'peer': False, u'guild': False, u'peep': False, u'disdains': False, u'explainable': False, u'peet': False, u'menage': False, u'stoker': False, u'deathless': False, u'ferraris': False, u'caraciture': False, u'restlessness': False, u'benches': False, u'_____': False, u'filmcritic': False, u'bicentennial': False, u'oneness': False, u'mussenden': False, u'janitorial': False, u'hoenicker': False, u'stoked': False, u'whoaaaaaa': False, u'kilgore': False, u'gads': False, u'dahlings': False, u'jacques': False, u'guiness': False, u'8034': False, u'wasting': False, u'maxwell': False, u'marshall': False, u'honeymoon': False, u'profession': False, u'mba': False, u'liebes': False, u'rendering': False, u'beings': False, u'marshals': False, u'hallucinogenic': False, u'shoots': False, u'aggressivelly': False, u'stumble': False, u'familiarize': False, u'despised': False, u'deception': False, u'fabric': False, u'plod': False, u'suffice': False, u'unfocused': False, u'raped': False, u'grasping': False, u'despises': False, u'obserable': False, u'greatness': False, u'rapes': False, u'exacty': False, u'grooms': False, u'spurting': False, u'overjoyed': False, u'ballisitic': False, u'needles': False, u'catalyst': False, u'congratulations': False, u'humbled': False, u'masquerading': False, u'smashes': False, u'1600s': False, u'humbler': False, u'complications': False, u'exacts': False, u'smashed': False, u'verging': False, u'duet': False, u'dues': False, u'passenger': False, u'ayla': False, u'disgrace': False, u'barrymore': False, u'minah': False, u'unnerve': False, u'yankovich': False, u'borg': False, u'decapitation': False, u'paglia': False, u'triangles': False, u'slurring': False, u'spacemusic': False, u'biederman': False, u'dowling': False, u'cambodia': False, u'fuse': False, u'pasadena': False, u'role': False, u'telefixated': False, u'rolf': False, u'vegetative': False, u'wordlessly': False, u'roll': False, u'intend': False, u'palms': False, u'transported': False, u'palme': False, u'connote': False, u'comely': False, u'intent': False, u'smelling': False, u'variable': False, u'batmans': False, u'bacri': False, u'hawkes': False, u'explosions': False, u'loren': False, u'meteorologist': False, u'shootout': False, u'overturned': False, u'gown': False, u'hyperdrive': False, u'childs': False, u'cincinnati': False, u'chain': False, u'whoever': False, u'separates': False, u'bandits': False, u'unraveled': False, u'chair': False, u'macht': False, u'ballet': False, u'malintentioned': False, u'grapples': False, u'graph': False, u'freelance': False, u'crates': False, u'crater': False, u'silencers': False, u'obssessed': False, u'macho': False, u'oversight': False, u'tenacious': False, u'1991': False, u'downloading': False, u'paychecks': False, u'jerk': False, u'tnt': False, u'jere': False, u'prancer': False, u'prances': False, u'choice': False, u'aissa': False, u'embark': False, u'gloomy': False, u'ghostbusters': False, u'stays': False, u'1995': False, u'exact': True, u'minute': False, u'cooks': False, u'scholl': False, u'1997': False, u'adorable': False, u'masturbates': False, u'minnie': False, u'skewed': False, u'mathew': False, u'skewer': False, u'matthau': False, u'xenophobe': False, u'trails': False, u'heavyweight': False, u'chopping': False, u'shirts': False, u'ogled': False, u'biopic': False, u'headset': False, u'lavishness': False, u'massironi': False, u'antwerp': False, u'celebrated': False, u'karras': False, u'tizard': False, u'celebrates': False, u'unintentionally': False, u'drafted': False, u'erb': False, u'oldies': False, u'climbs': False, u'blunted': False, u'topicality': False, u'gladys': False, u'address': False, u'dwindling': False, u'benson': False, u'mafioso': False, u'accomplishes': False, u'dusty': False, u'impacted': False, u'cusack': False, u'accomplished': False, u'sprouted': False, u'expressively': False, u'enrols': False, u'influx': False, u'kasinsky': False, u'architectural': False, u'houseman': False, u'substitution': False, u'betraying': False, u'jose': False, u'pees': False, u'fakery': False, u'hallie': False, u'darnell': False, u'undergone': False, u'working': False, u'oldham': False, u'quivering': False, u'opposed': False, u'unjust': False, u'familar': False, u'perishes': False, u'ooooooo': False, u'assimilation': False, u'consoles': False, u'riders': False, u'rebounding': False, u'pooper': False, u'originally': False, u'abortion': False, u'americanised': False, u'harmonious': False, u'remaning': False, u'following': False, u'zippers': False, u'admired': False, u'mirrors': False, u'stetson': False, u'parachute': False, u'locks': False, u'sextette': False, u'admires': False, u'admirer': False, u'succumbing': False, u'listens': False, u'gentler': False, u'septic': False, u'vainly': False, u'thanking': False, u'edouard': False, u'rewatched': False, u'mintues': False, u'mat': False, u'pesimism': False, u'casualness': False, u'mythos': False, u'convincingly': False, u'islamic': False, u'meddled': False, u'horks': False, u'brainless': False, u'egotistical': False, u'surfing': False, u'jonnie': False, u'conscious': False, u'regressive': False, u'nebbish': False, u'skirmish': False, u'wolves': False, u'pulled': False, u'manga': False, u'impactful': False, u'years': True, u'professors': False, u'structuring': False, u'episodes': False, u'marshmallows': False, u'professory': False, u'overlord': False, u'disconnect': False, u'slimeball': False, u'jia': False, u'milked': False, u'jim': False, u'troubles': False, u'pulitzer': False, u'rudnick': False, u'roadster': False, u'jip': False, u'suspension': False, u'troubled': False, u'modestly': False, u'sparing': False, u'recipients': False, u'civilian': False, u'courageously': False, u'indigenous': False, u'overpowering': False, u'drilling': False, u'workmanlike': False, u'henpecked': False, u'sorted': False, u'\\': False, u'josh': False, u'materialized': False, u'didn': True, u'didi': False, u'dispite': False, u'fisherman': False, u'battleships': False, u'instability': False, u'quarter': False, u'quartet': False, u'materializes': False, u'retrieve': False, u'policed': False, u'bursting': False, u'receipt': False, u'sponsor': False, u'entering': False, u'salads': False, u'disasters': False, u'bouyant': False, u'rioters': False, u'interned': False, u'yojimbo': False, u'1992': False, u'wiseguys': False, u'disaster_': False, u'seriously': False, u'trauma': False, u'firorina': False, u'internet': False, u'merpeople': False, u'ladder': False, u'igniting': False, u'rebuilding': False, u'complicates': False, u'disintegrated': False, u'hairdresser': False, u'sympathize': False, u'existentialist': False, u'complicated': False, u'mcferran': False, u'grandma': False, u'marla': False, u'eggar': False, u'bared': False, u'tomahawk': False, u'tasting': False, u'modest': False, u'marlo': False, u'initiate': False, u'aboard': False, u'socking': False, u'domed': False, u'neglect': False, u'emotion': False, u'gunshot': False, u'saving': False, u'symmetry': False, u'spoken': False, u'velda': False, u'savini': False, u'westlake': False, u'reprisal': False, u'one': True, u'respecting': False, u'ony': False, u'punishable': False, u'periodical': False, u'haviland': False, u'tamara': False, u'onw': False, u'plotless': False, u'exaggerations': False, u'stifler': False, u'stifles': False, u'jugs': False, u'tenko': False, u'davidovitch': False, u'stifled': False, u'backwoods': False, u'oversimplified': False, u'lingering': False, u'featherbrained': False, u'beesley': False, u'cherbourg': False, u'shawn': False, u'surges': False, u'obtained': False, u'snatch': False, u'devito': False, u'anthesis': False, u'absorbs': False, u'thai': False, u'padre': False, u'rza': False, u'hoyle': False, u'gisbourne': False, u'crossroads': False, u'admitedly': False, u'rehab': False, u'wandering': False, u'fruity': False, u'disasterous': False, u'dilemnas': False, u'bulow': False, u'illness': False, u'aaaaaaaahhhh': False, u'stylings': False, u'sumptuous': False, u'premier': False, u'turned': False, u'locations': False, u'jewels': False, u'balsan': False, u'odile': False, u'uninterrupted': False, u'infomercial': False, u'breakfast': False, u'emmylou': False, u'politicos': False, u'jacks': False, u'pimply': False, u'zoe': False, u'wcw': False, u'cigarettes': False, u'warriors': False, u'reasonable': False, u'zoo': False, u'goodman': False, u'portents': False, u'martineau': False, u'_titus_andronicus_': False, u'mayer': False, u'pimple': False, u'topping': False, u'opposite': False, u'discerning': False, u'spewing': False, u'buffet': False, u'printed': False, u'knowingly': False, u'buffed': False, u'captivatingly': False, u'wacked': False, u'touchy': False, u'phil': False, u'toucha': False, u'jitters': False, u'messier': False, u'wreaked': False, u'slaver': False, u'jittery': False, u'atlantic': False, u'delroy': False, u'wynn': False, u'fakeouts': False, u'imagines': False, u'friction': False, u'fecal': False, u'oderkerk': False, u'inconsistent': False, u'copyrighted': False, u'soviets': False, u'imagined': False, u'wynt': False, u'geography': False, u'zahn': False, u'reconciling': False, u'coaxing': False, u'goregeous': False, u'sierra': False, u'fades': False, u'guarded': False, u'rejoiced': False, u'suitcases': False, u'revolutionized': False, u'tilting': False, u'undetected': False, u'simplistic': False, u'awaiting': False, u'miming': False, u'wahlberg': False, u'pimp': False, u'trys': False, u'carrion': False, u'ambling': False, u'recoiling': False, u'choudhury': False, u'vision': False, u'morose': False, u'attenuated': False, u'underbids': False, u'audaciously': False, u'impressions': False, u'intoxicating': False, u'aboslutely': False, u'defensively': False, u'retells': False, u'masturbatory': False, u'alarming': False, u'feuds': False, u'sponsorship': False, u'moons': False, u'nicest': False, u'enjoys': False, u'playhouse': False, u'caan': False, u'tsui': False, u'lyricized': False, u'faded': False, u'braggarts': False, u'punts': False, u'awards': False, u'menacing': False, u'innuendos': False, u'smoggy': False, u'uncharacteristically': False, u'concentrated': False, u'busting': False, u'confection': False, u'majestically': False, u'rhodes': False, u'matheson': False, u'millionaire': False, u'flipped': False, u's': True, u'workplace': False, u'concentrates': False, u'flipper': False, u'doctoring': False, u'loveliest': False, u'beowolf': False, u'adroit': False, u'mugshots': False, u'imbues': False, u'comparitive': False, u'collides': False, u'west': False, u'deuteronomy': False, u'collided': False, u'brancia': False, u'motives': False, u'sked': False, u'nntphub': False, u'spyglass': False, u'wants': False, u'vomits': False, u'tomei': False, u'formed': False, u'photon': False, u'readings': False, u'photos': False, u'tightened': False, u'abject': False, u'former': False, u'sedition': False, u'sommers': False, u'chauvinistic': False, u'defeatist': False, u'straighten': False, u'squeezes': False, u'shockwave': False, u'diverse': False, u'newspaper': False, u'situation': False, u'slapping': False, u'landslide': False, u'penthouse': False, u'unlikeable': False, u'rapier': False, u'surveying': False, u'engaged': False, u'zucker': False, u'dubious': False, u'_still_': False, u'menancing': False, u'twotg': False, u'engages': False, u'multitudes': False, u'debilitating': False, u'ingrained': False, u'nuptials': False, u'fistfights': False, u'quiclky': False, u'otto': False, u'jessalyn': False, u'bogglingly': False, u'visually': False, u'wires': False, u'edged': False, u'assigns': False, u'hideaway': False, u'sickness': False, u'krippendorf': False, u'defy': False, u'brassed': False, u'deflate': False, u'tolan': False, u'edges': False, u'amuck': False, u'advertisement': False, u'ratttz': False, u'_seven_nights_': False, u'tracking': False, u'droppingly': False, u'charges': False, u'nothin': False, u'peculiarities': False, u'delectably': False, u'penetration': False, u'dimension': False, u'persistently': False, u'recycles': False, u'being': False, u'bueller': False, u'recycled': False, u'dick_': False, u'parlay': False, u'lonesome': False, u'procreating': False, u'rover': False, u'grounded': False, u'cloris': False, u'lifelong': False, u'gloating': False, u'overthrow': False, u'haystack': False, u'dicks': False, u'absense': False, u'phelps': False, u'sportsmanship': False, u'rejoin': False, u'sums': False, u'unveil': False, u'sumo': False, u'traffic': False, u'preference': False, u'politely': False, u'world': True, u'embrassment': False, u'postal': False, u'reap': False, u'likeablity': False, u'sensational': False, u'malfunctions': False, u'unrepentant': False, u'benefit': False, u'superiority': False, u'glamor': False, u'dirtier': False, u'petrice': False, u'confrontatory': False, u'satisfactory': False, u'superintendent': False, u'affay': False, u'dilbert': False, u'tvs': False, u'magma': False, u'demeaning': False, u'diving': False, u'stagecoach': False, u'divine': False, u'bongos': False, u'dancefloor': False, u'painstakingly': False, u'bottlecaps': False, u'cavity': False, u'seaman': False, u'squirt': False, u'francois': False, u'911': False, u'restoring': False, u'process': False, u'squabble': False, u'arrow': True, u'macgowan': False, u'retains': False, u'cliquey': False, u'tv2': False, u'leadership': False, u'piscapo': False, u'thailand': False, u'demarco': False, u'exasperating': False, u'loyalties': False, u'hopkins': False, u'majorino': False, u'ob': False, u'stanleyville': False, u'wrestled': False, u'frights': False, u'niall': False, u'chow': False, u'internalize': False, u'johnston': False, u'locklear': False, u'sensitively': False, u'rabbinical': False, u'perturbed': False, u'shapely': False, u'burial': False, u'antidote': False, u'kroon': False, u'ineffable': False, u'lively': False, u'bukater': False, u'pivot': False, u'conceptually': False, u'rossi': False, u'uhhhhhm': False, u'complexly': False, u'distractedness': False, u'vaporize': False, u'hunks': False, u'gleam': False, u'glean': False, u'lounging': False, u'redirection': False, u'mindless': False, u'missy': False, u'sealed': False, u'brazilian': False, u'bubble': False, u'witt': False, u'continents': False, u'wits': False, u'bohemians': False, u'lane': False, u'societal': False, u'foreheads': False, u'attainable': False, u'with': True, u'abused': False, u'pull': False, u'rush': False, u'thumps': False, u'dominican': False, u'rage': False, u'tripe': False, u'claustral': False, u'chomped': False, u'rags': False, u'dirty': False, u'abuser': False, u'abuses': False, u'russ': False, u'trips': False, u'touchstone': False, u'patois': False, u'falseness': False, u'wormwood': False, u'gratuitous': False, u'watches': False, u'watcher': False, u'associating': False, u'toontown': False, u'watched': False, u'jargon': False, u'tremble': False, u'dampens': False, u'cream': False, u'moniker': False, u'ideally': False, u'administered': False, u'yogi': False, u'sympathetically': False, u'unwelcomed': False, u'introspection': False, u'hofstra': False, u'unparalleled': False, u'friggin': False, u'puppy': False, u'addictions': False, u'artillary': False, u'waving': False, u'falstaff': False, u'midget': False, u'brotherhood': False, u'whippersnappers': False, u'torches': False, u'linebackers': False, u'fedoras': False, u'tricky': False, u'mourned': False, u'natalie': False, u'thora': False, u'tricks': False, u'maliciously': False, u'dyed': False, u'uploaded': False, u'finklestein': False, u'dyer': False, u'humoring': False, u'sci': False, u'anette': False, u'lopped': False, u'caused': False, u'beware': False, u'slimming': False, u'zappa': False, u'upholds': False, u'causes': False, u'riots': False, u'nora': False, u'conciousness': False, u'carters': False, u'norm': False, u'gazarra': False, u'clubbed': False, u'powaqqatsi': False, u'floated': False, u'clubber': False, u'24th': False, u'suspensefully': False, u'sant': False, u'rootless': False, u'sans': False, u'boozed': False, u'shenanigans': False, u'sang': False, u'sand': False, u'sane': False, u'unwraps': False, u'small': False, u'sank': False, u'vanquish': False, u'courtesans': False, u'abbreviated': False, u'quicker': False, u'traditions': False, u'tardis': False, u'healed': False, u'past': False, u'displays': False, u'pass': False, u'healer': False, u'investment': False, u'amarcord': False, u'clock': False, u'skywalker': False, u'leit': False, u'colonists': False, u'leia': False, u'psychoanalysts': False, u'dwells': False, u'hasn': False, u'full': False, u'hash': False, u'diapers': False, u'portrays': False, u'civilians': False, u'november': False, u'hass': False, u'melancholic': False, u'contrastingly': False, u'ivey': False, u'experience': False, u'anthropologists': False, u'prior': False, u'beaman': False, u'periodic': False, u'holdover': False, u'cessation': False, u'divison': False, u'skepticism': False, u'hime': False, u'amadeus': False, u'uniquely': False, u'interactivity': False, u'norville': False, u'followed': False, u'retroactive': False, u'mediator': False, u'scharzenegger': False, u'traumatized': False, u'follower': False, u'analyzing': False, u'traumatizes': False, u'cynics': False, u'enlightened': False, u'automats': False, u'volcanos': False, u'silva': False, u'attendance': False, u'enliven': False, u'canoe': False, u'briesewitz': False, u'lollipop': False, u'mori': False, u'unrecognizable': False, u'certified': False, u'restraints': False, u'firth': False, u'mora': False, u'glowers': False, u'more': True, u'israel': False, u'barbarino': False, u'door': False, u'doos': False, u'initiated': False, u'chucky': False, u'company': False, u'corrected': False, u'tested': False, u'lameness': False, u'fumble': False, u'doom': False, u'negativity': False, u'leary': False, u'kaminski': False, u'fornicators': False, u'maniac': False, u'patriarch': False, u'kaminsky': False, u'learn': False, u'knocked': False, u'grope': False, u'scramble': False, u'barclay': False, u'allegra': False, u'bogs': False, u'memoirs': False, u'meaner': False, u'bogg': False, u'aatish': False, u'prostration': False, u'sikh': False, u'huge': False, u'respective': False, u'hickey': False, u'edgecomb': False, u'demolition': False, u'speedboat': False, u'hugo': False, u'hugh': False, u'dismissed': False, u'hugs': False, u'dismisses': False, u'isuro': False, u'sprinkle': False, u'lanky': False, u'intended': False, u'mendes': False, u'thickened': False, u'disgraced': False, u'greenwald': False, u'hackwork': False, u'maltese': False, u'dryland': False, u'malevolent': False, u'jiang': False, u'resemble': False, u'sublte': False, u'twisting': False, u'tiegs': False, u'dolph': False, u'overcooked': False, u'replied': False, u'weirdoes': False, u'depraved': False, u'peppy': False, u'installed': False, u'resorts': False, u'paper': False, u'scott': False, u'signs': False, u'smiling': False, u'signy': False, u'roots': False, u'saucy': False, u'mistreated': False, u'tantalizingly': False, u'ethnocentric': False, u'sublimated': False, u'hounds': False, u'isaak': False, u'blunderheaded': False, u'dolly': False, u'bummer': False, u'isaac': False, u'sauce': False, u'reintroduced': False, u'colleague': False, u'cartman': False, u'frizzi': False, u'abandons': False, u'universality': False, u'gadget': False, u'frizzy': False, u'balaban': False, u'weeds': False, u'idols': False, u'everytime': False, u'denny': False, u'courses': False, u'unbrewed': False, u'hatchette': False, u'repayment': False, u'shocking': False, u'reactions': False, u'brunette': False, u'another': False, u'numeric': False, u'scalvaging': False, u'chrissy': False, u'operation': False, u'centuries': False, u'inquired': False, u'lipstick': False, u'ernie': False, u'kensington': False, u'buzzsaw': False, u'research': False, u'inquires': False, u'illustrate': False, u'occurs': False, u'abnormally': False, u'poignantly': False, u'definition': False, u'pairs': False, u'2056': False, u'theroux': False, u'unjustifyably': False, u'testament': False, u'existential': False, u'porpoise': False, u'petra': False, u'horndog': False, u'seduction': False, u'trekkie': False, u'terrifyingly': False, u'brutally': False, u'preservation': False, u'burke': False, u'arlington': False, u'nomi': False, u'moderately': False, u'heartedness': False, u'bigscreen': False, u'excitable': False, u'bedridden': False, u'saint': False, u'kindergartner': False, u'essays': False, u'peaceably': False, u'nursery': False, u'justly': False, u'dethroned': False, u'interviewed': False, u'typhoon': False, u'cheekbones': False, u'chapelle': False, u'theater': False, u'stifle': False, u'oilrig': False, u'dethrones': False, u'pyroclastic': False, u'funicello': False, u'stormare': False, u'condolences': False, u'bruckheimer': False, u'getaway': False, u'dogs': False, u'dismantling': False, u'beneficial': False, u'prescott': False, u'labyrinthian': False, u'navigators': False, u'kret': False, u'mensch': False, u'organisations': False, u'swanky': False, u'waft': False, u'berle': False, u'guarding': False, u'graffiti': False, u'blond': False, u'cleon': False, u'cleverness': False, u'sell': False, u'nosebleeding': False, u'antagonizes': False, u'tarnish': False, u'self': False, u'sela': False, u'client': False, u'also': True, u'recognizing': False, u'sebastiano': False, u'conscription': False, u'sharpe': False, u'bastad': False, u'pringles': False, u'singles': False, u'raucous': False, u'virus': False, u'channeling': False, u'immediacy': False, u'singled': False, u'understands': False, u'omaha': False, u'seize': False, u'sometimes': False, u'flits': False, u'barred': False, u'cultivating': False, u'barren': False, u'barrel': False, u'shread': False, u'amusements': False, u'dragonflies': False, u'ambiguities': False, u'ugh': False, u'ugc': False, u'blended': False, u'accommodations': False, u'colorized': False, u'prodigious': False, u'naomi': False, u'overwhelmed': False, u'caruso': False, u'wraps': False, u'kinkiness': False, u'neophytes': False, u'turmoil': False, u'gooey': False, u'cassette': False, u'snobbish': False, u'crashlands': False, u'indifference': False, u'lombard': False, u'haskin': False, u'secular': False, u'ceasing': False, u'sunny': False, u'asssss': False, u'informants': False, u'yeager': False, u'remedy': False, u'compass': False, u'damnit': False, u'distraction': False, u'devoured': False, u'sects': False, u'pleasures': False, u'tanked': False, u'sojourn': False, u'tanker': False, u'pleasured': False, u'rumored': False, u'insane': False, u'delicately': False, u'bozo': False, u'activists': False, u'collectively': False, u'overboard': False, u'allie': False, u'glistening': False, u'richelieu': False, u'commensurately': False, u'ahmet': False, u'storyboarded': False, u'wopr': False, u'15th': False, u'goggins': False, u'untouched': False, u'coffee': False, u'canran': False, u'lass': False, u'last': False, u'legitimately': False, u'opal': False, u'swigert': False, u'connection': False, u'amoeba': False, u'opar': False, u'retarded': False, u'lash': False, u'onofrio': False, u'bell': False, u'2293': False, u'acted': False, u'adaptation': False, u'seldes': False, u'belt': False, u'unthrilling': False, u'warfield': False, u'unarguably': False, u'satire': False, u'suburbs': False, u'proprietor': False, u'initiation': False, u'portait': False, u'faulkner': False, u'patrolled': False, u'combatants': False, u'infect': False, u'amphibians': False, u'adaptable': False, u'awake': False, u'mournful': False, u'magwitch': False, u'exponential': False, u'caged': False, u'expanded': False, u'budget': False, u'admire': False, u'reopens': False, u'cagey': False, u'pressed': False, u'frighteners': False, u'bogan': False, u'cages': False, u'beatng': False, u'voe': False, u'agitation': False, u'mystic': False, u'von': False, u'binding': False, u'faceted': False, u'vow': False, u'underlining': False, u'cuisine': False, u'raiders': False, u'jerking': False, u'perpetrator': False, u'pridefully': False, u'matchmakers': False, u'cayman': False, u'ugliness': False, u'windmill': False, u'praising': False, u'flooded': False, u'everclear': False, u'reiser': False, u'fleischer': False, u'wunderkind': False, u'vargas': False, u'infamous': False, u'symbolise': False, u'doreen': False, u'prehensile': False, u'coachmen': False, u'dared': False, u'portillo': False, u'scoffs': False, u'thats': False, u'soaked': False, u'pepto': False, u'salva': False, u'cheddar': False, u'crackled': False, u'thaddeus': False, u'hercules': False, u'crackles': False, u'wage': False, u'hemingwayesque': False, u'cuffs': False, u'melinda': False, u'rangers': False, u'studious': False, u'parents': False, u'depravity': False, u'boardroom': False, u'eery': False, u'cormack': False, u'emergency': False, u'impaling': False, u'couple': False, u'bureaucrat': False, u'emanating': False, u'wives': False, u'ofcs': False, u'abound': False, u'emergence': False, u'thurman': False, u'marquee': False, u'spine': False, u'chorus': False, u'individuals': False, u'crookier': False, u'bogie': False, u'mediocrity': False, u'bamboo': False, u'turvy': False, u'alexandre': False, u'spins': False, u'crescendo': False, u'methods': False, u'goddamn': False, u'unsubstantial': False, u'bounce': False, u'ahern': False, u'bouncy': False, u'greener': False, u'underbelly': False, u'obliges': False, u'measurements': False, u'novelty': False, u'pell': False, u'behave': False, u'whodunit': False, u'metamorphoses': False, u'seclusion': False, u'inserting': False, u'dialogueless': False, u'hammond': False, u'jovovich': False, u'wayward': False, u'obscures': False, u'respite': False, u'grotesqe': False, u'janusz': False, u'obscured': False, u'cranked': False, u'deserved': False, u'simplify': False, u'goody': False, u'scorces': False, u'wrinkles': False, u'melbourne': False, u'deserves': False, u'scraggly': False, u'maude': False, u'wrinkled': False, u'gallagher': False, u'canning': False, u'laughton': False, u'tornatore': False, u'_dead_': False, u'terrorists': False, u'into': True, u'unredeemable': False, u'catchiness': False, u'middleton': False, u'controversies': False, u'remembrance': False, u'chirping': False, u'katie': False, u'realisation': False, u'span': False, u'harnessed': False, u'spam': False, u'tingles': False, u'sock': False, u'gases': False, u'bios': False, u'limburgher': False, u'grave': False, u'mishandle': False, u'spar': False, u'purred': False, u'spat': False, u'considerably': False, u'atlantis': False, u'invite': False, u'hawaiian': False, u'murphy': False, u'palentologist': False, u'_dragon_': False, u'deductions': False, u'lydia': False, u'carping': False, u'fanatasies': False, u'considerable': False, u'intestines': False, u'jacki': False, u'remastered': False, u'charmed': False, u'erich': False, u'pissant': False, u'testaments': False, u'eddie': False, u'erica': False, u'paired': False, u'tamahori': False, u'kihlstedt': False, u'awestruck': False, u'chad': False, u'pseudonymous': False, u'influence': False, u'haunt': False, u'portentuous': False, u'globally': False, u'thomsen': False, u'chap': False, u'revelatory': False, u'palisades': False, u'chat': False, u'apropos': False, u'chaz': False, u'frontgate': False, u'immeadiately': False, u'neilsen': False, u'intrepid': False, u'puzzling': False, u'copulate': False, u'thanks': False, u'excuses': False, u'conceptions': False, u'ellen': False, u'singed': False, u'heebie': False, u'aunt': False, u'rabal': False, u'interogation': False, u'oblige': False, u'gardenia': False, u'teck': False, u'strums': False, u'aussies': False, u'prepared': False, u'bianca': False, u'mckidd': False, u'flyboy': False, u'suppression': False, u'euphegenia': False, u'lang': False, u'guitry': False, u'land': False, u'ryan_': False, u'lana': False, u'advertisment': False, u'purged': False, u'reserve': False, u'modernizing': False, u'zzzzzzz': False, u'splashing': False, u'spielbergization': False, u'unbuttoning': False, u'broader': False, u'amiss': False, u'flashback': False, u'humpback': False, u'coffey': False, u'detectives': False, u'amalgamation': False, u'turkish': False, u'ditzism': False, u'horsing': False, u'dickinson': False, u'carelessly': False, u'resources': False, u'nervousness': False, u'lindner': False, u'boatload': False, u'undeterred': False, u'millieu': False, u'alienbusting': False, u'hockley': False, u'huddled': False, u'prakazrel': False, u'traumatised': False, u'scatology': False, u'koppelman': False, u'petitions': False, u'decorating': False, u'herzfeld': False, u'detested': False, u'yakov': False, u'lombardo': False, u'rifling': False, u'integrating': False, u'fewer': False, u'damning': False, u'yevgeny': False, u'disheveled': False, u'insubordinate': False, u'leonardi': False, u'leonardo': False, u'villiany': False, u'maclachlan': False, u'overblown': False, u'dysfuntion': False, u'cannibals': False, u'mishap': False, u'crook': False, u'video': True, u'dynamics': False, u'elisa': False, u'victor': False, u'improvisationaly': False, u'narrations': False, u'sweats': False, u'waning': False, u'harvests': False, u'sweaty': False, u'henceforth': False, u'royalist': False, u'turnaround': False, u'slickster': False, u'flowing': False, u'charade': False, u'harassing': False, u'guamo': False, u'forwarned': False, u'apace': False, u'squirming': False, u'fifteen': False, u'implicit': False, u'kriss': False, u'bakersfield': False, u'33': False, u'unwittingly': False, u'scatter': False, u'condescending': False, u'panned': False, u'survey': False, u'climb': False, u'makes': True, u'maker': False, u'looted': False, u'bumming': False, u'panicked': False, u'blizzard': False, u'formulates': False, u'dumbest': False, u'chilly': False, u'desiring': False, u'confidence': False, u'excising': False, u'pfarrer': False, u'gregor': False, u'zsigmond': False, u'next': False, u'eleven': False, u'assuring': False, u'mccleod': False, u'chu': False, u'tahoe': False, u'binges': False, u'phallus': False, u'yugoslavian': False, u'pencil': False, u'babe': False, u'spearing': False, u'tons': True, u'duper': False, u'babs': False, u'boondocks': False, u'prizes': False, u'losin': False, u'baby': False, u'antichrist': False, u'_escape': False, u'documentarian': False, u'customer': False, u'f': False, u'clients': False, u'unknowns': False, u'retell': False, u'harve': False, u'initation': False, u'rehabilitation': False, u'wedge': False, u'loca': False, u'painkiller': False, u'calculation': False, u'lock': False, u'coolness': False, u'loco': False, u'promotional': False, u'aughra': False, u'nears': False, u'bolstered': False, u'taj': False, u'cecilia': False, u'educational': False, u'afi': False, u'raeeyain': False, u'awkwardness': False, u'paled': False, u'schandling': False, u'tightrope': False, u'procured': False, u'neary': False, u'bilingual': False, u'hormones': False, u'burley': False, u'engagingly': False, u'intelligent': False, u'pales': False, u'incongruent': False, u'highs': False, u'huffs': False, u'retrograding': False, u'upstate': False, u'procures': False, u'infirm': False, u'realized': False, u'jolting': False, u'solon': False, u'clarkson': False, u'shout': False, u'robot': False, u'realizes': False, u'scrubs': False, u'sciorra': False, u'typicalness': False, u'marshalls': False, u'bartenders': False, u'houston': False, u'boxing': False, u'thigh': False, u'mute': False, u'muth': False, u'despite': True, u'frieberg': False, u'spatula': False, u'directs': False, u'bartusiak': False, u'hotcakes': False, u'perfect': False, u'anonymously': False, u'byline': False, u'rizzo': False, u'jetsons': False, u'meantime': True, u'thieves': False, u'derivative': False, u'90210': False, u'sabotaging': False, u'prosper': False, u'vocalized': False, u'impervious': False, u'overal': False, u'isacsson': False, u'guaspari': False, u'reinvents': False, u'snake': False, u'squabbling': False, u'realize': False, u'reconstruction': False, u'comedy': False, u'damian': False, u'scenic': False, u'zeist': False, u'denzel': False, u'shortage': False, u'weismuller': False, u'emo': False, u'glasses': False, u'goldsman': False, u'suitors': False, u'bump': False, u'poppins': False, u'bums': False, u'deficiency': False, u'leplastrier': False, u'books': False, u'resuscitate': False, u'gungan': False, u'bigfoot': False, u'witness': False, u'unoriginal': False, u'matrix': False, u"'": True, u'harrowingly': False, u'narratively': False, u'frowns': False, u'unprepared': False, u'hypnotist': False, u'red': False, u'unwieldy': False, u'benben': False, u'inferiority': False, u'greedy': False, u'gawain': False, u'disintegrating': False, u'initialize': False, u'pothead': False, u'mainland': False, u'fueled': False, u'blandy': False, u'gallons': False, u'could': False, u'genieveve': False, u'length': False, u'chills': False, u'babyzilla': False, u'fleshed': False, u'scene': False, u'reaches': False, u'soothing': False, u'affliction': False, u'leick': False, u'morice': False, u'scent': False, u'fleshes': False, u'braces': False, u'erstwhile': False, u'leder': False, u'festival': False, u'lumet': False, u'rediscovers': False, u'fanaro': False, u'stabbin': False, u'sergeant': False, u'henning': False, u'pervasive': False, u'enforcement': False, u'zookeeper': False, u'stomach': False, u'quarry': False, u'greenbaum': False, u'pulman': False, u'beastuality': False, u'incongruities': False, u'fatboy': False, u'egregious': False, u'roulette': False, u'commandant': False, u'gentile': False, u'orchestrated': False, u'daydreams': False, u'mackey': False, u'faulted': False, u'false': False, u'shrinks': False, u'chivalrous': False, u'tonight': False, u'ponders': False, u'richman': False, u'sufis': False, u'cecil': False, u'hessian': False, u'depict': False, u'venturing': False, u'dishes': False, u'fireballs': False, u'mia': False, u'dodie': False, u'mib': False, u'precipice': False, u'dished': False, u'bakshi': False, u'sandworms': False, u'worldwide': False, u'jimmies': False, u'closeups': False, u'manor': False, u'fujioka': False, u'petals': False, u'cipher': False, u'draws': False, u'unsexy': False, u'hoodwink': False, u'salutory': False, u'unsparing': False, u'doogie': False, u'auberjonois': False, u'placement': False, u'introversion': False, u'wuthering': False, u'bred': False, u'thanksgiving': False, u'lots': False, u'perceiving': False, u'undersea': False, u'brew': False, u'bret': False, u'rainbows': False, u'woolly': False, u'greenhouse': False, u'nominally': False, u'xvi': False, u'taps': False, u'jax': False, u'jay': False, u'jaw': False, u'consciouness': False, u'jar': False, u'risqueness': False, u'gascogne': False, u'jan': False, u'entities': False, u'jam': False, u'tape': False, u'jah': False, u'jai': False, u'riding': False, u'jab': False, u'abbe': False, u'insight': True, u'cooperation': False, u'abba': False, u'antagonism': False, u'prohibition': False, u'molasses': False, u'drawn': False, u'tossed': False, u'wring': False, u'styrofoam': False, u'abby': False, u'gertz': False, u'rapper': False, u'comprising': False, u'taxes': False, u'shields': False, u'coaxed': False, u'ocmic': False, u'stuff': False, u'ohio': False, u'rapped': False, u'raceway': False, u'exude': False, u'guessing': False, u'allusion': False, u'qinqin': False, u'frame': False, u'hijinx': False, u'arsed': False, u'kombat_': False, u'alessandro': False, u'trods': False, u'siegfried': False, u'deconstructs': False, u'dungeon': False, u'destiny': False, u'insulting': False, u'nuclear': False, u'comprehendably': False, u'melrose': False, u'comprehendable': False, u'repetitively': False, u'nesmith': False, u'preminger': False, u'keynote': False, u'quirkyness': False, u'ifans': False, u'refuting': False, u'lawsuit': False, u'staring': False, u'marty': False, u'hammy': False, u'swann': False, u'marts': False, u'flanery': False, u'doorway': False, u'unearthing': False, u'vadar': False, u'conclude': False, u'roughed': False, u'stylistically': False, u'confronts': False, u'novalee': False, u'mailman': False, u'midland': False, u'catholicism': False, u'kahl': False, u'kahn': False, u'eviscerated': False, u'yoram': False, u'feather': False, u'>': False, u'butchers': False, u'marcellus': False, u'altruist': False, u'sheepish': False, u'commuter': False, u'commutes': False, u'swarmed': False, u'coherence': False, u'hateful': False, u'swindling': False, u'banish': False, u'miscommunicated': False, u'lecherous': False, u'reminiscence': False, u'gamesmanship': False, u'eaton': False, u'farsical': False, u'hahaha': False, u'fielding': False, u'miyake': False, u'stuffing': False, u'lawerence': False, u'lascivious': False, u'incense': False, u'ransom': False, u'tattoo': False, u'ostentatious': False, u'moves': False, u'pauly': False, u'subtitling': False, u'paull': False, u'painstaking': False, u'gibbs': False, u'pickin': False, u'chronicling': False, u'paula': False, u'unimaginable': False, u'complemented': False, u'unsympathetic': False, u'censoring': False, u'fraidy': False, u'berardinelli': False, u'identity': False, u'ofa': False, u'off': True, u'shotgun': False, u'dissing': False, u'patterns': False, u'oft': False, u'audio': False, u'tactfully': False, u'quentin': False, u'braniff': False, u'newest': False, u'obscenity': False, u'dissatisfying': False, u'souped': False, u'coalwood': False, u'clocks': False, u'diedre': False, u'unmitigatedly': False, u'web': False, u'tong': False, u'wee': False, u'hauntingly': False, u'wei': False, u'wen': False, u'toyed': False, u'undulating': False, u'wes': True, u'toni': False, u'wet': False, u'practise': False, u'villagers': False, u'tics': False, u'pieh': False, u'pied': False, u'crud': False, u'stooped': False, u'falters': False, u'mutations': False, u'crux': False, u'cruz': False, u'zimbabwe': False, u'hallucinates': False, u'atrophy': False, u'piet': False, u'tick': False, u'pier': False, u'pies': False, u'debatable': False, u'emma': False, u'bulge': False, u'mistrustful': False, u'flickering': False, u'become': False, u'emmy': False, u'assasination': False, u'palladino': False, u'nutsy': False, u'knifepoint': False, u'brainerd': False, u'underwent': False, u'basque': False, u'_pick_chucky_up_': False, u'immortal': False, u'petey': False, u'gymnastics': False, u'choosing': False, u'flush': False, u'hissing': False, u'humming': False, u'recognition': False, u'delaurentiis': False, u'hipsters': False, u'mementos': False, u'buehler': False, u'passion': False, u'copulation': False, u'biology': False, u'uhhhm': False, u'brokering': False, u'pressure': False, u'infiltrating': False, u'imaginary': False, u'coldly': False, u'homemaker': False, u'iwai': False, u'lifestyle': False, u'langer': False, u'burroughs': False, u'outshines': False, u'blackness': False, u'sadoski': False, u'documentary': False, u'swimming': False, u'promiss': False, u'letters': False, u'miscegenation': False, u'rochon': False, u'hang': False, u'compadre': False, u'mojorino': False, u'privates': False, u'terminated': False, u'letter_': False, u'brides': False, u'pairing': False, u'peters': False, u'heresy': False, u'indoctrinated': False, u'rhythmic': False, u'yarns': False, u'moonstruck': False, u'progenitor': False, u'contradictory': False, u'jerkish': False, u'bagger': False, u'counterparts': False, u'zaniness': False, u'unformal': False, u'places': False, u'bloodline': False, u'congresswoman': False, u'excitement': False, u'placed': False, u'mouseketeer': False, u'tosses': False, u'problem': True, u'unsupportive': False, u'nurses': False, u'_lot_': False, u'lobotomise': False, u'walters': False, u'effected': False, u'compared': False, u'nonetheless': False, u'deadly': False, u'purproses': False, u'lately': False, u'kerrigans': False, u'compares': False, u'details': False, u'boon': False, u'behold': False, u'vulgarize': False, u'illusion': False, u'ponytail': False, u'rebelled': False, u'repeat': False, u'zhou': False, u'treason': False, u'allotting': False, u'impregnating': False, u'tinier': False, u'trunchbull': False, u'laude': False, u'exposure': False, u'searches': False, u'ustinov': False, u'disatisfaction': False, u'mishears': False, u'torrid': False, u'compete': False, u'lestat': False, u'villainous': False, u'searched': False, u'gardens': False, u'homerian': False}
    


```python
featuresets = [(find_features(rev), category) for (rev, category) in documents]
featuresets
```

## 13. [Naive Bayes Classifier with NLTK](https://pythonprogramming.net/naive-bayes-classifier-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15355406/)


```python
# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
```

    ('Classifier accuracy percent:', 68.0)
    


```python
classifier.show_most_informative_features(15)
```

    Most Informative Features
                   insulting = True              neg : pos    =     10.7 : 1.0
                      doubts = True              pos : neg    =      9.5 : 1.0
                        sans = True              neg : pos    =      8.4 : 1.0
                  mediocrity = True              neg : pos    =      7.8 : 1.0
                     wasting = True              neg : pos    =      7.8 : 1.0
                refreshingly = True              pos : neg    =      7.6 : 1.0
                   dismissed = True              pos : neg    =      6.9 : 1.0
                 bruckheimer = True              neg : pos    =      6.4 : 1.0
                       wires = True              neg : pos    =      6.4 : 1.0
                      fabric = True              pos : neg    =      6.3 : 1.0
                 overwhelmed = True              pos : neg    =      6.3 : 1.0
                         ugh = True              neg : pos    =      5.9 : 1.0
                   uplifting = True              pos : neg    =      5.8 : 1.0
                      bounce = True              neg : pos    =      5.7 : 1.0
                        lang = True              pos : neg    =      5.6 : 1.0
    

## 14. [Saving Classifiers with NLTK](https://pythonprogramming.net/pickle-classifier-save-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15355460/)


```python
import pickle

save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

classifier.show_most_informative_features(15)
```

    Most Informative Features
                   insulting = True              neg : pos    =     10.7 : 1.0
                      doubts = True              pos : neg    =      9.5 : 1.0
                        sans = True              neg : pos    =      8.4 : 1.0
                  mediocrity = True              neg : pos    =      7.8 : 1.0
                     wasting = True              neg : pos    =      7.8 : 1.0
                refreshingly = True              pos : neg    =      7.6 : 1.0
                   dismissed = True              pos : neg    =      6.9 : 1.0
                 bruckheimer = True              neg : pos    =      6.4 : 1.0
                       wires = True              neg : pos    =      6.4 : 1.0
                      fabric = True              pos : neg    =      6.3 : 1.0
                 overwhelmed = True              pos : neg    =      6.3 : 1.0
                         ugh = True              neg : pos    =      5.9 : 1.0
                   uplifting = True              pos : neg    =      5.8 : 1.0
                      bounce = True              neg : pos    =      5.7 : 1.0
                        lang = True              pos : neg    =      5.6 : 1.0
    

## 15. [Scikit-Learn Sklearn with NLTK](https://pythonprogramming.net/sklearn-scikit-learn-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15355745/)


```python
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

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
```

    ('Original Naive Bayes Algo accuracy percent:', 68.0)
    Most Informative Features
                   insulting = True              neg : pos    =     10.7 : 1.0
                      doubts = True              pos : neg    =      9.5 : 1.0
                        sans = True              neg : pos    =      8.4 : 1.0
                  mediocrity = True              neg : pos    =      7.8 : 1.0
                     wasting = True              neg : pos    =      7.8 : 1.0
                refreshingly = True              pos : neg    =      7.6 : 1.0
                   dismissed = True              pos : neg    =      6.9 : 1.0
                 bruckheimer = True              neg : pos    =      6.4 : 1.0
                       wires = True              neg : pos    =      6.4 : 1.0
                      fabric = True              pos : neg    =      6.3 : 1.0
                 overwhelmed = True              pos : neg    =      6.3 : 1.0
                         ugh = True              neg : pos    =      5.9 : 1.0
                   uplifting = True              pos : neg    =      5.8 : 1.0
                      bounce = True              neg : pos    =      5.7 : 1.0
                        lang = True              pos : neg    =      5.6 : 1.0
    ('MNB_classifier accuracy percent:', 68.0)
    ('BernoulliNB_classifier accuracy percent:', 69.0)
    ('LogisticRegression_classifier accuracy percent:', 66.0)
    ('SGDClassifier_classifier accuracy percent:', 63.0)
    ('SVC_classifier accuracy percent:', 44.0)
    ('LinearSVC_classifier accuracy percent:', 55.00000000000001)
    ('NuSVC_classifier accuracy percent:', 61.0)
    

## 16. [Combining Algorithms with NLTK](https://pythonprogramming.net/combine-classifier-algorithms-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15355927/)


```python
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
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

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
        
training_set = featuresets[:1900]
testing_set =  featuresets[1900:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()




print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(training_set)
##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)
```

    ('Original Naive Bayes Algo accuracy percent:', 89.0)
    Most Informative Features
                   insulting = True              neg : pos    =     10.7 : 1.0
                      doubts = True              pos : neg    =      9.5 : 1.0
                        sans = True              neg : pos    =      8.4 : 1.0
                  mediocrity = True              neg : pos    =      7.8 : 1.0
                     wasting = True              neg : pos    =      7.8 : 1.0
                refreshingly = True              pos : neg    =      7.6 : 1.0
                   dismissed = True              pos : neg    =      6.9 : 1.0
                 bruckheimer = True              neg : pos    =      6.4 : 1.0
                       wires = True              neg : pos    =      6.4 : 1.0
                      fabric = True              pos : neg    =      6.3 : 1.0
                 overwhelmed = True              pos : neg    =      6.3 : 1.0
                         ugh = True              neg : pos    =      5.9 : 1.0
                   uplifting = True              pos : neg    =      5.8 : 1.0
                      bounce = True              neg : pos    =      5.7 : 1.0
                        lang = True              pos : neg    =      5.6 : 1.0
    ('MNB_classifier accuracy percent:', 74.0)
    ('BernoulliNB_classifier accuracy percent:', 74.0)
    ('LogisticRegression_classifier accuracy percent:', 70.0)
    ('SGDClassifier_classifier accuracy percent:', 64.0)
    ('LinearSVC_classifier accuracy percent:', 70.0)
    ('NuSVC_classifier accuracy percent:', 71.0)
    ('voted_classifier accuracy percent:', 75.0)
    ('Classification:', u'pos', 'Confidence %:', 0)
    ('Classification:', u'pos', 'Confidence %:', 0)
    ('Classification:', u'neg', 'Confidence %:', 100)
    ('Classification:', u'pos', 'Confidence %:', 100)
    ('Classification:', u'neg', 'Confidence %:', 0)
    ('Classification:', u'neg', 'Confidence %:', 0)
    

## 17. [Investigating bias with NLTK](https://pythonprogramming.net/investigating-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15356098/)

## 18. [Improving Training Data for sentiment analysis with NLTK](https://pythonprogramming.net/new-data-set-training-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15356221/)


```python
import codecs

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize


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
        
short_pos = open("nlp/hello_nltk/short_reviews/positive.txt","r").read()
short_neg = open("nlp/hello_nltk/short_reviews/negative.txt","r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append( (r, "pos") )

for r in short_neg.split('\n'):
    documents.append( (r, "neg") )


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

# positive data example:      
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

##
### negative data example:      
##training_set = featuresets[100:]
##testing_set =  featuresets[:100]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(training_set)
##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


voted_classifier = VoteClassifier(
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
```


    

    UnicodeDecodeErrorTraceback (most recent call last)

    <ipython-input-5-3add54aa8726> in <module>()
         38         return conf
         39 
    ---> 40 short_pos = codecs.open("nlp/hello_nltk/short_reviews/positive.txt","r",encoding = "utf8").read()
         41 short_neg = codecs.open("nlp/hello_nltk/short_reviews/negative.txt","r",encoding = "utf8").read()
         42 
    

    D:\Anaconda2\lib\codecs.pyc in read(self, size)
        684     def read(self, size=-1):
        685 
    --> 686         return self.reader.read(size)
        687 
        688     def readline(self, size=None):
    

    D:\Anaconda2\lib\codecs.pyc in read(self, size, chars, firstline)
        490             data = self.bytebuffer + newdata
        491             try:
    --> 492                 newchars, decodedbytes = self.decode(data, self.errors)
        493             except UnicodeDecodeError, exc:
        494                 if firstline:
    

    UnicodeDecodeError: 'utf8' codec can't decode byte 0xf3 in position 4645: invalid continuation byte


## 19. [Creating a module for Sentiment Analysis with NLTK](https://pythonprogramming.net/sentiment-analysis-module-nltk-tutorial/) | [video](https://www.bilibili.com/video/av15356391/)


```python
import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



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
    
short_pos = open("short_reviews/positive.txt","r").read()
short_neg = open("short_reviews/negative.txt","r").read()

# move this up here
all_words = []
documents = []


#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())



save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()


all_words = nltk.FreqDist(all_words)


word_features = list(all_words.keys())[:5000]


save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

###############
save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


##NuSVC_classifier = SklearnClassifier(NuSVC())
##NuSVC_classifier.train(training_set)
##print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)

save_classifier = open("pickled_algos/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()
```


```python
import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



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


documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()




word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



featuresets_f = open("pickled_algos/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]



open_file = open("pickled_algos/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()



open_file = open("pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()




voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)




def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
```


    

    IOErrorTraceback (most recent call last)

    <ipython-input-7-2e7e2625bac8> in <module>()
         35 
         36 
    ---> 37 documents_f = open("pickled_algos/documents.pickle", "rb")
         38 documents = pickle.load(documents_f)
         39 documents_f.close()
    

    IOError: [Errno 2] No such file or directory: 'pickled_algos/documents.pickle'



```python
print(sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
print(sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))
```


```python
## 20. []() | []()
```


```python
## 21. []() | []()
```
