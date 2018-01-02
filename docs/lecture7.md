
# 7. 从文本提取信息

## 待回答的问题？
1. 提取结构化数据

2. 识别实体与关系

3. 合适的语料库


## 7.1 信息提取
分句 - 分词 - 词性标注


```
import nltk


def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)  # 分句             
    sentences = [nltk.word_tokenize(sent) for sent in sentences]  # 分词
    sentences = [nltk.pos_tag(sent) for sent in sentences]  # 词性标注
    return sentences
```

## 7.2 分块
