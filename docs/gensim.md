
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gensim


```python
# -*- coding: utf-8 -*-
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```


```python
from gensim import corpora

documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]
```

    e:\ProgramData\Anaconda3\lib\site-packages\gensim\utils.py:860: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    2017-12-25 14:51:28,440 : INFO : 'pattern' package not found; tag filters are not available for English
    


```python
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in documents]
```


```python
import jieba

content = """面对当前挑战，我们应该落实2030年可持续发展议程，促进包容性发展"""
content = list(jieba.cut(content, cut_all=False))
for word in content:
    print(word)
```

    Building prefix dict from the default dictionary ...
    2017-12-25 14:51:32,170 : DEBUG : Building prefix dict from the default dictionary ...
    Loading model from cache C:\Users\THREEP~1\AppData\Local\Temp\jieba.cache
    2017-12-25 14:51:32,178 : DEBUG : Loading model from cache C:\Users\THREEP~1\AppData\Local\Temp\jieba.cache
    Loading model cost 0.710 seconds.
    2017-12-25 14:51:32,881 : DEBUG : Loading model cost 0.710 seconds.
    Prefix dict has been built succesfully.
    2017-12-25 14:51:32,883 : DEBUG : Prefix dict has been built succesfully.
    

    面对
    当前
    挑战
    ，
    我们
    应该
    落实
    2030
    年
    可
    持续
    发展
    议程
    ，
    促进
    包容性
    发展
    


```python
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (15.0, 8.0)
% matplotlib inline

import numpy as np
import pandas as pd

from scipy.misc import imread
from wordcloud import WordCloud
```


```python
name = "docs/assets/择天记.txt"
with open(name, encoding="utf-8") as fp:
    book = fp.read()

with open(name, encoding="utf-8") as fp:
    lines = [line for line in fp.readlines() if len(line) > 2]

print(book[:100])
print(lines[:10])
```

    择天记
    
    猫腻
    
    玄幻奇幻
    
    太始元年，有神石自太空飞来，分散落在人间，其中落在东土大陆的神石，上面镌刻着奇怪的图腾，人因观其图腾而悟道，后立国教。
    数千年后，十四岁的少年孤儿陈长生，为治病改命离开自
    ['择天记\n', '猫腻\n', '玄幻奇幻\n', '太始元年，有神石自太空飞来，分散落在人间，其中落在东土大陆的神石，上面镌刻着奇怪的图腾，人因观其图腾而悟道，后立国教。\n', '数千年后，十四岁的少年孤儿陈长生，为治病改命离开自己的师父，带着一纸婚约来到神都，从而开启了一个逆天强者的崛起征程。\n', '各位书友要是觉得《择天记》还不错的话请不要忘记向您QQ群和微博里的朋友推荐哦！\n', '序 下山\n', '世界是相对的。\n', '中土大6隔着海洋与大西洲遥遥相对。东方地势较高，那里的天空似乎也高了起来，云雾从海上6地上升腾而起，不停向着那处飘去，最终汇聚在一起，终年不散。\n', '这里便是云墓——世间所有云的坟墓。\n']
    


```python
import jieba
jieba.add_word("陈长生",tag="nz")
jieba.add_word("徐有容",tag="nz")
jieba.add_word("落落",tag="nz")
jieba.add_word("小黑龙",tag="nz")

segments = [seg for seg in jieba.cut(book) if len(seg) > 1]
df = pd.DataFrame({'segment': segments})
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>segment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>择天记</td>
    </tr>
    <tr>
      <th>1</th>
      <td>猫腻</td>
    </tr>
    <tr>
      <th>2</th>
      <td>玄幻</td>
    </tr>
    <tr>
      <th>3</th>
      <td>奇幻</td>
    </tr>
    <tr>
      <th>4</th>
      <td>太始</td>
    </tr>
    <tr>
      <th>5</th>
      <td>元年</td>
    </tr>
    <tr>
      <th>6</th>
      <td>有神</td>
    </tr>
    <tr>
      <th>7</th>
      <td>石自</td>
    </tr>
    <tr>
      <th>8</th>
      <td>太空</td>
    </tr>
    <tr>
      <th>9</th>
      <td>飞来</td>
    </tr>
    <tr>
      <th>10</th>
      <td>分散</td>
    </tr>
    <tr>
      <th>11</th>
      <td>人间</td>
    </tr>
    <tr>
      <th>12</th>
      <td>其中</td>
    </tr>
    <tr>
      <th>13</th>
      <td>东土</td>
    </tr>
    <tr>
      <th>14</th>
      <td>大陆</td>
    </tr>
    <tr>
      <th>15</th>
      <td>神石</td>
    </tr>
    <tr>
      <th>16</th>
      <td>上面</td>
    </tr>
    <tr>
      <th>17</th>
      <td>镌刻</td>
    </tr>
    <tr>
      <th>18</th>
      <td>奇怪</td>
    </tr>
    <tr>
      <th>19</th>
      <td>图腾</td>
    </tr>
    <tr>
      <th>20</th>
      <td>因观</td>
    </tr>
    <tr>
      <th>21</th>
      <td>图腾</td>
    </tr>
    <tr>
      <th>22</th>
      <td>悟道</td>
    </tr>
    <tr>
      <th>23</th>
      <td>立国</td>
    </tr>
    <tr>
      <th>24</th>
      <td>数千年</td>
    </tr>
    <tr>
      <th>25</th>
      <td>十四岁</td>
    </tr>
    <tr>
      <th>26</th>
      <td>少年</td>
    </tr>
    <tr>
      <th>27</th>
      <td>孤儿</td>
    </tr>
    <tr>
      <th>28</th>
      <td>陈长生</td>
    </tr>
    <tr>
      <th>29</th>
      <td>治病</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1024912</th>
      <td>常见</td>
    </tr>
    <tr>
      <th>1024913</th>
      <td>宫廷</td>
    </tr>
    <tr>
      <th>1024914</th>
      <td>故事</td>
    </tr>
    <tr>
      <th>1024915</th>
      <td>陈长生</td>
    </tr>
    <tr>
      <th>1024916</th>
      <td>说道</td>
    </tr>
    <tr>
      <th>1024917</th>
      <td>我要</td>
    </tr>
    <tr>
      <th>1024918</th>
      <td>圣城</td>
    </tr>
    <tr>
      <th>1024919</th>
      <td>我们</td>
    </tr>
    <tr>
      <th>1024920</th>
      <td>可能</td>
    </tr>
    <tr>
      <th>1024921</th>
      <td>顺路</td>
    </tr>
    <tr>
      <th>1024922</th>
      <td>铁面人</td>
    </tr>
    <tr>
      <th>1024923</th>
      <td>焦急</td>
    </tr>
    <tr>
      <th>1024924</th>
      <td>说道</td>
    </tr>
    <tr>
      <th>1024925</th>
      <td>一定</td>
    </tr>
    <tr>
      <th>1024926</th>
      <td>顺路</td>
    </tr>
    <tr>
      <th>1024927</th>
      <td>一定</td>
    </tr>
    <tr>
      <th>1024928</th>
      <td>顺路</td>
    </tr>
    <tr>
      <th>1024929</th>
      <td>就算</td>
    </tr>
    <tr>
      <th>1024930</th>
      <td>地狱</td>
    </tr>
    <tr>
      <th>1024931</th>
      <td>毫不犹豫</td>
    </tr>
    <tr>
      <th>1024932</th>
      <td>跟随</td>
    </tr>
    <tr>
      <th>1024933</th>
      <td>脚步</td>
    </tr>
    <tr>
      <th>1024934</th>
      <td>陈长生</td>
    </tr>
    <tr>
      <th>1024935</th>
      <td>说道</td>
    </tr>
    <tr>
      <th>1024936</th>
      <td>如果</td>
    </tr>
    <tr>
      <th>1024937</th>
      <td>我要</td>
    </tr>
    <tr>
      <th>1024938</th>
      <td>地方</td>
    </tr>
    <tr>
      <th>1024939</th>
      <td>神国</td>
    </tr>
    <tr>
      <th>1024940</th>
      <td>全文</td>
    </tr>
    <tr>
      <th>1024941</th>
      <td>本章</td>
    </tr>
  </tbody>
</table>
<p>1024942 rows × 1 columns</p>
</div>




```python
stopwords = pd.read_csv(u"docs/assets/stop_words.txt")
df = df[~df.segment.isin(stopwords.stopword)]
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>segment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>择天记</td>
    </tr>
    <tr>
      <th>1</th>
      <td>猫腻</td>
    </tr>
    <tr>
      <th>2</th>
      <td>玄幻</td>
    </tr>
    <tr>
      <th>3</th>
      <td>奇幻</td>
    </tr>
    <tr>
      <th>4</th>
      <td>太始</td>
    </tr>
    <tr>
      <th>5</th>
      <td>元年</td>
    </tr>
    <tr>
      <th>6</th>
      <td>有神</td>
    </tr>
    <tr>
      <th>7</th>
      <td>石自</td>
    </tr>
    <tr>
      <th>8</th>
      <td>太空</td>
    </tr>
    <tr>
      <th>9</th>
      <td>飞来</td>
    </tr>
    <tr>
      <th>10</th>
      <td>分散</td>
    </tr>
    <tr>
      <th>11</th>
      <td>人间</td>
    </tr>
    <tr>
      <th>13</th>
      <td>东土</td>
    </tr>
    <tr>
      <th>14</th>
      <td>大陆</td>
    </tr>
    <tr>
      <th>15</th>
      <td>神石</td>
    </tr>
    <tr>
      <th>16</th>
      <td>上面</td>
    </tr>
    <tr>
      <th>17</th>
      <td>镌刻</td>
    </tr>
    <tr>
      <th>18</th>
      <td>奇怪</td>
    </tr>
    <tr>
      <th>19</th>
      <td>图腾</td>
    </tr>
    <tr>
      <th>20</th>
      <td>因观</td>
    </tr>
    <tr>
      <th>21</th>
      <td>图腾</td>
    </tr>
    <tr>
      <th>22</th>
      <td>悟道</td>
    </tr>
    <tr>
      <th>23</th>
      <td>立国</td>
    </tr>
    <tr>
      <th>24</th>
      <td>数千年</td>
    </tr>
    <tr>
      <th>25</th>
      <td>十四岁</td>
    </tr>
    <tr>
      <th>26</th>
      <td>少年</td>
    </tr>
    <tr>
      <th>27</th>
      <td>孤儿</td>
    </tr>
    <tr>
      <th>28</th>
      <td>陈长生</td>
    </tr>
    <tr>
      <th>29</th>
      <td>治病</td>
    </tr>
    <tr>
      <th>30</th>
      <td>改命</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1024910</th>
      <td>推演出来</td>
    </tr>
    <tr>
      <th>1024911</th>
      <td>一个</td>
    </tr>
    <tr>
      <th>1024912</th>
      <td>常见</td>
    </tr>
    <tr>
      <th>1024913</th>
      <td>宫廷</td>
    </tr>
    <tr>
      <th>1024914</th>
      <td>故事</td>
    </tr>
    <tr>
      <th>1024915</th>
      <td>陈长生</td>
    </tr>
    <tr>
      <th>1024916</th>
      <td>说道</td>
    </tr>
    <tr>
      <th>1024917</th>
      <td>我要</td>
    </tr>
    <tr>
      <th>1024918</th>
      <td>圣城</td>
    </tr>
    <tr>
      <th>1024920</th>
      <td>可能</td>
    </tr>
    <tr>
      <th>1024921</th>
      <td>顺路</td>
    </tr>
    <tr>
      <th>1024922</th>
      <td>铁面人</td>
    </tr>
    <tr>
      <th>1024923</th>
      <td>焦急</td>
    </tr>
    <tr>
      <th>1024924</th>
      <td>说道</td>
    </tr>
    <tr>
      <th>1024925</th>
      <td>一定</td>
    </tr>
    <tr>
      <th>1024926</th>
      <td>顺路</td>
    </tr>
    <tr>
      <th>1024927</th>
      <td>一定</td>
    </tr>
    <tr>
      <th>1024928</th>
      <td>顺路</td>
    </tr>
    <tr>
      <th>1024929</th>
      <td>就算</td>
    </tr>
    <tr>
      <th>1024930</th>
      <td>地狱</td>
    </tr>
    <tr>
      <th>1024931</th>
      <td>毫不犹豫</td>
    </tr>
    <tr>
      <th>1024932</th>
      <td>跟随</td>
    </tr>
    <tr>
      <th>1024933</th>
      <td>脚步</td>
    </tr>
    <tr>
      <th>1024934</th>
      <td>陈长生</td>
    </tr>
    <tr>
      <th>1024935</th>
      <td>说道</td>
    </tr>
    <tr>
      <th>1024937</th>
      <td>我要</td>
    </tr>
    <tr>
      <th>1024938</th>
      <td>地方</td>
    </tr>
    <tr>
      <th>1024939</th>
      <td>神国</td>
    </tr>
    <tr>
      <th>1024940</th>
      <td>全文</td>
    </tr>
    <tr>
      <th>1024941</th>
      <td>本章</td>
    </tr>
  </tbody>
</table>
<p>905949 rows × 1 columns</p>
</div>




```python
segStat = df.groupby(by=["segment"])["segment"].agg({"count": np.size}).reset_index().sort_values(by=["count"], ascending=False);
segStat.head(20)
```

    e:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: using a dict on a Series for aggregation
    is deprecated and will be removed in a future version
      """Entry point for launching an IPython kernel.
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>segment</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>49602</th>
      <td>陈长生</td>
      <td>15939</td>
    </tr>
    <tr>
      <th>31806</th>
      <td>没有</td>
      <td>13387</td>
    </tr>
    <tr>
      <th>44735</th>
      <td>说道</td>
      <td>10080</td>
    </tr>
    <tr>
      <th>36711</th>
      <td>看着</td>
      <td>6958</td>
    </tr>
    <tr>
      <th>37136</th>
      <td>知道</td>
      <td>6245</td>
    </tr>
    <tr>
      <th>19516</th>
      <td>已经</td>
      <td>4787</td>
    </tr>
    <tr>
      <th>14009</th>
      <td>国教</td>
      <td>4271</td>
    </tr>
    <tr>
      <th>4017</th>
      <td>事情</td>
      <td>4187</td>
    </tr>
    <tr>
      <th>2416</th>
      <td>不是</td>
      <td>3760</td>
    </tr>
    <tr>
      <th>1568</th>
      <td>三十六</td>
      <td>3686</td>
    </tr>
    <tr>
      <th>17320</th>
      <td>学院</td>
      <td>3575</td>
    </tr>
    <tr>
      <th>21252</th>
      <td>徐有容</td>
      <td>3339</td>
    </tr>
    <tr>
      <th>21111</th>
      <td>很多</td>
      <td>3017</td>
    </tr>
    <tr>
      <th>35031</th>
      <td>现在</td>
      <td>2969</td>
    </tr>
    <tr>
      <th>543</th>
      <td>一个</td>
      <td>2819</td>
    </tr>
    <tr>
      <th>41362</th>
      <td>能够</td>
      <td>2416</td>
    </tr>
    <tr>
      <th>12033</th>
      <td>可能</td>
      <td>2403</td>
    </tr>
    <tr>
      <th>20163</th>
      <td>应该</td>
      <td>2352</td>
    </tr>
    <tr>
      <th>5529</th>
      <td>仿佛</td>
      <td>2170</td>
    </tr>
    <tr>
      <th>36634</th>
      <td>看到</td>
      <td>2137</td>
    </tr>
  </tbody>
</table>
</div>




```python
back_coloring = imread(u"docs/assets/mask.jpg")

wordcloud = WordCloud(font_path=u"docs/assets/wqywmh.ttf", background_color="white", mask=back_coloring)
plt.axis("off")
wordcloud = wordcloud.fit_words(dict([(s, g) for s, g in segStat.head(20).itertuples(index=False)]))

plt.imshow(wordcloud)
```




    <matplotlib.image.AxesImage at 0x2c0c23986a0>




![png](gensim_files/gensim_10_1.png)



```python
import jieba.posseg as pseg

words = pseg.cut(book[:100])
for w in words:
    if w.flag == 'nr':
        print(w.word, w.flag, w)
```

    玄幻 nr 玄幻/nr
    石自 nr 石自/nr
    孤儿 nr 孤儿/nr
    


```python
sentences = []
for line in lines:
    words = list(jieba.cut(line))
    sentences.append(words)
```


```python
考虑更加细腻的sentence
```


```python
import gensim
model = gensim.models.Word2Vec(sentences,size=200,window=5,min_count=5,workers=4)
```

    2017-12-25 15:56:37,001 : INFO : collecting all words and their counts
    2017-12-25 15:56:37,003 : INFO : PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
    2017-12-25 15:56:37,065 : INFO : PROGRESS: at sentence #10000, processed 260578 words, keeping 18494 word types
    2017-12-25 15:56:37,123 : INFO : PROGRESS: at sentence #20000, processed 542096 words, keeping 27616 word types
    2017-12-25 15:56:37,177 : INFO : PROGRESS: at sentence #30000, processed 847726 words, keeping 35161 word types
    2017-12-25 15:56:37,232 : INFO : PROGRESS: at sentence #40000, processed 1116154 words, keeping 40354 word types
    2017-12-25 15:56:37,275 : INFO : PROGRESS: at sentence #50000, processed 1347282 words, keeping 44113 word types
    2017-12-25 15:56:37,325 : INFO : PROGRESS: at sentence #60000, processed 1570782 words, keeping 47817 word types
    2017-12-25 15:56:37,376 : INFO : PROGRESS: at sentence #70000, processed 1784533 words, keeping 51083 word types
    2017-12-25 15:56:37,415 : INFO : PROGRESS: at sentence #80000, processed 1967748 words, keeping 53288 word types
    2017-12-25 15:56:37,450 : INFO : collected 55415 word types from a corpus of 2123096 raw words and 87625 sentences
    2017-12-25 15:56:37,451 : INFO : Loading a fresh vocabulary
    2017-12-25 15:56:37,599 : INFO : min_count=5 retains 16452 unique words (29% of original 55415, drops 38963)
    2017-12-25 15:56:37,600 : INFO : min_count=5 leaves 2060983 word corpus (97% of original 2123096, drops 62113)
    2017-12-25 15:56:37,749 : INFO : deleting the raw counts dictionary of 55415 items
    2017-12-25 15:56:37,752 : INFO : sample=0.001 downsamples 40 most-common words
    2017-12-25 15:56:37,754 : INFO : downsampling leaves estimated 1500692 word corpus (72.8% of prior 2060983)
    2017-12-25 15:56:37,755 : INFO : estimated required memory for 16452 words and 200 dimensions: 34549200 bytes
    2017-12-25 15:56:37,845 : INFO : resetting layer weights
    2017-12-25 15:56:38,082 : INFO : training model with 4 workers on 16452 vocabulary and 200 features, using sg=0 hs=0 sample=0.001 negative=5 window=5
    2017-12-25 15:56:39,091 : INFO : PROGRESS: at 13.98% examples, 1130832 words/s, in_qsize 7, out_qsize 0
    2017-12-25 15:56:40,094 : INFO : PROGRESS: at 32.37% examples, 1257027 words/s, in_qsize 7, out_qsize 0
    2017-12-25 15:56:41,099 : INFO : PROGRESS: at 49.64% examples, 1273680 words/s, in_qsize 7, out_qsize 0
    2017-12-25 15:56:42,103 : INFO : PROGRESS: at 68.29% examples, 1303971 words/s, in_qsize 7, out_qsize 0
    2017-12-25 15:56:43,108 : INFO : PROGRESS: at 86.07% examples, 1302467 words/s, in_qsize 7, out_qsize 0
    2017-12-25 15:56:43,882 : INFO : worker thread finished; awaiting finish of 3 more threads
    2017-12-25 15:56:43,884 : INFO : worker thread finished; awaiting finish of 2 more threads
    2017-12-25 15:56:43,891 : INFO : worker thread finished; awaiting finish of 1 more threads
    2017-12-25 15:56:43,898 : INFO : worker thread finished; awaiting finish of 0 more threads
    2017-12-25 15:56:43,899 : INFO : training on 10615480 raw words (7502987 effective words) took 5.8s, 1291613 effective words/s
    


```python
for k, s in model.most_similar(positive=[u"圣后"]):
    print(k, s)
```

    e:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: DeprecationWarning: Call to deprecated `most_similar` (Method will be removed in 4.0.0, use self.wv.most_similar() instead).
      """Entry point for launching an IPython kernel.
    2017-12-25 15:58:02,828 : INFO : precomputing L2-norms of word weight vectors
    

    胜雪 0.806833028793335
    皇后 0.7890491485595703
    承武 0.7554647326469421
    承文 0.7192850708961487
    家 0.7164201736450195
    沾衣 0.6703976392745972
    朝 0.6002320051193237
    先帝 0.5968987345695496
    白帝 0.5489777326583862
    旨意 0.5485547184944153
    
