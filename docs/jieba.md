

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import jieba.posseg as pseg
import jieba.analyse
```


```python
jieba.add_word(u"陈长生",tag="nz")
jieba.add_word(u"徐有容",tag="nz")
jieba.add_word(u"落落",tag="nz")
jieba.add_word(u"小黑龙",tag="nz")

with open("docs/assets/择天记.txt",encoding="utf-8") as fp:
    book = fp.read()
    
with open("docs/assets/择天记.txt",encoding="utf-8") as fp:
    lines = [line for line in fp.readlines() if len(line) > 2]
```


```python
book[:100]
```




    '择天记\n\n猫腻\n\n玄幻奇幻\n\n太始元年，有神石自太空飞来，分散落在人间，其中落在东土大陆的神石，上面镌刻着奇怪的图腾，人因观其图腾而悟道，后立国教。\n数千年后，十四岁的少年孤儿陈长生，为治病改命离开自'




```python
lines[50:55]
```




    ['就是这样轻微的接触，便产生了极为剧烈的变化——黄金巨龙眼瞳深处的两粒神火，轰的一声散开，变成万千星辰，那片星辰海洋里，赤裸裸地流露出冷酷而贪婪的欲望！\n',
     '那份欲望，是赞美，是动容。\n',
     '是对生命的赞美，是因为生命而动容。\n',
     '是生命最原始的渴望。\n',
     '黄金巨龙看着溪上的木盆，张开了嘴，龙息如碎玉般倾渲而出。\n']




```python
for line in lines[50:55]:
    words = jieba.cut(line)  # 默认模式
    print(" ".join(words))

    poswords = [(word, tag) for (word, tag) in pseg.cut(line)][:-1]  # 词性分词
    print("  ".join(["%s/%s" % (word, tag) for (word, tag) in poswords]))

    labels = [word for word in jieba.analyse.extract_tags(line, topK=12)]  # 提取标签
    print(" ".join(labels))
    
    print("="*50)
```

    就是 这样 轻微 的 接触 ， 便 产生 了 极为 剧烈 的 变化 — — 黄金 巨龙 眼瞳 深处 的 两粒 神火 ， 轰的一声 散开 ， 变成 万千 星辰 ， 那片 星辰 海洋 里 ， 赤裸裸 地 流露出 冷酷 而 贪婪 的 欲望 ！ 
    
    就是/d  这样/r  轻微/d  的/uj  接触/v  ，/x  便/d  产生/n  了/ul  极为/d  剧烈/a  的/uj  变化/vn  —/x  —/x  黄金/n  巨龙/nr  眼瞳/v  深处/s  的/uj  两粒/m  神火/n  ，/x  轰的一声/i  散开/v  ，/x  变成/v  万千/m  星辰/n  ，/x  那/r  片/q  星辰/n  海洋/ns  里/f  ，/x  赤裸裸/z  地/uv  流露出/i  冷酷/a  而/c  贪婪/a  的/uj  欲望/v  ！/x
    星辰 眼瞳 轰的一声 两粒 神火 巨龙 赤裸裸 万千 冷酷 散开 流露出 贪婪
    ==================================================
    那 份 欲望 ， 是 赞美 ， 是 动容 。 
    
    那/r  份/q  欲望/v  ，/x  是/v  赞美/ns  ，/x  是/v  动容/n  。/x
    动容 赞美 欲望
    ==================================================
    是 对 生命 的 赞美 ， 是因为 生命 而 动容 。 
    
    是/v  对/p  生命/vn  的/uj  赞美/ns  ，/x  是因为/c  生命/vn  而/c  动容/n  。/x
    生命 动容 赞美 是因为
    ==================================================
    是 生命 最 原始 的 渴望 。 
    
    是/v  生命/vn  最/d  原始/v  的/uj  渴望/v  。/x
    渴望 原始 生命
    ==================================================
    黄金 巨龙 看着 溪上 的 木盆 ， 张开 了 嘴 ， 龙息 如 碎玉 般倾 渲而出 。 
    
    黄金/n  巨龙/nr  看着/v  溪/n  上/f  的/uj  木盆/n  ，/x  张开/nr  了嘴/v  ，/x  龙息/n  如/c  碎玉/n  般/u  倾/v  渲/vg  而/c  出/v  。/x
    溪上 龙息 般倾 渲而出 碎玉 木盆 巨龙 张开 黄金 看着
    ==================================================
    
