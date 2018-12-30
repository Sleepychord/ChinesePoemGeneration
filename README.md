# ChinesePoemGeneration

使用了循环神经网络（GRU）和position embedding生成古诗。输入第一个汉字，生成剩余诗句，并且在生成过程中依概率在top5中选字。

## 基本框架

将古诗生成看成一个auto-regressive过程，下一个字的生成只与前面的字有关。将古诗看成plain character sequence，并使用position embedding来标志在句子中的位置。

![](data/model.png)

position embedding是[《Attention is all you need》](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)文章中提出的方法，汉字的embedding使用古代的四库全书训练的[(paper_link)](http://aclweb.org/anthology/P18-2023)。

## Current Version

生成样例（以春为例）:

> 春风飘雨霁，天地已无尘，水色连山色，山深水上清。
>
> ​	最相近诗句：朝来微有雨,天地爽无尘
>
> 春来江外事，何处觅人家，月落江山雨，秋风夜雨晴。
>
> 春风流水出，烟火柳营依，野店依依旧，山村夜宿稀。

> 春草迷津路，闲人倚地阴，无人墓山下，归去老家城。

> 春来无处去，一树逐秋鹰，不敢求己是，非道有人心。
>
> 春风吹竹风，高柳柳丝重，水落流沙外，风轻旅思通。
>
> 春色生春草，新晴似旧情，风流初觉夜，花落自伤情。
>
> 春草连天外，风波入帝乡，风摇天地落，月满水云长。

## 缺点和改进方向

1. 部分叠字会出现连环重叠现象
2. position embedding仅仅考虑了一句内的情况，导致前后句可能出现意象重复。

