# ChinesePoemGeneration

## Version 1.0
现在实现了最简单的RNN生成，问题主要在各种*重复*。

good case：
> 春风吹柳柳，花花拂翠旒。宝房开玉圃，香蕊拂金杯。

bad case:
> 春风吹吹吹，吹吹入浦洲。白云飞羽翼，飞飞凤凰飞。

实际上训练的时候并没有断句，也没有考虑押韵之类的。

## Version 1.1
使用了GRU，加入了positional embedding，并且在生成过程中依概率在top5中选字

some cases:
> 春草迷津路，闲人倚地阴，无人墓山下，归去老家城。
> 春来无处去，一树逐秋鹰，不敢求己是，非道有人心。
