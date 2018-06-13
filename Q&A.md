# 1. fpn scales assignment

box -> (fmap, center, scale, ratio)

1. single-scale ~~~ level-fmaps

fmap size bianxiao, scale-box size ye tonshi bianxiao!
 
scale yu fmap qiangxing ouhe! yi dui yi! 

2. level-scales ~~~ single-fmap

keyi wanmei fugai image shang de daxiao wuti !

zong you yige box neng qiahao zhaozhu wuti !

3. level-scales ~~~ level-fmaps

an fmap de bi li tiaozheng scales !

fei chang rong yu !

'特征层levels与scales不相同！'

4. x2 and y2 should not be part of the box. Increment by 1.  
    todo??? over border?
    
5. Conv(padding=Same)   SamePad2d()