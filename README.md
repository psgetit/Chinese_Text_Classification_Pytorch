# Chinese_Text_Classification_Pytorch

**分类任务怎么破！！！快来看看！！！实测复杂任务(61分类且数据不均匀)准确率高达92.37%！！！**

p.s.以上准确率任务是在基于此代码进行一部分优化情况下达成，此开源仓库为base版本，有任何问题欢迎issue

中文文本分类，基于pytorch使用ERNIE_3.0_base(中文文本分类之王)，新人友好，开箱即用。

## 介绍
机器：一块4090 ， 训练时间：15分钟。  

3060(6G显存笔记本版本)也能跑，慢一些，在model/ERINE.py降低batch_size就好了

因此理论上降低batch都可以跑，下限没测过



## 环境
pip install -r requirements.txt


## 数据集
实习工作中的数据不方面给出(总共2w条)

按指定格式制作数据集train.txt,test.txt,dev.txt,class.txt放在文件夹Dataset下

train.txt,test.txt,dev.txt内数据格式为：

中文文本    标签(int)

中文文本    标签(int)

中文文本    标签(int)

......

class.txt内数据格式为：

类别1

类别2

类别3

......



联系为标签(0)对应class.txt里的第一行类别1，如此递增

## 预训练语言模型
预训练模型放在 ERINE_pretain目录下，目录下是三个文件：
 - pytorch_model.bin  

 - bert_config.json  

 - vocab.txt  

   

网盘地址：链接：https://pan.baidu.com/s/1IQ_GDXgWXJFJI9Bfr15hzQ?pwd=42vx 

如果你可以科学上网，可以考虑进入huggingface[nghuyong (HuYong) (huggingface.co)](https://huggingface.co/nghuyong)选择适合你的ERNIE版本

解压后，按照上面说的放在对应目录下，文件名称确认无误即可。  

## 使用说明
下载好预训练模型放入指定位置就可以跑了。
```shell
# 训练并测试：
python run.py 
# 使用模型进行推理
python predict.py
```



## 未完待续
后续可能会更新...
