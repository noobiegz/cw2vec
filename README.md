# cw2vec 的 TensorFlow 实现

cw2vec 是一种基于 skip-gram，并辅以笔画信息来训练中文词向量的模型，详见文章：https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17444/16786

**声明**：此实现非 cw2vec 的官方实现，所有观点皆非 cw2vec 官方观点，如有错误请指正。

此实现力求忠于原文，除了以下：

1. 相似度的计算：此代码在原文的基础上，乘了`1 / |S(w)|`，这是因为在使用原文的公式时，观察到了嵌入中出现 NaN 的现象。
2. 在 skip window 中采样使用的是 `random.sample()`，原文未作说明。
3. batch 的处理并没有使用断句，原文未作说明。
4. 对于笔画长度太短的词，代码进行了 padding，原文未作说明，（这样的词较少）。

## 关于 commit 的数量
此代码库只有为数不多的几个 commit，这是因为内部的代码库使用了 Git LFS，有许多大文件，所以使用了新的代码库。

## 训练方法

```
git clone https://github.com/noobiegz/cw2vec.git
cd cw2vec

conda create -n cw2vec python=3.6
source activate cw2vec
conda install --file requirements.txt

# 编译，注意，这里要求 gcc >= 5，这包括用来编译 TensorFlow 的 gcc。
# 对于更早的 gcc 的支持请参照：
# https://www.tensorflow.org/extend/adding_an_op
./compile.sh

# 详细参数（如训练语料，超参数等）请参见 train.py
CUDA_VISIBLE_DEVICES='' python -m train --cmd=train
```

## 训练好的词向量（仍有提升空间）
https://github.com/noobiegz/cw2vec/releases

## 可提升的地方

- 运算速度
- 断句
- 参数调优

## 一个**不严谨**的比较

受限于时间预算，并没有用相同的参数、分词方式及语料训练各个模型来进行比较。下面的比较中：
- cw2vec: 使用本库 commit 8f789ed 训练，（其中包含详细参数），语料为中文维基百科，epochs=8
- word2vec: 使用 gensim 的 word2vec 训练，语料为中文维基百科（与 cw2vec 非同一份，但接近），epochs=5
- score: cosine 相似度

与“森林”相似的词：

 |cw2vec score   |cw2vec word   |word2vec word   |word2vec score
 |-------------- |------------- |--------------- |----------------
 |      0.750604 |林地          |对诺定          |        0.686841
 |      0.724008 |湿地          |沼泽            |        0.686402
 |      0.700475 |沼泽          |林地            |        0.677119
 |      0.697006 |雨林          |布雷顿          |        0.674078
 |      0.688
