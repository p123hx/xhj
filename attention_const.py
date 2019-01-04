# _*_ coding: UTF-8 _*_

HIDDEN_SIZE = 2048                  # LSTM的隐藏层规模。
DECODER_LAYERS = 2                     # 解码器中LSTM结构的层数。这个例子中编码器固定使用单层的双向LSTM。
SRC_VOCAB_SIZE = 6750                  # 源语言词汇表大小。
TRG_VOCAB_SIZE = 6750                  # 目标语言词汇表大小。
BATCH_SIZE = 35                      # 训练数据batch的大小。
NUM_EPOCH = 30                         # 使用训练数据的轮数。
KEEP_PROB = 0.99                        # 节点不被dropout的概率。
MAX_GRAD_NORM = 5                      # 用于控制梯度膨胀的梯度大小上限。
SHARE_EMB_AND_SOFTMAX = True           # 在Softmax层和词向量层之间共享参数。

MAX_LEN = 50   # 限定句子的最大单词数量。

# for seq2seq model
NUM_LAYERS = 2                         # 深层循环神经网络中LSTM结构的层数。

# 词汇表中<sos>和<eos>的ID。在解码过程中需要用<sos>作为第一步的输入，并将检查
# 是否是<eos>，因此需要知道这两个符号的ID。
SOS_ID = 1
EOS_ID = 2
