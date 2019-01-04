# _*_ coding: UTF-8 _*_

import tensorflow as tf
from _datetime import datetime
import sys
import traceback

from xhj_preprocess import VocabUtil
from attention_const import HIDDEN_SIZE, SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, SHARE_EMB_AND_SOFTMAX, \
    SOS_ID, EOS_ID, NUM_LAYERS


# 定义NMTModel类来描述模型。
class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量。
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构。
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)])

        # 为源语言和目标语言分别定义词向量。
        self.src_embedding = tf.get_variable(
            "src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable(
            "trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(
                "weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable(
            "softmax_bias", [TRG_VOCAB_SIZE])

    def inference(self, src_input):
        # 虽然输入只有一个句子，但因为dynamic_rnn要求输入是batch的形式，因此这里
        # 将输入句子整理为大小为1的batch。
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        # 使用dynamic_rnn构造编码器。这一步与训练时相同。
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32)

        # 设置解码的最大步数。这是为了避免在极端情况出现无限循环的问题。
        MAX_DEC_LEN = 100

        with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
            # 使用一个变长的TensorArray来存储生成的句子。
            init_array = tf.TensorArray(dtype=tf.int32, size=0,
                                        dynamic_size=True, clear_after_read=False)
            # 填入第一个单词<sos>作为解码器的输入。
            init_array = init_array.write(0, SOS_ID)
            # 构建初始的循环状态。循环状态包含循环神经网络的隐藏状态，保存生成句子的
            # TensorArray，以及记录解码步数的一个整数step。
            init_loop_var = (enc_state, init_array, 0)

            # tf.while_loop的循环条件：
            # 循环直到解码器输出<eos>，或者达到最大步数为止。
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), EOS_ID),
                    tf.less(step, MAX_DEC_LEN - 1)))

            def loop_body(state, trg_ids, step):
                # 读取最后一步输出的单词，并读取其词向量。
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,
                                                 trg_input)
                # 这里不使用dynamic_rnn，而是直接调用dec_cell向前计算一步。
                dec_outputs, next_state = self.dec_cell.call(
                    state=state, inputs=trg_emb)
                # 计算每个可能的输出单词对应的logit，并选取logit值最大的单词作为
                # 这一步的而输出。
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                logits = (tf.matmul(output, self.softmax_weight)
                          + self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                # 将这一步输出的单词写入循环状态的trg_ids中。
                trg_ids = trg_ids.write(step + 1, next_id[0])
                return next_state, trg_ids, step + 1

            # 执行tf.while_loop，返回最终状态。
            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()


def main(test_text, checkpoint_path):
    tf.reset_default_graph()

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("nmt_model", reuse=None):
        model = NMTModel()

        # 定义个测试句子。
    #     test_en_text = "This is a test . <eos>"
    #     print(test_en_text)

    vocabutil = VocabUtil("./data/xhj.vocab")

    # 根据英文词汇表，将测试句子转为单词ID。
    test_ids = vocabutil.get_ids_word(test_text)
    print(test_ids)

    # 建立解码所需的计算图。
    output_op = model.inference(test_ids)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # 读取翻译结果。
    output_ids = sess.run(output_op)
    print(output_ids)

    # 根据中文词汇表，将翻译结果转换为中文文字。
    answer = vocabutil.get_text(output_ids)

    # 输出翻译结果。
    print(datetime.now(), answer.encode('utf8').decode(sys.stdout.encoding))
    sess.close()

def _prompt_input():
    prompt = datetime.now().strftime("%H:%M:%S > ")
    q = input(prompt)
    q = q.strip()
    return q

def get_answer(text_ids, checkpoint_path):
    tf.reset_default_graph()

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("nmt_model", reuse=None):
        model = NMTModel()

    # 建立解码所需的计算图。
    output_op = model.inference(text_ids)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # 读取翻译结果。
    output_ids = sess.run(output_op)
    print(output_ids)

    sess.close()
    return output_ids

def talk_word(checkpoint_path):

    vocabutil = VocabUtil("./data/xhj.vocab")

    try:
        while True:
            q = _prompt_input()
            if q.lower() == 'exit':
                break

            ids = vocabutil.get_ids_word(q)

            output_ids = get_answer(ids, checkpoint_path)
            answer = vocabutil.get_text(output_ids)
            print(answer)

    except Exception:
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Ctrl+c exit.")

if __name__ == '__main__':
    checkpoint_path = "./model/xhj_seq2seq_ckpt-14210"

    # main("你是谁", checkpoint_path)

    talk_word(checkpoint_path)

    pass

