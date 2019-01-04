# _*_ coding: UTF-8 _*_

import tensorflow as tf
from datetime import datetime
import traceback

from xhj_preprocess import VocabUtil
from attention_test import NMTModel

def main(text, checkpoint_path):
    q_vocabutil = VocabUtil("./data/xhj_q.vocab")
    a_vocabutil =  VocabUtil("./data/xhj_a.vocab")

    tf.reset_default_graph()

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("nmt_model", reuse=None):
        model = NMTModel()

    print(datetime.now(), text)

    # 根据词汇表，将测试句子转为ids。
    text_ids = q_vocabutil.get_ids_word(text)
    print(datetime.now(), text_ids)

    # 建立解码所需的计算图。
    output_op = model.inference(text_ids)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # 读取翻译结果。
    output_ids = sess.run(output_op)
    print(datetime.now(), output_ids)

    output_text = a_vocabutil.get_text(output_ids)

    # 输出翻译结果。
    print(datetime.now(), output_text)
    sess.close()


def _prompt_input():
    prompt = datetime.now().strftime("%H:%M:%S > ")
    q = input(prompt)
    q = q.strip()
    return q

def get_answer(text_ids, checkpoint_path):
    # print("input ids", text_ids)

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
    # print("output ids:", output_ids)

    sess.close()
    return output_ids

class TestModel():
    def __init__(self, vocabfile, checkpoint_path):
        self.vocabutil = VocabUtil(vocabfile)

        tf.reset_default_graph()

        # 定义训练用的循环神经网络模型。
        with tf.variable_scope("nmt_model", reuse=tf.AUTO_REUSE):
            self.model = NMTModel()

        # 建立解码所需的计算图。
        text_ids = self.vocabutil.get_ids_word("你是谁")
        output_op = self.model.inference(text_ids)

        # self.saver = tf.train.import_meta_graph(checkpoint_path+".meta")
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, checkpoint_path)


        # self.sess.run(tf.global_variables_initializer())

        answer = self.vocabutil.get_text(self.sess.run(output_op))
        print(answer)

        # print(self.predict("你是谁"))

    def close(self):
        self.sess.close()

    def predict(self, q):
        # tf.reset_default_graph()
        text_ids = self.vocabutil.get_ids_word(q)
        output_op = self.model.inference(text_ids)
        self.sess.run(output_op)
        answer = self.vocabutil.get_text(output_op)
        return answer

def talk(checkpoint_path):

    q_vocabutil = VocabUtil("./data/xhj_q.vocab")
    a_vocabutil = VocabUtil("./data/xhj_a.vocab")

    try:
        while True:
            q = _prompt_input()
            if q.lower() == 'exit':
                break

            ids = q_vocabutil.get_ids_word(q)

            answer = get_answer(ids, checkpoint_path)
            answer = a_vocabutil.get_text(answer)
            print(answer)

    except Exception:
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Ctrl+c exit.")

def talk_word(checkpoint_path):
    """

    :rtype: object
    """

    vocabutil = VocabUtil("./data/xhj.vocab")

    #time consumption: 00.002783

    try:
        while True:
            q = _prompt_input()
            if q.lower() == 'exit':
                break
            t1 = datetime.now()
            ids = vocabutil.(q)
            print("time_vacabUtil", datetime.now() - t1)
            print(ids)
            answer = get_answer(ids, checkpoint_path)
            answer = vocabutil.get_text(answer)
            print(answer)

    except Exception:
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Ctrl+c exit.")

def talk_word_2(checkpoint_path):

    model = TestModel("./data/xhj.vocab", checkpoint_path)
    try:
        while True:
            q = _prompt_input()
            if q.lower() == 'exit':
                break

            answer = model.predict(q)
            print(answer)

    except Exception:
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Ctrl+c exit.")



def _gettextlines(filename):
    import codecs
    with codecs.open(filename, "r", "utf-8") as fin:
        return [w.strip() for w in fin.readlines()]

def check_xhj_precision(checkpoint_path):
    a_raw_file = "./data/xhj_a_train"
    q_raw_file = "./data/xhj_q_train"
    # vocabutil = VocabUtil("./data/xhj.vocab")

    questions = _gettextlines(q_raw_file)
    answers = _gettextlines(a_raw_file)

    for i in range(len(questions[:20])):
        q = questions[i]
        ids = [int(w) for w in q.split()]
        # ids = vocabutil.get_ids_word(q)
        a = get_answer(ids, checkpoint_path)
        # a = vocabutil.get_text(a)
        print("q:%s, a:%s, model:%s" % (q, answers[i], a))


if __name__ == "__main__":

    checkpoint_path = "./model/xhj_attention_ckpt-13801"

    # time consumption: .000007

    # check_xhj_precision(checkpoint_path)
    # talk_word(checkpoint_path)
    talk_word(checkpoint_path)

    # test_en_text = "大懒鸡"
    # test_en_text = "你有男朋友没"
    # main(test_en_text, checkpoint_path)
    pass
