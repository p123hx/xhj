# _*_ coding: UTF-8 _*_
from datetime import datetime
import tensorflow as tf
import traceback
from xhj_preprocess import VocabUtil
from attention_test import NMTModel



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
    #0:00:00.050704from datetime import datetime

    # 建立解码所需的计算图。


    t=datetime.now()
    output_op = model.inference(text_ids)
    print(datetime.now()-t)


    sess = tf.Session()


    saver = tf.train.Saver()


    saver.restore(sess, checkpoint_path)
    #0:00:01.142495

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


def talk_word(checkpoint_path):
    vocabutil = VocabUtil("./data/xhj.vocab")
    #time consumption: 00.002783
    try:
        while True:
            q = _prompt_input()
            if q.lower() == 'exit':
                break
            ids = vocabutil.get_ids_word(q)
            #time_consumption:00.000027

            answer = get_answer(ids, checkpoint_path)
            #time_consumption:03.158722

            answer = vocabutil.get_text(answer)
            #modi_answser 0:00:00.000066

            print(answer)

    except Exception:
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Ctrl+c exit.")


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
