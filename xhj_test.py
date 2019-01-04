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
    tf.reset_default_graph()
    with tf.variable_scope("nmt_model", reuse=None):
        model = NMTModel()
    output_op = model.inference(text_ids)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    output_ids = sess.run(output_op)
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
            answer = get_answer(ids, checkpoint_path)
            answer = vocabutil.get_text(answer)
            print(answer)
    except Exception:
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Ctrl+c exit.")
if __name__ == "__main__":
    checkpoint_path = "./model/xhj_attention_ckpt-1600-200"
    talk_word(checkpoint_path)
    pass
