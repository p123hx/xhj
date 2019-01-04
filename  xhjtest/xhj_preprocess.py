# _*_ coding: UTF-8 _*_
import jieba
import codecs
import collections
from operator import itemgetter

SPACE_CHARS = ["\u00A0", "\u0020", "\u3000", chr(12288), chr(20), "\u0009", "\u000a", "\u000d", "\u2006", "\u2005"]

def _is_space(word):
    return word in SPACE_CHARS

def _cut(line):
    '''
    调用jieba分词
    :param line:
    :return:
    '''
    line = line.strip()
    return [word for word in jieba.cut(line) if not _is_space(word)]

def _save_vocab(counter, vocab_file):
    '''
    根据统计的词频, 创建字典
    :param vocab_file:
    :return:
    '''

    # 按词频顺序对单词进行排序。
    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
    print("vocab len", len(sorted_words))

    for i in range(10):
        print(sorted_words[i], counter[sorted_words[i]])

    with codecs.open(vocab_file, 'w', 'utf-8') as file_output:
        for word in sorted_words:
            file_output.write(word + "\n")

def _get_vocabs(raw_data_file, vocabdict=None):
    '''
    调用jieba分词, 按词的方式创建字典
    :param raw_data_file:
    :param vocabdict:
    :return:
    '''
    if vocabdict is None:
        counter = collections.Counter()
    else:
        counter = vocabdict

    with codecs.open(raw_data_file, "r", "utf-8") as f:
        for line in f:
            for word in _cut(line):
                counter[word] += 1
    return counter

def _get_vocabs_word(raw_data_file, vocabdict=None):
    '''
    按字的方式分割字符, 按字的方式创建字典
    :param raw_data_file:
    :param vocabdict:
    :return:
    '''
    if vocabdict is None:
        counter = collections.Counter()
    else:
        counter = vocabdict

    with codecs.open(raw_data_file, "r", "utf-8") as f:
        for line in f:
            line = line.strip()
            for word in line:
                if not _is_space(word):
                    counter[word] += 1
    return counter


def parse_xhj_all_vocab():
    '''
    解析小黄鸡的语料数据, 生成字典, 问题和答案一个字典
    :return:
    '''
    a_raw_data_file = "./data/xhj_a"
    q_raw_data_file = "./data/xhj_q"

    counter = _get_vocabs(a_raw_data_file)
    counter = _get_vocabs(q_raw_data_file, counter)

    vocab_file = "./data/xhj.vocab"
    _save_vocab(counter, vocab_file)

def create_xhj_vocab(raw_file, vocab_file):
    '''创建独立的字典'''
    counter = _get_vocabs(raw_file)
    _save_vocab(counter, vocab_file)

def convert_xhj_data_to_ids(vocab_file, raw_file, output_file):
    # 读取词汇表，并建立词汇到单词编号的映射。
    with codecs.open(vocab_file, "r", "utf-8") as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    # 如果出现了不在词汇表内的低频词，则替换为"unk"。
    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

    fin = codecs.open(raw_file, "r", "utf-8")
    fout = codecs.open(output_file, 'w', 'utf-8')
    for line in fin:
        # jieba分词, 读取单词并添加<eos>结束符
        words = _cut(line) + ["<eos>"]
        # 将每个单词替换为词汇表中的编号
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        fout.write(out_line)
    fin.close()
    fout.close()

def convert_xhj_data_to_ids_byword(vocab_file, raw_file, output_file):
    '''
    按字的方式转换文办到ids, 读取词汇表，并建立字到单词编号的映射。
    :param vocab_file:
    :param raw_file:
    :param output_file:
    :return:
    '''
    with codecs.open(vocab_file, "r", "utf-8") as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    # 如果出现了不在词汇表内的低频词，则替换为"unk"。
    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

    fin = codecs.open(raw_file, "r", "utf-8")
    fout = codecs.open(output_file, 'w', 'utf-8')
    for line in fin:
        # 按词分割并添加<eos>结束符
        line = line.strip()
        words = [word for word in line if not _is_space(word)] + ["<eos>"]
        # 将每个单词替换为词汇表中的编号
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        fout.write(out_line)
    fin.close()
    fout.close()


def preprocess1():
    parse_xhj_all_vocab()

    vocab_file = "./data/xhj.vocab"
    a_raw_file = "./data/xhj_a"
    a_out_file = "./data/xhj_a_train"
    q_raw_file = "./data/xhj_q"
    q_out_file = "./data/xhj_q_train"

    convert_xhj_data_to_ids(vocab_file, a_raw_file, a_out_file)
    convert_xhj_data_to_ids(vocab_file, q_raw_file, q_out_file)

# def get_question_ids(text):
#     return get_vocab_ids(text, "./data/xhj_q.vocab")
#
# def get_answer_text(ids):
#     return get_vocab_text(ids, "./data/xhj_a.vocab")

def preprocess2():
    a_vocab_file = "./data/xhj_a.vocab"
    a_raw_file = "./data/xhj_a"
    a_out_file = "./data/xhj_a_train"

    q_vocab_file = "./data/xhj_q.vocab"
    q_raw_file = "./data/xhj_q"
    q_out_file = "./data/xhj_q_train"

    create_xhj_vocab(q_raw_file, q_vocab_file)
    create_xhj_vocab(a_raw_file, a_vocab_file)

    convert_xhj_data_to_ids(a_vocab_file, a_raw_file, a_out_file)
    convert_xhj_data_to_ids(q_vocab_file, q_raw_file, q_out_file)

def preprocess_word():
    '''
    按字的方式建字典
    :return:
    '''
    vocab_file = "./data/xhj.vocab"
    a_raw_file = "./data/xhj_a_s"
    a_out_file = "./data/xhj_a_train"
    q_raw_file = "./data/xhj_q_s"
    q_out_file = "./data/xhj_q_train"

    counter = _get_vocabs_word(q_raw_file)
    print("q vocab len", len(counter))
    counter = _get_vocabs_word(a_raw_file, counter)
    _save_vocab(counter, vocab_file)

    convert_xhj_data_to_ids_byword(vocab_file, a_raw_file, a_out_file)
    convert_xhj_data_to_ids_byword(vocab_file, q_raw_file, q_out_file)


class VocabUtil():
    def __init__(self, vocab_file):
        with codecs.open(vocab_file, "r", "utf-8") as f_vocab:
            self.vocab = [w.strip() for w in f_vocab.readlines()]

        self.word_to_id = {k: v for (k, v) in zip(self.vocab, range(len(self.vocab)))}
        self.vocab_file = vocab_file
        self.eos_id = self.word_to_id["<eos>"]
        self.unk_id = self.word_to_id["<unk>"]
        self.sos_id = self.word_to_id["<sos>"]

    def _get_id(self, word):
        return self.word_to_id[word] if word in self.word_to_id else self.unk_id

    def get_ids_jieba_cut(self, text):
        return [self._get_id(word) for word in _cut(text)] + [self.eos_id, ]

    def get_ids_word(self, text):
        text = text.strip()
        return [self._get_id(word) for word in text if not _is_space(word)] + [self.eos_id, ]

    def get_text(self, ids):
        result = []
        for x in ids:
            # 删除 <sos>, <eos>
            if x in [self.sos_id, self.eos_id]:
                continue
            result.append(self.vocab[x])

        return ''.join(result)

if __name__ == '__main__':
    # q_raw_file = "./data/xhj_q"
    # counter = _get_vocabs(q_raw_file)
    # word = "张慧婷"
    # print(word, counter.get(word))

    # vocabutil = VocabUtil("./data/xhj.vocab")
    # print(vocabutil.get_ids_word(chr(8197)))

    preprocess_word()

    pass






