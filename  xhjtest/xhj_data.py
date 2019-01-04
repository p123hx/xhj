# _*_ coding: UTF-8 _*_

import re
import sys
import codecs
import collections
from operator import itemgetter


def prepare():
    with open("./data/xiaohuangji50w_nofenci.conv") as fopen:
        reg = re.compile("E\nM (.*?)\nM (.*?)\n")
        match_dialogs = re.findall(reg, fopen.read())
        dialogs = match_dialogs
        print("dialog len", len(match_dialogs))

        questions = []
        answers = []
        for que, ans in dialogs:
            questions.append(que)
            answers.append(ans)

        save(questions, "./data/xhj_q")
        save(answers, "./data/xhj_a")

def save(dialogs, file):
    with open(file, "w") as fopen:
        fopen.write("\n".join(dialogs))

def question_len(filepath):
    counter = collections.Counter()
    linecount = 0
    with codecs.open(filepath, "r", "utf-8") as f:
        for line in f:
            line = line.strip()
            counter[len(line)] += 1
            linecount += 1
    print(linecount)
    return counter

def question_dup(filepath):
    counter = collections.Counter()
    linecount = 0
    with codecs.open(filepath, "r", "utf-8") as f:
        for line in f:
            line = line.strip()
            counter[line] += 1
            linecount += 1
    print(linecount)
    return counter

'''
小黄鸡数据统计
xhj_q
总行数: 454051
唯一行数: 220471
重复行数: 67702
重复总数: 233580 
重复总数-重复行数: 165878

xhj_a
总行数: 454051
唯一行数: 45104
重复行数: 57864
重复总数: 408947
重复总数-重复行数: 351083

'''

def dupquestions():
    question_file = "./data/xhj_q"

    allquestion = {}
    linecount = 0
    with codecs.open(question_file, "r", "utf-8") as f:
        for line in f:
            line = line.strip()

            if line in allquestion:
                allquestion[line].append(linecount)
            else:
                allquestion[line] = [linecount, ]
            linecount += 1

    return allquestion

def print_dupquesiton(text):
    question_file = "./data/xhj_q"
    answer_file = "./data/xhj_a"
    with codecs.open(question_file, "r", "utf-8") as f:
        allquestion = [w.strip() for w in f.readlines()]

    with codecs.open(answer_file, "r", "utf-8") as f:
        allanswer = [w.strip() for w in f.readlines()]

    answers = dupquestions()[text]
    for i in answers:
        print(i, allquestion[i], allanswer[i])

def unique_question():
    dupq = dupquestions()

    # question_file = "./data/xhj_q"
    answer_file = "./data/xhj_a"
    # with codecs.open(question_file, "r", "utf-8") as f:
    #     allquestion = [w.strip() for w in f.readlines()]

    with codecs.open(answer_file, "r", "utf-8") as f:
        allanswer = [w.strip() for w in f.readlines()]

    from random import randint

    questions = []
    answers = []
    for q, indexs in dupq.items():
        questions.append(q)

        # a = randint(0, len(indexs) - 1)
        # answers.append(allanswer[indexs[a]])

        answers.append(allanswer[indexs[0]])

    save(questions, "./data/xhj_q_s")
    save(answers, "./data/xhj_a_s")
    print(len(questions), len(answers))

if __name__ == '__main__':
    # counter = question_len("./data/xhj_q")
    # sorted_counter = sorted(counter.items(), key=itemgetter(1), reverse=True)
    # print(sum([v for _, v in sorted_counter if _ > 50]))

    counter = question_dup("./data/xhj_a_s")
    dupline = 0
    dupsum = 0
    single = 0
    for k, v in counter.items():
        if v > 1:
            dupline += 1
            dupsum += v
            # print(k, ":", v)
        else:
            single += 1
    print(single, dupline, dupsum, dupsum-dupline)

    #
    # sorted_counter = sorted(counter.items(), key=itemgetter(1), reverse=True)
    # for k, v in sorted_counter[:100]:
    #     print(k, v)

    # print_dupquesiton("小通")



    pass

