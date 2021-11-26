import fasttext
import json

from wabot.parser import parse, get_sessions


def get_messages(chat_txt='chat.txt'):
    lines = open(chat_txt).readlines()
    return list(parse(lines))


def word_embedding(chat_txt='chat.txt', model_bin='model.bin'):
    msgs = get_messages(chat_txt)
    msgs_txt = 'msgs.txt'

    with open(msgs_txt, 'w') as f:
        for msg in msgs:
            print(msg.text, file=f)

    model = fasttext.train_unsupervised(
        msgs_txt,
        model='cbow',
        dim=16)

    model.save_model(model_bin)
