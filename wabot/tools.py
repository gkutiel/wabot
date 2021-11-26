import fasttext
import json

from wabot.parser import parse, get_sessions


def word_embedding(chat_txt='chat.txt', model_bin='model.bin'):
    model = fasttext.train_unsupervised(
        chat_txt,
        model='cbow',
        dim=16,
        epoch=100)

    model.save_model(model_bin)
