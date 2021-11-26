from wabot.parser import parse
from fire import Fire


def dump_text():
    from wabot.tools import dump_text as main
    Fire(main)


def word_embedding():
    from wabot.tools import word_embedding as main
    Fire(main)


def train_simple_predictor():
    from wabot.train import train_simple_predictor as main
    Fire(main)
