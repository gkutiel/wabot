from wabot.parser import parse
from fire import Fire


def train():
    from wabot.train import train as main
    Fire(main)


def train_hp():
    from wabot.train import train_hp as main
    Fire(main)
