from torch import Tensor
from dataclasses import dataclass
from wabot.SimpleSenderPredictor import SimpleSenderPredictor
from wabot.parser import Msg, get_messages, get_sessions


@dataclass(frozen=True)
class Prediction(Msg):
    pred: Tensor


def quiz():
    model = SimpleSenderPredictor.load_from_checkpoint('model.ckpt')
    msgs = get_messages()

    predictions = [
        Prediction(
            pred=model.predict(m.text),
            time=m.time,
            sender=m.sender,
            text=m.text,)
        for m in msgs
        if len(model.text_encoder.tokenize(m.text))]

    with open('quiz.txt', 'w', encoding='utf-8') as f:
        for s in get_sessions(predictions):
            if 2 <= len(s) <= 10:

                print(file=f)
                print('=' * 10, file=f)
                for m in s:
                    print(m, file=f)


if __name__ == '__main__':
    quiz()
