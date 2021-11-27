import json
import torch

from wabot.SimpleSenderPredictor import SimpleSenderPredictor

from wabot.SimpleSenderPredictor import SimpleSenderPredictor
from wabot.parser import get_messages


def predict():
    model = SimpleSenderPredictor.load_from_checkpoint('model.ckpt')
    msgs = get_messages()

    senders = json.load(open('senders.json'))
    senders = {v: k for k, v in senders.items()}

    with torch.no_grad():
        model.eval()
        with open('predictions.txt', 'w') as f:
            for msg in msgs:
                tokens = model.text_encoder.tokenize(msg.text)
                if 10 <= len(tokens) <= 20:
                    pred = model(tokens)
                    v, i = pred.max(0)
                    if v > .9:
                        print(msg.text, senders[i.item()], msg.sender, file=f)


if __name__ == '__main__':
    predict()
