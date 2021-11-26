import fasttext
import numpy as np


def _test_model():
    model = fasttext.load_model('model.bin')
    assert model.get_dimension() == 16

    words, _ = model.get_line('המלצה לסרט טוב')
    assert words[0] == 'המלצה'
    assert words[1] == 'לסרט'
    assert words[2] == 'טוב'
