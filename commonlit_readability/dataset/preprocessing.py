import re
from textacy import preprocessing

preproc = preprocessing.make_pipeline(
    preprocessing.normalize.unicode,
    preprocessing.normalize.whitespace,
    preprocessing.remove.accents
)

def preprocess(text):
    pattern = '\n'
    text = re.sub(pattern, ' ', text)
    text = preproc(text)
    return text