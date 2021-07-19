import re
from typing import List, Optional
from textacy import preprocessing

class TextPreprocessor:

    def __init__(self, pipelines: List, remove_chars: Optional[List] = None):
        self.pipelines = list()
        if "unicode" in pipelines: self.pipelines.append(preprocessing.normalize.unicode)
        if "whitespace" in pipelines: self.pipelines.append(preprocessing.normalize.whitespace)
        self.preprocessor = preprocessing.make_pipeline(*self.pipelines)

        self.remove_chars = remove_chars

    def run(self, text):
        for pattern in self.remove_chars:
            text = re.sub(pattern, ' ', text)
        text = self.preprocessor(text)
        return text
    
if __name__ == "__main__":
    text_preprocessor = TextPreprocessor(pipelines=["whitespace", "unicode"], remove_chars=["\n"])
    text = "Hello    this is text example.\n"
    assert "Hello this is text example." == text_preprocessor.run(text)