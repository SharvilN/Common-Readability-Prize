import numpy as np
import pandas as pd
import spacy


if __name__ == '__main__':
    pass

def spacy_features(df: pd.DataFrame):
    """
    This function generates features using spacy en_core_wb_lg
    I learned about this from these resources:
    https://www.kaggle.com/konradb/linear-baseline-with-cv
    https://www.kaggle.com/anaverageengineer/comlrp-baseline-for-complete-beginners
    """
    
    nlp = spacy.load('en_core_web_lg')
    with nlp.disable_pipes():
        vectors = np.array([nlp(text).vector for text in df.excerpt])
        
    return vectors

def get_spacy_col_names():
    names = list()
    for i in range(300):
        names.append(f"spacy_{i}")
        
    return names