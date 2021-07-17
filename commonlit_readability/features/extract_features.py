import spacy
import textacy
import argparse
import collections
import numpy as np
import pandas as pd

from textacy import text_stats
from textacy.text_stats import readability

from commonlit_readability.utils import utils
from commonlit_readability.dataset import preprocessing

class FeatureGenerator:

    def __init__(self, df: pd.DataFrame, spacy_model=None, preprocess: bool=True) -> None:
        self.raw = df
        self.output = self.raw.copy(deep=True)
        self.nlp = spacy_model
        self.preprocess = preprocess

    def extract_features(self):
        if self.preprocess: self._preprocess()
        self._extract_traditional_features()
        self._extract_syntatic_features()
        self._extract_pos_tag_features()
        self._spacy_vectors()

        return self.output

    def _preprocess(self):
        print('Preprocessing data...')
        self.raw.loc[:, 'preprocessed_excerpt'] = self.raw['excerpt'].apply(preprocessing.preprocess)
        self.output.loc[:, 'preprocessed_excerpt'] = self.raw['preprocessed_excerpt']

    def _extract_traditional_features(self):

        def n_longest_sent(doc) -> int:
            return max([len(sent) for sent in list(doc.sents)])

        print('Extracting traditional features...')
        for idx, row in self.raw.iterrows():
            doc = textacy.make_spacy_doc(row['preprocessed_excerpt'], lang='en_core_web_lg')
            ts = text_stats.TextStats(doc)
            self.output.loc[idx, 'n_sents'] = ts.n_sents                
            self.output.loc[idx, 'n_words'] = ts.n_words                
            self.output.loc[idx, 'n_words_per_sent'] = ts.n_words / ts.n_sents             
            self.output.loc[idx, 'n_unique_words'] = ts.n_unique_words
            self.output.loc[idx, 'n_unique_words_per_sent'] = ts.n_unique_words / ts.n_sents
            self.output.loc[idx, 'n_chars_per_word'] = ts.n_chars / ts.n_words
            self.output.loc[idx, 'n_syllables'] = ts.n_syllables
            self.output.loc[idx, 'n_syllables_per_word'] = ts.n_syllables / ts.n_words
            self.output.loc[idx, 'n_syllables_per_sent'] = ts.n_syllables / ts.n_sents
            self.output.loc[idx, 'n_monosyllable_words'] = ts.n_monosyllable_words
            self.output.loc[idx, 'n_polysyllable_words'] = ts.n_polysyllable_words
            self.output.loc[idx, 'n_long_words'] = ts.n_long_words
            self.output.loc[idx, 'n_long_words_ratio'] = ts.n_long_words / ts.n_words
            self.output.loc[idx, 'entropy'] = ts.entropy
            self.output.loc[idx, 'n_longest_sent'] = n_longest_sent(doc)

            self.output.loc[idx, 'automated_readability_index'] \
                = readability.automated_readability_index(ts.n_chars, ts.n_words, ts.n_sents)
            self.output.loc[idx, 'coleman_liau_index'] \
                = readability.coleman_liau_index(ts.n_chars, ts.n_words, ts.n_sents)
            self.output.loc[idx, 'flesch_kincaid_grade_level'] \
                = readability.flesch_kincaid_grade_level(ts.n_syllables, ts.n_words, ts.n_sents)
            self.output.loc[idx, 'flesch_reading_ease'] \
                = readability.flesch_reading_ease(ts.n_syllables, ts.n_words, ts.n_sents)
            self.output.loc[idx, 'lix'] \
                = readability.lix(ts.n_syllables, ts.n_long_words, ts.n_sents)
            self.output.loc[idx, 'smog_index'] \
                = readability.smog_index(ts.n_polysyllable_words, ts.n_sents)
            self.output.loc[idx, 'gunning_fog_index'] \
                = readability.gunning_fog_index(ts.n_words, ts.n_polysyllable_words, ts.n_sents)

    def _extract_syntatic_features(self):
        print('Extracting Syntactic features...')
        def tree_height(root):
            if not list(root.children):
                return 1
            else:
                return 1 + max(tree_height(x) for x in root.children)

        def get_average_height(paragraph):
            doc = self.nlp(paragraph) if type(paragraph) == str else paragraph
            roots = [sent.root for sent in doc.sents]
            return np.mean([tree_height(root) for root in roots])

        def count_subtrees(root):
            if not list(root.children):
                return 0
            else:
                return 1 + sum(count_subtrees(x) for x in root.children)

        def get_mean_subtrees(paragraph):
            doc = self.nlp(paragraph) if type(paragraph) == str else paragraph
            roots = [sent.root for sent in doc.sents]
            return np.mean([count_subtrees(root) for root in roots])

        def get_averge_noun_chunks(paragraph):
            doc = self.nlp(paragraph) if type(paragraph) == str else paragraph
            return len(list(doc.noun_chunks))
            
        def get_noun_chunks_size(paragraph):
            doc = self.nlp(paragraph) if type(paragraph) == str else paragraph
            noun_chunks_size = [len(chunk) for chunk in doc.noun_chunks]
            return np.mean(noun_chunks_size)
        
        self.output['avg_parse_tree_height'] = self.output.preprocessed_excerpt.apply(get_average_height)
        self.output['mean_parse_subtrees'] = self.output.preprocessed_excerpt.apply(get_mean_subtrees)
        self.output['noun_chunks'] = self.output.preprocessed_excerpt.apply(get_averge_noun_chunks)
        self.output['noun_chunk_size'] = self.output.preprocessed_excerpt.apply(get_noun_chunks_size)
        self.output['avg_noun_chunks'] = self.output['noun_chunks'] / self.output['n_sents']
        self.output['mean_noun_chunk_size'] = self.output['noun_chunk_size'] / self.output['avg_noun_chunks']
    
    def _extract_pos_tag_features(self):
        print('Extracting POS Tag features...')
        def get_pos_freq_per_word(paragraph, tag):
            doc = self.nlp(paragraph) if type(paragraph) == str else paragraph
            pos_counter = collections.Counter(([token.pos_ for token in doc]))
            pos_count_by_tag = pos_counter[tag]
            total_pos_counts = sum(pos_counter.values())
            return pos_count_by_tag / total_pos_counts

        self.output['nouns_per_word'] = self.raw.preprocessed_excerpt.apply(lambda x: get_pos_freq_per_word(x, 'NOUN'))
        self.output['proper_nouns_per_word'] = self.raw.preprocessed_excerpt.apply(lambda x: get_pos_freq_per_word(x, 'PROPN'))
        self.output['pronouns_per_word'] = self.raw.preprocessed_excerpt.apply(lambda x: get_pos_freq_per_word(x, 'PRON'))
        self.output['adj_per_word'] = self.raw.preprocessed_excerpt.apply(lambda x: get_pos_freq_per_word(x, 'ADJ'))
        self.output['adv_per_word'] = self.raw.preprocessed_excerpt.apply(lambda x: get_pos_freq_per_word(x, 'ADV'))
        self.output['verbs_per_word'] = self.raw.preprocessed_excerpt.apply(lambda x: get_pos_freq_per_word(x, 'VERB'))
        self.output['cconj_per_word'] = self.raw.preprocessed_excerpt.apply(lambda x: get_pos_freq_per_word(x, 'CCONJ'))
        self.output['sconj_per_word'] = self.raw.preprocessed_excerpt.apply(lambda x: get_pos_freq_per_word(x, 'SCONJ'))

    def _spacy_vectors(self):
        print('Extracting Spacy vectors...')
        with self.nlp.disable_pipes():
            vectors = np.array([self.nlp(text).vector for text in self.output.preprocessed_excerpt])
            cols = utils.get_col_names('spacy', len(vectors[0]))
            self.output[cols] = vectors

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="to provide csv/text file path")
    parser.add_argument("--output_path", help="to provide path for storing output file")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_path)
    fe = FeatureGenerator(df, spacy.load('en_core_web_lg'))
    output = fe.extract_features()
    print(output.head())
    output.to_csv(args.output_path, index=False)