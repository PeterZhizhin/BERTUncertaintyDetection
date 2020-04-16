import re
import spacy
import spacy.tokens


def _tokenizer_without_inside_word_splits(nlp):
    return spacy.tokenizer.Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                     suffix_search=nlp.tokenizer.suffix_search,
                                     # Regex that doesn't match anything to avoid inside word splits
                                     infix_finditer=re.compile('a^').finditer,
                                     token_match=nlp.tokenizer.token_match,
                                     # Remove rules that require to split such words as "don't" to "do" "n't"
                                     rules={})


class SpacyNlpSingleton(object):
    instance = None

    @classmethod
    def get(cls):
        if cls.instance is None:
            cls.instance = spacy.load("en_core_web_sm")
            spacy.tokens.Token.set_extension('is_uncertain', default=False)
            spacy.tokens.Token.set_extension('uncertainty_type', default='')
            spacy.tokens.Token.set_extension('uncertainty_span_idx', default=None)
            cls.instance.tokenizer = _tokenizer_without_inside_word_splits(cls.instance)
        return cls.instance


def nlp():
    return SpacyNlpSingleton.get()
