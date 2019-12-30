import spacy
import spacy.tokens


class SpacyNlpSingleton(object):
    instance = None

    @classmethod
    def get(cls):
        if cls.instance is None:
            cls.instance = spacy.load("en_core_web_sm")
            spacy.tokens.Token.set_extension('is_uncertain', default=False)
            spacy.tokens.Token.set_extension('uncertainty_type', default='')
        return cls.instance


def nlp():
    return SpacyNlpSingleton.get()
