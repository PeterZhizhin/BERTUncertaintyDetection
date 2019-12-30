import csv

import spacy.tokens
from lxml import etree
import tqdm

from utils import nlp_module


def _match_uncertainty_tokens(doc, uncertain_spans, uncertainty_types):
    current_uncertain_span_i = 0
    for token in doc:
        token_idx = token.idx
        # Ensure that current token index is in span range
        while current_uncertain_span_i < len(uncertain_spans) and (
                token_idx >= uncertain_spans[current_uncertain_span_i][1]):
            current_uncertain_span_i += 1
        # If we are out of tokens, then there is nothing to mark
        if current_uncertain_span_i >= len(uncertain_spans):
            break

        span_start, span_end = uncertain_spans[current_uncertain_span_i]
        if span_start <= token_idx < span_end:
            token._.is_uncertain = True
            token._.uncertainty_type = uncertainty_types[current_uncertain_span_i]


class HedgeDataset(object):
    def __init__(self, root):
        self.root = root

    def convert_to_csv(self, out_file):
        # Convert to a list to know the full size
        documents_iterator = list(self.root.iterdescendants('Document'))
        csv_file = csv.writer(out_file)
        csv_file.writerow(
            ('doc_id', 'doc_type', 'sentence_id', 'token_id', 'token', 'is_uncertain', 'uncertainty_type')
        )
        for document in tqdm.tqdm(documents_iterator, desc='Converting documents'):
            document_id = next(document.iterchildren('DocID'))
            document_type = document_id.attrib['type']
            document_id = document_id.text

            all_sentences_iter = document.iterdescendants('Sentence')
            for sentence_i, sentence in tqdm.tqdm(enumerate(all_sentences_iter)):
                parsed_sentence = self.sentence_to_tagged_tokens(sentence)
                for token in parsed_sentence:
                    token_i = token.i
                    token_text = token.text
                    is_untertain = token._.is_uncertain
                    uncertainty_type = token._.uncertainty_type
                    row = (document_id, document_type, sentence_i, token_i, token_text, is_untertain, uncertainty_type)
                    csv_file.writerow(row)

    @classmethod
    def from_xml_file(cls, source):
        parsed_file = etree.parse(source)
        root = parsed_file.getroot()
        return cls(root)

    @classmethod
    def sentence_to_tagged_tokens(cls, element: etree.Element) -> spacy.tokens.Doc:
        nlp = nlp_module.nlp()

        full_text = element.text or ''
        uncertainty_spans = []
        uncertainty_types = []
        for ccue_element in element.iterchildren():
            uncertainty_types.append(ccue_element.attrib['type'])

            span_begin = len(full_text)
            full_text += ccue_element.text or ''
            span_end = len(full_text)
            full_text += ccue_element.tail or ''
            uncertainty_spans.append((span_begin, span_end))

        parsed_text = nlp(full_text)
        _match_uncertainty_tokens(parsed_text, uncertainty_spans, uncertainty_types)

        return parsed_text
