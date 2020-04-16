import csv

import spacy.tokens
from lxml import etree
import tqdm

from utils import nlp_module

UNCERTAINTY_TO_CLASS = {
    'speculation_hypo_condition _': 'COND',
    'speculation_hypo_doxastic _': 'DOX',
    'speculation_hypo_investigation _': 'INV',
    'speculation_modal_possible_': 'EPIST',
    'speculation_modal_probable_': 'EPIST',
}


def _match_uncertainty_tokens(doc, uncertain_spans, uncertainty_types):
    current_uncertain_span_i = 0
    for token in doc:
        token_idx = token.idx
        # Ensure that current span is at current token first symbol index or after it.
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
            token._.uncertainty_span_idx = current_uncertain_span_i


def _get_token_class(is_uncertain, current_uncertainty_span_idx, prev_uncertainty_span_idx, uncertainty_type):
    if not is_uncertain:
        return 'O'
    class_string = 'I' if current_uncertainty_span_idx == prev_uncertainty_span_idx else 'B'
    uncertainty_class = UNCERTAINTY_TO_CLASS[uncertainty_type]
    return class_string + '-' + uncertainty_class


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

    def convert_to_ner_txt(self, out_file_obj, documents_iterator=None, use_tqdm=True):
        if documents_iterator is None:
            documents_iterator = list(self.root.iterdescendants('Document'))
        if use_tqdm:
            documents_iterator = tqdm.tqdm(documents_iterator, desc='Converting documents')
        for document in documents_iterator:
            all_sentences_iter = document.iterdescendants('Sentence')
            for sentence_i, sentence in enumerate(all_sentences_iter):
                parsed_sentence = self.sentence_to_tagged_tokens(sentence)
                prev_uncertainty_span_idx = None
                for token in parsed_sentence:
                    token_text = token.text
                    is_uncertain = token._.is_uncertain
                    current_uncertainty_span_idx = token._.uncertainty_span_idx
                    uncertainty_type = token._.uncertainty_type
                    uncertainty_class = _get_token_class(is_uncertain, current_uncertainty_span_idx,
                                                         prev_uncertainty_span_idx, uncertainty_type)

                    print(token_text, uncertainty_class, file=out_file_obj)
                    prev_uncertainty_span_idx = current_uncertainty_span_idx

                # Sentence end, print empty line
                print(file=out_file_obj)

    def convert_to_multiple_ner_files(self, out_file_pattern):
        documents_iterator = list(self.root.iterdescendants('Document'))
        for i, document in enumerate(tqdm.tqdm(documents_iterator, desc='Converting documents'), 1):
            with open(out_file_pattern.format(i), 'w', encoding='utf-8') as out_file_i:
                self.convert_to_ner_txt(out_file_i, documents_iterator=[document], use_tqdm=False)

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
