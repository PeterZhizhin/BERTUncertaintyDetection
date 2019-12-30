import io
import unittest
from lxml import etree

from data import hedge_dataset


class MyTestCase(unittest.TestCase):
    def test_sentence_to_tagged_sentence_returns_parsed_sent_no_ccue(self):
        element = etree.fromstring("<Sentence>A cat does a barrel roll.</Sentence>")
        parsed_sentence = hedge_dataset.HedgeDataset.sentence_to_tagged_tokens(element)
        sent_converted_back = ' '.join(str(i) for i in parsed_sentence)
        self.assertEqual(sent_converted_back, 'A cat does a barrel roll .')

    def test_sentence_to_tagged_sentence_with_ccue_returns_full_sentence(self):
        element = etree.fromstring(
            ('<Sentence>A cat <ccue type="speculation one">maybe, but not sure</ccue> '
             'does a <ccue type="speculation two">not sure</ccue> barrel roll. </Sentence>')
        )
        parsed_sentence = hedge_dataset.HedgeDataset.sentence_to_tagged_tokens(element)
        sent_converted_back = ' '.join(str(i) for i in parsed_sentence)
        self.assertEqual(
            sent_converted_back,
            'A cat maybe , but not sure does a not sure barrel roll .'
        )

    def test_to_tagged_sentence_with_ccue_returns_correct_is_uncertain_tokens(self):
        element = etree.fromstring(
            '<Sentence>A cat <ccue type="speculation">maybe</ccue> does a roll.</Sentence>'
        )
        parsed_sentence = hedge_dataset.HedgeDataset.sentence_to_tagged_tokens(element)
        self.assertTrue(parsed_sentence[2]._.is_uncertain)

    def test_to_tagged_sentence_with_ccue_has_correct_ccue_tokenization_when_multiple_spans(self):
        element = etree.fromstring(
            ('<Sentence>A cat <ccue type="speculation one">maybe, but not sure</ccue> '
             'does a <ccue type="speculation two">not sure</ccue> barrel roll. </Sentence>')
        )
        parsed_sentence = hedge_dataset.HedgeDataset.sentence_to_tagged_tokens(element)
        expected_is_uncertain_mapping = [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0]
        actual_is_uncertain_mapping = [int(tok._.is_uncertain) for tok in parsed_sentence]
        self.assertSequenceEqual(actual_is_uncertain_mapping, expected_is_uncertain_mapping)

    def test_to_tagged_sentence_with_ccue_has_correct_ccue_tokenization_when_starts_with_span(self):
        element = etree.fromstring(
            ('<Sentence><ccue type="speculation one">Maybe, but not sure</ccue> a cat '
             'does a barrel roll, <ccue type="speculation two">not sure</ccue>.</Sentence>')
        )
        parsed_sentence = hedge_dataset.HedgeDataset.sentence_to_tagged_tokens(element)
        expected_is_uncertain_mapping = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
        actual_is_uncertain_mapping = [int(tok._.is_uncertain) for tok in parsed_sentence]
        self.assertSequenceEqual(actual_is_uncertain_mapping, expected_is_uncertain_mapping)

    def test_to_tagged_sentence_with_ccue_has_expected_uncertainty_type(self):
        element = etree.fromstring(
            '<Sentence><ccue type="speculation">maybe</ccue></Sentence>'
        )
        parsed_sentence = hedge_dataset.HedgeDataset.sentence_to_tagged_tokens(element)
        self.assertEqual(parsed_sentence[0]._.uncertainty_type, 'speculation')

    def test_convert_to_csv_returns_tagged_sentences(self):
        element = etree.fromstring(
            '''<Annotation>
                <DocumentSet>
                    <Document>
                        <DocID type="test_doc">test_doc_id1</DocID>
                        <Sentence>A <ccue type="spec1">maybe1</ccue></Sentence>
                        <Sentence>B <ccue type="spec2">maybe2</ccue></Sentence>
                    </Document>
                    <Document>
                        <DocID type="test_doc">test_doc_id2</DocID>
                        <Sentence>C <ccue type="spec3">maybe3</ccue></Sentence>
                        <Sentence>D <ccue type="spec4">maybe4</ccue></Sentence>
                    </Document>
                </DocumentSet>
            </Annotation>'''
        )
        test_output_file = io.StringIO()
        ds = hedge_dataset.HedgeDataset(element)
        ds.convert_to_csv(test_output_file)
        resulted_csv = test_output_file.getvalue().strip()
        expected_csv = ('doc_id,doc_type,sentence_id,token_id,token,is_uncertain,uncertainty_type\r\n'
                        'test_doc_id1,test_doc,0,0,A,False,\r\n'
                        'test_doc_id1,test_doc,0,1,maybe1,True,spec1\r\n'
                        'test_doc_id1,test_doc,1,0,B,False,\r\n'
                        'test_doc_id1,test_doc,1,1,maybe2,True,spec2\r\n'
                        'test_doc_id2,test_doc,0,0,C,False,\r\n'
                        'test_doc_id2,test_doc,0,1,maybe3,True,spec3\r\n'
                        'test_doc_id2,test_doc,1,0,D,False,\r\n'
                        'test_doc_id2,test_doc,1,1,maybe4,True,spec4')
        self.assertEqual(resulted_csv, expected_csv)


if __name__ == '__main__':
    unittest.main()
