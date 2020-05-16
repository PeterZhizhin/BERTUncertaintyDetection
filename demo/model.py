import logging
import transformers
import torch
import numpy as np
import spacy.lang.en

logger = logging.getLogger(__name__)

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class Model(object):
    def __init__(self, model_path, labels_path, max_seq_len=512):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = transformers.AutoModelForTokenClassification.from_pretrained(model_path)
        self.max_seq_len = max_seq_len
        self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

        labels = get_labels(labels_path)
        self.label_map = {i: label for i, label in enumerate(labels)}

        self.nlp = spacy.lang.en.English()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

        self.model.eval()

    def predict_sentence(self, texts):
        encoded_text = self.tokenizer.batch_encode_plus(texts,
                                                        max_length=self.max_seq_len,
                                                        pad_to_max_length=True,
                                                        return_input_lengths=True,
                                                        return_offsets_mapping=True)

        with torch.no_grad():
            input_ids = torch.tensor(encoded_text['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(encoded_text['attention_mask'], dtype=torch.long)
            token_type_ids = torch.tensor(encoded_text['token_type_ids'], dtype=torch.long)

            if input_ids.ndimension() == 1:
                input_ids = input_ids.reshape((1, -1))
                attention_mask = attention_mask.reshape((1, -1))
                token_type_ids = token_type_ids.reshape((1, -1))

            sequence_lengths = (input_ids != 0).sum(axis=1)
            total_len = sequence_lengths.max()

            input_ids = input_ids[:, :total_len]
            attention_mask = attention_mask[:, :total_len]
            token_type_ids = token_type_ids[:, :total_len]

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs[0]

        sequence_lengths = sequence_lengths.detach().cpu().numpy()
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2)

        preds_list = [[] for _ in range(preds.shape[0])]
        for i, total_len in enumerate(sequence_lengths):
            for j in range(total_len):
                preds_list[i].append(self.label_map[preds[i][j]])

        preds_list = [prediction[1:-1] for prediction in preds_list]

        offset_mapping = encoded_text['offset_mapping']
        if len(texts) == 1:
            offset_mapping = [offset_mapping]

        offset_mapping = [offset[1:len(prediction) + 1]
                          for prediction, offset
                          in zip(preds_list, offset_mapping)]

        return preds_list, offset_mapping

    def predict_text(self, text):
        doc = self.nlp(text)

        sentences = list(doc.sents)
        sentences_texts = [sent.text for sent in sentences]
        preds_lists, offset_mappings = self.predict_sentence(sentences_texts)

        full_preds_list = []
        full_offset_mapping = []
        for predictions, offset_mapping, sentence in zip(preds_lists, offset_mappings, sentences):
            full_preds_list += predictions
            sentence_start = sentence.start_char
            full_offset_mapping += [(span_start + sentence_start,
                                     span_end + sentence_start) for span_start, span_end in offset_mapping]

        result = [{
            'start': offset_mapping[0],
            'end': offset_mapping[1],
            'type': prediction,
        } for prediction, offset_mapping in zip(full_preds_list, full_offset_mapping) if prediction != 'O']

        return result
