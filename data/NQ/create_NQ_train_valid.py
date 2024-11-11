import datasets
import random
import numpy as np
import json
import string

random.seed(313)

NUM_TRAIN = 4000
NUM_EVAL = 1000

data = datasets.load_dataset('natural_questions', cache_dir='cache')['train']
rand_inds = list(range(len(data)))
random.shuffle(rand_inds)

title_set = set()
current_docid = 0

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix((remove_punc(lower(s))))

with open('NQ_10k_multi_task_train.jsonl', 'w', encoding='utf-8') as tf, \
        open('NQ_10k_valid.jsonl', 'w', encoding='utf-8') as vf:
    for ind in rand_inds:
        title = data[ind]['document']['title']  # we use title as the doc identifier to prevent two docs have the same text
        if title not in title_set:
            title_set.add(title)
            token_inds = np.where(np.array(data[ind]['document']['tokens']['is_html']) == False)[0]
            tokens = np.array(data[ind]['document']['tokens']['token'])[token_inds]
            doc_text = " ".join(tokens)
            question_text = data[ind]['question']['text']

            jitem = json.dumps({'text_id': str(current_docid), 'text': 'document: ' + normalize_answer(doc_text)})
            tf.write(jitem + '\n')
            jitem = json.dumps({'text_id': str(current_docid), 'text': 'question: ' + normalize_answer(question_text)})
            if len(title_set) <= NUM_TRAIN:
                tf.write(jitem + '\n')
            else:
                vf.write(jitem + '\n')
            current_docid += 1
            if len(title_set) == NUM_TRAIN + NUM_EVAL:
                break
        print(f"Creating training and validation dataset: {'{:.1%}'.format(len(title_set)/(NUM_TRAIN + NUM_EVAL))}", end='\r')