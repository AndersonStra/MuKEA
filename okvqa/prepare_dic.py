import json
import pickle
import collections

word_counts = collections.Counter()
with open('/kb-vqa/data/okvqa_train.json','r') as f:
    train_row = json.load(f)

for qid, item in train_row.items():
    ans = item['multi_answers']
    for a in ans:
        word_counts[a] += 1

vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

with open('/kb-vqa/data/ans_dic_raw.pickle', 'wb') as f:
    pickle.dump(vocabulary, f)
