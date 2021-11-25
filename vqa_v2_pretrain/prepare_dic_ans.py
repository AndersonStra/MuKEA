import json
import pickle
import collections

word_counts = collections.Counter()
with open('/kb-vqa/data/vqa_train.json') as f:
    vqa2 = json.load(f)
# with open('/kb-vqa/data/vqa_val.json') as f:
#     vqa2_val = json.load(f)
# vqa2.update(vqa2_val)
with open('/kb-vqa/data/fvqa/fvqa-ans_dic.pickle', 'rb') as f:
    a_dic = pickle.load(f)
# for a in a_dic.keys():
#     word_counts[a] += 1

for qid, item in vqa2.items():
    # anss = list(set(item['multi_answers']))
    # for ans in anss:
    #     word_counts[ans] += 1
    ans = item['answer']
    word_counts[ans] += 1

vocabulary_inv = [x[0] for x in word_counts.most_common()[:3500]]
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

# with open('/kb-vqa/data/vqav2/vqav2_dic_3500.pickle', 'wb') as f:
#     pickle.dump(vocabulary, f)
