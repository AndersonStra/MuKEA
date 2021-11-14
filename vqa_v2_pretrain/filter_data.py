import json

with open('/kb-vqa/data/vqa_train.json','r') as f:
    qa_raw = json.load(f)

for qid, item in list(qa_raw.items()):
    ques = item['question']
    m_ans = item['multi_answers']
    if 'yes' in m_ans or 'no' in m_ans or 'How many' in ques or 'where' in ques:
        del qa_raw[qid]
    else:
        for ans in m_ans:
            if ans.isdigit():
                del qa_raw[qid]
                break

print(len(qa_raw))
with open('/kb-vqa/data/vqa_train_filter.json','w') as f:
    json.dump(qa_raw, f, indent=4)
