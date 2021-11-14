import json
import pickle

train={}
val={}
test={}

with open('/data2/yjgroup/dy/kb-vqa/data/kr-vqa/question_answer_reason.json','r') as f:
    row = json.load(f)
with open('/data2/yjgroup/dy/kb-vqa/data/kr-vqa/splits.json','r') as f:
    split = json.load(f)

for i in row:
    qid = i['question_id']
    if '#' in i['answer']:
        i['answer'] = i['answer'].split('#')[0]
    if '/' in i['answer']:
        i['answer'] = i['answer'].split('/')[-1]
    if int(qid) in split['train']:
        train[qid] = i
    elif int(qid) in split['val']:
        val[qid] = i
    elif int(qid) in split['test']:
        test[qid] = i
    else:
        print('error')

with open('/data2/yjgroup/dy/kb-vqa/data/kr-vqa/krvqa_train.json','w') as f:
    json.dump(train, f, indent=4)
with open('/data2/yjgroup/dy/kb-vqa/data/kr-vqa/krvqa_val.json','w') as f:
    json.dump(val, f, indent=4)
with open('/data2/yjgroup/dy/kb-vqa/data/kr-vqa/krvqa_test.json','w') as f:
    json.dump(test, f, indent=4)
print(len(train),len(val),len(test))

