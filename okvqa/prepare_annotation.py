import json
import pickle

row = {}

with open('/data2/yjgroup/lmz/pr*/data/okvqa6_1/raw_data/train_36bbox_res_okvqa2.pickle', 'rb') as f:
    image_raw = pickle.load(f, encoding='iso-8859-1')
with open('/data2/yjgroup/dy/kb-vqa/data/ok-vqa/OpenEnded_mscoco_train2014_questions.json') as f:
    ques = json.load(f)
with open('/data2/yjgroup/dy/kb-vqa/data/ok-vqa/mscoco_train2014_annotations.json') as f:
    annotation = json.load(f)

ques = ques['questions']
annotation = annotation['annotations']

for q, a in zip(ques, annotation):
    question = q['question']
    image_id = q['image_id']
    image_file = 'COCO_train2014_' + '0' * (12 - len(str(image_id))) + str(image_id) + '.jpg'
    origin_labels = image_raw[image_file]['labels']
    multi_answers = []
    for ans in a['answers']:
        multi_answers.append(ans['raw_answer'])
    row[q['question_id']] = {'question':question, 'image_id':image_id, 'multi_answers':multi_answers,'label':origin_labels}

with open('/data2/yjgroup/dy/kb-vqa/data/okvqa_train_raw_answer.json','w') as f:
    json.dump(row, f, indent=4)