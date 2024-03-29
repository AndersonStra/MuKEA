import json


dic = {}
with open('/kb-vqa/data/vqav2/v2_OpenEnded_mscoco_test2015_questions.json') as f:
    a = json.load(f)
ques_list = a['questions']
with open('/kb-vqa/data/vqav2/v2_mscoco_test2015_annotations.json') as f:
    b = json.load(f)
annotation = b['annotations']

for q_a in zip(ques_list, annotation):
    q_id = str(q_a[0]['question_id']) + 'f'
    question = q_a[0]['question']
    image_id = q_a[0]['image_id']
    ans = q_a[1]['multiple_choice_answer']
    multi_ans = q_a[1]['answers']
    # for vqa_test, there is no answer annotation
    # ans = ['unknown']
    # multi_list = ans * 10
    # for vqa_train or vqa_val, add answer annotation
    multi_list = []
    for i in multi_ans:
        multi_list.append(i['answer'])
    item = {'question': question, 'image_id': image_id, 'answer': ans,'multi_answers': multi_list}
    dic[q_id] = item
with open('/kb-vqa/data/vqa_test.json','w') as f:
    json.dump(dic, f, indent=4)
