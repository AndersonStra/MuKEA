import json


dic = {}
with open('/data2/yjgroup/dy/kb-vqa/data/vqav2/v2_OpenEnded_mscoco_test2015_questions.json') as f:
    a = json.load(f)
ques_list = a['questions']
with open('/data2/yjgroup/dy/kb-vqa/data/vqav2/v2_mscoco_val2014_annotations.json') as f:
    b = json.load(f)
annotation = b['annotations']

for q_a in zip(ques_list, ques_list):
    q_id = str(q_a[0]['question_id']) + 'f'
    question = q_a[0]['question']
    image_id = q_a[0]['image_id']
    # ans = q_a[1]['multiple_choice_answer']
    # multi_ans = q_a[1]['answers']
    ans = ['unknown']
    multi_list = ans * 10
    # for i in multi_ans:
    #     multi_list.append(i['answer'])
    item = {'question': question, 'image_id': image_id, 'answer': ans,'multi_answers': multi_list}
    dic[q_id] = item
with open('/data2/yjgroup/dy/kb-vqa/data/vqa_test.json','w') as f:
    json.dump(dic, f, indent=4)
