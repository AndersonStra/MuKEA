import json
no_coco = []
image_dic = {}

with open('/data/dy/GRUC/vilbert/KRVQA/dataset/question_answer_reason.json','r') as f:
    kr = json.load(f)
with open('/data/dy/GRUC/vilbert/VG/image_data.json','r') as f:
    image_data = json.load(f)
print(len(kr))
for item in image_data:
    image_id = item['image_id']
    image_dic[image_id] = item

for item in kr:
    image_id = item['image_id']
    coco_id = image_dic[image_id]['coco_id']
    if coco_id is None:
        no_coco.append(image_id)

no_coco = list(set(no_coco))
with open('no_coco.json', 'w') as f:
    json.dump(no_coco, f, indent=4)

print(len(no_coco))