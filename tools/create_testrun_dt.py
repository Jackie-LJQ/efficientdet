
# dataDir = "/nas/home/biaoye/liujiaqi/datasets/coco/"
# annFile = "/nas/home/biaoye/liujiaqi/datasets/coco/annotations/instances_train2017.json"

# for img_id in 
"""
First NUM images of coco train 2017
"""
import json
from pycocotools.coco import COCO
NUM = 100
annFile = "/nas/home/biaoye/liujiaqi/datasets/coco/annotations/instances_train2017.json"
savePath = "/nas/home/biaoye/liujiaqi/datasets/coco/annotations/cocoTestRun.json"
with open(annFile, "r") as f:
    annotation = json.load(f)
target_images = []
target_annotations = []
coco = COCO(annFile)
for images_ann in annotation["images"][:NUM]:
    _ann_id = coco.getAnnIds(imgIds=[images_ann["id"]])
    _anns = coco.loadAnns(ids=_ann_id)
    target_images.append(images_ann)
    target_annotations += _anns
    # break
_coco_ann = {"images":target_images, "annotations":target_annotations, "categories":annotation["categories"]}
with open(savePath, "w") as f:
    f.write(json.dumps(_coco_ann))
    
    
    

