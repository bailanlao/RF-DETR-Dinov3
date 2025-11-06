coco_id_to_name={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
           9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
           16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
           25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
           35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
           41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
           48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
           56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
           64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
           75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
           82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
           90: 'toothbrush'}
mapping={}
coco_mapping = {
    1: 0,   # person
    2: 1,   # bicycle
    3: 2,   # car
    4: 3,   # motorcycle
    5: 4,   # airplane
    6: 5,   # bus
    7: 6,   # train
    8: 7,   # truck
    9: 8,   # boat
    10: 9,  # traffic light
    11: 10, # fire hydrant
    13: 11, # stop sign
    14: 12, # parking meter
    15: 13, # bench
    16: 14, # bird
    17: 15, # cat
    18: 16, # dog
    19: 17, # horse
    20: 18, # sheep
    21: 19, # cow
    22: 20, # elephant
    23: 21, # bear
    24: 22, # zebra
    25: 23, # giraffe
    27: 24, # backpack
    28: 25, # umbrella
    31: 26, # handbag
    32: 27, # tie
    33: 28, # suitcase
    34: 29, # frisbee
    35: 30, # skis
    36: 31, # snowboard
    37: 32, # sports ball
    38: 33, # kite
    39: 34, # baseball bat
    40: 35, # baseball glove
    41: 36, # skateboard
    42: 37, # surfboard
    43: 38, # tennis racket
    44: 39, # bottle
    46: 40, # wine glass
    47: 41, # cup
    48: 42, # fork
    49: 43, # knife
    50: 44, # spoon
    51: 45, # bowl
    52: 46, # banana
    53: 47, # apple
    54: 48, # sandwich
    55: 49, # orange
    56: 50, # broccoli
    57: 51, # carrot
    58: 52, # hot dog
    59: 53, # pizza
    60: 54, # donut
    61: 55, # cake
    62: 56, # chair
    63: 57, # couch
    64: 58, # potted plant
    65: 59, # bed
    67: 60, # dining table
    70: 61, # toilet
    72: 62, # tv
    73: 63, # laptop
    74: 64, # mouse
    75: 65, # remote
    76: 66, # keyboard
    77: 67, # cell phone
    78: 68, # microwave
    79: 69, # oven
    80: 70, # toaster
    81: 71, # sink
    82: 72, # refrigerator
    84: 73, # book
    85: 74, # clock
    86: 75, # vase
    87: 76, # scissors
    88: 77, # teddy bear
    89: 78, # hair drier
    90: 79, # toothbrush
}
for index, (coco_id, name) in enumerate(coco_id_to_name.items()):
    mapping[coco_id]=index
    if coco_mapping[coco_id]!=index:
        print(f"coco_id: {coco_id}, coco_mapping: {coco_mapping[coco_id]}, index: {index}")
