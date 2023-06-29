import json
import os
import numpy as np

path = 'images/dataset/wt_data/resize_1024'
# path = 'images/dataset/wt_data/resize_1024/val'
# path = 'images/dataset/wtott_data/'
# path = 'images/dataset/wtott_data/val'


files = []
for file in os.listdir(path):
    if file[-5:] == '.json':
        files.append(file)
        # print(files)
# print(json.load(open('dataset/train/'+files[0])))


# Load annotations
# VGG Image Annotator (up to version 1.6) saves each image in the form:
# { 'filename': '28503151_5b5b7ec140_b.jpg',
#   'regions': {
#       '0': {
#           'region_attributes': {},
#           'shape_attributes': {
#               'all_points_x': [...],
#               'all_points_y': [...],
#               'name': 'polygon'}},
#       ... more regions ...
#   },
#   'size': 100202
# }

via_region_data = {}

for file in files:
    one_json = json.load(open(path+'/'+file))
    # print(one_json)
    one_image = {}
    one_image['filename'] = file.split('.')[0] + '.jpg'
    shape = one_json['shapes']

    regions = {}  # 字典
    for i in range(len(shape)):
        points = np.array(shape[i]['points'])  # 多行两列
        # print(points)
        all_points_x = list(points[:, 0])  # 矩阵转化为列表，提取x坐标
        all_points_y = list(points[:, 1])

        regions[str(i)] = {}  # 用str将i转化为字符型
        regions[str(i)]['region_attributes'] = {}
        regions[str(i)]['shape_attributes'] = {}

        regions[str(i)]['shape_attributes']['all_points_x'] = all_points_x
        regions[str(i)]['shape_attributes']['all_points_y'] = all_points_y
        regions[str(i)]['shape_attributes']['name'] = shape[i]['label']

    one_image['regions'] = regions
    one_image['size'] = 0

    via_region_data[file] = one_image
    # print(via_region_data)

# with open('images/train/via_region_data.json', 'w') as f:
#     json.dump(via_region_data, f, sort_keys=False, ensure_ascii=True)
with open('images/train/via_region_data.json', 'w') as f:
    json.dump(via_region_data, f, sort_keys=False, ensure_ascii=True)


# #############测试集
# path = 'images/dataset/val'
# files = []
# for file in os.listdir(path):
#     if file[-5:] == '.json':
#         files.append(file)
#         # print(files)
# # print(json.load(open('dataset/train/'+files[0])))
#
#
# # Load annotations
# # VGG Image Annotator (up to version 1.6) saves each image in the form:
# # { 'filename': '28503151_5b5b7ec140_b.jpg',
# #   'regions': {
# #       '0': {
# #           'region_attributes': {},
# #           'shape_attributes': {
# #               'all_points_x': [...],
# #               'all_points_y': [...],
# #               'name': 'polygon'}},
# #       ... more regions ...
# #   },
# #   'size': 100202
# # }
#
# via_region_data = {}
#
# for file in files:
#     one_json = json.load(open(path+'/'+file))
#     # print(one_json)
#     one_image = {}
#     one_image['filename'] = file.split('.')[0] + '.jpg'
#     shape = one_json['shapes']
#
#     regions = {}  # 字典
#     for i in range(len(shape)):
#         points = np.array(shape[i]['points'])  # 多行两列
#         # print(points)
#         all_points_x = list(points[:, 0])  # 矩阵转化为列表，提取x坐标
#         all_points_y = list(points[:, 1])
#
#         regions[str(i)] = {}  # 用str将i转化为字符型
#         regions[str(i)]['region_attributes'] = {}
#         regions[str(i)]['shape_attributes'] = {}
#
#         regions[str(i)]['shape_attributes']['all_points_x'] = all_points_x
#         regions[str(i)]['shape_attributes']['all_points_y'] = all_points_y
#         regions[str(i)]['shape_attributes']['name'] = shape[i]['label']
#
#     one_image['regions'] = regions
#     one_image['size'] = 0
#
#     via_region_data[file] = one_image
#     # print(via_region_data)
#
# with open('images/val/via_region_data.json', 'w') as f:
#     json.dump(via_region_data, f, sort_keys=False, ensure_ascii=True)
