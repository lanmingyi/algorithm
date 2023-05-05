import os
import yaml  # "Yet Another Markup Language"（仍是一种标记语言）
from easydict import EasyDict as edict


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """
    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read(), Loader=yaml.FullLoader))
                # print(cfg_dict)

        # python3直接写成 super().方法名（参数）。python2必须写成 super（父类，self）.方法名（参数）
        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self,  config_file):
        with open(config_file, 'r') as fo:
            # print(yaml.load(fo.read(), Loader=yaml.FullLoader))
            # {'DEEPSORT': {'REID_CKPT': './deep_sort/deep/checkpoint/ckpt.t7', 'MAX_DIST': 0.2, 'MIN_CONFIDENCE': 0.3,
            # 'NMS_MAX_OVERLAP': 0.5, 'MAX_IOU_DISTANCE': 0.7, 'MAX_AGE': 70, 'N_INIT': 3, 'NN_BUDGET': 100}}
            # self.update(yaml.load(fo.read())
            self.update(yaml.load(fo.read(), Loader=yaml.FullLoader))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)


if __name__ == "__main__":
    cfg = YamlParser(config_file="../config/yolov3.yaml")
    cfg.merge_from_file("../config/deep_sort.yaml")
    print(cfg)

    # import ipdb; ipdb.set_trace()
