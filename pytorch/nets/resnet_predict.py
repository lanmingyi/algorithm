"""
Copyright (c) 2021 BrightMind. All rights reserved.
Written by MingYi Lan
"""

import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from resnet import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    img_path = "../../dataset/tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = "../../models/pt/resnet34.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        # predict_cla = torch.argmax(predict.numpy())  # 'input' (position 1) must be Tensor, not numpy.ndarray
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}  prob: {:.3}".format(class_indict[str(predict_cla)],
                                                predict[predict_cla].numpy())

    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
