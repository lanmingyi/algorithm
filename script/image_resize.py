# 提取目录下所有图片,更改尺寸后保存到另一目录
from PIL import Image
import os.path
import glob


# target_path = "images/dataset/wt_data/resize_1024"        #输出目标文件路径
target_path = "./images_resize"        # 输出目标文件路径
if not os.path.exists(target_path):
    os.makedirs(target_path)


def convertjpg(jpgfile,outdir, width=412, height=197):
    img=Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


for jpgfile in glob.glob("images/*.jpg"):
    convertjpg(jpgfile, target_path)
