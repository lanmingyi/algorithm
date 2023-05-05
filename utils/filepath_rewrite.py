import os

paths = 'D:/darknet/build/darknet/x64/data/obj'
f = open('D:/darknet/build/darknet/x64/data/train.txt', 'r+')
filenames = os.listdir(paths)
# print(filenames)

for filename in filenames:
    if os.path.splitext(filename)[1] == '.jpg':
        out_path = "data/obj/" + filename
        # print(out_path)
        f.write(out_path+'\n')
f.close()
