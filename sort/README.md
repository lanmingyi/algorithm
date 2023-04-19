Note
=====
SORT（Simple Online and Realtime Tracking），在2016年被提出，其性能超过了同时代的其他多目标跟踪器。然后，正如它的名字，SORT的原理非常简单，源码只有不到300行！SORT主要由三部分组成：目标检测，卡尔曼滤波，匈牙利算法。

1. 目标检测（Object Detection）
```
  目前主流的跟踪算法常常与目标检测算法相结合，SORT也是其中之一。目标检测的任务就是找到目标的位置，常用一个矩形（bounding box,简称为bbox）
将目标框出来，来表示目标的位置。在SORT中，作者使用了Faster RCNN来得到bbox，具体原理这里不展开，大家当一个黑箱来使用就好了。
当然，除了Faster RCNN以外还可以用其他的检测算法，如YOLO等。值得一提的是，作者发现目标跟踪质量的好坏与检测算法的性能有很大的关系，
通过使用先进的检测算法，跟踪结果质量能够显著提升。
```

2. 卡尔曼滤波（Kalman Filter）
```
  得到bbox后，我们是否就知道目标的准确位置了呢？从严格上来说，不是的。因为测量总是存在误差的，我们通过目标检测得到的bbox会不可避免地带有噪声，
导致bbox的位置不够精确。这时，卡尔曼滤波的作用就体现出来了。卡尔曼滤波可以通过利用数学模型预测的值和测量得到的观测值进行数据融合，
找到“最优”的估计值（这里的最优指的是均方差最小）。比方说，我们现在要知道t帧时某一目标准确的bbox（即，计算估计值），记为bbox[optimal]。
我们已知的是1~t-1帧中目标的bbox。现在我们有两种方法得到t帧的bbox：一是通过数学建模，根据1~t-1的信息来预测出t帧的bbox，记为bbox[prediction]；
二是通过检测算法，直接测量出t帧的bbox，记为bbox[measurement]。卡尔曼滤波做的事情就是利用bbox[prediction]和bbox[measurement]来得到bbox[optimal]，
具体分两步实现：预测(predict)，即通过数学模型计算出bbox[prediction]；更新(update)，结合测量值bbox[measurement]得到当前状态(state)的最优估计值。
卡尔曼滤波是一种去噪技术，能够在目标检测的基础上，得到更加准确的bbox。
```

3. 匈牙利算法（Hungarian Algorithm）
```
  匈牙利算法是一种数据关联(Data Association)算法，其实从本质上讲，跟踪算法要解决的就是数据关联问题。假设有两个集合S和T，集合S中有m个元素，
集合T中有n个元素，匈牙利算法要做的是把S中的元素和T中的元素两两匹配（可能匹配不上）。结合跟踪的情景，匈牙利算法的任务就是把t帧的bbox与t-1帧的bbox两两匹配，
这样跟踪就完成了。要想匹配就需要一定的准则，匈牙利算法依据的准则是“损失最小”。损失由损失矩阵的形式来表示，损失矩阵描述了匹配两个集合中某两个元素所要花费的代价。
```

4. SORT的具体实现
```
  理解了上面这些组件后，SORT的原理就显得特别简单了。SORT将卡尔曼滤波预测的bbox[prediction]和目标检测算法得到的bbox[measurement]，
用匈牙利算法进行匹配，再用bbox[prediction]和bbox[measurement]更新当前状态，得到bbox[optimal]，作为追踪的结果。
匈牙利算法在使用前需要定义损失矩阵，SORT利用bbox[prediction]和bbox[measurement]的IOU（交并比）来定义损失矩阵。
比如损失矩阵cost[ij]表示前一帧第i个bbox与这一帧第j个bbox的IOU。
  SORT中的Kalman滤波采用线性匀速模型，状态向量描述成 x = [u, v, s, r, u',v',s']T ，其中u,v表示bbox的中心坐标，s表示面积，
r表示横纵比（SORT认为对每个目标而言，r是不变的常数；而在Deep SORT中则不是），头上带点的是相应的变化率（速度）。
```
      

SORT
=====
A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.
See an example [video here](https://alex.bewley.ai/misc/SORT-MOT17-06-FRCNN.webm).

By Alex Bewley  

### Introduction

SORT is a barebones implementation of a visual multiple object tracking framework based on rudimentary data association and state estimation techniques. It is designed for online tracking applications where only past and current frames are available and the method produces object identities on the fly. While this minimalistic tracker doesn't handle occlusion or re-entering objects its purpose is to serve as a baseline and testbed for the development of future trackers.

SORT was initially described in [this paper](http://arxiv.org/abs/1602.00763). At the time of the initial publication, SORT was ranked the best *open source* multiple object tracker on the [MOT benchmark](https://motchallenge.net/results/2D_MOT_2015/).

**Note:** A significant proportion of SORT's accuracy is attributed to the detections.
For your convenience, this repo also contains *Faster* RCNN detections for the MOT benchmark sequences in the [benchmark format](https://motchallenge.net/instructions/). To run the detector yourself please see the original [*Faster* RCNN project](https://github.com/ShaoqingRen/faster_rcnn) or the python reimplementation of [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) by Ross Girshick.

**Also see:**
A new and improved version of SORT with a Deep Association Metric implemented in tensorflow is available at [https://github.com/nwojke/deep_sort](https://github.com/nwojke/deep_sort) .

### License

SORT is released under the GPL License (refer to the LICENSE file for details) to promote the open use of the tracker and future improvements. If you require a permissive license contact Alex (alex@bewley.ai).

### Citing SORT

If you find this repo useful in your research, please consider citing:

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }


### Dependencies:

To install required dependencies run:
```
$ pip install -r requirements.txt
```


### Demo:

To run the tracker with the provided detections:

```
$ cd path/to/sort
$ python sort.py
```

To display the results you need to:

1. Download the [2D MOT 2015 benchmark dataset](https://motchallenge.net/data/2D_MOT_2015/#download)
0. Create a symbolic link to the dataset
  ```
  $ ln -s /path/to/MOT2015_challenge/data/2DMOT2015 mot_benchmark
  ```
0. Run the demo with the ```--display``` flag
  ```
  $ python sort.py --display
  ```


### Main Results

Using the [MOT challenge devkit](https://motchallenge.net/devkit/) the method produces the following results (as described in the paper).

 Sequence       | Rcll | Prcn |  FAR | GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL
--------------- |:----:|:----:|:----:|:-------------:|:-------------------:|:------------------:
 TUD-Campus     | 68.5 | 94.3 | 0.21 |  8   6   2   0|   15   113    6    9|  62.7  73.7  64.1
 ETH-Sunnyday   | 77.5 | 81.9 | 0.90 | 30  11  16   3|  319   418   22   54|  59.1  74.4  60.3
 ETH-Pedcross2  | 51.9 | 90.8 | 0.39 | 133  17  60  56|  330  3014   77  103|  45.4  74.8  46.6
 ADL-Rundle-8   | 44.3 | 75.8 | 1.47 | 28   6  16   6|  959  3781  103  211|  28.6  71.1  30.1
 Venice-2       | 42.5 | 64.8 | 2.75 | 26   7   9  10| 1650  4109   57  106|  18.6  73.4  19.3
 KITTI-17       | 67.1 | 92.3 | 0.26 |  9   1   8   0|   38   225    9   16|  60.2  72.3  61.3
 *Overall*      | 49.5 | 77.5 | 1.24 | 234  48 111  75| 3311 11660  274  499|  34.0  73.3  35.1


### Using SORT in your own project

Below is the gist of how to instantiate and update SORT. See the ['__main__'](https://github.com/abewley/sort/blob/master/sort.py#L239) section of [sort.py](https://github.com/abewley/sort/blob/master/sort.py#L239) for a complete example.
    
    from sort import *
    
    #create instance of SORT
    mot_tracker = Sort() 
    
    # get detections
    ...
    
    # update SORT
    track_bbs_ids = mot_tracker.update(detections)

    # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
    ...
    
 
