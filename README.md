# complex_YOLO

Portfolio의 output 실행하는 코드에 대해서 설명하도록 하겠습니다.

## 그림 1
```
from utils import kitti_utils
$ ipython --gui=qt  ### 그림을 그리기 위해서 반드시 terminal에 해당 코드를 입력해야한다.
from utils.mayavi_viewer import draw_lidar_simple

velo_filename ='.../complex_yolo/data/KITTI/object/testing/velodyne/%.bin'
P = kitti_utils.load_velo_scan(velo_filename)  ## Point Cloud 계산

draw_lidar_simple(P)
```
![그림 1-1](https://user-images.githubusercontent.com/79675366/140274529-41d0765c-bd88-4655-a7e9-2f9040b49a6a.png)
![그림 1-2](https://user-images.githubusercontent.com/79675366/140274656-a67f584c-270e-467a-ad64-5c83b2726c7e.png)

마우스로 위치 크기 조절이 가능하다.

## 그림 2
```
from utils import kitti_utils
from utils import kitti_bev_utils
from utils.mayavi_viewer import draw_lidar
$ ipython --gui=qt  ### 그림을 그리기 위해서 반드시 terminal에 해당 코드를 입력해야한다.

velo_filename ='.../complex_yolo/data/KITTI/object/testing/velodyne/%.bin'
P = kitti_utils.load_velo_scan(velo_filename)  ## Point Cloud 계산
R = kitti_bev_utils.removePoints(P, cnf.boundary)  #원하는 Space 구성

draw_lidar(R, bgcolor = (0, 0, 0))
```
![그림 2-1](https://user-images.githubusercontent.com/79675366/140275174-d381fbd6-2cd3-451e-b6e8-4883be62f03c.png)
![그림 2-2](https://user-images.githubusercontent.com/79675366/140275208-b5aa2ff5-f6b2-4a22-be5e-211988273b74.png)

## 그림3

```
import numpy as np
import cv2
import utils.config as cnf
from utils import kitti_bev_utils
from utils import kitti_utils
from utils import kitti_bev_utils
from utils.mayavi_viewer import draw_lidar

velo_filename ='.../complex_yolo/data/KITTI/object/testing/velodyne/%.bin'
P = kitti_utils.load_velo_scan(velo_filename)  ## Point Cloud 계산
R = kitti_bev_utils.removePoints(P, cnf.boundary)  #원하는 Space 구성

rgb_data_1 = makeBVFeature(R, cnf.DISCRETIZATION, cnf.boundary)
rgb_data_1.resize(608,608,3)

cv2.imshow('img',rgb_data_1)

```
<img width="301" alt="RGB" src="https://user-images.githubusercontent.com/79675366/140276000-f5f6f2d4-3119-44b3-adc5-19514d8a7dc0.PNG">

## training

```
$ python train.py --epochs= --batch_size= --gradient_accumulations= --n_cpu= --evaluation_interval= --pretrained_weights=''           #[--model_def MODEL_DEF]
            --img_size=IMG_SIZE
            --multiscale_training 
```
설명 참조
if __name__ == "__main__":\
    parser = argparse.ArgumentParser()\
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")\
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")\
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")\
    parser.add_argument("--model_def", type=str, default="config/complex_yolov3.cfg", help="path to model definition file")\
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")\
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")\
    parser.add_argument("--img_size", type=int, default=cnf.BEV_WIDTH, help="size of each image dimension")\
    parser.add_argument("--evaluation_interval", type=int, default=2, help="interval evaluations on validation set")\
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")\
    opt = parser.parse_args()\
    print(opt)

## Test

```
$ python test_detection.py
```
![Testing](https://user-images.githubusercontent.com/79675366/140277127-62ceed09-0d84-4be2-b635-0cae8f877584.jpg)
![Testing_BEV](https://user-images.githubusercontent.com/79675366/140277156-d7e3c2c5-1ca0-4755-9ff9-138e6007246e.jpg)



