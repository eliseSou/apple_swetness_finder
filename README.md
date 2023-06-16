# apple_swetness_finder

Hello ! this project has been originally developed as part of the Data Engineering course in GIST (2023)

For my teammates, quick guide to use gitHub : 
  1. IF you have not already, add a ssh key to your account. (Go to your setting in the access section) 
  2. Git clone the reporitory with ssh agent. 
  3. To upload your code : 
        - git fetch
        - git add * (to add all modified files, you can also add each file individually)
        - git commit (please write a small description)
        - git push 

For dataset, use data/main.py (We only use arisu data in AI hub)

Download apple dataset: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=490

For Training, use this command
```
python train.py --data data/arisu.yaml --epochs 300 --weights 'yolov5n.pt' --cfg yolov5n.yaml  --batch-size 128 --name results
```

YOLOv5 model: https://github.com/ultralytics/yolov5
