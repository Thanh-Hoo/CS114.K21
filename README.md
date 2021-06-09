# AIC-2021

# evaluate mAP

## 1. formant file:
```
    File is included in json format 
```
```python
[
    {"text": "ELIS NGOC", 
        "score": 0.8800175189971924, 
        "polygon": [374, 312, 960, 312, 960, 536, 374, 536],
        "BBox": [391, 317, 546, 207]}, 
    {"text": "2aio facebok", 
        "score": 0.6789452433586121, 
        "polygon": [875, 522, 1093, 522, 1093, 559, 875, 559], 
        "BBox": [870, 523, 224, 31]}, 
    {"text": "323 Le Van Sy, P1, &.Tan Binh", 
        "score": 0.6625133156776428, 
        "polygon": [206, 538, 810, 538, 810, 594, 206, 594], 
        "BBox": [200, 537, 621, 49]},
    {"text": "0898 408 916", 
        "score": 0.5985706448554993, 
        "polygon": [876, 552, 1095, 552, 1095, 593, 876, 593], 
        "BBox": [873, 554, 223, 33]},
    {"text": "demo", 
        "score": 0.5985706448554993, 
        "polygon": [304, 588, 564, 588, 564, 614, 304, 614], 
        "BBox": [873, 554, 223, 33]}
]

```

## 2. Setup folder
* Ground truth folder
* Predicted results folder
```
    The folders will contain the JSON file that is the content of the image.
  For each image, there will be a JSON file  
```
