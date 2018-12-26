#!/bin/bash
# Reference: https://github.com/priya-dwivedi/Deep-Learning/issues/14
ls images | grep ".png" | sed s/.png// > annotations/trainval.txt
