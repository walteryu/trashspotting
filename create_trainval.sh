#!/bin/bash
ls images | grep ".png" | sed s/.png// > annotations/trainval.txt
