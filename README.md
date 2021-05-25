# faster_rcnn_project
project to make implementation of faster rcnn (https://arxiv.org/abs/1506.01497) in pytorch.

This network learns to identify a set of objects in images and surround them with a bounding box.

There are two networks: a region proposal network which suggests initial boxes, which are fed to the classification network, which determines what are inside of those boxes, if anything at all. 

See tcm3_RPN_final.ipynb for sample of code to run this code.
