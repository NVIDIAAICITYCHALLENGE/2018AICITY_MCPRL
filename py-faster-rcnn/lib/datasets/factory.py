# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_test import pascal_voc_test
from datasets.pascal_voc_resnet import pascal_voc_resnet
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
#for year in ['2007', '2012']:
#    for split in ['train', 'val', 'trainval', 'test']:
#        name = 'voc_{}_{}'.format(year, split)
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
#for year in ['2014']:
#    for split in ['train', 'val', 'minival', 'valminusminival']:
#        name = 'coco_{}_{}'.format(year, split)
#        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
#for year in ['2015']:
#    for split in ['test', 'test-dev']:
#        name = 'coco_{}_{}'.format(year, split)
#        __sets[name] = (lambda split=split, year=year: coco(split, year))

#set the information about imdb
name = 'pascal_voc'
devkit = './nvidia/'
__sets['pascal_voc'] = (lambda name = name ,devkit = devkit:pascal_voc(name,devkit))

name = 'pascal_voc_2'
__sets['pascal_voc_2'] = (lambda name = name, devkit = devkit:pascal_voc(name,devkit))

name = 'pascal_voc_truck'
__sets['pascal_voc_truck'] = (lambda name = name,devkit = devkit:pascal_voc(name,devkit))

name = 'pascal_voc_all'
__sets['pascal_voc_all'] = (lambda name = name,devkit = devkit:pascal_voc(name,devkit))

name = 'pascal_voc_dark'
__sets['pascal_voc_dark'] = (lambda name = name, devkit = devkit:pascal_voc(name,devkit))

name = 'pascal_voc_vgg'
__sets['pascal_voc_vgg'] = (lambda name = name, devkit = devkit:pascal_voc(name,devkit))

name = 'pascal_voc_test'
__sets['pascal_voc_test'] = (lambda name = name,devkit = devkit:pascal_voc_test(name,devkit))

name = 'pascal_voc_resnet'
__sets['pascal_voc_resnet'] = (lambda name = name,devkit = devkit:pascal_voc_resnet(name,devkit))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
