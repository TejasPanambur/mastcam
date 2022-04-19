# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:25:17 2019

@author: TEJAS
"""

from torchvision import datasets

class TrainImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, target, path)