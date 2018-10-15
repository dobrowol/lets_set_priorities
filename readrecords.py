#!/usr/bin/env python3
import re
import os
import numpy as np
class obj(object):
    def __init__(self):
        self.label='';
        self.orglabel='';
        self.bbox=[];
        self.polygon=[];
        self.mask='';
        self.det=False;

class record(object):
    def __init__(self):
    	self.imagename='';
    	self.imgsize=[];
    	self.database='';
    	self.objects=[];


pattern="Image filename : \"(.*)\"|Bounding box for object (\d+) (\".*\") \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)|Image size \(X x Y x C\) : (\d+) x (\d+) x (\d+)|Database : (.*)|Pixel mask for object (\d+) (.*) : \"(.*)\"|Original label for object (\d+) \"(.*)\" : \"(.*)\""
def readAnnotations():
    recs = []
    for root, subdirs, files in os.walk('./VOCdata/VOC2005_1/Annotations/'):
        for f in files:
            filename = os.path.join(root,f)
            recs.append(readrecord(filename))
    return recs

def constructYvector(rec, input_size):
    number_of_cells=7
    Y=np.zeros((number_of_cells, number_of_cells,9))
    cell_size=[rec.imgsize[0]/number_of_cells, rec.imgsize[1]/number_of_cells]
    print("imgsize ", rec.imgsize)
    print("cellsize ", cell_size)
    for o in rec.objects:
        y=[0,0,0,0,0,0,0,0,0]
        if o.label=='VOCmotorbikes':
            y[5]=1
        if o.label=='VOCbicycles':
            y[6]=1
        if o.label=='VOCpeople':
            y[7]=1
        if o.label=='VOCcars':
            y[8]=1
        if o.bbox:
            width=(o.bbox[2]-o.bbox[0])
            height=(o.bbox[3]-o.bbox[1])
            center_of_box=[o.bbox[2]-width/2, o.bbox[3]-height/2]
            
            row_where_center_is=int(center_of_box[0]/cell_size[0])
            col_where_center_is=int(center_of_box[1]/cell_size[1])
            print("center of box", center_of_box)
            print ("row: ", row_where_center_is, " col: ", col_where_center_is)
            relative_center_of_box_x = (center_of_box[0]%cell_size[0])/cell_size[0]
            relative_center_of_box_y = (center_of_box[1]%cell_size[1])/cell_size[1]
            y[0]=1
            y[1]=relative_center_of_box_x
            y[2]=relative_center_of_box_y
            y[3]=width/rec.imgsize[0]
            y[4]=height/rec.imgsize[1]
            Y[row_where_center_is,col_where_center_is,:]=y
        
    return Y

def readrecord(filename):
    with open(filename,'rt') as fd:
        rec=record() 
        for line in fd:
            a=re.findall(pattern, line)
            if a :
                print (a)
                if a[0][0] is not '':
                    rec.imagename=a[0][0]
                if a[0][1] is not '':
                    boundedObject=obj()
                    if len(rec.objects) >= int(a[0][1]):
                        boundedObject = rec.objects[int(a[0][1])-1] 
                    boundedObject.label = a[0][2]
                    boundedObject.bbox=[int(a[0][3]), int(a[0][4]), int(a[0][5]), int(a[0][6])]
                    print(boundedObject.bbox)
                    if len(rec.objects) < int(a[0][1]):
                        rec.objects.append(boundedObject)
                if a[0][7] is not '':
                    rec.imgsize=[int(a[0][7]), int(a[0][8]), int(a[0][9])]
                    print(rec.imgsize)
                if a[0][10] is not '':
                    rec.database=a[0][10]
                if a[0][11] is not '':
                    boundedObject=obj()
                    if len(rec.objects) >= int(a[0][11]):
                        boundedObject = rec.objects[int(a[0][11])-1] 
                    boundedObject.mask=a[0][13]   
                    if len(rec.objects) < int(a[0][11]):
                        rec.objects.append(boundedObject)
                if a[0][14] is not '':
                    boundedObject = obj()
                    if len(rec.objects) >= int(a[0][14]):
                        boundedObject = rec.objects[int(a[0][14])-1] 
                    boundedObject.orglabel=a[0][16]
                    if len(rec.objects) < int(a[0][14]):
                        rec.objects.append(boundedObject)
    return rec
