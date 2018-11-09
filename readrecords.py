#!/usr/bin/env python3
import re
import os
import numpy as np
import re
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
def readAnnotations(annotationDirectory):
    recs = []
    for root, subdirs, files in os.walk(annotationDirectory):
        for f in files:
            filename = os.path.join(root,f)
            recs.append(readrecord(filename))
    return recs

def constructYvector(rec, input_size, output_size):
    number_of_cells=7
    Y=np.zeros((number_of_cells, number_of_cells,output_size))
    cell_size=[rec.imgsize[0]/number_of_cells, rec.imgsize[1]/number_of_cells]

    for o in rec.objects:
        y=np.zeros((output_size))
        if re.search('motorbike', o.label):
            y[6]=1
        if re.search('bicycle', o.label):
            y[7]=1
        if re.search('people', o.label):
            y[8]=1
        if re.search('car', o.label):
            y[9]=1
        if o.bbox:
            y[0] = 1
            width=(o.bbox[2]-o.bbox[0])
            height=(o.bbox[3]-o.bbox[1])
            center_of_box=[(o.bbox[2]-width)/2, (o.bbox[3]-height)/2]
            row_where_center_is=int(center_of_box[0]/cell_size[0])
            col_where_center_is=int(center_of_box[1]/cell_size[1])

            relative_center_of_box_x = (center_of_box[0]%cell_size[0])/cell_size[0]
            relative_center_of_box_y = (center_of_box[1]%cell_size[1])/cell_size[1]
            y[1]=1
            y[2]=relative_center_of_box_x
            y[3]=relative_center_of_box_y
            y[4]=width/rec.imgsize[0]
            y[5]=height/rec.imgsize[1]
        
            if row_where_center_is == 7 or col_where_center_is == 7:
                print ("corrupted record ", o.bbox, " center box ", center_of_box, " cell size ", cell_size)
            Y[row_where_center_is,col_where_center_is,:]=y
        
    return Y

def readrecord(filename):
    with open(filename,'rt') as fd:
        rec=record() 
        for line in fd:
            a=re.findall(pattern, line)
            if a :

                if a[0][0] is not '':
                    rec.imagename=a[0][0]
                if a[0][1] is not '':
                    boundedObject=obj()
                    if len(rec.objects) >= int(a[0][1]):
                        boundedObject = rec.objects[int(a[0][1])-1] 
                    boundedObject.label = a[0][2]
                    boundedObject.bbox=[int(a[0][3]), int(a[0][4]), int(a[0][5]), int(a[0][6])]

                    if len(rec.objects) < int(a[0][1]):
                        rec.objects.append(boundedObject)
                if a[0][7] is not '':
                    rec.imgsize=[int(a[0][7]), int(a[0][8]), int(a[0][9])]

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
