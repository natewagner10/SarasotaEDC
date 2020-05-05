#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:22:57 2020

@author: natewagner
"""



from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
#%matplotlib inline
from os import listdir
from xml.etree import ElementTree




class myMaskRCNNConfig(Config):
    NAME = "MaskRCNN_config"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    NUM_CLASSES = 2+1
   
    STEPS_PER_EPOCH = 240
    
    LEARNING_RATE=0.006
     
    DETECTION_MIN_CONFIDENCE = 0.40
    
    RPN_NMS_THRESHOLD = 0.20
    
    MAX_GT_INSTANCES=63
    
    DETECTION_MAX_INSTANCES = 63
    
    VALIDATION_STEPS = 60
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    

config = myMaskRCNNConfig()
config.display()


class surveyDataset(Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        

        self.add_class("dataset", 1, "box")
        self.add_class("dataset", 2, "checked_box")
        

        images_dir = dataset_dir + '/images_sharp/'
        annotations_dir = dataset_dir + '/annotations/'
        
        cnt = 0
        for filename in listdir(images_dir):
        
            cnt += 1
      
            image_id = filename[:-4]
                  
            if image_id in have:
                continue
            if image_id in ['.DS_Store']:
                continue
            if image_id in ['.DS_S.xml']:
                continue
            
            if is_train and cnt >= 241:
                continue
           
            if not is_train and cnt <= 241:
                continue
     
            img_path = images_dir + filename
            
            ann_path = annotations_dir + image_id + '.xml'
            
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def extract_boxes(self, filename):
        

        tree = ElementTree.parse(filename)

        root = tree.getroot()

        obj_class = list()
        for cl in root.findall('.//name'):
            obj_class.append(cl.text)
        
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        
       
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height, obj_class

    """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
     """
    def load_mask(self, image_id):

        info = self.image_info[image_id]
        
        path = info['annotation']
        
        # load XML
        boxes, w, h, ob_cls = self.extract_boxes(path)
       
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            if ob_cls[i] == "box":
                class_ids.append(self.class_names.index('box'))
            else:
                class_ids.append(self.class_names.index('checked_box'))
        return masks, asarray(class_ids, dtype='int32')
# load an image reference
     #"""Return the path of the image."""
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']



# prepare train set
train_set = surveyDataset()
train_set.load_dataset('/Users/natewagner/Documents/ML_Final_Project/', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

have = []
for x in range(0, 240):
    have.append(train_set.image_info[x]['id'])
    
               
# prepare test/val set
test_set = surveyDataset()
test_set.load_dataset('/Users/natewagner/Documents/ML_Final_Project/', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))





print("Loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="training", config=config, model_dir='./')




#load the weights for COCO
model.load_weights('/Users/natewagner/MASK_RCNN/mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])




## train heads with higher lr to speedup the learning
model.train(train_set, test_set, learning_rate=2*config.LEARNING_RATE, epochs=2, layers='heads')
history = model.keras_model.history.history


################################
###        model eval        ###   
################################

import matplotlib.pyplot as plt
#plt.plot(history['accuracy'])
plt.plot(history['val_loss'])
plt.plot(history['mrcnn_bbox_loss'])
plt.plot(history['mrcnn_mask_loss'])
plt.plot(history['mrcnn_class_loss'])
plt.plot(history['rpn_bbox_loss'])
plt.plot(history['rpn_class_loss'])
plt.plot(history['val_mrcnn_mask_loss'])
plt.title('Model loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()









#import time
model_path = '/Users/natewagner/Documents/ML_Final_Project/mask_rcnn_'  + '.' + str(time.time()) + '.h5'
model.keras_model.save_weights(model_path)

# good
#model_path = '/Users/natewagner/Documents/ML_Final_Project/mask_rcnn_.1587772339.828717.h5'
#model_path = '/Users/natewagner/Documents/ML_Final_Project/mask_rcnn_.1588432666.050406.h5'
#model_path = '/Users/natewagner/Documents/ML_Final_Project/mask_rcnn_.1588546554.8807812.h5'
#model_path = '/Users/natewagner/Documents/ML_Final_Project/mask_rcnn_.1588559854.111884.h5'
# best so far 94%
#model_path = '/Users/natewagner/Documents/ML_Final_Project/mask_rcnn_.1588546554.8807812.h5'

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#Loading the model in the inference mode
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='/Users/natewagner/Documents/ML_Final_Project/mask_rcnn_.1588559854.111884.h5')
# loading the trained weights o the custom dataset
model.load_weights(model_path, by_name=True)
img = load_img("/Users/natewagner/Documents/ML_Final_Project/images/survey-page-B16-52.jpg")
#img = load_img("/Users/natewagner/Documents/ML_Final_Project/new_image_arranged.jpg")

img = img_to_array(img)
# detecting objects in the image
results = model.detect([img], verbose = 1)






r = results[0]
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                            ['BG', 'box', 'checked_box'], r['scores'], 
                            title="Predictions")


############################################################################################################

  
################################
###        crop from rois    ###   
################################    

from PIL import Image
im = Image.open('/Users/natewagner/Documents/ML_Final_Project/images16/survey-page-B16-42.jpg')    
rl = results[0]['rois'][35]
c_dims = list(rl)
left = c_dims[1]
upper = c_dims[0]
right = c_dims[3]
lower = c_dims[2]

im.crop([left, upper, right, lower])    
im.show()
    
    



################################
###        extract data      ###   
################################
im = Image.open('/Users/natewagner/Documents/ML_Final_Project/images16/survey-page-B16-56.jpg')    
im.resize((1024,1024)).show()
#im.filter(ImageFilter.EDGE_ENHANCE_MORE).show()

import operator

rois_list = list(r['rois'])
mark = list(r['class_ids'])
len(rois_list)


np.append(rois_list[0], mark[0])

new_arrs = []
for x in range(0, len(mark)):
    new_arrs.append(np.append(rois_list[x], mark[x]))




new_arrs
sorted_rois = sorted(new_arrs, key=operator.itemgetter(0))

c_dims = sorted_rois[0]
left = c_dims[1]
upper = c_dims[0]
right = c_dims[3]
lower = c_dims[2]
im.crop([left, upper, right, lower])    




q5_r1 = sorted([sorted_rois[0], sorted_rois[1], sorted_rois[2]], key=operator.itemgetter(1))
q5_r2 = sorted([sorted_rois[3], sorted_rois[4], sorted_rois[5]], key=operator.itemgetter(1))
q5_r3 = sorted([sorted_rois[6], sorted_rois[7], sorted_rois[8]], key=operator.itemgetter(1))
q5_r4 = sorted([sorted_rois[9], sorted_rois[10], sorted_rois[11]], key=operator.itemgetter(1))
q5_r5 = sorted([sorted_rois[12], sorted_rois[13], sorted_rois[14]], key=operator.itemgetter(1))
q5_r6 = sorted([sorted_rois[15], sorted_rois[16], sorted_rois[17]], key=operator.itemgetter(1))
q5_r7 = sorted([sorted_rois[18], sorted_rois[19], sorted_rois[20]], key=operator.itemgetter(1))
q5_r8 = sorted([sorted_rois[21], sorted_rois[22], sorted_rois[23]], key=operator.itemgetter(1))

q10 = sorted( [sorted_rois[24], sorted_rois[25], sorted_rois[26], sorted_rois[27],
               sorted_rois[28], sorted_rois[29], sorted_rois[30],
               sorted_rois[31], sorted_rois[32], sorted_rois[33] ], key=operator.itemgetter(1))

q10_p1 = q10[0]
q10_p2 = q10[1]
q10_p3 = q10[2]
q10_p4 = q10[3]
q10_p5 = q10[4]
q10_p6 = q10[5]
q10_p7 = q10[6]
q10_p8 = q10[7]
q10_p9 = q10[8]
q10_p10 = q10[9]


q7_p11 = sorted([sorted_rois[34], sorted_rois[35]], key=operator.itemgetter(1))
q7_p22 = sorted([sorted_rois[36], sorted_rois[37]], key=operator.itemgetter(1))
q7_p33 = sorted([sorted_rois[38], sorted_rois[39]], key=operator.itemgetter(1))
q7_p44 = sorted([sorted_rois[42], sorted_rois[43]], key=operator.itemgetter(1))

q7_p1 = q7_p11[0]
q7_p2 = q7_p11[1]
q7_p3 = q7_p22[0]
q7_p4 = q7_p22[1]
q7_p5 = q7_p33[0]
q7_p6 = q7_p33[1]
q7_p7 = q7_p44[0]
q7_p8 = q7_p44[1]



q2_p = sorted([sorted_rois[40], sorted_rois[41]], key=operator.itemgetter(1))
q2_yes = q2_p[0]
q2_no = q2_p[1]



############################################################################################################



################################
###        enhance edges     ###   
################################

from PIL import ImageFilter    

image_dir = '/Users/natewagner/Documents/ML_Final_Project/images/'
for filename in listdir(image_dir):
    if '.DS_Store' in filename:
        continue
    im = Image.open(image_dir + filename)
    im_a = np.array(im)
    im_a[im_a < 200] = 0
    im2 = Image.fromarray(im_a.astype(np.uint8))
    #im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
    im2.save('/Users/natewagner/Documents/ML_Final_Project/images_sharp/' + filename)






################################
###         model eval       ###   
################################

from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from numpy import expand_dims
from mrcnn.utils import compute_ap
from numpy import mean


def evaluate_model(dataset, model, cfg, io):
    APs, prs, rcs = list(), list(), list()
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)
        r = yhat[0]
        AP, precisions, recalls, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold = io)
        APs.append(AP)
        rcs.append(recalls)
        prs.append(precisions)
    mAP = mean(APs)
    return mAP, APs, prs, rcs



# evaluate model on training dataset
train_mAP, aps, prs, rcs = evaluate_model(train_set, model, config)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP, aps, prs, rcs = evaluate_model(test_set, model, config)
print("Test mAP: %.3f" % test_mAP)

test_mAP_40, aps_40, prs_40, rcs_40 = evaluate_model(test_set, model, config, 0.40)
test_mAP_50, aps_50, prs_50, rcs_50 = evaluate_model(test_set, model, config, 0.50)
test_mAP_60, aps_60, prs_60, rcs_60 = evaluate_model(test_set, model, config, 0.60)
test_mAP_70, aps_70, prs_70, rcs_70 = evaluate_model(test_set, model, config, 0.70)
test_mAP_80, aps_80, prs_80, rcs_80 = evaluate_model(test_set, model, config, 0.80)
test_mAP_90, aps_90, prs_90, rcs_90 = evaluate_model(test_set, model, config, 0.90)



#from mrcnn.visualize import plot_precision_recall
#plot_precision_recall(test_mAP_50, prs_50[1], rcs_50[1])
#plot_precision_recall(test_mAP_75, prs_75[1], rcs_75[1])
#plot_precision_recall(test_mAP_90, prs_90[1], rcs_90[1])
#plt.show()

pyplot.plot(prs_90[0], rcs_90[0], label = "IoU: 0.90 - mAP: 0.16")
pyplot.plot(prs_80[0], rcs_80[0], label = "IoU: 0.80 - mAP: 0.58")
pyplot.plot(prs_70[0], rcs_70[0], label = "IoU: 0.70 - mAP: 0.83")
pyplot.plot(prs_60[0], rcs_60[0], label = "IoU: 0.60 - mAP: 0.93")
pyplot.plot(prs_50[0], rcs_50[0], label = "IoU: 0.50 - mAP: 0.94")
pyplot.plot(prs_40[0], rcs_40[0], label = "IoU: 0.40 - mAP: 0.94")
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.title("Precision-Recall Curve")
pyplot.legend()
pyplot.show()





################################
###   new design template    ###   
################################

im = Image.open('/Users/natewagner/Documents/ML_Final_Project/images/survey-page-B16-51.jpg')    
im.size

left_dims = (70, 545, 500, 1190)
left = im.crop(left_dims)
left

q6_10_dims = (520, 435, 1660, 1000)
q6_10 = im.crop(q6_10_dims)
q6_10.show()

question5_dims = (500, 125, 1620, 420)
question5 = im.crop(question5_dims)
question5.show()

new_image = Image.new('L', (1700, 2200), 'white')
new_image.paste(q6_10, (50, 50))
new_image.paste(left, (1200, 50))
new_image.paste(question5, (10, 650))
new_image = new_image.filter(ImageFilter.EDGE_ENHANCE)
new_image.show()
#new_image.save('/Users/natewagner/Documents/ML_Final_Project/new_image_arranged.jpg')


np.round(r['scores'], 2)



im_a = np.array(im)



im_a[im_a < 200] = 0

Image.fromarray(im_a.astype(np.uint8)).show()














