#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:35:47 2020

@author: natewagner
"""


from pdf2image import convert_from_path
import pandas as pd
import pytesseract
from PIL import Image
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
import cv2 as cv
from matplotlib import pyplot as plt
from pdf2image import convert_from_path




path = '/Users/natewagner/Documents/Surveys/batch10/survey-page-3.pdf'


def parseQuestion5_meanFunc(page, temp1, temp2):
    # set path
    setpath = '/Users/natewagner/Documents/Surveys/'
    path = setpath + page


    #path = '/Users/natewagner/Documents/Surveys/batch6/survey-page-100.pdf'
    images = convert_from_path(path)
    images_bw = images[0].convert('L') 
    images_bw = images_bw.transpose(Image.ROTATE_270)
    images_bw_array = np.array(images_bw)
    #images_bw.show()
    
    
    
    
    # convert to binary
    wd, ht = images_bw.size
    pix = np.array(images_bw.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
    #plt.imshow(bin_img, cmap='gray')
    #plt.savefig('binary.png')
    
    
    def find_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score
    
    
    delta = 1
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)
    
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    #print('Best angle: {}'.formate(best_angle))
    
    # correct skew
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    images_bw = im.fromarray((255 * data).astype("uint8")).convert("L")
    
    #images_bw.show()
    images_bw_array = np.array(images_bw)

    
    #question5_dims = (1100, 170, 1575, 470)
    #question5 = images_bw.crop(question5_dims)
    #question5
    #question5.save("/Users/natewagner/Documents/temp5after_deskew.jpg") 
    
    
    
    
    
    
    
    #Image to pass in:
    img = images_bw_array
    img2 = img.copy()
    
    # set template
    #template = cv.imread('/Users/natewagner/Documents/temp5after_deskew.jpg',0)
    template = temp1
    
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    #methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
    #            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    
    img = img2.copy()
    method = eval('cv.TM_CCOEFF')
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 0, 5)
    #plt.subplot(121),plt.imshow(res,cmap = 'gray')
    #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(img,cmap = 'gray')
    #plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    #plt.suptitle('cv.TM_CCOEFF')
    #plt.show()
    
    
    crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    #plt.imshow(crop_img,cmap = 'gray')
    #plt.show()
    
    Q5_straight = Image.fromarray(crop_img)
    
    
    #question5_dims1 = (60, 40, 450, 290)
    #question51 = Q5_straight.crop(question5_dims1)
    #question51
    #question51.save("/Users/natewagner/Documents/temp5after_deskewZOOM.jpg") 
    
    
    Q5_straight_array = np.array(Q5_straight)
    
    
    
    
    
    
    
    #Image to pass in:
    img = Q5_straight_array
    img2 = img.copy()
    
    # set template
    #template = cv.imread('/Users/natewagner/Documents/temp5after_deskewZOOM.jpg',0)
    template = temp2
    
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    #methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
    #            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    
    img = img2.copy()
    method = eval('cv.TM_CCOEFF')
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 0, 5)
    #plt.subplot(121),plt.imshow(res,cmap = 'gray')
    #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(img,cmap = 'gray')
    #plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    #plt.suptitle('cv.TM_CCOEFF')
    #plt.show()
    
    
    crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    #plt.imshow(crop_img,cmap = 'gray')
    #plt.show()
    
    
    
    Q5_straight_zoomed = Image.fromarray(crop_img)
    Q5_straight_zoomed_arr = np.array(Q5_straight_zoomed)
    #Q5_straight_zoomed.show()
    Q5_straight_zoomed_arr[Q5_straight_zoomed_arr != 0] = 255 
    Q5_straight_zoomed = Image.fromarray(Q5_straight_zoomed_arr)
    
    
    
    actual_guessed = []
     
       
    # question 5_p1
    question5_p1_dims1 = (5, 10, 100, 40)
    question5_p1_dims2 = (150, 10, 245, 40)
    question5_p1_dims3 = (285, 10, 380, 40)
    
    question5_p11 = Q5_straight_zoomed.crop(question5_p1_dims1)
    question5_p12 = Q5_straight_zoomed.crop(question5_p1_dims2)
    question5_p13 = Q5_straight_zoomed.crop(question5_p1_dims3)
    
    
    mean_pix1 = np.mean(np.array(question5_p11))
    mean_pix2 = np.mean(np.array(question5_p12))
    mean_pix3 = np.mean(np.array(question5_p13))
    
    
    d1 = abs(mean_pix1 - mean_pix2)+0.00001 
    d2 = abs(mean_pix1 - mean_pix3)+0.00002
    d3 = abs(mean_pix2 - mean_pix3)+0.00003
    
    if d1 and d2 and d3 < 2:
        actual_guessed.append(0)
    else:
        if mean_pix1 > mean_pix2 and mean_pix1 > mean_pix3:
            actual_guessed.append(1)
        if mean_pix2 > mean_pix1 and mean_pix2 > mean_pix3:
            actual_guessed.append(2)
        if mean_pix3 > mean_pix1 and mean_pix3 > mean_pix2:
            actual_guessed.append(3)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # question 5_p2
    question5_p2_dims1 = (5, 40, 100, 70)
    question5_p2_dims2 = (150, 40, 245, 70)
    question5_p2_dims3 = (285, 40, 380, 70)
    
    question5_p21 = Q5_straight_zoomed.crop(question5_p2_dims1)
    question5_p22 = Q5_straight_zoomed.crop(question5_p2_dims2)
    question5_p23 = Q5_straight_zoomed.crop(question5_p2_dims3)
    
    mean_pix1 = np.mean(np.array(question5_p21))
    mean_pix2 = np.mean(np.array(question5_p22))
    mean_pix3 = np.mean(np.array(question5_p23))
    
    d1 = abs(mean_pix1 - mean_pix2)+0.00001 
    d2 = abs(mean_pix1 - mean_pix3)+0.00002
    d3 = abs(mean_pix2 - mean_pix3)+0.00003
    
    if d1 and d2 and d3 < 2:
        actual_guessed.append(0)
    else:
        if mean_pix1 > mean_pix2 and mean_pix1 > mean_pix3:
            actual_guessed.append(1)
        if mean_pix2 > mean_pix1 and mean_pix2 > mean_pix3:
            actual_guessed.append(2)
        if mean_pix3 > mean_pix1 and mean_pix3 > mean_pix2:
            actual_guessed.append(3)
    
    
    
    
    
    
    
    # question 5_p3
    question5_p3_dims1 = (5, 70, 100, 100)
    question5_p3_dims2 = (150, 70, 245, 100)
    question5_p3_dims3 = (285, 70, 380, 100)
    
    question5_p31 = Q5_straight_zoomed.crop(question5_p3_dims1)
    question5_p32 = Q5_straight_zoomed.crop(question5_p3_dims2)
    question5_p33 = Q5_straight_zoomed.crop(question5_p3_dims3)
    
    mean_pix1 = np.mean(np.array(question5_p31))
    mean_pix2 = np.mean(np.array(question5_p32))
    mean_pix3 = np.mean(np.array(question5_p33))
    
    d1 = abs(mean_pix1 - mean_pix2)+0.00001
    d2 = abs(mean_pix1 - mean_pix3)+0.00002
    d3 = abs(mean_pix2 - mean_pix3)+0.00003
    
    if d1 and d2 and d3 < 2:
        actual_guessed.append(0)
    else:
        if mean_pix1 > mean_pix2 and mean_pix1 > mean_pix3:
            actual_guessed.append(1)
        if mean_pix2 > mean_pix1 and mean_pix2 > mean_pix3:
            actual_guessed.append(2)
        if mean_pix3 > mean_pix1 and mean_pix3 > mean_pix2:
            actual_guessed.append(3)
            
    
    
    
    
    
    # question 5_p4
    question5_p4_dims1 = (5, 100, 100, 130)
    question5_p4_dims2 = (150, 100, 245, 130)
    question5_p4_dims3 = (285, 100, 380, 130)
    
    question5_p41 = Q5_straight_zoomed.crop(question5_p4_dims1)
    question5_p42 = Q5_straight_zoomed.crop(question5_p4_dims2)
    question5_p43 = Q5_straight_zoomed.crop(question5_p4_dims3)
    
    mean_pix1 = np.mean(np.array(question5_p41))
    mean_pix2 = np.mean(np.array(question5_p42))
    mean_pix3 = np.mean(np.array(question5_p43))
    
    
    d1 = abs(mean_pix1 - mean_pix2)+0.00001
    d2 = abs(mean_pix1 - mean_pix3)+0.00002
    d3 = abs(mean_pix2 - mean_pix3)+0.00003
    
    if d1 and d2 and d3 < 2:
        actual_guessed.append(0)
    else:
        if mean_pix1 > mean_pix2 and mean_pix1 > mean_pix3:
            actual_guessed.append(1)
        if mean_pix2 > mean_pix1 and mean_pix2 > mean_pix3:
            actual_guessed.append(2)
        if mean_pix3 > mean_pix1 and mean_pix3 > mean_pix2:
            actual_guessed.append(3)
    
    
    
    
    
    
    # question 5_p5
    question5_p5_dims1 = (5, 130, 100, 160)
    question5_p5_dims2 = (150, 130, 245, 160)
    question5_p5_dims3 = (285, 130, 380, 160)
    
    question5_p51 = Q5_straight_zoomed.crop(question5_p5_dims1)
    question5_p52 = Q5_straight_zoomed.crop(question5_p5_dims2)
    question5_p53 = Q5_straight_zoomed.crop(question5_p5_dims3)
    
    mean_pix1 = np.mean(np.array(question5_p51))
    mean_pix2 = np.mean(np.array(question5_p52))
    mean_pix3 = np.mean(np.array(question5_p53))
    
    d1 = abs(mean_pix1 - mean_pix2)+0.00001
    d2 = abs(mean_pix1 - mean_pix3)+0.00002
    d3 = abs(mean_pix2 - mean_pix3)+0.00003
    
    if d1 and d2 and d3 < 2:
        actual_guessed.append(0)
    else:
        if mean_pix1 > mean_pix2 and mean_pix1 > mean_pix3:
            actual_guessed.append(1)
        if mean_pix2 > mean_pix1 and mean_pix2 > mean_pix3:
            actual_guessed.append(2)
        if mean_pix3 > mean_pix1 and mean_pix3 > mean_pix2:
            actual_guessed.append(3)
    
    
    
    
    
    # question 5_p6
    question5_p6_dims1 = (5, 160, 100, 190)
    question5_p6_dims2 = (150, 160, 245, 190)
    question5_p6_dims3 = (285, 160, 380, 190)
    
    question5_p61 = Q5_straight_zoomed.crop(question5_p6_dims1)
    question5_p62 = Q5_straight_zoomed.crop(question5_p6_dims2)
    question5_p63 = Q5_straight_zoomed.crop(question5_p6_dims3)
    
    mean_pix1 = np.mean(np.array(question5_p61))
    mean_pix2 = np.mean(np.array(question5_p62))
    mean_pix3 = np.mean(np.array(question5_p63))
    
    d1 = abs(mean_pix1 - mean_pix2)+0.00001
    d2 = abs(mean_pix1 - mean_pix3)+0.00002
    d3 = abs(mean_pix2 - mean_pix3)+0.00003
    
    if d1 and d2 and d3 < 2:
        actual_guessed.append(0)
    else:
        if mean_pix1 > mean_pix2 and mean_pix1 > mean_pix3:
            actual_guessed.append(1)
        if mean_pix2 > mean_pix1 and mean_pix2 > mean_pix3:
            actual_guessed.append(2)
        if mean_pix3 > mean_pix1 and mean_pix3 > mean_pix2:
            actual_guessed.append(3)
    
    
    
    
    
    
    
    # question 5_p7
    question5_p7_dims1 = (5, 190, 100, 220)
    question5_p7_dims2 = (150, 190, 245, 220)
    question5_p7_dims3 = (285, 190, 380, 220)
    
    question5_p71 = Q5_straight_zoomed.crop(question5_p7_dims1)
    question5_p72 = Q5_straight_zoomed.crop(question5_p7_dims2)
    question5_p73 = Q5_straight_zoomed.crop(question5_p7_dims3)
    
    mean_pix1 = np.mean(np.array(question5_p71))
    mean_pix2 = np.mean(np.array(question5_p72))
    mean_pix3 = np.mean(np.array(question5_p73))
    
    d1 = abs(mean_pix1 - mean_pix2)+0.00001 
    d2 = abs(mean_pix1 - mean_pix3)+0.00002
    d3 = abs(mean_pix2 - mean_pix3)+0.00003
    
    if d1 and d2 and d3 < 2:
        actual_guessed.append(0)
    else:
        if mean_pix1 > mean_pix2 and mean_pix1 > mean_pix3:
            actual_guessed.append(1)
        if mean_pix2 > mean_pix1 and mean_pix2 > mean_pix3:
            actual_guessed.append(2)
        if mean_pix3 > mean_pix1 and mean_pix3 > mean_pix2:
            actual_guessed.append(3)
    
    
    
    
    
    
    # question 5_p8
    question5_p8_dims1 = (5, 220, 100, 250)
    question5_p8_dims2 = (150, 220, 245, 250)
    question5_p8_dims3 = (285, 220, 380, 250)
    
    question5_p81 = Q5_straight_zoomed.crop(question5_p8_dims1)
    question5_p82 = Q5_straight_zoomed.crop(question5_p8_dims2)
    question5_p83 = Q5_straight_zoomed.crop(question5_p8_dims3)
    
    mean_pix1 = np.mean(np.array(question5_p81))
    mean_pix2 = np.mean(np.array(question5_p82))
    mean_pix3 = np.mean(np.array(question5_p83))
    
    d1 = abs(mean_pix1 - mean_pix2)+0.00001 
    d2 = abs(mean_pix1 - mean_pix3)+0.00002
    d3 = abs(mean_pix2 - mean_pix3)+0.00003
    
    if d1 and d2 and d3 < 2:
        actual_guessed.append(0)
    else:
        if mean_pix1 > mean_pix2 and mean_pix1 > mean_pix3:
            actual_guessed.append(1)
        if mean_pix2 > mean_pix1 and mean_pix2 > mean_pix3:
            actual_guessed.append(2)
        if mean_pix3 > mean_pix1 and mean_pix3 > mean_pix2:
            actual_guessed.append(3)
            
    
    return actual_guessed



temp1 = cv.imread('/Users/natewagner/Documents/temp5after_deskew.jpg',0)
temp2 = cv.imread('/Users/natewagner/Documents/temp5after_deskewZOOM.jpg',0)

test = parseQuestion5_meanFunc("batch10/survey-page-3.pdf", temp1, temp2)


pdfs = [100, 100, 100]
bats = [10, 16, 6]

batch = 'batch'
survey_num = '/survey-page-'
end = '.pdf'
question5_data = []
for pages in bats:
    for x in range(1, 101):        
        survey = batch + str(pages) + survey_num + str(x) + end         
        df = parseQuestion5_meanFunc(survey, temp1, temp2)
        if len(df) != 8:
            print(survey)
        question5_data.append(df)
        #print(survey)
    #checkYN['batch'] = str(pages)
    print(pages)

flat_list = []
for sublist in question5_data:
    for item in sublist:
        flat_list.append(float(item))


actual_yes_no = pd.read_csv("/Users/natewagner/Documents/Surveys/train_data_with_actualQ5.csv")
act = actual_yes_no['actual']
act_list = list(act)

error = np.mean( np.array(flat_list[0:301]) != np.array(act_list[0:301]) )
accuracy = 1 - error

print(flat_list[:50])
print(act_list[:50])



