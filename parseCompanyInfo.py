#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:33:50 2020

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
#from matplotlib import pyplot as plt
#from pdf2image import convert_from_path


def templateMatch(img, template):
    #Image to pass in:
    img2 = img.copy()    
    w, h = template.shape[::-1]
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
    cv.rectangle(img,top_left, bottom_right, 255, 5)
    #plt.subplot(121),plt.imshow(res,cmap = 'gray')
    #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(img,cmap = 'gray')
    #plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    #plt.suptitle('cv.TM_CCOEFF')
    #plt.show()
    
    
    crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    im_from_arr = Image.fromarray(crop_img)
    
    return(im_from_arr)

def straightenImage(pic):
    # convert to binary
    wd, ht = pic.size
    pix = np.array(pic.convert('1').getdata(), np.uint8)
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
    pic_straight = im.fromarray((255 * data).astype("uint8")).convert("L")
    
    return pic_straight








def getCompInfo(page):
    
    # set path
    setpath = '/Users/natewagner/Documents/Surveys/'
    path = setpath + page
    
    
    images = convert_from_path(path)
    images_bw = images[0].convert('L') 
    images_bw = images_bw.transpose(Image.ROTATE_270)
    images_bw_array = np.array(images_bw)
    #images_bw.show()
    
    
    
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'
    
    
    # extract account number
    #accnt_num_dims = (1000, 10, 1550, 100)
    #accnt_num = images_bw.crop(accnt_num_dims) 
    #accnt_num.save("/Users/natewagner/Documents/surveys/accnt_num.jpg") 
    
    # extract name
    #name_dims = (500, 1020, 1700, 1290)
    #name_1 = images_bw.crop(name_dims) 
    #name_1.save("/Users/natewagner/Documents/surveys/comp_info.jpg") 
    
    
    
    
    
    
    img = images_bw_array
        
    # initial crop
    comp_info = templateMatch(
        img, template = cv.imread("/Users/natewagner/Documents/surveys/comp_info.jpg",0))
        
    #comp_info.show()
    
    # extract name
    name_dims = (620, 1060, 1650, 1120)
    name_1 = images_bw.crop(name_dims) 
    name_1
    name = pytesseract.image_to_string(name_1)
    
    
    # extract company name
    comp_info_dims = (720, 1115, 1650, 1170)
    comp_info = images_bw.crop(comp_info_dims) 
    comp_info
    bus_name = pytesseract.image_to_string(comp_info)
    
    
    # extract phone number
    phone_number_dims = (730, 1160, 1650, 1220)
    phone_info = images_bw.crop(phone_number_dims) 
    phone_info
    phone_number = pytesseract.image_to_string(phone_info)
    
    
    # extract email
    email_dims = (730, 1200, 1650, 1280)
    email_info1 = images_bw.crop(email_dims) 
    email_info = pytesseract.image_to_string(email_info1)
    
    
    
    account_number = templateMatch(
        img, template = cv.imread("/Users/natewagner/Documents/surveys/accnt_num.jpg",0))
        
    account_number_dims = (160, 0, 500, 80)
    account_number_zoom = account_number.crop(account_number_dims)
    #account_number_zoom
    
    accnt_number_text = pytesseract.image_to_string(account_number_zoom)
    
    
    
    
    company_info = pd.DataFrame([accnt_number_text, name, 
                                bus_name, phone_number, 
                                email_info, page]).transpose()
    

    
    return company_info






pdfs = [66, 99, 123, 94, 137, 92, 138, 164, 82, 85, 46, 81, 120, 100, 91]
batches = [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18] 
        
   
batch = 'batch'
survey_num = '/survey-page-'
end = '.pdf'

comp_info = pd.DataFrame()

cnt = 0
for pages in batches:
    for x in range(1, pdfs[cnt]+1):  
        survey = batch + str(pages) + survey_num + str(x) + end
        df = getCompInfo(survey)
        comp_info = comp_info.append(df)
    cnt += 1
    print(pages)
    
    
    
#comp_info.to_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/comp_info.csv', encoding='utf-8', index = False) 
    
    









