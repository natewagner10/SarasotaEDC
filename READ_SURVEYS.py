#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:05:54 2020

@author: natewagner
"""


from pdf2image import convert_from_path
import pandas as pd
#import pytesseract
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


def parseYES_NO(page):
    # set path
    setpath = '/Users/natewagner/Documents/Surveys/'
    path = setpath + page

    # convert from pdf to png
    images = convert_from_path(path)
    
    # convert to greyscale
    images_bw = images[0].convert('L') 
    
    # rotate image
    images_bw = images_bw.transpose(Image.ROTATE_270)
    
    # convert to numpy array
    images_bw_array = np.array(images_bw)

    
    # set up image
    img = images_bw_array
    
    # initial crop
    survey_zoomed = templateMatch(
        img, template = cv.imread('/Users/natewagner/Documents/full_survey_temp.jpg',0))
    
    
    # set up image
    img = np.array(survey_zoomed)
    
    
    # template match Q2
    q2_zoomed = templateMatch(img, template = cv.imread('/Users/natewagner/Documents/Surveys/Q2_zoom.jpg',0))
    
    # template match Q3
    q3_zoomed = templateMatch(img, template = cv.imread('/Users/natewagner/Documents/Surveys/Q3_zoom.jpg',0))

    # template match Q4    
    q4_zoomed = templateMatch(img, template = cv.imread('/Users/natewagner/Documents/Surveys/Q4_zoom.jpg',0))

    # template match Q8 and Q9    
    q8_9_zoomed = templateMatch(img, template = cv.imread('/Users/natewagner/Documents/Surveys/Q89_zoom.jpg',0))


    
    
    def getData():
    
        #Q2_dims = (70, 550, 500, 800)
        #Q2 = images_bw.crop(Q2_dims)
        #Q2.save("/Users/natewagner/Documents/Surveys/Q2_zoom.jpg") 
        
        #Q3_dims = (70, 775, 500, 1075)
        #Q3 = images_bw.crop(Q3_dims)
        #Q3
        #Q3.save("/Users/natewagner/Documents/Surveys/Q3_zoom.jpg") 
        
        #Q4_dims = (70, 1050, 510, 1230)
        #Q4 = images_bw.crop(Q4_dims)
        #Q4
        #Q4.save("/Users/natewagner/Documents/Surveys/Q4_zoom.jpg") 
        
        #Q89_dims = (520, 735, 1560, 890)
        #Q89 = images_bw.crop(Q89_dims)
        #Q89.show()
        #Q89.save("/Users/natewagner/Documents/Surveys/Q89_zoom.jpg") 
        
        #Q9_dims = (520, 790, 1500, 875)
        #Q9 = images_bw.crop(Q9_dims)
        #Q9.show()
        #Q9.save("/Users/natewagner/Documents/Surveys/Q9_zoom.jpg") 
        return 5+5
    

    
    # straighten Q2
    q2_zoomed_str8 = straightenImage(q2_zoomed)

    # straighten Q3    
    q3_zoomed_str8 = straightenImage(q3_zoomed)

    # straighten Q4
    q4_zoomed_str8 = straightenImage(q4_zoomed)

    # straighten Q8 and Q9
    q8_9_zoomed_str8 = straightenImage(q8_9_zoomed)

    
    
    
    
    # crop Q2
    Q2yes_crop_dims = (40, 110, 130, 200)
    Q2yes_crop = q2_zoomed_str8.crop(Q2yes_crop_dims)
    Q2yes_crop
    
    Q2no_crop_dims = (130, 110, 220, 200)
    Q2no_crop = q2_zoomed_str8.crop(Q2no_crop_dims)
    Q2no_crop
    
    
    Q2_yes = []
    for pixel in iter(Q2yes_crop.getdata()):
        Q2_yes.append(pixel)
        
    Q2_yes_df = pd.DataFrame(Q2_yes).transpose()
    
    Q2_no = []
    for pixel in iter(Q2no_crop.getdata()):
        Q2_no.append(pixel)
        
    Q2_no_df = pd.DataFrame(Q2_no).transpose()
    
    
    
    
    
    # crop Q3    
    Q3yes_crop_dims = (20, 210, 110, 300)
    Q3yes_crop = q3_zoomed_str8.crop(Q3yes_crop_dims)
    Q3yes_crop
    
    Q3no_crop_dims = (110, 210, 200, 300)
    Q3no_crop = q3_zoomed_str8.crop(Q3no_crop_dims)
    Q3no_crop
    
    
    Q3_yes = []
    for pixel in iter(Q3yes_crop.getdata()):
        Q3_yes.append(pixel)
        
    Q3_yes_df = pd.DataFrame(Q3_yes).transpose()
    
    Q3_no = []
    for pixel in iter(Q3no_crop.getdata()):
        Q3_no.append(pixel)
        
    Q3_no_df = pd.DataFrame(Q3_no).transpose()
    
    

    
    
    # crop Q4 
    Q4yes_crop_dims = (20, 80, 110, 170)
    Q4yes_crop = q4_zoomed_str8.crop(Q4yes_crop_dims)
    Q4yes_crop
    
    Q4no_crop_dims = (130, 80, 220, 170)
    Q4no_crop = q4_zoomed_str8.crop(Q4no_crop_dims)
    Q4no_crop
    
    
    Q4_yes = []
    for pixel in iter(Q4yes_crop.getdata()):
        Q4_yes.append(pixel)
        
    Q4_yes_df = pd.DataFrame(Q4_yes).transpose()
    
    Q4_no = []
    for pixel in iter(Q4no_crop.getdata()):
        Q4_no.append(pixel)
        
    Q4_no_df = pd.DataFrame(Q4_no).transpose()
      
    
    
  
    
    # crop 8
    Q8yes_crop_dims = (820, 0, 910, 90)
    Q8yes_crop = q8_9_zoomed_str8.crop(Q8yes_crop_dims)
    Q8yes_crop
    
    Q8no_crop_dims = (910, 0, 1000, 90)
    Q8no_crop = q8_9_zoomed_str8.crop(Q8no_crop_dims)
    Q8no_crop
    
    
    Q8_yes = []
    for pixel in iter(Q8yes_crop.getdata()):
        Q8_yes.append(pixel)
        
    Q8_yes_df = pd.DataFrame(Q8_yes).transpose()
    
    Q8_no = []
    for pixel in iter(Q8no_crop.getdata()):
        Q8_no.append(pixel)
        
    Q8_no_df = pd.DataFrame(Q8_no).transpose()
    
    
    
    
    
    # crop Q9
    Q9yes_crop_dims = (640, 70, 730, 160)
    Q9yes_crop = q8_9_zoomed_str8.crop(Q9yes_crop_dims)
    Q9yes_crop
    
    Q9no_crop_dims = (730, 70, 820, 160)
    Q9no_crop = q8_9_zoomed_str8.crop(Q9no_crop_dims)
    Q9no_crop
    
    
    
    Q9_yes = []
    for pixel in iter(Q9yes_crop.getdata()):
        Q9_yes.append(pixel)
        
    Q9_yes_df = pd.DataFrame(Q9_yes).transpose()
    
    Q9_no = []
    for pixel in iter(Q9no_crop.getdata()):
        Q9_no.append(pixel)
        
    Q9_no_df = pd.DataFrame(Q9_no).transpose()
    
    
    
    # save all to pandas df
    yes_no_data = pd.concat([Q2_yes_df, Q2_no_df, Q3_yes_df, Q3_no_df, Q4_yes_df,
                             Q4_no_df, Q8_yes_df, Q8_no_df, Q9_yes_df, Q9_no_df])
    
    yes_no_data['survey'] = page
    return yes_no_data
    
def parseQ5_checks(page):
    # set path
    setpath = '/Users/natewagner/Documents/Surveys/'
    path = setpath + page
    
    # convert pdf to png
    images = convert_from_path(path)
    
    # convert to greyscale
    images_bw = images[0].convert('L') 
    
    # rotate image
    images_bw = images_bw.transpose(Image.ROTATE_270)
    
    # convert to numpy array
    images_bw_array = np.array(images_bw)

    

    #Image to pass in:
    img = images_bw_array

    # inital template match
    Q5_straight = templateMatch(
        img, template = cv.imread('/Users/natewagner/Documents/full_survey_temp.jpg',0))
    
    # convert to array
    Q5_straight_array = np.array(Q5_straight)
    
    
    def buildTemplates():
        
        #question5_dims = (1160, 105, 1575, 370)
        #question5 = Q5_straight.crop(question5_dims)
        #question5
        #question5.save("/Users/natewagner/Documents/temp5after_deskew.jpg") 
        #images_bw_crop_dim = (30, 60, 1700, 1400)
        #images_bw_crop = clean_arr.crop(images_bw_crop_dim)
        #images_bw_crop.save("/Users/natewagner/Documents/full_survey_temp.jpg") 
    
        #question5_dims = (1100, 170, 1575, 470)
        #question5 = images_bw.crop(question5_dims)
        #question5
        #question5.save("/Users/natewagner/Documents/temp5after_deskew.jpg") 
        return 5+5
    
    
    #Image to pass in:
    img = Q5_straight_array

    # second template match
    Q5_straight_zoomed = templateMatch(
        img, template = cv.imread('/Users/natewagner/Documents/temp5after_deskew.jpg',0))
    
    # straighten image
    Q5_zoomed = straightenImage(Q5_straight_zoomed)
    
    # convert to array
    images_bw_array = np.array(Q5_zoomed)

    
    # set all non 0 values to 255
    images_bw_array[images_bw_array != 0] = 255 
    
    # convert back to PIL
    Q5_straight_zoomed = Image.fromarray(images_bw_array)
    
    
    

       
    # question 5_p1
    question5_p1_dims1 = (5, 10, 100, 40)
    question5_p1_dims2 = (150, 10, 245, 40)
    question5_p1_dims3 = (285, 15, 380, 45)
    
    question5_p11 = Q5_straight_zoomed.crop(question5_p1_dims1)
    question5_p12 = Q5_straight_zoomed.crop(question5_p1_dims2)
    question5_p13 = Q5_straight_zoomed.crop(question5_p1_dims3)
    
    question5_p11_new = Image.new('L', (95, 95))
    question5_p11_new.paste(question5_p11, (0, 32))
    
    question5_p12_new = Image.new('L', (95, 95))
    question5_p12_new.paste(question5_p12, (0, 32))
    
    question5_p13_new = Image.new('L', (95, 95))
    question5_p13_new.paste(question5_p13, (0, 32))
    
    
    question5_p11_data = []
    for pixel in iter(question5_p11_new.getdata()):
        question5_p11_data.append(pixel)
        
    question5_p11_dataF = pd.DataFrame(question5_p11_data).transpose()
    
    
    question5_p12_data = []
    for pixel in iter(question5_p12_new.getdata()):
        question5_p12_data.append(pixel)
        
    question5_p12_dataF = pd.DataFrame(question5_p12_data).transpose()
    
    
    question5_p13_data = []
    for pixel in iter(question5_p13_new.getdata()):
        question5_p13_data.append(pixel)
        
    question5_p13_dataF = pd.DataFrame(question5_p13_data).transpose()
        
    
    
  
    
    # question 5_p2
    question5_p2_dims1 = (5, 40, 100, 70)
    question5_p2_dims2 = (150, 40, 245, 70)
    question5_p2_dims3 = (285, 45, 380, 75)
    
    question5_p21 = Q5_straight_zoomed.crop(question5_p2_dims1)
    question5_p22 = Q5_straight_zoomed.crop(question5_p2_dims2)
    question5_p23 = Q5_straight_zoomed.crop(question5_p2_dims3)
    
    question5_p21_new = Image.new('L', (95, 95))
    question5_p21_new.paste(question5_p21, (0, 32))
    
    question5_p22_new = Image.new('L', (95, 95))
    question5_p22_new.paste(question5_p22, (0, 32))
    
    question5_p23_new = Image.new('L', (95, 95))
    question5_p23_new.paste(question5_p23, (0, 32))
    
    
    question5_p21_data = []
    for pixel in iter(question5_p21_new.getdata()):
        question5_p21_data.append(pixel)
        
    question5_p21_dataF = pd.DataFrame(question5_p21_data).transpose()
    
    
    question5_p22_data = []
    for pixel in iter(question5_p22_new.getdata()):
        question5_p22_data.append(pixel)
        
    question5_p22_dataF = pd.DataFrame(question5_p22_data).transpose()
    
    
    question5_p23_data = []
    for pixel in iter(question5_p23_new.getdata()):
        question5_p23_data.append(pixel)
        
    question5_p23_dataF = pd.DataFrame(question5_p23_data).transpose()
    
    
    
    
    
    # question 5_p3
    question5_p3_dims1 = (5, 70, 100, 100)
    question5_p3_dims2 = (150, 70, 245, 100)
    question5_p3_dims3 = (285, 75, 380, 105)
    
    question5_p31 = Q5_straight_zoomed.crop(question5_p3_dims1)
    question5_p32 = Q5_straight_zoomed.crop(question5_p3_dims2)
    question5_p33 = Q5_straight_zoomed.crop(question5_p3_dims3)
    
    question5_p31_new = Image.new('L', (95, 95))
    question5_p31_new.paste(question5_p31, (0, 32))
    
    question5_p32_new = Image.new('L', (95, 95))
    question5_p32_new.paste(question5_p32, (0, 32))
    
    question5_p33_new = Image.new('L', (95, 95))
    question5_p33_new.paste(question5_p33, (0, 32))
    
    question5_p31_data = []
    for pixel in iter(question5_p31_new.getdata()):
        question5_p31_data.append(pixel)
        
    question5_p31_dataF = pd.DataFrame(question5_p31_data).transpose()
    
    
    question5_p32_data = []
    for pixel in iter(question5_p32_new.getdata()):
        question5_p32_data.append(pixel)
        
    question5_p32_dataF = pd.DataFrame(question5_p32_data).transpose()
    
    
    question5_p33_data = []
    for pixel in iter(question5_p33_new.getdata()):
        question5_p33_data.append(pixel)
        
    question5_p33_dataF = pd.DataFrame(question5_p33_data).transpose()
    
    
    
    
    
    # question 5_p4
    question5_p4_dims1 = (5, 100, 100, 130)
    question5_p4_dims2 = (150, 100, 245, 130)
    question5_p4_dims3 = (285, 105, 380, 135)
    
    question5_p41 = Q5_straight_zoomed.crop(question5_p4_dims1)
    question5_p42 = Q5_straight_zoomed.crop(question5_p4_dims2)
    question5_p43 = Q5_straight_zoomed.crop(question5_p4_dims3)
    
    question5_p41_new = Image.new('L', (95, 95))
    question5_p41_new.paste(question5_p41, (0, 32))
    
    question5_p42_new = Image.new('L', (95, 95))
    question5_p42_new.paste(question5_p42, (0, 32))
    
    question5_p43_new = Image.new('L', (95, 95))
    question5_p43_new.paste(question5_p43, (0, 32))
    
    
    
    question5_p41_data = []
    for pixel in iter(question5_p41_new.getdata()):
        question5_p41_data.append(pixel)
        
    question5_p41_dataF = pd.DataFrame(question5_p41_data).transpose()
    
    
    question5_p42_data = []
    for pixel in iter(question5_p42_new.getdata()):
        question5_p42_data.append(pixel)
        
    question5_p42_dataF = pd.DataFrame(question5_p42_data).transpose()
    
    
    question5_p43_data = []
    for pixel in iter(question5_p43_new.getdata()):
        question5_p43_data.append(pixel)
        
    question5_p43_dataF = pd.DataFrame(question5_p43_data).transpose()
    
    
    
    
    
    # question 5_p5
    question5_p5_dims1 = (5, 130, 100, 160)
    question5_p5_dims2 = (150, 130, 245, 160)
    question5_p5_dims3 = (285, 135, 380, 165)
    
    question5_p51 = Q5_straight_zoomed.crop(question5_p5_dims1)
    question5_p52 = Q5_straight_zoomed.crop(question5_p5_dims2)
    question5_p53 = Q5_straight_zoomed.crop(question5_p5_dims3)
    
    question5_p51_new = Image.new('L', (95, 95))
    question5_p51_new.paste(question5_p51, (0, 32))
    
    question5_p52_new = Image.new('L', (95, 95))
    question5_p52_new.paste(question5_p52, (0, 32))
    
    question5_p53_new = Image.new('L', (95, 95))
    question5_p53_new.paste(question5_p53, (0, 32))
    
    
    question5_p51_data = []
    for pixel in iter(question5_p51_new.getdata()):
        question5_p51_data.append(pixel)
        
    question5_p51_dataF = pd.DataFrame(question5_p51_data).transpose()
    
    
    question5_p52_data = []
    for pixel in iter(question5_p52_new.getdata()):
        question5_p52_data.append(pixel)
        
    question5_p52_dataF = pd.DataFrame(question5_p52_data).transpose()
    
    
    question5_p53_data = []
    for pixel in iter(question5_p53_new.getdata()):
        question5_p53_data.append(pixel)
        
    question5_p53_dataF = pd.DataFrame(question5_p53_data).transpose()
    
    
    
    
    
    # question 5_p6
    question5_p6_dims1 = (5, 160, 100, 190)
    question5_p6_dims2 = (150, 160, 245, 190)
    question5_p6_dims3 = (285, 165, 380, 195)
    
    question5_p61 = Q5_straight_zoomed.crop(question5_p6_dims1)
    question5_p62 = Q5_straight_zoomed.crop(question5_p6_dims2)
    question5_p63 = Q5_straight_zoomed.crop(question5_p6_dims3)
    
    question5_p61_new = Image.new('L', (95, 95))
    question5_p61_new.paste(question5_p61, (0, 32))
    
    question5_p62_new = Image.new('L', (95, 95))
    question5_p62_new.paste(question5_p62, (0, 32))
    
    question5_p63_new = Image.new('L', (95, 95))
    question5_p63_new.paste(question5_p63, (0, 32))
    
    
    question5_p61_data = []
    for pixel in iter(question5_p61_new.getdata()):
        question5_p61_data.append(pixel)
        
    question5_p61_dataF = pd.DataFrame(question5_p61_data).transpose()
    
    
    question5_p62_data = []
    for pixel in iter(question5_p62_new.getdata()):
        question5_p62_data.append(pixel)
        
    question5_p62_dataF = pd.DataFrame(question5_p62_data).transpose()
    
    
    question5_p63_data = []
    for pixel in iter(question5_p63_new.getdata()):
        question5_p63_data.append(pixel)
        
    question5_p63_dataF = pd.DataFrame(question5_p63_data).transpose()
    
    
    
    
    
    # question 5_p7
    question5_p7_dims1 = (5, 190, 100, 220)
    question5_p7_dims2 = (150, 190, 245, 220)
    question5_p7_dims3 = (285, 195, 380, 225)
    
    question5_p71 = Q5_straight_zoomed.crop(question5_p7_dims1)
    question5_p72 = Q5_straight_zoomed.crop(question5_p7_dims2)
    question5_p73 = Q5_straight_zoomed.crop(question5_p7_dims3)
    
    question5_p71_new = Image.new('L', (95, 95))
    question5_p71_new.paste(question5_p71, (0, 32))
    
    question5_p72_new = Image.new('L', (95, 95))
    question5_p72_new.paste(question5_p72, (0, 32))
    
    question5_p73_new = Image.new('L', (95, 95))
    question5_p73_new.paste(question5_p73, (0, 32))
    
    
    question5_p71_data = []
    for pixel in iter(question5_p71_new.getdata()):
        question5_p71_data.append(pixel)
        
    question5_p71_dataF = pd.DataFrame(question5_p71_data).transpose()
    
    
    question5_p72_data = []
    for pixel in iter(question5_p72_new.getdata()):
        question5_p72_data.append(pixel)
        
    question5_p72_dataF = pd.DataFrame(question5_p72_data).transpose()
    
    
    question5_p73_data = []
    for pixel in iter(question5_p73_new.getdata()):
        question5_p73_data.append(pixel)
        
    question5_p73_dataF = pd.DataFrame(question5_p73_data).transpose()
    
    
    
    
    
    # question 5_p8
    question5_p8_dims1 = (5, 220, 100, 250)
    question5_p8_dims2 = (150, 220, 245, 250)
    question5_p8_dims3 = (285, 225, 380, 255)
    
    question5_p81 = Q5_straight_zoomed.crop(question5_p8_dims1)
    question5_p82 = Q5_straight_zoomed.crop(question5_p8_dims2)
    question5_p83 = Q5_straight_zoomed.crop(question5_p8_dims3)
    
    question5_p81_new = Image.new('L', (95, 95))
    question5_p81_new.paste(question5_p81, (0, 32))
    
    question5_p82_new = Image.new('L', (95, 95))
    question5_p82_new.paste(question5_p82, (0, 32))
    
    question5_p83_new = Image.new('L', (95, 95))
    question5_p83_new.paste(question5_p83, (0, 32))
    
    
    question5_p81_data = []
    for pixel in iter(question5_p81_new.getdata()):
        question5_p81_data.append(pixel)
        
    question5_p81_dataF = pd.DataFrame(question5_p81_data).transpose()
    
    
    question5_p82_data = []
    for pixel in iter(question5_p82_new.getdata()):
        question5_p82_data.append(pixel)
        
    question5_p82_dataF = pd.DataFrame(question5_p82_data).transpose()
    
    
    question5_p83_data = []
    for pixel in iter(question5_p83_new.getdata()):
        question5_p83_data.append(pixel)
        
    question5_p83_dataF = pd.DataFrame(question5_p83_data).transpose()
    
    
    
    # set up final pandas df
    question5DATA = pd.concat([question5_p11_dataF, question5_p12_dataF, question5_p13_dataF, 
                               question5_p21_dataF, question5_p22_dataF, question5_p23_dataF, 
                               question5_p31_dataF, question5_p32_dataF, question5_p33_dataF, 
                               question5_p41_dataF, question5_p42_dataF, question5_p43_dataF, 
                               question5_p51_dataF, question5_p52_dataF, question5_p53_dataF, 
                               question5_p61_dataF, question5_p62_dataF, question5_p63_dataF, 
                               question5_p71_dataF, question5_p72_dataF, question5_p73_dataF, 
                               question5_p81_dataF, question5_p82_dataF, question5_p83_dataF])
    

    question5DATA['survey'] = page
        
    return question5DATA
    
def parseQuestion6(page):
    # set path
    setpath = '/Users/natewagner/Documents/Surveys/'
    path = setpath + page


    # convert from pdf to png
    images = convert_from_path(path)
    
    # convert to greyscale
    images_bw = images[0].convert('L') 
    
    # rotate image
    images_bw = images_bw.transpose(Image.ROTATE_270)
    
    # turn to array
    images_bw_array = np.array(images_bw)

    
    def buildTemplates():
        
    
        #images_bw_crop_dim = (30, 70, 1700, 1400)
        #images_bw_crop = images_bw.crop(images_bw_crop_dim)
        #images_bw_crop.show()
        #images_bw_crop.save("/Users/natewagner/Documents/full_survey_temp.jpg") 
        #question5_dims = (450, 370, 1550, 550)
        #question5 = Q5_straight.crop(question5_dims)
        #question5.show()
        #question5.save("/Users/natewagner/Documents/question_6_template2.jpg") 
        return 5+5
        
        
    # Image to pass in:
    img = images_bw_array

    # initial crop
    survey_crop = templateMatch(
        img, template = cv.imread('/Users/natewagner/Documents/full_survey_temp.jpg',0))
    
    # covert to array
    survey_crop_array = np.array(survey_crop)
    
    

    
    # Image to pass in:
    img = survey_crop_array
    
    # second template match
    Q6_zoomed = templateMatch(
        img, template = cv.imread('/Users/natewagner/Documents/question_6_template2.jpg',0))

    # straighten image
    Q6_zoomed_more = straightenImage(Q6_zoomed)




    # crop out question 6
    question6_dims = (200, 83, 1025, 133)
    question6 = Q6_zoomed_more.crop(question6_dims)
    
    # convert to array
    question6_arr = np.array(question6)
    
    # make all non 0 pixels 255
    question6_arr[question6_arr != 0] = 255 


    
    # de noise image
    #clean = cv.fastNlMeansDenoising(question6_arr, question6_arr, h = 40.0)
    
    # convert from array
    Q6_straight_zoomed = Image.fromarray(question6_arr)

    # part 1
    question6_p1_dims = (20, 0, 70, 50)
    question6_p1 = Q6_straight_zoomed.crop(question6_p1_dims)
    question6_p1
    
        
    Q6p1 = []
    for pixel in iter(question6_p1.getdata()):
        Q6p1.append(pixel)
    
    Q6p1df = pd.DataFrame(Q6p1).transpose()
    
    
    # part 2
    question6_p2_dims = (140, 0, 190, 50)
    question6_p2 = Q6_straight_zoomed.crop(question6_p2_dims)
    question6_p2
    
    Q6p2 = []
    for pixel in iter(question6_p2.getdata()):
        Q6p2.append(pixel)
    
    Q6p2df = pd.DataFrame(Q6p2).transpose()
    
    
    
    # part 4
    question6_p3_dims = (185, 0, 235, 50)
    question6_p3 = Q6_straight_zoomed.crop(question6_p3_dims)
    question6_p3
    
    Q6p3 = []
    for pixel in iter(question6_p3.getdata()):
        Q6p3.append(pixel)
    
    Q6p3df = pd.DataFrame(Q6p3).transpose()
    
    
    
    
    # part 4
    question6_p4_dims = (235, 0, 285, 50)
    question6_p4 = Q6_straight_zoomed.crop(question6_p4_dims)
    question6_p4
    
    Q6p4 = []
    for pixel in iter(question6_p4.getdata()):
        Q6p4.append(pixel)
    
    Q6p4df = pd.DataFrame(Q6p4).transpose()
    
    
    
    
    
    # part 5
    question6_p5_dims = (345, 0, 395, 50)
    question6_p5 = Q6_straight_zoomed.crop(question6_p5_dims)
    question6_p5
    
    Q6p5 = []
    for pixel in iter(question6_p5.getdata()):
        Q6p5.append(pixel)
    
    Q6p5df = pd.DataFrame(Q6p5).transpose()
    
    
    
    # part 6
    question6_p6_dims = (455, 0, 505, 50)
    question6_p6 = Q6_straight_zoomed.crop(question6_p6_dims)
    question6_p6
    
    Q6p6 = []
    for pixel in iter(question6_p6.getdata()):
        Q6p6.append(pixel)
    
    Q6p6df = pd.DataFrame(Q6p6).transpose()
    
    
    
    # part 7
    question6_p7_dims = (500, 0, 550, 50)
    question6_p7 = Q6_straight_zoomed.crop(question6_p7_dims)
    question6_p7
    
    Q6p7 = []
    for pixel in iter(question6_p7.getdata()):
        Q6p7.append(pixel)
    
    Q6p7df = pd.DataFrame(Q6p7).transpose()
    
    
    
    
    # part 8
    question6_p8_dims = (550, 0, 600, 50)
    question6_p8 = Q6_straight_zoomed.crop(question6_p8_dims)
    question6_p8
    
    Q6p8 = []
    for pixel in iter(question6_p8.getdata()):
        Q6p8.append(pixel)
    
    Q6p8df = pd.DataFrame(Q6p8).transpose()
    
    
    
    # part 9
    question6_p9_dims = (590, 0, 640, 50)
    question6_p9 = Q6_straight_zoomed.crop(question6_p9_dims)
    question6_p9
    
    Q6p9 = []
    for pixel in iter(question6_p9.getdata()):
        Q6p9.append(pixel)
    
    Q6p9df = pd.DataFrame(Q6p9).transpose()
    
    
    
    
    # part 10
    question6_p10_dims = (730, 0, 780, 50)
    question6_p10 = Q6_straight_zoomed.crop(question6_p10_dims)
    question6_p10
    
    Q6p10 = []
    for pixel in iter(question6_p10.getdata()):
        Q6p10.append(pixel)
    
    Q6p10df = pd.DataFrame(Q6p10).transpose()
    
    
    # set up final pandas df
    Q6_data = pd.concat([Q6p1df, Q6p2df, Q6p3df, Q6p4df, Q6p5df, 
                         Q6p6df, Q6p7df, Q6p8df, Q6p9df, Q6p10df])
    
    Q6_data['survey'] = page
    return(Q6_data)

def parseQ10_checks(page):
    # set path
    setpath = '/Users/natewagner/Documents/Surveys/'
    path = setpath + page
    
    
    # convert from pdf to png
    images = convert_from_path(path)
    
    # convert to greyscale
    images_bw = images[0].convert('L') 
    
    # rotate image
    images_bw = images_bw.transpose(Image.ROTATE_270)
    
    # save image as array
    images_bw_array = np.array(images_bw)

    
    
    # image to pass in
    img = images_bw_array
    
    # first template match
    full_survey_cropped = templateMatch(
        img, template = cv.imread('/Users/natewagner/Documents/full_survey_temp.jpg',0))
    
    # convert to array
    full_survey_cropped_array = np.array(full_survey_cropped)

    
    
    def templatesHolder():
        #question5_dims = (490, 675, 1650, 950)
        #question5 = Q5_straight.crop(question5_dims)
        #question5.show()
        #question5.save("/Users/natewagner/Documents/surveys/question5zoomed.jpg") 
        
        #question5_dims = (20, 120, 1150, 260)
        #question5 = Q5_straight.crop(question5_dims)
        #question5.show()
        #question5.save("/Users/natewagner/Documents/surveys/question5zoomed_more.jpg") 
        return 5+5
    
    
    # image to pass in
    img = full_survey_cropped_array
    
    # second template match
    Q10_first_crop = templateMatch(
        img, template = cv.imread('/Users/natewagner/Documents/surveys/question5zoomed.jpg',0))    
    
    # convert to array
    Q10_first_crop_array = np.array(Q10_first_crop)

    
    
    
    # image to pass in
    img = Q10_first_crop_array
    
    # third template match
    Q10_final_crop = templateMatch(
        img, template = cv.imread('/Users/natewagner/Documents/surveys/question5zoomed_more.jpg',0))
    
    # straighten final image to crop
    Q10_final_crop_straight = straightenImage(Q10_final_crop)

    
    
    # question 10 part 1
    question10_p1_dims = (40, 45, 100, 105)
    question10_p1 = Q10_final_crop_straight.crop(question10_p1_dims)
    question10_p1
    
    question10_p1_data = []
    for pixel in iter(question10_p1.getdata()):
        question10_p1_data.append(pixel)
        
    question10_p1_dataF = pd.DataFrame(question10_p1_data).transpose()
    
    
    
    
    # question 10 part 2   
    question10_p2_dims = (310, 40, 370, 100)
    question10_p2 = Q10_final_crop_straight.crop(question10_p2_dims)
    question10_p2
    
    question10_p2_data = []
    for pixel in iter(question10_p2.getdata()):
        question10_p2_data.append(pixel)
        
    question10_p2_dataF = pd.DataFrame(question10_p2_data).transpose()
    
    
    
    
    # question 10 part 3   
    question10_p3_dims = (480, 40, 540, 100)
    question10_p3 = Q10_final_crop_straight.crop(question10_p3_dims)
    question10_p3
    
    question10_p3_data = []
    for pixel in iter(question10_p3.getdata()):
        question10_p3_data.append(pixel)
        
    question10_p3_dataF = pd.DataFrame(question10_p3_data).transpose()
    
    
    
    
    # question 10 part 4
    question10_p4_dims = (660, 40, 720, 100)
    question10_p4 = Q10_final_crop_straight.crop(question10_p4_dims)
    question10_p4
    
    question10_p4_data = []
    for pixel in iter(question10_p4.getdata()):
        question10_p4_data.append(pixel)
        
    question10_p4_dataF = pd.DataFrame(question10_p4_data).transpose()
    
    
    
    
    # question 10 part 5
    question10_p5_dims = (895, 40, 955, 100)
    question10_p5 = Q10_final_crop_straight.crop(question10_p5_dims)
    question10_p5
    
    question10_p5_data = []
    for pixel in iter(question10_p5.getdata()):
        question10_p5_data.append(pixel)
        
    question10_p5_dataF = pd.DataFrame(question10_p5_data).transpose()
    
    
    
    
    # question 10 part 6
    question10_p6_dims = (40, 80, 100, 140)
    question10_p6 = Q10_final_crop_straight.crop(question10_p6_dims)
    question10_p6
    
    question10_p6_data = []
    for pixel in iter(question10_p6.getdata()):
        question10_p6_data.append(pixel)
        
    question10_p6_dataF = pd.DataFrame(question10_p6_data).transpose()
    
    
    
    
    # question 10 part 7
    question10_p7_dims = (230, 80, 290, 140)
    question10_p7 = Q10_final_crop_straight.crop(question10_p7_dims)
    question10_p7
    
    
    question10_p7_data = []
    for pixel in iter(question10_p7.getdata()):
        question10_p7_data.append(pixel)
        
    question10_p7_dataF = pd.DataFrame(question10_p7_data).transpose()
    
    
    
    
    # question 10 part 8
    question10_p8_dims = (445, 75, 505, 135)
    question10_p8 = Q10_final_crop_straight.crop(question10_p8_dims)
    question10_p8
    
    
    question10_p8_data = []
    for pixel in iter(question10_p8.getdata()):
        question10_p8_data.append(pixel)
        
    question10_p8_dataF = pd.DataFrame(question10_p8_data).transpose()
    
    
    
    
    # question 10 part 9
    question10_p9_dims = (590, 75, 650, 135)
    question10_p9 = Q10_final_crop_straight.crop(question10_p9_dims)
    question10_p9
    
    question10_p9_data = []
    for pixel in iter(question10_p9.getdata()):
        question10_p9_data.append(pixel)
        
    question10_p9_dataF = pd.DataFrame(question10_p9_data).transpose()
    
    
    
    
    # question 10 part 10
    question10_p10_dims = (770, 80, 830, 140)
    question10_p10 = Q10_final_crop_straight.crop(question10_p10_dims)
    question10_p10
    
    question10_p10_data = []
    for pixel in iter(question10_p10.getdata()):
        question10_p10_data.append(pixel)
        
    question10_p10_dataF = pd.DataFrame(question10_p10_data).transpose()
    
    
    
    
    # question 10 part 11
    question10_p11_dims = (980, 80, 1040, 140)
    question10_p11 = Q10_final_crop_straight.crop(question10_p11_dims)
    question10_p11
    
    
    question10_p11_data = []
    for pixel in iter(question10_p11.getdata()):
        question10_p11_data.append(pixel)
        
    question10_p11_dataF = pd.DataFrame(question10_p11_data).transpose()
    
    
    
    
    # final pandas df of pixel values
    question10DATA = pd.concat([question10_p1_dataF,
                                question10_p2_dataF,
                                question10_p3_dataF,
                                question10_p4_dataF,
                                question10_p5_dataF,
                                question10_p6_dataF,
                                question10_p7_dataF,
                                question10_p8_dataF,
                                question10_p9_dataF,
                                question10_p10_dataF,
                                question10_p11_dataF])
    
    question10DATA['survey'] = page
    
    return question10DATA
    







pdfs = [66, 99, 123, 94, 137, 92, 138, 164, 82, 85, 46, 81, 120, 100, 91]
batches = [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18] 
        
   
batch = 'batch'
survey_num = '/survey-page-'
end = '.pdf'

YesNoData = pd.DataFrame()
question5_data = pd.DataFrame()
question6_data = pd.DataFrame()
question10_data = pd.DataFrame()

cnt = 0
for pages in batches:
    for x in range(1, pdfs[cnt]+1):  
        survey = batch + str(pages) + survey_num + str(x) + end 
        df = parseQ10_checks(survey)
        question10_data = question10_data.append(df)
    cnt += 1
    print(pages)



#YesNoData.to_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/yes_no_pixel.csv', encoding='utf-8', index = False)
#question5_data.to_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/question5_pixel.csv', encoding='utf-8', index = False)
#question6_data.to_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/question6_pixel.csv', encoding='utf-8', index = False)
#question10_data.to_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/question10_pixel.csv', encoding='utf-8', index = False)




from tensorflow import keras


################################# YES / NO #################################


# to reload model
YesNo_CNN = keras.models.load_model('/Users/natewagner/Documents/Surveys/Models/YesNo_CNN.h5')

YesNoData = pd.read_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/yes_no_pixel.csv', encoding='utf-8')

# set all non zero pixels to 255
all_train_data1 = YesNoData.copy()
all_train_data1 = all_train_data1[all_train_data1.columns[:8100]]
all_train_data1[all_train_data1 != 0] = 255 

images = []
for index, row in all_train_data1.iterrows():
    images.append(np.array(row[:8100]).reshape(90, 90))

images_stacked = np.stack(images, axis=0)

#reshape data to fit model
yes_no_images = images_stacked.reshape(15180,90,90,1)

# standardize
yes_no_images =  yes_no_images/255


probability = []
predicted_answers_YesNo = []
for pixels in yes_no_images:
    image = pixels.reshape(1,90,90,1)
    prob = YesNo_CNN.predict(image)
    if prob[0][0] > prob[0][1]:
        predicted_answers_YesNo.append(0)
        probability.append(prob[0][0])
    if prob[0][0] < prob[0][1]:
        predicted_answers_YesNo.append(1)
        probability.append(prob[0][1])
    
        
survey = list(YesNoData['survey'])
YesNo_predicted = pd.DataFrame([survey, predicted_answers_YesNo, probability]).transpose()
YesNo_predicted.columns = ["survey", "predicted", "probability"]

#YesNo_predicted.to_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/yes_no_predictions.csv', encoding='utf-8', index = False)



################################# Question 5 #################################

from tensorflow import keras

# to reload model
Question5_CNN = keras.models.load_model('/Users/natewagner/Documents/Surveys/Models/Question5_CNN.h5')

question5_data = pd.read_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/question5_pixel.csv', encoding='utf-8')

all_train_data1 = question5_data.copy()
all_train_data1 = all_train_data1[all_train_data1.columns[:9025]]
all_train_data1[all_train_data1 != 0] = 255 


images = []
for index, row in all_train_data1.iterrows():
    images.append(np.array(row[:9025]).reshape(95,95))


images_stacked = np.stack(images, axis=0)

#reshape data to fit model
question5_images = images_stacked.reshape(36432,95,95,1)

question5_images =  question5_images/255

probability = []
predicted_answers_Q5 = []
for pixels in question5_images:
    image = pixels.reshape(1,95,95,1)
    prob = Question5_CNN.predict(image)
    if prob[0][0] > prob[0][1]:
        predicted_answers_Q5.append(0)
        probability.append(prob[0][0])
    if prob[0][0] < prob[0][1]:
        predicted_answers_Q5.append(1)
        probability.append(prob[0][1])


survey = list(question5_data['survey'])
Question5_predicted = pd.DataFrame([survey, predicted_answers_Q5, probability]).transpose()
Question5_predicted.columns = ["survey", "predicted", "probability"]


#Question5_predicted.to_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/Question5_predictions.csv', encoding='utf-8', index = False)



################################# Question 6 #################################

# load model
Question6_CNN = keras.models.load_model('/Users/natewagner/Documents/Surveys/Models/Question6_CNN.h5')

all_train_data1 = question6_data.copy()
all_train_data1 = all_train_data1[all_train_data1.columns[:2500]]
all_train_data1[all_train_data1 != 0] = 255 

images = []
for index, row in all_train_data1.iterrows():
    images.append(np.array(row[:2500]).reshape(50, 50))


# stack images
images_stacked = np.stack(images, axis=0)


#reshape data to fit model
question6_images = images_stacked.reshape(15180,50,50,1)

# standardize
question6_images = question6_images/255


probability = []
predicted_answers_Q6 = []
for pixels in question6_images:
    image = pixels.reshape(1,50,50,1)
    prob = Question6_CNN.predict(image)
    if prob[0][0] > prob[0][1]:
        predicted_answers_Q6.append(0)
        probability.append(prob[0][0])
    if prob[0][0] < prob[0][1]:
        predicted_answers_Q6.append(1)
        probability.append(prob[0][1])


survey = list(question6_data['survey'])
Question6_predicted = pd.DataFrame([survey, predicted_answers_Q6, probability]).transpose()
Question6_predicted.columns = ["survey", "predicted", "probability"]


#Question6_predicted.to_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/Question6_predictions.csv', encoding='utf-8', index = False)




################################# Question 10 #################################

from tensorflow import keras

# to reload model
Q10_CNN = keras.models.load_model('/Users/natewagner/Documents/Surveys/Models/Question10_CNN.h5')



all_train_data1 = question10_data.copy()
all_train_data1 = all_train_data1[all_train_data1.columns[:3600]]
all_train_data1[all_train_data1 != 0] = 255 


images = []
for index, row in all_train_data1.iterrows():
    images.append(np.array(row[:3600]).reshape(60,60))


images_stacked = np.stack(images, axis=0)


#reshape data to fit model
question10_images = images_stacked.reshape(16698,60,60,1)


question10_images = question10_images/255


probability = []
predicted_answers_Q10 = []
for pixels in question10_images:
    image = pixels.reshape(1,60,60,1)
    prob = Q10_CNN.predict(image)
    if prob[0][0] > prob[0][1]:
        predicted_answers_Q10.append(0)
        probability.append(prob[0][0])
    if prob[0][0] < prob[0][1]:
        predicted_answers_Q10.append(1)
        probability.append(prob[0][1])


survey = list(question10_data['survey'])
Question10_predicted = pd.DataFrame([survey, predicted_answers_Q10, probability]).transpose()
Question10_predicted.columns = ["survey", "predicted", "probability"]



#Question10_predicted.to_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/Question10_predictions.csv', encoding='utf-8', index = False)


