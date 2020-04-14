#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:15:56 2020

@author: natewagner
"""

from pdf2image import convert_from_path
import pandas as pd
import pytesseract
from PIL import Image
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter



def parseQuestion5(page):
    # set path
    setpath = '/Users/natewagner/Documents/Surveys/'
    path = setpath + page
    
    # Convert to png and to greyscale / rotate
    images = convert_from_path(path)
    images_bw = images[0].convert('L') 
    images_bw = images_bw.transpose(Image.ROTATE_270)
    
    # set path to pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'
    
    
    # extract account number
    accnt_num_dims = (1000+155, 150-140, 1600-200, 60)
    accnt_num = images_bw.crop(accnt_num_dims)    
    accnt_number = pytesseract.image_to_string(accnt_num)

    
    
    # extract name
    name_dims = (465+160, 1060, 1250-100, 1095)
    name_1 = images_bw.crop(name_dims) 
    name = pytesseract.image_to_string(name_1)

    
    # extract company name
    comp_info_dims = (465+250, 1100, 1250+400, 1145)
    comp_info = images_bw.crop(comp_info_dims) 
    bus_name = pytesseract.image_to_string(comp_info)

    
    # extract phone number
    phone_number_dims = (465+250, 1155, 1150, 1200)
    phone_info = images_bw.crop(phone_number_dims) 
    phone_number = pytesseract.image_to_string(phone_info)

    
    # extract email
    email_dims = (465+250, 1225, 1100, 1250)
    email_info = images_bw.crop(email_dims) 
    email = pytesseract.image_to_string(email_info)
    
    question5_dims = (1130, 125, 1650, 445)
    question5 = images_bw.crop(question5_dims)
    
    # convert to binary
    wd, ht = question5.size
    pix = np.array(question5.convert('1').getdata(), np.uint8)
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
    Q5_straight = im.fromarray((255 * data).astype("uint8")).convert("L")
    Q5_straight
    
    
    
    
    
    
    # question 5_p1
    question5_p1_dims = (50, 50, 500, 90)
    question5_p1 = Q5_straight.crop(question5_p1_dims) 
    question5_p1
    
    q5p1 = Image.new('L', (450, 450))
    q5p1.paste(question5_p1, (0, 225))
    q5p1
    
    Q5p1p = []
    for pixel in iter(q5p1.getdata()):
        Q5p1p.append(pixel)
    
    Q5p1df = pd.DataFrame(Q5p1p).transpose()
    Q5p1df['question'] = 'question5p1'
    Q5p1df['accnt_num'] = accnt_number
    Q5p1df['bus_name'] = bus_name
    Q5p1df['name'] = name    
    Q5p1df['phone_number'] = phone_number
    Q5p1df['email'] = email
    
    
    
    # question 5_p2
    question5_p2_dims = (50, 85, 500, 120)
    question5_p2 = Q5_straight.crop(question5_p2_dims) 
    question5_p2
    
    q5p2 = Image.new('L', (450, 450))
    q5p2.paste(question5_p2, (0, 225))
    q5p2
    
    Q5p2p = []
    for pixel in iter(q5p2.getdata()):
        Q5p2p.append(pixel)
    
    Q5p2df = pd.DataFrame(Q5p1p).transpose()
    Q5p2df['question'] = 'question5p2'
    Q5p2df['accnt_num'] = accnt_number
    Q5p2df['bus_name'] = bus_name
    Q5p2df['name'] = name    
    Q5p2df['phone_number'] = phone_number
    Q5p2df['email'] = email
    
    
    
    # question 5_p3
    question5_p3_dims = (50, 120, 500, 150)
    question5_p3 = Q5_straight.crop(question5_p3_dims) 
    question5_p3
    
    q5p3 = Image.new('L', (450, 450))
    q5p3.paste(question5_p3, (0, 225))
    q5p3
    
    Q5p3p = []
    for pixel in iter(q5p3.getdata()):
        Q5p3p.append(pixel)
    
    Q5p3df = pd.DataFrame(Q5p1p).transpose()
    Q5p3df['question'] = 'question5p3'
    Q5p3df['accnt_num'] = accnt_number
    Q5p3df['bus_name'] = bus_name
    Q5p3df['name'] = name    
    Q5p3df['phone_number'] = phone_number
    Q5p3df['email'] = email
    
    
    # question 5_p4
    question5_p4_dims = (50, 150, 500, 180)
    question5_p4 = Q5_straight.crop(question5_p4_dims) 
    question5_p4
    
    q5p4 = Image.new('L', (450, 450))
    q5p4.paste(question5_p4, (0, 225))
    q5p4
    
    Q5p4p = []
    for pixel in iter(q5p4.getdata()):
        Q5p4p.append(pixel)
    
    Q5p4df = pd.DataFrame(Q5p1p).transpose()
    Q5p4df['question'] = 'question5p4'
    Q5p4df['accnt_num'] = accnt_number
    Q5p4df['bus_name'] = bus_name
    Q5p4df['name'] = name    
    Q5p4df['phone_number'] = phone_number
    Q5p4df['email'] = email
    
    
    
    # question 5_p5
    question5_p5_dims = (50, 180, 500, 210)
    question5_p5 = Q5_straight.crop(question5_p5_dims) 
    question5_p5
    
    q5p5 = Image.new('L', (450, 450))
    q5p5.paste(question5_p5, (0, 225))
    q5p5
    
    Q5p5p = []
    for pixel in iter(q5p5.getdata()):
        Q5p5p.append(pixel)
    
    Q5p5df = pd.DataFrame(Q5p1p).transpose()
    Q5p5df['question'] = 'question5p5'
    Q5p5df['accnt_num'] = accnt_number
    Q5p5df['bus_name'] = bus_name
    Q5p5df['name'] = name    
    Q5p5df['phone_number'] = phone_number
    Q5p5df['email'] = email
    
    
    # question 5_p6
    question5_p6_dims = (50, 210, 500, 240)
    question5_p6 = Q5_straight.crop(question5_p6_dims) 
    question5_p6
    
    q5p6 = Image.new('L', (450, 450))
    q5p6.paste(question5_p6, (0, 225))
    q5p6
    
    Q5p6p = []
    for pixel in iter(q5p6.getdata()):
        Q5p6p.append(pixel)
    
    Q5p6df = pd.DataFrame(Q5p1p).transpose()
    Q5p6df['question'] = 'question5p6'
    Q5p6df['accnt_num'] = accnt_number
    Q5p6df['bus_name'] = bus_name
    Q5p6df['name'] = name    
    Q5p6df['phone_number'] = phone_number
    Q5p6df['email'] = email
    
    
    
    # question 5_p7
    question5_p7_dims = (50, 240, 500, 270)
    question5_p7 = Q5_straight.crop(question5_p7_dims) 
    question5_p7
    
    q5p7 = Image.new('L', (450, 450))
    q5p7.paste(question5_p7, (0, 225))
    q5p7
    
    Q5p7p = []
    for pixel in iter(q5p7.getdata()):
        Q5p7p.append(pixel)
    
    Q5p7df = pd.DataFrame(Q5p1p).transpose()
    Q5p7df['question'] = 'question5p7'
    Q5p7df['accnt_num'] = accnt_number
    Q5p7df['bus_name'] = bus_name
    Q5p7df['name'] = name    
    Q5p7df['phone_number'] = phone_number
    Q5p7df['email'] = email
    
    
    
    # question 5_p8
    question5_p8_dims = (50, 270, 500, 300)
    question5_p8 = Q5_straight.crop(question5_p8_dims) 
    question5_p8
    
    q5p8 = Image.new('L', (450, 450))
    q5p8.paste(question5_p1, (0, 225))
    q5p8
    
    Q5p8p = []
    for pixel in iter(q5p8.getdata()):
        Q5p8p.append(pixel)
    
    Q5p8df = pd.DataFrame(Q5p8p).transpose()
    Q5p8df['question'] = 'question5p8'
    Q5p8df['accnt_num'] = accnt_number
    Q5p8df['bus_name'] = bus_name
    Q5p8df['name'] = name    
    Q5p8df['phone_number'] = phone_number
    Q5p8df['email'] = email
    
    
    
    Q5_data = pd.concat([Q5p1df, Q5p2df, Q5p3df, Q5p4df, Q5p5df, Q5p6df, Q5p7df, Q5p8df])
    Q5_data['survey'] = str(page)
        
        
        
    return(Q5_data)
        
        
    
pdfs = [66, 99, 123, 94, 137, 100, 92, 138, 164, 100, 82, 85, 46, 81, 120, 100, 100, 91]


batch = 'batch'
survey_num = '/survey-page-'
end = '.pdf'
question5_data = pd.DataFrame()
for pages in range(10, 19):
    for x in range(1, pdfs[pages-1]+1):        
        survey = batch + str(pages) + survey_num + str(x) + end                 
        df = parseQuestion5(survey)
        question5_data = question5_data.append(df)
    question5_data['batch'] = str(pages)
    print(pages)


question5_data.head(10)
question5_data.shape
    
#question5_data.to_csv('/Users/natewagner/Documents/Surveys/question5_data_second_half.csv', encoding='utf-8', index = False)
    
    
    
    
    
    
    
