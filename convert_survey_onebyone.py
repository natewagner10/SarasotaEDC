#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:16:14 2020

@author: natewagner
"""

from pdf2image import convert_from_path
import pandas as pd
import pytesseract
from PIL import Image

# set path
path = '/Users/natewagner/Documents/Surveys/batch1/survey-page-8.pdf'


# Convert to png and to greyscale / rotate
images = convert_from_path(path)
images_bw = images[0].convert('L') 
images_bw = images_bw.transpose(Image.ROTATE_270)
images_bw.show()
# set path to pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'


# extract account number
accnt_num_dims = (1000+155, 150-140, 1600-200, 75)
accnt_num = images_bw.crop(accnt_num_dims)    
accnt_number = pytesseract.image_to_string(accnt_num)


# extract company name
comp_info_dims = (465+250, 1100, 1250+400, 1145)
comp_info = images_bw.crop(comp_info_dims) 
bus_name = pytesseract.image_to_string(comp_info)


# question 2
question2_dims = (150-40, 783+145, 290+10, 923+195)
question2 = images_bw.crop(question2_dims)    
Q2p = []
for pixel in iter(question2.getdata()):
    Q2p.append(pixel)

q2df = pd.DataFrame(Q2p).transpose()
q2df['question'] = 'question2'
q2df['accnt_num'] = accnt_number
q2df['bus_name'] = bus_name
#q2df.head(20)



# question 3
question3_dims = (135-30, 860+80, 275+20, 1000+130)
question3 = images_bw.crop(question3_dims) 
Q3p = []
for pixel in iter(question3.getdata()):
    Q3p.append(pixel)

q3df = pd.DataFrame(Q3p).transpose()
q3df['question'] = 'question3'
q3df['accnt_num'] = accnt_number
q3df['bus_name'] = bus_name



# question 4
question4_dims = (115, 1070, 305, 1070+190)
question4 = images_bw.crop(question4_dims) 
Q4p = []
for pixel in iter(question4.getdata()):
    Q4p.append(pixel)

q4df = pd.DataFrame(Q4p).transpose()
q4df['question'] = 'question4'
q4df['accnt_num'] = accnt_number
q4df['bus_name'] = bus_name



# question 8
question8_dims = (1060+300, 928-250, 1250+300, 1118-250)
question8 = images_bw.crop(question8_dims) 
Q8p = []
for pixel in iter(question8.getdata()):
    Q8p.append(pixel)

q8df = pd.DataFrame(Q8p).transpose()
q8df['question'] = 'question8'
q8df['accnt_num'] = accnt_number
q8df['bus_name'] = bus_name



# question 9
question9_dims = (1060+112, 928-180, 1250+112, 1118-180)
question9 = images_bw.crop(question9_dims) 
Q9p = []
for pixel in iter(question9.getdata()):
    Q9p.append(pixel)

q9df = pd.DataFrame(Q9p).transpose()
q9df['question'] = 'question9'
q9df['accnt_num'] = accnt_number
q9df['bus_name'] = bus_name

check_YN_data = pd.concat([q2df, q3df, q4df, q8df, q9df])
