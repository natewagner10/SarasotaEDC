#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:53:11 2020

@author: natewagner
"""

#import matplotlib as plt
from PIL import Image

#from PIL import Image
#import pandas as pd
#from matplotlib import image
#from matplotlib import pyplot
#from numpy import asarray
#import numpy as np
import pytesseract



from PyPDF2 import PdfFileWriter, PdfFileReader

pdf_document = "/Users/natewagner/Documents/Surveys/batch01.pdf"
pdf = PdfFileReader(pdf_document)

# for page in range(pdf.getNumPages()):
#     pdf_writer = PdfFileWriter()
#     current_page = pdf.getPage(page)
#     pdf_writer.addPage(current_page)

#     outputFilename = "survey-page-{}.pdf".format(page + 1)
#     with open(outputFilename, "wb") as out:
#         pdf_writer.write(out)

#         print("created", outputFilename)





from pdf2image import convert_from_path
import pandas as pd



def parseCheckYN(page):

    # set path
    setpath = '/Users/natewagner/Documents/Surveys/batch1/'
    path = setpath + page
    
    # Convert to png and to greyscale / rotate
    images = convert_from_path(path)
    images_bw = images[0].convert('L') 
    images_bw = images_bw.transpose(Image.ROTATE_270)
    
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
    check_YN_data['survey'] = str(page)
    
    return(check_YN_data)
    




survey_num = 'survey-page-'
end = '.pdf'
checkYN_batch1 = pd.DataFrame()
for x in range(1, 67):
    survey = survey_num + str(x) + end
    df = parseCheckYN(survey)
    checkYN_batch1 = checkYN_batch1.append(df)
    

checkYN_batch1.head(10)
checkYN_batch1.shape
checkYN_batch1['batch'] = 'batch1'

checkYN_batch1.to_csv('/Users/natewagner/Documents/Surveys/checkYN_batch1.csv', encoding='utf-8', index = False)






# checking to see if we have accnt num or bus name
acc = checkYN_batch1['accnt_num']
bn = checkYN_batch1['bus_name']
dff = pd.concat([acc,bn], axis = 1)
dff['batch'] = 'batch1'
dff.to_csv('/Users/natewagner/Documents/Surveys/dff.csv', encoding='utf-8', index = False)
























