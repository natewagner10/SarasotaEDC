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



# from PyPDF2 import PdfFileWriter, PdfFileReader

# pdf_document = "/Users/natewagner/Documents/Surveys/batch18.pdf"
# pdf = PdfFileReader(pdf_document)

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
    
    
    # extract company name
    comp_info_dims = (465+250, 1100, 1250+400, 1145)
    comp_info = images_bw.crop(comp_info_dims) 
    bus_name = pytesseract.image_to_string(comp_info)
    
    
    # question 2
    question2_dims = (150-40, 783+210, 290+10, 923+125)
    question2 = images_bw.crop(question2_dims)    
    q2 = Image.new('L', (190, 190))
    q2.paste(question2, (0, 60))
    
    Q2p = []
    for pixel in iter(q2.getdata()):
        Q2p.append(pixel)
        
    q2df = pd.DataFrame(Q2p).transpose()
    q2df['question'] = 'question2'
    q2df['accnt_num'] = accnt_number
    q2df['bus_name'] = bus_name
    
    
    
    # question 3
    question3_dims = (135-30, 860+130, 275+20, 1000+60)
    question3 = images_bw.crop(question3_dims)
    q3 = Image.new('L', (190, 190))
    q3.paste(question3, (0, 60))
    Q3p = []

    for pixel in iter(q3.getdata()):
        Q3p.append(pixel)

    q3df = pd.DataFrame(Q3p).transpose()
    q3df['question'] = 'question3'
    q3df['accnt_num'] = accnt_number
    q3df['bus_name'] = bus_name
    
    
    
    # question 4
    question4_dims = (115, 1070+63, 305, 1070+135)
    question4 = images_bw.crop(question4_dims) 
    q4 = Image.new('L', (190, 190))
    q4.paste(question4, (0, 60))
    
    Q4p = []
    for pixel in iter(q4.getdata()):
        Q4p.append(pixel)

    q4df = pd.DataFrame(Q4p).transpose()
    q4df['question'] = 'question4'
    q4df['accnt_num'] = accnt_number
    q4df['bus_name'] = bus_name
    
    
    
    # question 8
    question8_dims = (1060+300, 928-185, 1250+300, 1118-315)
    question8 = images_bw.crop(question8_dims) 
    q8 = Image.new('L', (190, 190))
    q8.paste(question8, (0, 60))
    
    Q8p = []
    for pixel in iter(q8.getdata()):
        Q8p.append(pixel)

    q8df = pd.DataFrame(Q8p).transpose()
    q8df['question'] = 'question8'
    q8df['accnt_num'] = accnt_number
    q8df['bus_name'] = bus_name
    
    
    
    # question 9
    question9_dims = (1060+112, 928-100, 1250+112, 1118-240)
    question9 = images_bw.crop(question9_dims) 
    q9 = Image.new('L', (190, 190))
    q9.paste(question9, (0, 60))
    
    Q9p = []
    for pixel in iter(q9.getdata()):
        Q9p.append(pixel)
        
    q9df = pd.DataFrame(Q9p).transpose()
    q9df['question'] = 'question9'
    q9df['accnt_num'] = accnt_number
    q9df['bus_name'] = bus_name

    check_YN_data = pd.concat([q2df, q3df, q4df, q8df, q9df])
    check_YN_data['survey'] = str(page)
    
    
    return(check_YN_data)
    




sum([100, 82, 85, 46, 81, 120, 100, 100, 91])

pdfs = [66, 99, 123, 94, 137, 100, 92, 138, 164, 100, 82, 85, 46, 81, 120, 100, 100, 91]
#pdfs = [100, 82, 85, 46, 81, 120, 100, 100, 91]

batch = 'batch'
survey_num = '/survey-page-'
end = '.pdf'
checkYN = pd.DataFrame()
for pages in range(10, 19):
    for x in range(1, pdfs[pages-1]+1):
        survey = batch + str(pages) + survey_num + str(x) + end        
        df = parseCheckYN(survey)
        checkYN = checkYN.append(df)
    checkYN['batch'] = str(pages)
    print(pages)


checkYN.head(10)
checkYN.shape


#checkYN.to_csv('/Users/natewagner/Documents/Surveys/checkYN_second_half.csv', encoding='utf-8', index = False)






# checking to see if we have accnt num or bus name
acc = checkYN['accnt_num']
bn = checkYN['bus_name']
batch = checkYN['batch']
survey = checkYN['survey']

dff2 = pd.concat([acc,bn,batch,survey], axis = 1)
dff2 = dff2.drop_duplicates()
dff2.shape
#dff2.to_csv('/Users/natewagner/Documents/Surveys/checkYN_second_half_compslist.csv', encoding='utf-8', index = False)


#scan1_company_list = dff.append(dff2)
#scan1_company_list.to_csv('/Users/natewagner/Documents/Surveys/scan1_company_list.csv', encoding='utf-8', index = False)





# just playing around here

dff.head(10)

items = []
for line in dff.iloc[:,0]:
    num = line.strip().replace("_", "")
    num = num.replace("-", "")
    num = num.replace(",", "")
    num = num.replace(":", "")
    num = num.replace(".", "")
    items.append(num)
 
    
test = "BVUYT To4de"
abc = "abcdefghijklmnop"
for x in test.lower():
    print(x in abc)
    





