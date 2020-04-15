#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 07:14:10 2020

@author: natewagner
"""

import pandas as pd

YesNo_predictions = pd.read_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/yes_no_predictions.csv')


YesNo_predictions.head(10)

predicted = list(YesNo_predictions['predicted'])
survey = list(YesNo_predictions['survey'])
probability = list(YesNo_predictions['probability'])


Q2 = []
Q3 = []
Q4 = []
Q8 = []
Q9 = []

Q2_prob = []
Q3_prob = []
Q4_prob = []
Q8_prob = []
Q9_prob = []

Q2_surv = []
Q3_surv = []
Q4_surv = []
Q8_surv = []
Q9_surv = []


probs = []
answers = []
previous = []
cnt = 0
for i in range(0, len(predicted)):
    previous.append(predicted[i])
    if cnt == 4:
        if len(previous) == 2:
            if previous[0] == 1 and previous[1] == 1:
                Q9.append("check me")
                Q9_prob.append(probability[i])
                Q9_surv.append(survey[i])
            elif previous[0] == 1:
                Q9.append(1)
                Q9_prob.append(probability[i])
                Q9_surv.append(survey[i])
            elif previous[1] == 1:
                Q9.append(2)
                Q9_prob.append(probability[i])
                Q9_surv.append(survey[i])
            elif previous[0] == 0 and previous[1] == 0:
                Q9.append(3)
                Q9_prob.append(probability[i])
                Q9_surv.append(survey[i])
            previous = []
            cnt += 1
        #previous.append(predicted[i])
    if cnt == 3:
        if len(previous) == 2:
            if previous[0] == 1 and previous[1] == 1:
                Q8.append("check me")
                Q8_prob.append(probability[i])
                Q8_surv.append(survey[i])
            elif previous[0] == 1:
                Q8.append(1)
                Q8_prob.append(probability[i])
                Q8_surv.append(survey[i])
            elif previous[1] == 1:
                Q8.append(2)
                Q8_prob.append(probability[i])
                Q8_surv.append(survey[i])
            elif previous[0] == 0 and previous[1] == 0:
                Q8.append(3)
                Q8_prob.append(probability[i])
                Q8_surv.append(survey[i])
            previous = []
            cnt += 1
        #previous.append(predicted[i])   
    if cnt == 2:
        if len(previous) == 2:
            if previous[0] == 1 and previous[1] == 1:
                Q4.append("check me")
                Q4_prob.append(probability[i])
                Q4_surv.append(survey[i])
            elif previous[0] == 1:
                Q4.append(1)
                Q4_prob.append(probability[i])
                Q4_surv.append(survey[i])
            elif previous[1] == 1:
                Q4.append(2)
                Q4_prob.append(probability[i])
                Q4_surv.append(survey[i])
            elif previous[0] == 0 and previous[1] == 0:
                Q4.append(3)
                Q4_prob.append(probability[i])
                Q4_surv.append(survey[i])
            previous = []
            cnt += 1
        #previous.append(predicted[i])   
    if cnt == 1:
        if len(previous) == 2:
            if previous[0] == 1 and previous[1] == 1:
                Q3.append("check me")
                Q3_prob.append(probability[i])
                Q3_surv.append(survey[i])
            elif previous[0] == 1:
                Q3.append(1)
                Q3_prob.append(probability[i])
                Q3_surv.append(survey[i])
            elif previous[1] == 1:
                Q3.append(2)
                Q3_prob.append(probability[i])
                Q3_surv.append(survey[i])
            elif previous[0] == 0 and previous[1] == 0:
                Q3.append(3)
                Q3_prob.append(probability[i])
                Q3_surv.append(survey[i])
            previous = []
            cnt += 1
        #previous.append(predicted[i])
    if cnt == 0:
        if len(previous) == 2:
            if previous[0] == 1 and previous[1] == 1:
                Q2.append("check me")
                Q2_prob.append(probability[i])
                Q2_surv.append(survey[i])
            elif previous[0] == 1:
                Q2.append(1)
                Q2_prob.append(probability[i])
                Q2_surv.append(survey[i])
            elif previous[1] == 1:
                Q2.append(2)
                Q2_prob.append(probability[i])
                Q2_surv.append(survey[i])
            elif previous[0] == 0 and previous[1] == 0:
                Q2.append(3)
                Q2_prob.append(probability[i])
                Q2_surv.append(survey[i])
            previous = []
            cnt += 1
        #previous.append(predicted[i])
    if cnt == 5:
        cnt = 0


YesNo_predictions_clean = pd.DataFrame([Q2, Q2_prob, Q3, Q3_prob, Q4, Q4_prob,
                                       Q8, Q8_prob, Q9, Q9_prob, Q2_surv]).transpose()
YesNo_predictions_clean.columns = ["Q2", "Q2_prob", "Q3", "Q3_prob", 
                                   "Q4", "Q4_prob", "Q8", "Q8_prob",
                                   "Q9", "Q9_prob", "survey"]


#YesNo_predictions_clean.to_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/YesNo_predictions_clean.csv', encoding='utf-8', index = False)







question5_predictions = pd.read_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/Question5_predictions.csv')


question5_predictions.head(10)

predicted = list(question5_predictions['predicted'])
survey = list(question5_predictions['survey'])
probability = list(question5_predictions['probability'])


Q51 = []
Q52 = []
Q53 = []
Q54 = []
Q55 = []
Q56 = []
Q57 = []
Q58 = []


Q5_1_prob = []
Q5_2_prob = []
Q5_3_prob = []
Q5_4_prob = []
Q5_5_prob = []
Q5_6_prob = []
Q5_7_prob = []
Q5_8_prob = []

Q5_1_surv = []
Q5_2_surv = []
Q5_3_surv = []
Q5_4_surv = []
Q5_5_surv = []
Q5_6_surv = []
Q5_7_surv = []
Q5_8_surv = []


probs = []
answers = []
previous = []
cnt = 0
for i in range(0, len(predicted)):
    previous.append(predicted[i])
    if cnt == 7:
        if len(previous) == 3:
            if previous.count(1) > 1:
                Q58.append("check me")
                Q5_8_prob.append(probability[i])
                Q5_8_surv.append(survey[i])
            elif previous[0] == 1:
                Q58.append(1)
                Q5_8_prob.append(probability[i])
                Q5_8_surv.append(survey[i])
            elif previous[1] == 1:
                Q58.append(2)
                Q5_8_prob.append(probability[i])
                Q5_8_surv.append(survey[i])
            elif previous[2] == 1:
                Q58.append(3)
                Q5_8_prob.append(probability[i])
                Q5_8_surv.append(survey[i])                
            elif previous[0] == 0 and previous[1] == 0 and previous[2] == 0:
                Q58.append(0)
                Q5_8_prob.append(probability[i])
                Q5_8_surv.append(survey[i])
            previous = []
            cnt += 1
    if cnt == 6:
        if len(previous) == 3:
            if previous.count(1) > 1:
                Q57.append("check me")
                Q5_7_prob.append(probability[i])
                Q5_7_surv.append(survey[i])
            elif previous[0] == 1:
                Q57.append(1)
                Q5_7_prob.append(probability[i])
                Q5_7_surv.append(survey[i])
            elif previous[1] == 1:
                Q57.append(2)
                Q5_7_prob.append(probability[i])
                Q5_7_surv.append(survey[i])
            elif previous[2] == 1:
                Q57.append(3)
                Q5_7_prob.append(probability[i])
                Q5_7_surv.append(survey[i])                
            elif previous[0] == 0 and previous[1] == 0 and previous[2] == 0:
                Q57.append(0)
                Q5_7_prob.append(probability[i])
                Q5_7_surv.append(survey[i])
            previous = []
            cnt += 1  
    if cnt == 5:
        if len(previous) == 3:
            if previous.count(1) > 1:
                Q56.append("check me")
                Q5_6_prob.append(probability[i])
                Q5_6_surv.append(survey[i])
            elif previous[0] == 1:
                Q56.append(1)
                Q5_6_prob.append(probability[i])
                Q5_6_surv.append(survey[i])
            elif previous[1] == 1:
                Q56.append(2)
                Q5_6_prob.append(probability[i])
                Q5_6_surv.append(survey[i])
            elif previous[2] == 1:
                Q56.append(3)
                Q5_6_prob.append(probability[i])
                Q5_6_surv.append(survey[i])                
            elif previous[0] == 0 and previous[1] == 0 and previous[2] == 0:
                Q56.append(0)
                Q5_6_prob.append(probability[i])
                Q5_6_surv.append(survey[i])
            previous = []
            cnt += 1
    if cnt == 4:
        if len(previous) == 3:
            if previous.count(1) > 1:
                Q55.append("check me")
                Q5_5_prob.append(probability[i])
                Q5_5_surv.append(survey[i])
            elif previous[0] == 1:
                Q55.append(1)
                Q5_5_prob.append(probability[i])
                Q5_5_surv.append(survey[i])
            elif previous[1] == 1:
                Q55.append(2)
                Q5_5_prob.append(probability[i])
                Q5_5_surv.append(survey[i])
            elif previous[2] == 1:
                Q55.append(3)
                Q5_5_prob.append(probability[i])
                Q5_5_surv.append(survey[i])                
            elif previous[0] == 0 and previous[1] == 0 and previous[2] == 0:
                Q55.append(0)
                Q5_5_prob.append(probability[i])
                Q5_5_surv.append(survey[i])
            previous = []
            cnt += 1
    if cnt == 3:
        if len(previous) == 3:
            if previous.count(1) > 1:
                Q54.append("check me")
                Q5_4_prob.append(probability[i])
                Q5_4_surv.append(survey[i])
            elif previous[0] == 1:
                Q54.append(1)
                Q5_4_prob.append(probability[i])
                Q5_4_surv.append(survey[i])
            elif previous[1] == 1:
                Q54.append(2)
                Q5_4_prob.append(probability[i])
                Q5_4_surv.append(survey[i])
            elif previous[2] == 1:
                Q54.append(3)
                Q5_4_prob.append(probability[i])
                Q5_4_surv.append(survey[i])                
            elif previous[0] == 0 and previous[1] == 0 and previous[2] == 0:
                Q54.append(0)
                Q5_4_prob.append(probability[i])
                Q5_4_surv.append(survey[i])
            previous = []
            cnt += 1
    if cnt == 2:
        if len(previous) == 3:
            if previous.count(1) > 1:
                Q53.append("check me")
                Q5_3_prob.append(probability[i])
                Q5_3_surv.append(survey[i])
            elif previous[0] == 1:
                Q53.append(1)
                Q5_3_prob.append(probability[i])
                Q5_3_surv.append(survey[i])
            elif previous[1] == 1:
                Q53.append(2)
                Q5_3_prob.append(probability[i])
                Q5_3_surv.append(survey[i])
            elif previous[2] == 1:
                Q53.append(3)
                Q5_3_prob.append(probability[i])
                Q5_3_surv.append(survey[i])                
            elif previous[0] == 0 and previous[1] == 0 and previous[2] == 0:
                Q53.append(0)
                Q5_3_prob.append(probability[i])
                Q5_3_surv.append(survey[i])
            previous = []
            cnt += 1
    if cnt == 1:
        if len(previous) == 3:
            if previous.count(1) > 1:
                Q52.append("check me")
                Q5_2_prob.append(probability[i])
                Q5_2_surv.append(survey[i])
            elif previous[0] == 1:
                Q52.append(1)
                Q5_2_prob.append(probability[i])
                Q5_2_surv.append(survey[i])
            elif previous[1] == 1:
                Q52.append(2)
                Q5_2_prob.append(probability[i])
                Q5_2_surv.append(survey[i])
            elif previous[2] == 1:
                Q52.append(3)
                Q5_2_prob.append(probability[i])
                Q5_2_surv.append(survey[i])                
            elif previous[0] == 0 and previous[1] == 0 and previous[2] == 0:
                Q52.append(0)
                Q5_2_prob.append(probability[i])
                Q5_2_surv.append(survey[i])
            previous = []
            cnt += 1
    if cnt == 0:
        if len(previous) == 3:
            if previous.count(1) > 1:
                Q51.append("check me")
                Q5_1_prob.append(probability[i])
                Q5_1_surv.append(survey[i])
            elif previous[0] == 1:
                Q51.append(1)
                Q5_1_prob.append(probability[i])
                Q5_1_surv.append(survey[i])
            elif previous[1] == 1:
                Q51.append(2)
                Q5_1_prob.append(probability[i])
                Q5_1_surv.append(survey[i])
            elif previous[2] == 1:
                Q51.append(3)
                Q5_1_prob.append(probability[i])
                Q5_1_surv.append(survey[i])                
            elif previous[0] == 0 and previous[1] == 0 and previous[2] == 0:
                Q51.append(0)
                Q5_1_prob.append(probability[i])
                Q5_1_surv.append(survey[i])
            previous = []
            cnt += 1
    if cnt == 8:
        cnt = 0




question5_predictions_clean = pd.DataFrame([Q51, Q5_1_prob, Q52, Q5_2_prob,
                                            Q53, Q5_3_prob, Q54, Q5_4_prob,
                                            Q55, Q5_5_prob, Q56, Q5_6_prob,
                                            Q57, Q5_7_prob, Q58, Q5_8_prob,
                                            Q5_1_surv]).transpose()

question5_predictions_clean.columns = ["Q5_1", "Q5_1_prob", "Q5_2", "Q5_2_prob",
                                            "Q5_3", "Q5_3_prob", "Q5_4", "Q5_4_prob",
                                            "Q5_5", "Q5_5_prob", "Q5_6", "Q5_6_prob",
                                            "Q5_7", "Q5_7_prob", "Q5_8", "Q5_8_prob",
                                            "survey"]




#question5_predictions_clean.to_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/Question5_predictions_clean.csv', encoding='utf-8', index = False)








question6_predictions = pd.read_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/Question6_predictions.csv')

question6_predictions.head(20)

predicted = list(question6_predictions['predicted'])
survey = list(question6_predictions['survey'])
probability = list(question6_predictions['probability'])



Q6 = []
Q6_prob =[]
Q6_surv = []



probs = []
answers = []
previous = []
cnt = 0
for i in range(0, len(predicted)):
    previous.append(predicted[i])
    if len(previous) == 10:
        if previous.count(1) > 1:
            Q6.append("check me")
            Q6_prob.append(probability[i])
            Q6_surv.append(survey[i])
        elif previous[0] == 1:
            Q6.append(1)
            Q6_prob.append(probability[i])
            Q6_surv.append(survey[i])  
        elif previous[1] == 1:
            Q6.append(2)
            Q6_prob.append(probability[i])
            Q6_surv.append(survey[i])
        elif previous[2] == 1:
            Q6.append(3)
            Q6_prob.append(probability[i])
            Q6_surv.append(survey[i])
        elif previous[3] == 1:
            Q6.append(4)
            Q6_prob.append(probability[i])
            Q6_surv.append(survey[i])
        elif previous[4] == 1:
            Q6.append(5)
            Q6_prob.append(probability[i])
            Q6_surv.append(survey[i])
        elif previous[5] == 1:
            Q6.append(6)
            Q6_prob.append(probability[i])
            Q6_surv.append(survey[i])
        elif previous[6] == 1:
            Q6.append(7)
            Q6_prob.append(probability[i])
            Q6_surv.append(survey[i])
        elif previous[7] == 1:
            Q6.append(8)
            Q6_prob.append(probability[i])
            Q6_surv.append(survey[i])
        elif previous[8] == 1:
            Q6.append(9)
            Q6_prob.append(probability[i])
            Q6_surv.append(survey[i])
        elif previous[9] == 1:
            Q6.append(10)
            Q6_prob.append(probability[i])
            Q6_surv.append(survey[i])
        elif previous.count(1) < 1:
            Q6.append(0)
            Q6_prob.append(probability[i])
            Q6_surv.append(survey[i])   
        previous = []
        cnt += 1
    if cnt == 11:
        cnt = 0



question6_predictions_clean = pd.DataFrame([Q6, Q6_prob, Q6_surv]).transpose()

question6_predictions_clean.columns = ["Q6", "probability", "survey"]

#question6_predictions_clean.to_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/Question6_predictions_clean.csv', encoding='utf-8', index = False)








question10_predictions = pd.read_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/Question10_predictions.csv')

question10_predictions.head(20)

predicted = list(question10_predictions['predicted'])
survey = list(question10_predictions['survey'])
probability = list(question10_predictions['probability'])




Q10_a = []
Q10_b = []
Q10_c = []
Q10_d = []
Q10_e = []
Q10_f = []
Q10_g = []
Q10_h = []
Q10_i = []
Q10_j = []
Q10_k = []

Q10_a_prob = []
Q10_b_prob = []
Q10_c_prob = []
Q10_d_prob = []
Q10_e_prob = []
Q10_f_prob = []
Q10_g_prob = []
Q10_h_prob = []
Q10_i_prob = []
Q10_j_prob = []
Q10_k_prob = []

Q10_a_surv = []
Q10_b_surv = []
Q10_c_surv = []
Q10_d_surv = []
Q10_e_surv = []
Q10_f_surv = []
Q10_g_surv = []
Q10_h_surv = []
Q10_i_surv = []
Q10_j_surv = []
Q10_k_surv = []


probs = []
answers = []
previous = []
cnt = 0
for i in range(0, len(predicted)):
    previous.append(predicted[i])
    if len(previous) == 11:
        if previous[0] == 1:
            Q10_a.append(1)
            Q10_a_prob.append(probability[i])
            Q10_a_surv.append(survey[i])
        else:
            Q10_a.append(0)
            Q10_a_prob.append(probability[i])
            Q10_a_surv.append(survey[i])                
        if previous[1] == 1:
            Q10_b.append(1)
            Q10_b_prob.append(probability[i])
            Q10_b_surv.append(survey[i])
        else:
            Q10_b.append(0)
            Q10_b_prob.append(probability[i])
            Q10_b_surv.append(survey[i])                
        if previous[2] == 1:
            Q10_c.append(1)
            Q10_c_prob.append(probability[i])
            Q10_c_surv.append(survey[i])
        else:
            Q10_c.append(0)
            Q10_c_prob.append(probability[i])
            Q10_c_surv.append(survey[i])                 
        if previous[3] == 1:
            Q10_d.append(1)
            Q10_d_prob.append(probability[i])
            Q10_d_surv.append(survey[i])
        else:
            Q10_d.append(0)
            Q10_d_prob.append(probability[i])
            Q10_d_surv.append(survey[i])  
        if previous[4] == 1:
            Q10_e.append(1)
            Q10_e_prob.append(probability[i])
            Q10_e_surv.append(survey[i])
        else:
            Q10_e.append(0)
            Q10_e_prob.append(probability[i])
            Q10_e_surv.append(survey[i])
        if previous[5] == 1:
            Q10_f.append(1)
            Q10_f_prob.append(probability[i])
            Q10_f_surv.append(survey[i])
        else:
            Q10_f.append(0)
            Q10_f_prob.append(probability[i])
            Q10_f_surv.append(survey[i])
        if previous[6] == 1:
            Q10_g.append(1)
            Q10_g_prob.append(probability[i])
            Q10_g_surv.append(survey[i])
        else:
            Q10_g.append(0)
            Q10_g_prob.append(probability[i])
            Q10_g_surv.append(survey[i])
        if previous[7] == 1:
            Q10_h.append(1)
            Q10_h_prob.append(probability[i])
            Q10_h_surv.append(survey[i])
        else:
            Q10_h.append(0)
            Q10_h_prob.append(probability[i])
            Q10_h_surv.append(survey[i])
        if previous[8] == 1:
            Q10_i.append(1)
            Q10_i_prob.append(probability[i])
            Q10_i_surv.append(survey[i])
        else:
            Q10_i.append(0)
            Q10_i_prob.append(probability[i])
            Q10_i_surv.append(survey[i])
        if previous[9] == 1:
            Q10_j.append(1)
            Q10_j_prob.append(probability[i])
            Q10_j_surv.append(survey[i])
        else:
            Q10_j.append(0)
            Q10_j_prob.append(probability[i])
            Q10_j_surv.append(survey[i])
        if previous[10] == 1:
            Q10_k.append(1)
            Q10_k_prob.append(probability[i])
            Q10_k_surv.append(survey[i])
        else:
            Q10_k.append(0)
            Q10_k_prob.append(probability[i])
            Q10_k_surv.append(survey[i])                  
        previous = []
        cnt += 1
if cnt == 12:
    cnt = 0



question10_predictions_clean = pd.DataFrame([Q10_a, Q10_a_prob,
                                             Q10_b, Q10_b_prob,
                                             Q10_c, Q10_c_prob,
                                             Q10_d, Q10_d_prob,
                                             Q10_e, Q10_e_prob,
                                             Q10_f, Q10_f_prob,
                                             Q10_g, Q10_g_prob,
                                             Q10_h, Q10_h_prob,
                                             Q10_i, Q10_i_prob,
                                             Q10_j, Q10_j_prob,
                                             Q10_k, Q10_k_prob,
                                             Q10_a_surv]).transpose()

question10_predictions_clean.columns = ["Q10_a", "Q10_a_prob",
                                             "Q10_b", "Q10_b_prob",
                                             "Q10_c", "Q10_c_prob",
                                             "Q10_d", "Q10_d_prob",
                                             "Q10_e", "Q10_e_prob",
                                             "Q10_f", "Q10_f_prob",
                                             "Q10_g", "Q10_g_prob",
                                             "Q10_h", "Q10_h_prob",
                                             "Q10_i", "Q10_i_prob",
                                             "Q10_j", "Q10_j_prob",
                                             "Q10_k", "Q10_k_prob",
                                             "Q10_a_surv"]

#question10_predictions_clean.to_csv('/Users/natewagner/Documents/Surveys/ALL_SURVEY_DATA/Question10_predictions_clean.csv', encoding='utf-8', index = False)




