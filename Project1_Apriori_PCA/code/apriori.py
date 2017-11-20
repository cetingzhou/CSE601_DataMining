#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:19:24 2017
"""

import pandas as pd
import copy
from itertools import combinations

import time
start_time = time.time()

def Apriori(D, min_sup):
    C1, L = find_frequent_1_itemsets(D, min_sup)
    L1 = scan_count(D, L)
    
    freq_itemsets = {}   # use dictionary to record count
    freq_itemsets.update(L1)
    
    k = 1
    
    while L != []:
        k += 1
        C = apriori_gen(L, k)
        temp_save = []
        temp_remove = []
        temp_freq = scan_count(D, C)
        for key in temp_freq.keys():
            if temp_freq[key] / len(data) >= min_sup:
                temp_save.append(key)
            else:
                temp_remove.append(key)
        for key in temp_remove:
            temp_freq.pop(key)
            
        L = temp_save
        L.sort()
        freq_itemsets.update(temp_freq)
        #print(temp_freq)
        #print(L)
        
    return freq_itemsets
        
def find_frequent_1_itemsets(D, min_sup):
    C1 = {}
    for t in D:
        for item in t:
            if item in C1:
                C1[item] += 1.0
            else:
                C1[item] = 1.0
    L1 = [];
    for key in C1.keys():
        if C1[key] / len(data) >= min_sup:
            L1.append(key)
    L1.sort()
    return C1, L1

def apriori_gen(pre_freq_itemsets, k):
    candidates = []
    if k == 2:
        for l1 in pre_freq_itemsets:
            for l2 in pre_freq_itemsets:
                if l1 < l2:
                    c = [l1]
                    c.append(l2)
                    candidates.append(c)

    else:
        # for iteration we need to convert tuple to list
        pre_freq_itemsets = [list(e) for e in pre_freq_itemsets]
        for l1 in pre_freq_itemsets:
            for l2 in pre_freq_itemsets:
                if l1[:-1] == l2[:-1] and l1[-1] < l2[-1]:
                    temp1 = l1[:]
                    temp2 = l2[:]
                    temp1.append(temp2[-1])
                    c = temp1
                    # prune
                    if has_infrequent_subset(c, pre_freq_itemsets, k-1) == False:
                        candidates.append(c)
                    
        # for the key of dictionary, we need to convert list to tuple
    C = [tuple(e) for e in candidates]
    return C

def has_infrequent_subset(c, pre_freq_itemsets, k):
    s = list(combinations(c, k))
    s = [list(l) for l in s]
    for i in s:
        if i not in pre_freq_itemsets:
            return True
    return False

def scan_count(D, C):
    result = {}
    for c in C:
        result[c] = 0
    # c is tuple or str; the key of dictionary should be tuple or str
    if C != [] and type(C[0]) == str:
        for t in D:
            for c in C:
                if c in t:
                    result[c] += 1.0

    else:
        for t in D:
            for c in C:
                if check_sublist(list(c), t):
                    result[c] += 1.0
    return result

def check_sublist(sub, sup):
    for element in sub:
        if element not in sup:
            return False
    return True

def rule_gen(Data, freq_itemsets, freq_set, min_conf):
    # get the frequent items sets respect to min_sup
    L = freq_itemsets
    # get the support of the frequent itemset
    freq_set_sup = L[freq_set]
    # get all possibel rule bodies (heads)
    rules_body = []
    for i in range(len(freq_set)-1):
        temp = list(combinations(freq_set, i+1))
        for j in range(len(temp)):
            rules_body.append(temp[j])
    # compute the support of the bodies and record in dictionary
    asso_rules = scan_count(Data, rules_body)
    # compute the confidence of each association rule
    rules_remove = []
    for key in rules_body:
        asso_rules[key] = freq_set_sup / asso_rules[key]
        if asso_rules[key] < min_conf:
            rules_remove.append(key)
    # remove the rule below min_conf
    for rule in rules_remove:
        asso_rules.pop(rule)
        
    return asso_rules

def find_rules(Data, freq_itemsets, min_conf):
    L = list(freq_itemsets.keys())
    all_asso_rules = {}
    for key in L:
        # only consider frequent itemset with length-2 or more 
        # the key of all_ass_rules is a frequent itemset
        if type(key) != str:
            all_asso_rules[key] = rule_gen(Data, freq_itemsets, key, min_conf)
    return all_asso_rules

def template1(all_asso_rules, RBH, ANN, items):
    all_rules = copy.deepcopy(all_asso_rules)
    asso_rules = {}
    if RBH == "RULE":
        if ANN == "ANY":
            for key in all_rules.keys():
                for i in items:
                    if i in key:
                        asso_rules[key] = all_rules[key]
                        break
        if ANN == "NONE":
            for key in all_rules.keys():
                asso_rules[key] = all_rules[key]
                for i in items:
                    if i in key:
                        asso_rules.pop(key)
                        break
        if ANN == 1:
            for key in all_rules.keys():
                for i in items:
                    if i in key:
                        asso_rules[key] = all_rules[key]
            for key in list(asso_rules.keys()):
                count = 0
                for i in items:
                    if i in key:
                        count += 1
                if count != 1:
                    asso_rules.pop(key)
                    
    if RBH == "BODY":
        if ANN == "ANY":
            for key in all_rules.keys():
                # the key in dictionary freq_set is the rule body
                freq_set = all_rules[key]
                for body in list(freq_set.keys()):
                    count = 0
                    for i in items:
                        if i in body:
                            count += 1
                    if count == 0:
                        freq_set.pop(body)
                asso_rules[key] = all_rules[key]
        if ANN == "NONE":
            for key in all_rules.keys():
                freq_set = all_rules[key]
                for body in list(freq_set.keys()):
                    for i in items:
                        if i in body:
                            freq_set.pop(body)
                            break
                asso_rules[key] = all_rules[key]
        if ANN == 1:
            for key in all_rules.keys():
                freq_set = all_rules[key]
                for body in list(freq_set.keys()):
                    count = 0
                    for i in items:
                        if i in body:
                            count += 1
                    if count != 1:
                        freq_set.pop(body)
                asso_rules[key] = all_rules[key]
                
    if RBH == "HEAD":
        all_rules_copy = {}
        asso_rules_copy = {}
        # change the head as the dictionary key
        for key in list(all_rules.keys()):
            head_dict = {}
            all_rules_copy[key] = head_dict
            inner = all_rules[key]
            for body in list(inner.keys()):
                head = tuple(head for head in key if head not in body)
                head_dict[head] = inner[body]
        if ANN == "ANY":
            for key in all_rules_copy.keys():
                # the key in dictionary freq_set is the rule head
                freq_set = all_rules_copy[key]
                for head in list(freq_set.keys()):
                    count = 0
                    for i in items:
                        if i in head:
                            count += 1
                    if count == 0:
                        freq_set.pop(head)
                asso_rules_copy[key] = all_rules_copy[key]
        if ANN == "NONE":
            for key in all_rules_copy.keys():
                freq_set = all_rules_copy[key]
                for head in list(freq_set.keys()):
                    for i in items:
                        if i in head:
                            freq_set.pop(head)
                            break
                asso_rules_copy[key] = all_rules_copy[key]
        if ANN == 1:
            for key in all_rules_copy.keys():
                freq_set = all_rules_copy[key]
                for head in list(freq_set.keys()):
                    count = 0
                    for i in items:
                        if i in head:
                            count += 1
                    if count != 1:
                        freq_set.pop(head)
                asso_rules_copy[key] = all_rules_copy[key]
        
        # change the body as the dictionary head
        for key in list(asso_rules_copy.keys()):
            body_dict = {}
            asso_rules[key] = body_dict
            inner = asso_rules_copy[key]
            for head in list(inner.keys()):
                body = tuple(body for body in key if body not in head)
                body_dict[body] = inner[head]
                
    return asso_rules

    
def template2(all_asso_rules, RBH, N):
    all_rules = copy.deepcopy(all_asso_rules)
    asso_rules = {}
    if RBH == "RULE":
        # key is freq_itemset
        for key in all_rules.keys():
            if len(key) >= N:
                asso_rules[key] = all_rules[key]
                
    if RBH == "BODY":
        # key is freq_itemset
        # remove the rules with length <= N 
        for key in all_rules.keys():
            if len(key) >= N+1:
                asso_rules[key] = all_rules[key]
        
        for key in asso_rules.keys():
            # the key in dictionary freq_set is the rule body
            freq_set = asso_rules[key]
            for body in list(freq_set.keys()):
                if len(body) < N:
                    freq_set.pop(body)
        
    if RBH == "HEAD":
        for key in all_rules.keys():
            if len(key) >= N+1:
                asso_rules[key] = all_rules[key]
                
        for key in asso_rules.keys():
            freq_set = asso_rules[key]
            for body in list(freq_set.keys()):
                if len(body) > len(key) - N:
                    freq_set.pop(body)
                    
    return asso_rules


def template3(all_rules, comb, *args):
    if comb == "1or1":
        asso1 = template1(all_rules, args[0], args[1], args[2])
        asso2 = template1(all_rules, args[3], args[4], args[5])
        rules1 = take_rules(asso1)
        rules2 = take_rules(asso2)
        r1r2 = rules1 + [r for r in rules2 if r not in rules1]
    if comb == "1and1":
        asso1 = template1(all_rules, args[0], args[1], args[2])
        asso2 = template1(all_rules, args[3], args[4], args[5])
        rules1 = take_rules(asso1)
        rules2 = take_rules(asso2)
        r1r2 = [r for r in rules1 if r in rules2]
    if comb == "1or2":
        asso1 = template1(all_rules, args[0], args[1], args[2])
        asso2 = template2(all_rules, args[3], args[4])
        rules1 = take_rules(asso1)
        rules2 = take_rules(asso2)
        r1r2 = rules1 + [r for r in rules2 if r not in rules1]
    if comb == "1and2":
        asso1 = template1(all_rules, args[0], args[1], args[2])
        asso2 = template2(all_rules, args[3], args[4])
        rules1 = take_rules(asso1)
        rules2 = take_rules(asso2)
        r1r2 = [r for r in rules1 if r in rules2]
    if comb == "2or2":
        asso1 = template2(all_rules, args[0], args[1])
        asso2 = template2(all_rules, args[2], args[3])
        rules1 = take_rules(asso1)
        rules2 = take_rules(asso2)
        r1r2 = rules1 + [r for r in rules2 if r not in rules1]
    if comb == "2and2":
        asso1 = template2(all_rules, args[0], args[1])
        asso2 = template2(all_rules, args[2], args[3])
        rules1 = take_rules(asso1)
        rules2 = take_rules(asso2)
        r1r2 = [r for r in rules1 if r in rules2]
    count = 0
    for i in range(len(r1r2)):
        count += 1
        print (r1r2[i][0], '-->', r1r2[i][1])
    print ("the number of association rule: {}".format(count))
    return r1r2, count


def print_rules(asso_rules):
    # remove all empyty {}
    for key in list(asso_rules.keys()):
        if asso_rules[key] == {}:
            asso_rules.pop(key)
    # seperate body and head
    count = 0
    body_list = []
    head_list = []
    rules = []
    for key in asso_rules.keys():
        rule_body = asso_rules[key]
        for body in list(rule_body.keys()):
            head = tuple(head for head in key if head not in body)
            body_list.append(body)
            head_list.append(head)
            print (body, "-->", head)
            count += 1
    for i in range(len(body_list)):
        rules.append([body_list[i], head_list[i]])
        
    print ("the number of association rule: {}".format(count))
    return rules, count

def take_rules(asso_rules):
    for key in list(asso_rules.keys()):
        if asso_rules[key] == {}:
            asso_rules.pop(key)
    # seperate body and head
    body_list = []
    head_list = []
    rules = []
    for key in asso_rules.keys():
        rule_body = asso_rules[key]
        for body in list(rule_body.keys()):
            head = tuple(head for head in key if head not in body)
            body_list.append(body)
            head_list.append(head)
    for i in range(len(body_list)):
        rules.append([body_list[i], head_list[i]])
    return rules

###############################################################################
#--------------------------------LOAD DATA------------------------------------#
###############################################################################
# load data and data preprocessing
data = pd.read_csv('/Users/Jeremy/Desktop/Pre-CS-Courses/'+
                   '601DataMining/project1/associationruletestdata.txt',\
                   header = None, sep = '\t')
print (data.info())
# change the data to the format as "G0_Up"
for i in range(data.shape[1]-1):
    for j in range(data.shape[0]):
        data[i][j] = 'G%s_'%(i+1)+data[i][j]
data = data.values


###############################################################################
#----------------------------------PART-1-------------------------------------#
###############################################################################
"""
   if the last key of freq_itemsets is string, which means the max length
   of frequent itemset is 1.
   from the function of Apriori, length of the last key of freq_itemsets
   should be the max-length of frequent itemsets.
"""

"""
min_sup_list = [0.7, 0.6, 0.5, 0.4, 0.3]
for min_sup in min_sup_list:
    freq_itemsets = Apriori(data, min_sup)
    keys = list(freq_itemsets.keys())
    
    print ('Support is set to be {}:'.format(min_sup))
    
    if type(keys[-1]) == str:
        max_length = 1
        print ('the max-length of frequent itemsets is 1')
    else:
        max_length = len(keys[-1])
        print ('the max-length of frequent itemsets is {}'.format(max_length))
        
    for l in range(max_length):
        count = 0
        if l+1 == 1:
            for k in keys:
                if type(k) == str:
                    count += 1
            print ('the number of length-1 frequent itemsets: {}'.format(count))
        
        if l+1 > 1:
            for k in keys:
                if type(k) == tuple:
                    if len(k) == l+1:
                        count += 1
            print ('the number of length-{} frequent itemsets: {}'.format(l+1, count))
"""

###############################################################################
#----------------------------------PART-2-------------------------------------#
###############################################################################
# Given minimum support to find all frequent itemsets
freq_itemsets = Apriori(data, 0.5)
# Given minimum confidence to find all association rules
all_asso_rules = find_rules(data, freq_itemsets, 0.7)

#################################-Template1-###################################
print ('-------------------result11--------------------')
asso11 = template1(all_asso_rules, "RULE", "ANY", ['G59_Up'])
result11, cnt11 = print_rules(asso11)

print ('-------------------result12--------------------')
asso12 = template1(all_asso_rules, "RULE", "NONE", ['G59_Up'])
result12, cnt12 = print_rules(asso12)

print ('-------------------result13--------------------')
asso13 = template1(all_asso_rules, "RULE", 1, ('G59_Up', 'G10_Down'))
result13, cnt13 = print_rules(asso13)

print ('-------------------result14--------------------')
asso14 = template1(all_asso_rules, "BODY", "ANY", ['G59_Up'])
result14, cnt14 = print_rules(asso14)

print ('-------------------result15--------------------')
asso15 = template1(all_asso_rules, "BODY", "NONE", ['G59_Up'])
result15, cnt15 = print_rules(asso15)

print ('-------------------result16--------------------')
asso16 = template1(all_asso_rules, "BODY", 1, ('G59_Up', 'G10_Down'))
result16, cnt16 = print_rules(asso16)

print ('-------------------result17--------------------')
asso17 = template1(all_asso_rules, "HEAD", "ANY", ['G59_Up'])
result17, cnt17 = print_rules(asso17)

print ('-------------------result18--------------------')
asso18 = template1(all_asso_rules, "HEAD", "NONE", ['G59_Up'])
result18, cnt18 = print_rules(asso18)

print ('-------------------result19--------------------')
asso19 = template1(all_asso_rules, "HEAD", 1, ('G59_Up', 'G10_Down'))
result19, cnt19 = print_rules(asso19)


#################################-Template2-###################################
print ('-------------------result21--------------------')
asso21 = template2(all_asso_rules, "RULE", 3)
result21, cnt21 = print_rules(asso21)

print ('-------------------result22--------------------')
asso22 = template2(all_asso_rules, "BODY", 2)
result22, cnt22 = print_rules(asso22)

print ('-------------------result23--------------------')
asso23 = template2(all_asso_rules, "HEAD", 1)
result23, cnt23 = print_rules(asso23)


#################################-Template3-###################################
print ('-------------------result31--------------------')
result31, cnt31 = template3(all_asso_rules, "1or1", "BODY", "ANY", ['G10_Down'],\
                            "HEAD", 1, ['G59_Up'])

print ('-------------------result32--------------------')
result32, cnt32 = template3(all_asso_rules, "1and1", "BODY", "ANY", ['G10_Down'],\
                            "HEAD", 1, ['G59_Up'])

print ('-------------------result33--------------------')
result33, cnt33 = template3(all_asso_rules, "1or2", "BODY", "ANY", ['G10_Down'],\
                            "HEAD", 2)

print ('-------------------result34--------------------')
result34, cnt34 = template3(all_asso_rules, "1and2", "BODY", "ANY", ['G10_Down'],\
                            "HEAD", 2)

print ('-------------------result35--------------------')
result35, cnt35 = template3(all_asso_rules, "2or2", "BODY", 1,\
                            "HEAD", 2)

print ('-------------------result36--------------------')
result36, cnt36 = template3(all_asso_rules, "2and2", "BODY", 1,\
                            "HEAD", 2)

print("--- %s seconds ---" % (time.time() - start_time))