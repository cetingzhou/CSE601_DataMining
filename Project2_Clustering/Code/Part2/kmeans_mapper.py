#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:26:15 2017

"""
# mapper
import csv
import sys
import math
from itertools import izip
with open('centers', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter = "\t")
    centroids = [[float(e) for e in r] for r in reader]

for line in sys.stdin:
    point = map(float, line.strip().split('\t'))
    dist = [math.sqrt(sum((x - y)**2 for x, y in izip(point, cent))) for cent in centroids]
    cluster = min(izip(dist, range(len(dist))))[1]
    emit_key = '%d' % cluster
    emit_value = ','.join('%.4f' % v for v in point)
    print ('\t'.join((emit_key, emit_value)))
