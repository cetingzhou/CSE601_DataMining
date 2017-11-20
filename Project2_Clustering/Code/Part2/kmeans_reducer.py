#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:38:00 2017

"""
import sys
from itertools import izip
#reducer
cost = 0
current_clusid = -1
cluster = {}
for line in sys.stdin:
    v = line.strip().split('\t', 1)
    if len(v) > 1:
        cluster_id = int(v[0])
        emit_value = map(float, v[1].split(','))
        point = emit_value
        if current_clusid == cluster_id:
            cluster[current_clusid].append(point)
        else:
            current_clusid = cluster_id
            cluster.setdefault(current_clusid, [])
            cluster[current_clusid].append(point)
for key, point in cluster.iteritems():
    sum_points = map(sum, izip(*point))
    avg_points = [p/float(len(point)) for p in sum_points]
    print ('\t'.join('%.4f'%p for p in avg_points))
