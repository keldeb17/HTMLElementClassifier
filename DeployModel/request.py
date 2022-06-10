# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 01:55:23 2022

@author: kelse
"""

import requests as r

# add Search Field
search = ['<input class="quer" type="text" value="" style="border: solid #b9b9b9 2px ; height: 3em; width: 78%; border-radius:10px; padding-left: 8%; font-size: 1.5em; font-weight: 600; outline: none; margin-top: 2%" placeholder="Search">']
 
keys = {"search": search}

prediction = r.get("http://127.0.0.1:8000/predict-element/", params=keys)

results = prediction.json()

print(results["prediction"])