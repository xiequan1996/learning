'''
Descripttion: 
version: 
Author: xiequan
Date: 2021-04-24 11:35:32
LastEditors: Please set LastEditors
LastEditTime: 2021-07-18 16:37:20
'''
import sys
import os
import time
from selenium import webdriver
from tqdm import tqdm

# print(sys.path)
# driver=webdriver.Edge('C:\Program Files (x86)\Microsoft\Edge\Application\msedgedriver.exe')
# driver.get("https://www.ptpress.com.cn/")

with tqdm(total=200) as pbar:
  pbar.set_description("Processing")
  for i in range(10):
    time.sleep(1)
    pbar.update(i*10)
