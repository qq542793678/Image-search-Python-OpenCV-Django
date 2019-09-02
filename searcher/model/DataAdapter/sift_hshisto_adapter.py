#!/usr/bin/env python
# coding:utf-8

import json

import glob
import platform_sys


#http://m.ccmall.cn/api/product_img.aspx
##http://m.ccmall.cn/api/product_img.aspx?num=4

class Sift_HsHisto_adapter:
    def __init__(self, url=""):
        if url == "":
            url = platform_sys.root + '/' + platform_sys.sift_hshisto_pool

        self.pri_mat_url = url
        #pass

    def pri_mat_extract(self, url_root=""):
        if url_root == "":
            url_root = self.pri_mat_url
        json_lst =[]
        for pri_image in glob.glob(url_root + "/*.*"):
            pri_image=pri_image.replace('\\','/')
            elem_json={}
            elem_json.__setitem__("image",pri_image)
            elem_json.__setitem__("urladdress",pri_image)
            json_lst.append(elem_json)

        return json_lst
