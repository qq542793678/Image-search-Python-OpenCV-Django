#!/usr/bin/env python
# coding:utf-8

import json

import glob
import platform_sys
import sqlite3
import numpy as np


class Sift_HsHisto_sqlite3_adapter:
    def __init__(self, url=""):
        if url == "":
            url = platform_sys.root + '/' + platform_sys.sift_hshisto_sqlite3

        self.pri_mat_url = platform_sys.root + '/' + platform_sys.sift_NOSQL_pool
        self.db_url=url
        self.__sqlite3_init__()
        #pass

    def __sqlite3_init__(self):
        self.db_conn=sqlite3.connect(self.db_url)
        self.cursor=self.db_conn.cursor()
        pass

    def __look__(self,pool,keystr):
        for itm in pool:
            if itm['image'].find(keystr)>=0:
                return itm
        return None

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
        sql_str='select * from descriptors'
        if platform_sys.__FULL_FEATURE__==True:
            sql_str='select descriptors.image_id as id,' \
                'descriptors.rows as desc_rows,' \
                'descriptors.cols as desc_cols,' \
                'descriptors.data as descriptors,' \
                ' keypoints.rows as keypts_rows,' \
                'keypoints.cols as keypts_cols,' \
                'keypoints.data as keypoints '\
                ' from descriptors,keypoints ' \
                'where descriptors.image_id=keypoints.image_id'
        row_records=self.cursor.execute(sql_str)
        json_rec_lst=[]
        for rec in row_records:
            elems_json={}
            str='image%02d.jpg' % rec[0]

            elems_json.__setitem__('id',rec[0])
            elems_json.__setitem__('image_name',str)
            elems_json.__setitem__('desc_rows',rec[1])
            elems_json.__setitem__('desc_cols',rec[2])
            temp_np=np.array(rec[3],dtype='uint8')
            temp_np=temp_np.reshape(rec[1],rec[2])
            temp_np=temp_np.astype(np.float32)
            elems_json.__setitem__('desc',temp_np)
            if platform_sys.__FULL_FEATURE__==True:
                elems_json.__setitem__('kpts_rows',rec[4])
                elems_json.__setitem__('kpts_cols',rec[5])
                #temp_np=np.array(rec[6],dtype='float32')
                #temp_np=temp_np.reshape(rec[4],rec[5])
                elems_json.__setitem__('keypoints',rec[6])

            itm=self.__look__(json_lst,str)
            if itm is None:
                continue
            elems_json.__setitem__('image',itm['image'])
            elems_json.__setitem__('urladdress',itm['urladdress'])

            json_rec_lst.append(elems_json)



        return json_rec_lst
