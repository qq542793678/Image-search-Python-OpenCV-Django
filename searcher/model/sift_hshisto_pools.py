#!/usr/bin/env python
# coding:utf-8

import platform_sys
from DataAdapter.sift_hshisto_adapter import Sift_HsHisto_adapter
from DataAdapter.sift_hshisto_sqlite3_adapter import Sift_HsHisto_sqlite3_adapter


class sift_hshisto_pools(list):
    def __init__(self, url):
        if url.find('db')<0:
            url=platform_sys.root+'/'+platform_sys.sift_hshisto_pool
            mall_adapter_obj = Sift_HsHisto_adapter(url)
        else :
            url = platform_sys.root + '/' + platform_sys.sift_hshisto_sqlite3
            mall_adapter_obj = Sift_HsHisto_sqlite3_adapter(url)

        super(sift_hshisto_pools,self).__init__()
        self.extend( mall_adapter_obj.pri_mat_extract(''))
        pass

