#!/usr/bin/env python
# coding: utf-8


import argparse

import cv2

import platform_sys

from searcher.Sift_HSHisto.Sift_HSHistoLib import sso_match


def search( query_img,img_url, image_hub,root_dir):
    if platform_sys.sift_Mode.find('SQLITE3')<0:
        results=sso_match.shs_search(query_img,img_url,image_hub,root_dir)
    else:
        results=sso_match.shs_search_sqlite3(query_img,img_url,image_hub,root_dir)
    return results