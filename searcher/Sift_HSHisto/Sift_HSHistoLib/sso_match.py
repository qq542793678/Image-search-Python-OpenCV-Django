#!/usr/bin/env python
#coding:utf-8


import numpy as np
import cv2
import platform_sys
from common import anorm, getsize
import os as os
import urllib
import multi_hists as mhist
from searcher.model import sift_hshisto_pools
import platform_sys
import csv
import pandas as pd
import codecs
from PIL import Image



FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

CURRENT_MODE_INDEX=0#'HS-Histo'
MODE_NAME=['HS-Histo','Hue-Histo','Hsv-Histo']

DEBUG_MODE=False

mode_dir=['HSHisto/','HueHisto/','HsvHisto/']
samples_parent=platform_sys.root_dir+'/static/NoSQL/CLSMats/' #'/media/D/mat_pool/search_obj/'
source_sample_dir=samples_parent+'src/'#'/media/D/mat_pool/'
search_samples=samples_parent+'search_objsamples.png'
out_sift_directory='match_sift/'
out_joint_directory='match_Sift_Chi/'
#source_list=os.listdir(samples_parent)#+'src/')
res_postfix=''

math_res_list=[]


class sift_hsHisto_gear:
    def __init__(self,name=None):
        if name is  None:
            pass
        self.init_feature(name)
        pass
    def init_feature(self,name):
        self.chunks = name.split('-')
        if self.chunks[0] == 'sift':
            self.detector = cv2.SIFT()
            self.res_postfix='-sift'
            self.norm = cv2.NORM_L2
        elif self.chunks[0] == 'surf':
            self.detector = cv2.SURF(800)
            self.res_postfix='-surf'
            self.norm = cv2.NORM_L2
        elif self.chunks[0] == 'orb':
            self.detector = cv2.ORB(400)
            self.res_postfix='-orb'
            self.norm = cv2.NORM_HAMMING
        else:
            return None, None
        if 'flann' in self.chunks:
            if self.norm == cv2.NORM_L2:
                self.flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            else:
                self.flann_params= dict(algorithm = FLANN_INDEX_LSH,
                                   table_number = 6, # 12
                                   key_size = 12,     # 20
                                   multi_probe_level = 1) #2
            self.matcher = cv2.FlannBasedMatcher(self.flann_params, {})  # bug : need to pass empty dict (#1329)
        else:
            self.matcher = cv2.BFMatcher(self.norm)
        return self.detector, self.matcher

    def __filter_matches__(self,kp1, kp2, matches, ratio = 0.75):
        mkp1, mkp2 = [], []

        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append(kp1[m.queryIdx])

        p1 = np.float32([kp.pt for kp in mkp1])

        kp_pairs = zip(mkp1, mkp1)
        return p1, None, kp_pairs

    def morphBoundRect(self,rect,point=[0,0]):
        if rect is None:rect=np.int32([point,[1,1]])
        if point[0] < rect[0][0]: rect[0][0] = point[0]
        if point[0] > (rect[0][0] + rect[1][0]):
            rect[1][0] = point[0]- rect[0][0]
        if point[1] < rect[0][1]: rect[0][1] = point[1]
        if point[1] > (rect[0][1] + rect[1][1]):
            rect[1][1] = point[1] - rect[0][1]
        return rect

    def getMatchBoundary(self,kp_pairs,status=None,H=None):

        if status is None:
            status = np.ones(len(kp_pairs), np.bool_)
        p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
        p2 = np.int32([kpp[1].pt for kpp in kp_pairs])

        src_bound= None#np.float32([, [1, 1]])
        dst_bound=None#np.float32([p2, [1, 1]])

        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                src_bound=self.morphBoundRect(src_bound,[x1,y1])
                dst_bound=self.morphBoundRect(dst_bound,[x2,y2])

        return (src_bound,dst_bound)

    def compressImage(self,image_file):
        image = Image.open(image_file)  # 通过cp.picture 获得图像
        width = image.width
        height = image.height
        rate = 1.0  # 压缩率

        # 根据图像大小设置压缩率
        if width >= 3000 or height >= 3000:
            rate = 0.2
        elif width >= 2000 or height >= 2000:
            rate = 0.3
        elif width >= 1000 or height >= 1000:
            rate = 0.5
        elif width >= 500 or height >= 500:
            rate = 0.9

        width = int(width * rate)  # 新的宽
        height = int(height * rate)  # 新的高

        image.thumbnail((width, height), Image.ANTIALIAS)  # 生成缩略图
        image.save(image_file, 'JPEG')  # 保存到原路径

    def getImageFeature(self,image_file):
        self.compressImage(image_file)
        img = cv2.imread(image_file)#, 0)

        if self.detector != None:
            pass#print 'using', feature_name
        else:
            #print 'unknown feature:', feature_name
            return None
        kp1, desc1 = self.detector.detectAndCompute(img, None)
        return (img,kp1,desc1)

    def getImageObjFeature(self, image=None):
        if image is None:
            return None
        img = image

        if self.detector != None:
            pass  # print 'using', feature_name
        else:
            # print 'unknown feature:', feature_name
            return None
        kp1, desc1 = self.detector.detectAndCompute(img, None)
        return (img, kp1, desc1)

    def match_and_bound(self,kp1,desc1,kp2,desc2):
        #print 'matching...'
        raw_matches = self.matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
        p1, p2, kp_pairs = self.__filter_matches__(kp1, kp2, raw_matches)
        if len(p1) >= 100:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            #print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        else:
            H, status = None, None
            #print '%d matches found, not enough for homography estimation' % len(p1)

        srcBound, dstBound = self.getMatchBoundary(kp_pairs, status, H)
        #print 'srcBound=', srcBound, '\ndstBound=', dstBound

        if (status is None):
            match_ratio = 0
        else:
            match_ratio = np.sum(status) / len(status)

        return ( match_ratio,srcBound,dstBound)

    def write_histogram(self,hist,fn):
        hist_im = np.zeros((256, 240))
        # hist_im[:240,:256]=hist[:,:]#[:,:]
        for i in range(0, 31):
            for j in range(0, 29):
                for k in range(0, 7):
                    for l in range(0, 7):
                        hist_im[i * 8 + k, j * 8 + l] = hist[j, i]

        hist_im = hist_im / hist_im.max()

        hist_im = np.log10(1+hist_im)

        hist_im = hist_im/hist_im.max()*255

        hist_im = hist_im.astype(np.uint8)
        hist_im=cv2.equalizeHist(hist_im)
        #hist_im = 255 - hist_im

        cv2.imwrite(fn, hist_im)
        return None

    def calc_area_hist(self,im,rectBound=[[0,0],[1,1]],pfn=None):
        goal_area=None
        rect_str=rectBound.__str__()
        if im.shape.__len__()>2:
            goal_area= np.zeros((rectBound[1][1],rectBound[1][0],im.shape[2]))
        else:
            goal_area = np.zeros((rectBound[1][1], rectBound[1][0]))

        if goal_area is None:return None
        goal_area=im[rectBound[0][1]:rectBound[0][1]+rectBound[1][1],
                      rectBound[0][0]:rectBound[0][0]+rectBound[1][0]
                  ]

        if DEBUG_MODE:
            cv2.imwrite( samples_parent + \
                         mode_dir[CURRENT_MODE_INDEX] + \
                         'HSHisto/'+pfn+rect_str+'.png',goal_area)

        if CURRENT_MODE_INDEX == 0:# 'HS-Histo'
            hist=mhist.calcHSHist(goal_area,64)#goal_area,64)
            if pfn is  None:
                pfn = 'samples'

            if DEBUG_MODE:
                hist_im_fn = samples_parent + \
                             mode_dir[CURRENT_MODE_INDEX] + \
                             'HSHisto/' + pfn +rect_str+ '-hshisto' + '.png'
                self.write_histogram(hist,hist_im_fn)
            return hist
        elif CURRENT_MODE_INDEX == 1:#'Hue-Histo'
            return mhist.calcHueHist(goal_area,64)
        else :
            return mhist.calcHist(goal_area)

    def explore_match(self,win, img1, img2, kp_pairs, status=None, H=None):
        h1, w1, d1 = img1.shape[:3]
        h2, w2, d2 = img2.shape[:3]
        vis = np.zeros((max(h1, h2), w1 + w2, d1 if d1 > d2 else d2), np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1 + w2] = img2
        # vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        if H is not None:
            corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
            cv2.polylines(vis, [corners], True, (255, 255, 255))

        if status is None:
            status = np.ones(len(kp_pairs), np.bool_)
        p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
        p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

        green = (0, 255, 0)
        red = (0, 0, 255)
        white = (255, 255, 255)
        kp_color = (51, 103, 236)
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                col = green
                cv2.circle(vis, (x1, y1), 2, col, -1)
                cv2.circle(vis, (x2, y2), 2, col, -1)
            else:
                col = red
                r = 2
                thickness = 3
                cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
                cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
                cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
                cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
        vis0 = vis.copy()
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                cv2.line(vis, (x1, y1), (x2, y2), green)

        return vis

    def explore_feature(self,win, img, kp, H=None):
        h1, w1, dept = img.shape[:3]

        vis = np.zeros((h1, w1, dept), np.uint8)
        vis = img.copy()


        if H is not None:
            corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
            cv2.polylines(vis, [corners], True, (255, 255, 255))

        p1 = np.int32([kpp.pt for kpp in kp])

        green = (0, 255, 0)
        red = (0, 0, 255)
        white = (255, 255, 255)
        kp_color = (51, 103, 236)
        for (x1, y1), (x2, y2) in zip(p1, p1):
            col = red
            cv2.circle(vis, (x1, y1), 3, col, 0)

        vis0 = vis.copy()

        return vis

    def match_and_draw(self,win, fn2, img1, img2,kp1,desc1,kp2,desc2):
        #print 'matching...'
        raw_matches = self.matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
        p1, p2, kp_pairs =self.__filter_matches__(kp1, kp2, raw_matches)
        '''
        if len(p1) >= 100:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            #print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        else:
            H, status = None, None
            #print '%d matches found, not enough for homography estimation' % len(p1)

        vis=None
        if DEBUG_MODE:
            vis = self.explore_match(win, img1, img2, kp_pairs, status, H)

        srcBound,dstBound=self.getMatchBoundary(kp_pairs,status,H)
        #print 'srcBound=',srcBound,'\ndstBound=',dstBound

        if (status is None):
            match_ratio = 0
        else:
            match_ratio = np.sum(status) / len(status)
        return (fn2, vis, match_ratio)
        '''
        p1_len=p1.__len__()
        match_ratio=float(p1_len)/kp1.__len__()

        return (fn2, None, match_ratio)


def __init_DESC__( root_dir):
    results = []

    with open(root_dir+'index.csv','rU') as f:
        reader = csv.reader(f)

        for row in reader:
            elem_json = {}
            elem_json.__setitem__('image',row[0])
            elem_json.__setitem__('urladdress',row[1])
            elem_json.__setitem__('desc',row[2])
            results.append(elem_json)
        f.close()

    return results
def shs_search(qimg,image_url,mats_hub,root_dir):
    tmp_chunks = image_url.split('.')
    tmp_chunks[0]=tmp_chunks[0].replace('\\','/')
    tmp_chunks=tmp_chunks[0].split('/')
    pure_fn1=tmp_chunks[len(tmp_chunks)-1]

    mats_list=None
    matDESC=None
    zip_mat_list=None
    root_dir=os.getcwd().replace('\\','/')
    #mats_list = sift_hshisto_pools("db.sqlite3")
    if mats_list is None:
        mats_list=os.listdir(root_dir+'/static/NoSQL/CLSMats/')
        matDESC=__init_DESC__(root_dir+'/static/NoSQL/')
        zip_mat_list=zip(mats_list,matDESC)

    feature_name = 'sift-flann'
    shs_gear_obj=sift_hsHisto_gear(feature_name)
    detector, matcher = shs_gear_obj.init_feature(feature_name)
    (img1,kp1,desc1) = shs_gear_obj.getImageObjFeature(qimg)
    if DEBUG_MODE:
        struc_vis=shs_gear_obj.explore_feature('find_obj',img1,kp1)

        cv2.imwrite('/meida/D/strucvis.jpg',struc_vis)

    if img1.shape.__len__()>2:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


    if DEBUG_MODE:
        print 'Match Mode:',MODE_NAME[CURRENT_MODE_INDEX], \
            'match result is:'
    math_res_list=[]
    mat_path = root_dir
    for (image_dict,imDESC_json) in zip_mat_list:#mats_list:
        image_dict=imDESC_json['image']
        image_path = image_dict#['image']

        fname = image_path.split('/')[-1]
        fn2 = mat_path + '/static/NoSQL/CLSMats/' + fname
	#print 'fn2=' , fn2
        #if not os.path.exists(fn2):
        #    urllib.urlretrieve(image_path, fn2)

        tmp_chunks=fn2.split('.')
        pure_fn2=tmp_chunks[0]
        #fn2=source_sample_dir+fn2
        if (fn2.find(".jpg") > 0):

            (img2,kp2,desc2) = shs_gear_obj.getImageFeature(fn2)
            if DEBUG_MODE:
                struc_vis2=shs_gear_obj.explore_feature('find_obj',img2,kp2)
                cv2.imwrite(samples_parent+
                        mode_dir[CURRENT_MODE_INDEX]+
                        'src_mark/'+pure_fn2+'.png',struc_vis2)


            if img2.shape.__len__() > 2:
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            (pn,vis,match_ratio)=shs_gear_obj.match_and_draw('find_obj',pure_fn2,
                                                img1,img2,
                                                kp1,desc1,
                                                kp2,desc2
                                                )
            if match_ratio > 0.1 and DEBUG_MODE:
                dst_file = samples_parent + \
                           mode_dir[CURRENT_MODE_INDEX]+\
                           out_sift_directory + pn + '-' +\
                           feature_name + '.png'
                #print 'writting to ', dst_file
                cv2.imwrite(dst_file, vis)

            chisq_dist=float("INF")
            if(match_ratio>0.1):
                chisq_dist=0
                '''
                (match_ratio,srcBound,dstBound)=\
                    shs_gear_obj.match_and_bound(kp1,desc1,kp2,desc2)

                color_channel_hist1=shs_gear_obj.calc_area_hist(img1,
                                                                srcBound,
                                                                'template/'+pure_fn1)
                color_channel_hist2 = shs_gear_obj.calc_area_hist(img2,
                                                                  dstBound,'src/'+pure_fn2)

                chisq_dist=mhist.calcHSHistChiDist(color_channel_hist1,
                                                       color_channel_hist2)
                '''
                if chisq_dist<1 :
                    if DEBUG_MODE:
                        dst_file=samples_parent+ \
                             mode_dir[CURRENT_MODE_INDEX]+\
                             out_joint_directory+pn+'-'+feature_name+'-hs.png'
                        cv2.imwrite(dst_file,vis)
                    math_res_list.append({'result_id': '../static/NoSQL/CLSMats/'+image_dict,#['image'],
                                          'result_url':imDESC_json['urladdress'],
                                          'result_desc':imDESC_json['desc'],
                                          #'./static/NoSQL/CLSMats/'+image_dict,#image_dict['urladdress'],
                                          'match_ratio':match_ratio,
                                          'hshisto_chidist':chisq_dist})


    return math_res_list

def shs_search_sqlite3(qimg,image_url,mats_hub,root_dir):
    tmp_chunks = image_url.split('.')
    tmp_chunks[0]=tmp_chunks[0].replace('\\','/')
    tmp_chunks=tmp_chunks[0].split('/')
    pure_fn1=tmp_chunks[len(tmp_chunks)-1]

    mats_list = sift_hshisto_pools("db.sqlite3")
    #if mats_list is None:
    #    mats_list=source_list
    feature_name = 'sift-flann'
    shs_gear_obj=sift_hsHisto_gear(feature_name)
    detector, matcher = shs_gear_obj.init_feature(feature_name)
    (img1,kp1,desc1) = shs_gear_obj.getImageObjFeature(qimg)
    if DEBUG_MODE:
        struc_vis=shs_gear_obj.explore_feature('find_obj',img1,kp1)

        cv2.imwrite('/meida/D/strucvis.jpg',struc_vis)

    if img1.shape.__len__()>2:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


    if DEBUG_MODE:
        print 'Match Mode:',MODE_NAME[CURRENT_MODE_INDEX], \
            'match result is:'
    math_res_list=[]
    mat_path = root_dir
    for image_dict in mats_list:
        image_path = image_dict['image']
        fname = image_path.split('/')[-1]
        fn2 = mat_path + '/' + fname

        tmp_chunks=fn2.split('.')
        pure_fn2=tmp_chunks[0]
        #fn2=source_sample_dir+fn2
        if (fn2.find(".jpg") > 0):

            #(img2,kp2,desc2) = shs_gear_obj.getImageFeature(fn2)
            kp2=kp1
            desc2=image_dict['desc']
            kp2=(image_dict['keypoints'] if platform_sys.__FULL_FEATURE__==True else None)
            img2=cv2.imread(fn2)

            #if img2.shape.__len__() > 2:
            #    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            (pn,vis,match_ratio)=shs_gear_obj.match_and_draw('find_obj',pure_fn2,
                                                img1,img2,
                                                kp1,desc1,
                                                kp2,desc2
                                                )
            if match_ratio > 0.1 and DEBUG_MODE:
                dst_file = samples_parent + \
                           mode_dir[CURRENT_MODE_INDEX]+\
                           out_sift_directory + pn + '-' +\
                           feature_name + '.png'
                #print 'writting to ', dst_file
                cv2.imwrite(dst_file, vis)

            chisq_dist=float("INF")
            if(match_ratio>0.1):
                chisq_dist=0
                '''
                (match_ratio,srcBound,dstBound)=\
                    shs_gear_obj.match_and_bound(kp1,desc1,kp2,desc2)

                color_channel_hist1=shs_gear_obj.calc_area_hist(img1,
                                                                srcBound,
                                                                'template/'+pure_fn1)
                color_channel_hist2 = shs_gear_obj.calc_area_hist(img2,
                                                                  dstBound,'src/'+pure_fn2)

                chisq_dist=mhist.calcHSHistChiDist(color_channel_hist1,
                                                       color_channel_hist2)
                '''
                if chisq_dist<1 :
                    if DEBUG_MODE:
                        dst_file=samples_parent+ \
                             mode_dir[CURRENT_MODE_INDEX]+\
                             out_joint_directory+pn+'-'+feature_name+'-hs.png'
                        cv2.imwrite(dst_file,vis)
                    math_res_list.append({'result_id': image_dict['image'],
                                          'result_url':image_dict['urladdress'],
                                          'match_ratio':match_ratio,
                                          'hshisto_chidist':chisq_dist})


    return math_res_list
