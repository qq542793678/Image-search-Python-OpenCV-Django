#!/usr/bin/env python
#coding:utf-8


import cv2
import numpy as np


szBins=256
hist_sz=[30,32]#,64]
valRange=[0,180,0,256]#,[0,255]]
channels = [0, 1 ]
bins = np.arange(szBins).reshape(szBins,1)


def calcHist(im,szBins=256):
    channels=1
    im1=im
    if(im.shape.__len__()>2):
        (w,h,d)=im.shape
        im1=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    else :
        (w,h)=im.shape
        d=1

    hists={}
    hists.__setitem__('chs',d)
    hists.__setitem__('hist',{})
    indexName=['H','S','V']


    for i in range(0,d):
        hist_item=cv2.calcHist([im1],[i],None,[hist_sz[i]],valRange[i])
        #cv2.normalize(hist_item,hist_item,0,1,cv2.NORM_MINMAX)
        sumval=sum(hist_item)
        hist_item=hist_item/sum(hist_item)
        hists['hist'].__setitem__(indexName[i],hist_item)
    return hists

def calcHueHist(im,szBins=256):
    channels=1
    im1=im
    if(im.shape.__len__()>2):
        (w,h,d)=im.shape
        im1=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    else :
        (w,h)=im.shape
        d=1

    hist_item = cv2.calcHist([im1], [0], None, [30], [0,180])
    # cv2.normalize(hist_item,hist_item,0,1,cv2.NORM_MINMAX)
    sumval = sum(hist_item)
    hist_item = hist_item / sum(hist_item)
    return hist_item


def calcHSHist(im,szBins=256):
    #channels=1
    im1=im
    if(im.shape.__len__()>2):
        (w,h,d)=im.shape
        im1=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    else :
        (w,h)=im.shape
        d=1

    hists=cv2.calcHist([im1],channels,None,hist_sz,valRange)
    sumval = sum(sum(hists))
    #cv2.normalize(hists, hists, 0, 1, cv2.NORM_MINMAX)
    hists =hists/sumval

    #print 'h-s histogram sum:'
    #print sum(sum(hists))
    return np.array(hists)


def calcHSHistChiDist(histDict1,histDict2):
    chi1 = cv2.compareHist(histDict1, histDict2, cv2.cv.CV_COMP_CHISQR)

    '''
    #histDict1=np.where(histDict1>0,histDict1,1e-9)
    #histDict2 = np.where(histDict2 > 0, histDict2, 1e-9)

    dist_hist=(histDict1-histDict2)**2
    add_hist=(histDict1+histDict2)
    add_hist=np.where(add_hist==0,1e-9,add_hist)
    chiVal=dist_hist/add_hist
    chi1=np.sum(chiVal)#/2.0
    '''

    return chi1


def calcSubHistEntropy(histDict):

    ks=histDict['hist'].keys()
    subHlist=[]
    for j in range(0,len(ks)):
        subHDesc=[]
        for i in range(0,len(ks)):
            sub_hist=histDict['hist'][ks[i]]-histDict['hist'][ks[j]]
            abs_sub_hist=sum(sub_hist)
            subHDesc.append(abs_sub_hist[0])
        subHlist.append(subHDesc)

    #print subHlist
    #subHlist.append(subHDesc)

    return np.array(subHlist)

def calcHistDist(histDict1,histDict2):

    ks=histDict1['hist'].keys()
    subHlist=[]
    for j in range(0,len(ks)):
        if ks[j]=='V':
            continue
        sub_hist = histDict1['hist'][ks[j]] - histDict2['hist'][ks[j]]
        if subHlist.__len__()==0:
            subHlist.append(sub_hist)
        else:
            subHlist += abs(sub_hist)
        #subHlist.append(sub_hist)


    #print subHlist
    #subHlist.append(subHDesc)

    return np.array(subHlist)


def calcHistChiSqDist(histDict1,histDict2):

    ks=histDict1['hist'].keys()
    subHlist=[]
    for j in range(0,len(ks)):
        if ks[j]=='V': continue
        sub_hist = (histDict1['hist'][ks[j]] - histDict2['hist'][ks[j]])**2
        sub_hist1= (histDict1['hist'][ks[j]] + histDict2['hist'][ks[j]])
        div_sub_hist=sub_hist/2*sub_hist1;
        chiSquareDist=sum(sum(div_sub_hist))
        subHlist.append(chiSquareDist)

        #subHlist.append(sub_hist)


    #print subHlist
    #subHlist.append(subHDesc)

    return np.array(subHlist)


def hist_curve(im):
    h = np.zeros((300,szBins,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([im],[ch],None,[szBins],[0,256])
        #print hist_item
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        #print hist_item
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        cv2.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

def hist_lines(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print "hist_lines applicable only for grayscale images"
        #print "so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return y


if __name__ == '__main__':

    import sys

    if len(sys.argv)>1:
        im = cv2.imread(sys.argv[1])
    else :
        im = cv2.imread('./lena.jpg')
        print "usage : python hist.py <image_file>"


    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)


    print ''' Histogram plotting \n
    Keymap :\n
    a - show histogram for color image in curve mode \n
    b - show histogram in bin mode \n
    c - show equalized histogram (always in bin mode) \n
    d - show histogram for color image in curve mode \n
    e - show histogram for a normalized image in curve mode \n
    Esc - exit \n
    '''

    cv2.imshow('image',im)
    while True:
        k = cv2.waitKey(0)&0xFF
        if k == ord('a'):
            hists=calcHist(im,64)
            subHists=calcSubHistEntropy(hists)
            curve = hist_curve(im)
            cv2.imshow('histogram',curve)
            cv2.imshow('image',im)
            print 'a'
        elif k == ord('b'):
            print 'b'
            lines = hist_lines(im)
            cv2.imshow('histogram',lines)
            cv2.imshow('image',gray)
        elif k == ord('c'):
            print 'c'
            equ = cv2.equalizeHist(gray)
            lines = hist_lines(equ)
            cv2.imshow('histogram',lines)
            cv2.imshow('image',equ)
        elif k == ord('d'):
            print 'd'
            curve = hist_curve(gray)
            cv2.imshow('histogram',curve)
            cv2.imshow('image',gray)
        elif k == ord('e'):
            print 'e'
            norm = cv2.normalize(gray,alpha = 0,beta = 255,norm_type = cv2.NORM_MINMAX)
            lines = hist_lines(norm)
            cv2.imshow('histogram',lines)
            cv2.imshow('image',norm)
        elif k == 27:
            print 'ESC'
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
