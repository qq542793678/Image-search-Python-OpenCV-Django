# coding:utf-8
from django.shortcuts import render
from django.http import HttpResponse
from .models import QueryHistory, PhotoGallery

import os
import cv2
import platform_sys
import time, datetime
import zlib
from PIL import Image
import json
import csv

import searcher.Sift_HSHisto.search


# Create your views here.


def filename(post_data):
    fname = 'static/NoSQL/%s' % (post_data)
    return fname


def timestamp(last_data):
    time_c = str(last_data.split(',')[2])
    time_z = str(time_c.split(";")[0])
    timeArray = time.strptime(time_z, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


def get_data(D_data):
    homedir = os.getcwd()
    filedir = '%s/static/NoSQL/' % homedir
    with open(filedir + 'index.csv', 'wb+') as f:
        pass
    for obj in D_data:
        D_img = str(obj.img)
        s_img = D_img.split('/')[1]
        D_url = obj.url
        D_des = obj.desc
        homedir = os.getcwd()
        filedir = '%s/static/NoSQL/' % homedir
        with open(filedir + 'index.csv', 'ab+') as f:
            data_str = '%s,%s,%s\n' % (s_img, D_url, D_des)
            f.write(data_str)
            f.close()


def compressImage(img_url):
    image = Image.open(os.getcwd() + '/' + img_url)  # 通过cp.picture 获得图像
    width = image.width
    height = image.height
    rate = 1.0  # 压缩率

    # 根据图像大小设置压缩率
    if width >= 3000 or height >= 3000:
        rate = 0.1
    elif width >= 2000 or height >= 2000:
        rate = 0.2
    elif width >= 1000 or height >= 1000:
        rate = 0.4
    elif width >= 500 or height >= 500:
        rate = 0.8

    width = int(width * rate)  # 新的宽
    height = int(height * rate)  # 新的高

    image.thumbnail((width, height), Image.ANTIALIAS)  # 生成缩略图
    image.save(os.getcwd() + '/' + img_url, 'JPEG')  # 保存到原路径


def top_search(search_results):
    if search_results:  # 判断search_results是否为空
        top = search_results[0]  # 读取数据
        t_img = top['result_id']
        t_url = top['result_url']
        t_desc = top['result_desc']
        homedir = os.getcwd()
        filedir = '%s/static/NoSQL/' % homedir
        with open(filedir + 'top_search.csv', 'ab+') as f:  # 每次查询的数据存到这个文件中
            data_str = '%s,%s,%s\n' % (t_img, t_url, t_desc)
            f.write(data_str)
            f.close()
        with open(filedir + 'top_search.csv', 'r') as f:  # 打开读取这个文件内容
            csvreader = csv.reader(f)
            final_list = list(csvreader)
            with open(filedir + 'center.csv', 'wb+') as f2:  # 初始化center文件
                f2.close()
            for item in final_list:
                cishu, shuju_id, shuju_url, shuju_desc = final_list.count(item), item[0], item[1], item[2]
                with open(filedir + 'center.csv', 'ab+') as f1:  # 统计每个数据出现的次数并加到每行第一位
                    data_str = '%s,%s,%s,%s\n' % (cishu, shuju_id, shuju_url, shuju_desc)
                    f1.write(data_str)
                    f1.close()
            f.close()
        with open(filedir + 'center.csv', 'r') as f3:
            csvreader_f3 = csv.reader(f3)
            shuju_list = list(csvreader_f3)
            list_new = []
            i = 0
            with open(filedir + 'top.csv', 'wb+') as f6:  # 初始化top文件
                f6.close()
            for x in shuju_list:  # 删除center文件中重复的数据并存到top文件中
                if x not in list_new:
                    i = i + 1
                    list_new.append(x)
                    new_list = list_new[i - 1]
                    new_cishu, new_id, new_url, new_desc = int(new_list[0]), new_list[1], new_list[2], new_list[3]
                    with open(filedir + 'top.csv', 'ab+') as f4:
                        data_str1 = '%d,%s,%s,%s\n' % (new_cishu, new_id, new_url, new_desc)
                        f4.write(data_str1)
                        f4.close()
            f3.close()
        with open(filedir + 'top.csv', 'ab+') as f5:  # 对top这个文件进行排序（降序）
            csvreader_f5 = csv.reader(f5)
            top_list = list(csvreader_f5)
            top_list.sort(reverse=True, key=lambda top: int(top[0]))
            add_list = list(top_list)
            with open(filedir + 'end_top.csv', 'wb+') as f9:  # 初始化end_top这个文件
                f9.close()
            for end_list in add_list:
                top_list1 = end_list
                end_cishu, end_id, end_url, end_desc = top_list1[0], top_list1[1], top_list1[2], top_list1[3]
                with open(filedir + 'end_top.csv', 'ab+') as f7:  # 将top里面的数据按排序的顺序存入end_top文件
                    data_str5 = '%s,%s,%s,%s\n' % (end_cishu, end_id, end_url, end_desc)
                    f7.write(data_str5)
                    f7.close()
            f5.close()


def getquery(request):
    if request.method == "POST":
        D_data = PhotoGallery.objects.all()
        get_data(D_data)
        new_img = QueryHistory(
            img=request.FILES.get('q_img'),  # 将查询的图片保存到数据
        )
        new_img.save()

        # post_data = request.FILES.get('q_img')
        data_list = QueryHistory.objects.last()  # 读取数据库最后一条数据（刚刚传入需要查询的图片）

        post_data = data_list.img  # 读取数据中的图片

        img_url = filename(post_data)  # 取到图片存储路径

        compressImage(img_url)  # 压缩图片

        cv_img = cv2.imread(os.getcwd() + '/' + img_url)

        module_hub = './' + platform_sys.sift_NOSQL_pool
        modmat_dir = os.getcwd().replace("\\", '/') + '/' + platform_sys.sift_NOSQL_pool
        search_results = searcher.Sift_HSHisto.search.search(
            cv_img,
            img_url,
            module_hub,
            modmat_dir
        )
        top_search(search_results)
        if search_results:
            return render(request, 'show_img.html', {'search_results': search_results})
        else:
            text = "匹配不到相关图片，小so会努力更新噢。请耐心等待。"
            return render(request, 'show_img.html', {'t': text})
    return render(request, "queryImg.html")


def __succeed_proc__(results):
    result_json = []
    for elm in results:
        elem_dict = {}
        elem_dict.__setitem__("match_val", elm['match_ratio'])
        elem_dict.__setitem__("image", elm['result_id'])
        elem_dict.__setitem__("urladdress", elm['result_url'])
        elem_dict.__setitem__('desc', elm['result_desc'])
        result_json.append(elem_dict)
    #######################
    return result_json


def getappquery(request):
    if request.method == "POST":
        D_data = PhotoGallery.objects.all()
        get_data(D_data)
        new_img = QueryHistory(
            img=request.FILES.get('q_img'),  # 将查询的图片保存到数据
        )
        new_img.save()

        # post_data = request.FILES.get('q_img')
        data_list = QueryHistory.objects.last()  # 读取数据库最后一条数据（刚刚传入需要查询的图片）

        post_data = data_list.img  # 读取数据中的图片

        img_url = filename(post_data)  # 取到图片存储路径

        compressImage(img_url)  # 压缩图片

        cv_img = cv2.imread(os.getcwd() + '/' + img_url)

        module_hub = './' + platform_sys.sift_NOSQL_pool
        modmat_dir = os.getcwd().replace("\\", '/') + '/' + platform_sys.sift_NOSQL_pool
        search_results = searcher.Sift_HSHisto.search.search(
            cv_img,
            img_url,
            module_hub,
            modmat_dir
        )
        top_search(search_results)
        search_results_succ = __succeed_proc__(search_results)
        search_results_json = json.dumps(search_results_succ, ensure_ascii=False)
    # return search_results_json
    return HttpResponse(search_results_json, content_type="application/json")


def rsb_l(rsb_list):  # 将rsb_list转为list{{},{}]形式
    end_list = []
    i = 0
    for rsb in rsb_list:
        i = i + 1
        elem_dict = {}
        elem_dict.__setitem__("number", rsb[0])
        elem_dict.__setitem__("image", rsb[1])
        elem_dict.__setitem__("urladdress", rsb[2])
        # elem_dict.__setitem__("desc", rsb[3])
        end_list.append(elem_dict)
        if i > 4:
            break
    return end_list


def getrsb(request):
    homedir = os.getcwd()
    filedir = '%s/static/NoSQL/' % homedir
    with open(filedir + 'end_top.csv', 'r') as f8:
        csvreader_f8 = csv.reader(f8)
        rsb_list = list(csvreader_f8)
        end_list = rsb_l(rsb_list)
    return render(request, 'most_searched_hashtags.html', {'rsb': end_list})
