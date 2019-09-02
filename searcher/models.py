# coding:utf-8
from __future__ import unicode_literals
from datetime import datetime

from django.db import models
from django.utils.translation import ugettext_lazy as _
from ImageSearch.settings import MEDIA_URL

from system.storage import ImageStorage

# Create your models here.


class PhotoGallery(models.Model):
    img = models.ImageField(upload_to="CLSMats", storage=ImageStorage(), verbose_name=u"图片")
    url = models.URLField(max_length=500, verbose_name=u"商品链接")
    desc = models.TextField(verbose_name=u"图片描述")
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u"添加时间")

    class Meta:
        verbose_name = u"图片库"
        verbose_name_plural = verbose_name


class QueryHistory(models.Model):
    img = models.ImageField(upload_to='upload', storage=ImageStorage(), verbose_name=u"图片")
    ip_addr = models.GenericIPAddressField(_('action ip'), blank=True, null=True)
    add_time = models.DateTimeField(default=datetime.now, verbose_name=u"查询时间")

    class Meta:
        verbose_name = u"查询历史"
        verbose_name_plural = verbose_name