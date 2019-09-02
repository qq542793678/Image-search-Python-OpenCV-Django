# coding:utf-8

import xadmin

from xadmin import views

from .models import PhotoGallery, QueryHistory
from django.utils.safestring import mark_safe


class BaseSetting(object):
    #设置主题功能
    enable_themes = True
    use_bootswatch = True


class GlobalSettings(object):
    site_title = "瓷soso后台管理系统"
    site_footer = "瓷soso在线"
    # menu_style = "accordion"


class PhotoGalleryAdmin(object):
    list_display = ['img', 'url', 'desc', 'add_time']
    search_fields =['img', 'url']
    list_filter = ['img', 'url', 'desc', 'add_time']


class QueryHistoryAdmin(object):
    list_display = ['img', 'ip_addr', 'add_time']
    search_fields = ['img', 'ip_addr']
    list_filter = ['img', 'ip_addr', 'add_time']


xadmin.site.register(PhotoGallery, PhotoGalleryAdmin)
xadmin.site.register(views.BaseAdminView, BaseSetting)
xadmin.site.register(views.CommAdminView, GlobalSettings)
xadmin.site.register(QueryHistory, QueryHistoryAdmin)