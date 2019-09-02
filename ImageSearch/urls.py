"""ImageSearch URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
# coding:utf-8
from django.conf.urls import url
from django.contrib import admin
from searcher.views import getquery, getappquery,getrsb
from django.conf.urls.static import static
from django.conf import settings

import xadmin

urlpatterns = [
    url(r'^xadmin/', xadmin.site.urls),
    url(r'query', getquery, name='getquery'),
    url(r'^appquery$', getappquery,),
    url(r'^rsb/$', getrsb, name='getrsb'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
