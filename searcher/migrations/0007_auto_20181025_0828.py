# -*- coding: utf-8 -*-
# Generated by Django 1.9.8 on 2018-10-25 08:28
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('searcher', '0006_auto_20181025_0825'),
    ]

    operations = [
        migrations.AlterField(
            model_name='queryhistory',
            name='img',
            field=models.ImageField(upload_to='media/Queryimg', verbose_name='\u56fe\u7247'),
        ),
    ]
