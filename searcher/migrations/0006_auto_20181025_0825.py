# -*- coding: utf-8 -*-
# Generated by Django 1.9.8 on 2018-10-25 08:25
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('searcher', '0005_queryhistory_ip'),
    ]

    operations = [
        migrations.RenameField(
            model_name='queryhistory',
            old_name='ip',
            new_name='ip_addr',
        ),
    ]
