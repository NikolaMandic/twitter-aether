#!/usr/bin/env python

import urllib2

import os
import urllib2
response = urllib2.urlopen('http://twitter-aether.herokuapp.com/tweet?p='+os.environ['tk'])
html = response.read()
print(html)
