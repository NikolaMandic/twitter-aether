#!/usr/bin/python
# -*- coding: utf8 -*-
from __future__ import unicode_literals


from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream, API
import os
import numpy as np
import moment
import pickle
import json
from gensim import corpora, models, similarities
from itertools import *
from pprint import pprint

from pattern.web    import Twitter
from pattern.en     import tag
from pattern.vector import KNN, count
import os.path
import os
from flask import Flask, render_template, request, redirect, url_for
import urllib
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
from owslib.wfs import WebFeatureService
from time import gmtime, strftime,time
import numpy
from os import listdir
from os.path import isfile, join
from collections import Counter
consumer_key=os.environ['CONSUMER_KEY']
consumer_secret=os.environ['CONSUMER_SECRET']
access_token=os.environ['TWITTER_ACCESS_TOKEN']
access_token_secret=os.environ['TWITTER_ACCESS_TOKEN_SECRET']
api_key=os.environ['WEATHER_API_KEY']


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = API(auth)
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'this_should_be_configured')

ws_labels = ["not_much_wind", "windy", "high wind"]
temperature_labels = ['<-10','-10-0','0-10','>10']


def get_soup(date=moment.now()):
  print date
  starttime=date.subtract(hours=3).strftime("%Y-%m-%dT%H:%M:%SZ")
  print starttime
  endtime=date.add(hours=3).strftime("%Y-%m-%dT%H:%M:%SZ")
  print endtime
  q='Temperature,TotalCloudCover,WindSpeedMS,Precipitation1h,Pressure,Humidity'
  from owslib.wfs import WebFeatureService
  wfs11 = WebFeatureService(url='http://data.fmi.fi/fmi-apikey/' + api_key + '/wfs',version='2.0.0')
  f=wfs11.getfeature(typename='BsWfs:BsWfsElement',storedQueryID='fmi::observations::weather::cities::simple',storedQueryParams={ 'place':'helsinki', 'parameters': q, 'starttime':starttime , 'endtime':endtime })
  rd=f.read()
  soup = BeautifulSoup(rd,'lxml')
  return soup

def get_params(soup,param="WS_10MIN",mean=False):
  v=[]
  for par in soup.find_all("bswfs:parametername", text=param):
    val=par.parent.find("bswfs:parametervalue").text
    if val!='NaN':
      v.append(float(val))
  if mean:
    return numpy.mean(v)
  else:
    return v

def rain_to_labels(rain):
    if rain!=0.0:
        return precipitation_labels[1]
    return precipitation_labels[0]

def press_to_labels(press):
    return (int(press/100) if not(math.isnan(press)) else 0)

def hum_to_labels(hum):
    return (int(hum/10) if not(math.isnan(hum)) else 0)

def ws_to_labels(ws):
    if ws>9:
        return ws_labels[2]
    if ws>4:
        return ws_labels[1]
    return ws_labels[0]

def temp_to_labels(temp):
    if temp<-10:
        return temperature_labels[0]
    if temp<0:
        return temperature_labels[1]
    if temp<10:
        return temperature_labels[2]
    return temperature_labels[0]

def get(date=moment.now()):
  q=u'#sää'
  r = api.search(q=q,
                 lang='fi',
                 geocode='60.1812755,24.9299129,30km',
                 count=100,
                 until=date.format("YYYY-M-D")
                 )
  def t(x): return x.user.name + ' ' + x.text
  return map(t,r)

def e():
  xx=[]
  for x in range(0, 6):
    xx.append(moment.now().subtract(days=x).replace(hours=6))
    xx.append(moment.now().subtract(days=x).replace(hours=13))
    xx.append(moment.now().subtract(days=x).replace(hours=20))
  return xx

def savetofile(fldr,d,w):
    f=open(fldr+d.format("YYYY-M-D-H"),'w')
    pickle.dump(w,f)

def extract_features(soup):
    tl=temp_to_labels(get_params(soup,"Temperature",mean=True))
    wl=ws_to_labels(get_params(soup,"WindSpeedMS",mean=True))
    rl=rain_to_labels(get_params(soup,"Precipitation1h",mean=True))
    pl=press_to_labels(get_params(soup,"Pressure",mean=True))
    hm=hum_to_labels(get_params(soup,"Humidity",mean=True))
    return [tl,wl,rl,pl,hm]

def save_weather(times):
  for x in times:
    r=get_soup(x)
    ww=extract_features(r)
    savetofile('weather/',x,ww)

def save_twitter(times):
  for x in times:
    r=get(x)
    savetofile('tweets/',x,r)

@app.route('/trainfd')
def trainfd():
    """Send your static text file."""
    tc=load("tk")
    wc=load("wk")
    pc=load("pk")
    prc=load("prk")
    hc=load("hk")
    mypath='weather/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for x in onlyfiles:
        [tl, wl, rl, pl, hm]=pickle.load(open(mypath+x,'rb'))
        data=pickle.load(open('tweets/'+x,'rb'))
        train(tc,data,tl)
        train(wc,data,wl)
        train(pc,data,rl)
        train(prc,data,pl)
        train(hc,data,hm)
    dump_to_disk(tc,"tk")
    dump_to_disk(wc,"wk")
    dump_to_disk(pc,"pk")
    dump_to_disk(prc,"prk")
    dump_to_disk(hc,"hk")
    return "success"

@app.route('/')
def home():
    """Render website's home page."""
    return render_template('home.html')

@app.route('/')
def send_text_file(file_name):
    """Send your static text file."""
    return app.send_static_file("index.html")


@app.route('/classify')
def classify():
    """Send your static text file."""
    import urllib
    tc=load("tk")
    wc=load("wk")
    pc=load("pk")
    prc=load("prk")
    hc=load("hk")

    tw=request.args['tweet']
    tw=urllib.unquote(tw).decode('utf8')
    print(tw)
    t=classify1(tc,tw) or ''
    w=classify1(wc,tw) or ''
    p=classify1(pc,tw) or ''
    pr=classify1(prc,tw) or ''
    hm=classify1(pc,tw) or ''
    print(t+" "+" "+w+" "+p+" "+pr)
    return t+" "+" "+w+" "+p+" "+pr+" "+pr

@app.route('/get_weather_detail')
def get_weather_detail():
    data=get()
    t=[]
    w=[]
    p=[]
    pr=[]
    hm=[]
    tc=load("tk")
    wc=load("wk")
    pc=load("pk")
    prc=load("prk")
    hc=load("hk")
    for d in data:
       t.append(classify1(tc,d))
       w.append(classify1(wc,d))
       p.append(classify1(pc,d))
       pr.append(classify1(prc,d))
       hm.append(classify1(hc,d))

    return t.__repr__()+"\n"+ w.__repr__() + "\n" + p.__repr__() + "\n" + pr.__repr__() + "\n"+ hm.__repr__() + "\n"

def pred(t):
    return Counter(t).most_common(1)[0][0]

@app.route('/get_weather')
def get_weather():
    data=get()
    t=[]
    w=[]
    p=[]
    pr=[]
    hm=[]
    tc=load("tk")
    wc=load("wk")
    pc=load("pk")
    prc=load("prk")
    hc=load("hk")
    for d in data:
       t.append(classify1(tc,d))
       w.append(classify1(wc,d))
       p.append(classify1(pc,d))
       pr.append(classify1(prc,d))
       hm.append(classify1(hc,d))

    return [pred(t),pred(w),pred(p),pred(pr),pred(hm)]



@app.route('/tweet')
def tweet():
    if os.environ['tk']==request.args['p']:
        data=get()
        t=[]
        w=[]
        p=[]
        pr=[]
        hm=[]
        tc=load("tk")
        wc=load("wk")
        pc=load("pk")
        prc=load("prk")
        hc=load("hk")
        for d in data:
           t.append(classify1(tc,d))
           w.append(classify1(wc,d))
           p.append(classify1(pc,d))
           pr.append(classify1(prc,d))
           hm.append(classify1(hc,d))


        print(t)
        print(w)
        print(p)
        print(pr)
        print(pred(hm))
        predd='Time: '+moment.now().strftime("%Y-%m-%dT%H:%M:%S")+ ' Temperature: '+ pred(t).__str__() +',Wind '+pred(w).__str__() +',Rain: '+pred(p).__str__() +',Presure: '+pred(pr).__str__() +',Humidity: '+pred(hm).__str__()
        api.update_status(predd)
        return predd
    return "wrong password"

def classify1(klas,t):
    return klas.classify(t)



@app.route('/train')
def train():
    """Send your static text file."""
    tc=load("tk")
    wc=load("wk")
    pc=load("pk")
    prc=load("ppk")

    soup=get_soup()
    tl=temp_to_labels(get_params(soup,"Temperature",mean=True))
    wl=ws_to_labels(get_params(soup,"WindSpeedMS"))
    rl=rain_to_labels(get_params(soup,"Precipitation1h"))
    pl=press_to_labels(get_params(soup,"Pressure"))
    data=get()
    train(tc,data,tl)
    train(wc,data,wl)
    train(pc,data,rl)
    train(prc,data,pl)

    dump_to_disk(tc,"tk")
    dump_to_disk(wc,"wk")
    dump_to_disk(pc,"pk")
    dump_to_disk(prc,"prk")
    return "success"

def train_one(knn,tweet,label):
    v = tag(tweet)
    v = [word for word, pos in v]
    v = count(v)
    if v:
        knn.train(v, type=label)

def load(fname):
    klas=KNN()
    if os.path.isfile(fname):
        return KNN.load(fname)
    return klas

def train(klas,data,label):
    for tweet in data:
        train_one(klas,tweet,label)

def dump_to_disk(klas,fname):
    klas.save(fname)

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=600'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.errorhandler(404)
def page_not_found(error):
    """Custom 404 page."""
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)
