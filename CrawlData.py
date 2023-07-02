import requests
from bs4 import BeautifulSoup
import os
import json
from tqdm import tqdm
from datetime import datetime, timedelta
import time
import re

PRE_URL = "https://vnexpress.net/"
OUTPUT = 'test/data/'

END_TIMESTAMP = 1684022400

# 6 month
MINUS_TIMESTAMP = 7776000 

LIST_TOPIC = [   
    
    {"name":"thoi-su", 'cateId': 1001005},
    {"name":"kinh-doanh", 'cateId': 1003159},
    {"name":"khoa-hoc", 'cateId': 1001009},
    {"name":"giai-tri", 'cateId': 1002691},
    {"name":"the-thao", 'cateId': 1002565},
    {"name":"phap-luat", 'cateId': 1001007},
    {"name":"giao-duc", 'cateId': 1003497},
    {"name":"suc-khoe",'cateId': 1003750},
    {"name":"doi-song",'cateId': 1002966},
    {"name":"du-lich", 'cateId': 1003231},
]


def get_content(url, params):
    headers = dict()
    
    headers["accept"]="text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    while True:
        try:
            res = requests.get(url, headers= headers, timeout=3
, params= params)
            break
        except:
            print("Try again")
    
    res.close()
    return res.content

check = {}

def get_list_article(name, cateId):
    path = OUTPUT + name
    os.makedirs( path, exist_ok=True)
    
    url = f"{PRE_URL}category/day"
    
    params = {}
    params["cateid"] = cateId
    
    cnt = 0
    current_timestamp = END_TIMESTAMP
    
    print(f"Tải chủ đề  {name}:\n")
    # print(f"   Lấy danh sách tên các bài báo\n")
    for i in range(32):
        
    
        params["todate"] = f"{current_timestamp}"
        current_timestamp -= MINUS_TIMESTAMP
        params["fromdate"] = f"{current_timestamp}"
        for page in range(20):
            params["page"] = page + 1
            
            while True:
                soup = BeautifulSoup(get_content(url= url, params= params), 'html.parser')
                article_list_url = []
                for item in soup.find_all('a'):
                    
                    if item.parent.name == 'div' and item.parent.parent.name == "article":
                        article_list_url.append(item.get('href'))
                
                print(f"\n   Tải danh sách các bài báo lần {i}\n")
                
                try:
                    check[article_list_url[0]]
                    print(url, params)
                    time.sleep(5)
                except:
                    if len(article_list_url):
                        check[article_list_url[0]] = 1
                    for article_url in tqdm(article_list_url):
                        print(article_url)
                        info_article = get_info_article(article_url)
                        cnt = cnt + 1
                        with open(f'{path}/{cnt}.txt','w+', encoding='UTF-8') as file:
                            file.write(json.dumps(info_article, ensure_ascii=False))
                        # time.sleep(1)
                    break
        if cnt > 13000:
            break           
                # time.sleep(40)
        
        
    
    print(f"\nTải xong {cnt} bài chủ đề  {name}\n-------------------------------------------------------------------------------\n")
    
    return article_list_url
    
def get_info_article(url):
    soup = BeautifulSoup(get_content(url, params=None), 'html.parser')
    title = get_text(soup.find('h1',{'class':'title-detail'}))
    description = get_text(soup.find('p',{'class':"description"}))
    content = ""
    content += title + ". "
    content += description 
    for p in soup.find_all('p',{'class':"Normal"}):
        kt = True
        for p_content in p:
            if p_content.name == "strong":
                kt = False
                break
        if kt:
            content += get_text(p)
    return re.sub(r'\s+', ' ', content)

def get_text(e):
    if e == None: 
        return ""
    else:
        return e.text
    

for topic in LIST_TOPIC:
    get_list_article(name= topic["name"], cateId=topic["cateId"])
