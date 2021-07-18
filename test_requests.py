'''
Descripttion: 学习requsets模块
version: 
Author: xiequan
Date: 2021-05-23 17:28:19
LastEditors: Please set LastEditors
LastEditTime: 2021-07-18 16:06:09
'''
from posix import listdir
import requests  # 导入requests库
import os
import time
from bs4 import BeautifulSoup, element  # 导入BeautifulSoup模块
from tqdm import tqdm  # 导入进度条模块
from selenium import webdriver  # 导入Selenium


def mkdir(path):  # 创建文件夹
    path = path.strip()
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def get_response(html_url, encoding='utf-8'):
    headers = {
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip,deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9'
    }
    response = requests.get(url=html_url, headers=headers, timeout=20)
    response.encoding = encoding
    return response


class BeautifulNovel():     # 爬笔趣看小说网站的小说
    def save(self, title, content):
        # file_name = f"{novel_name}.txt"  # 一定要记得加后缀.txt
        # file_name=f"2.txt"
        # mode 保存方式 a是追加保存 encoding是保存编码
        if os.path.exists(title+'.txt'):
            os.remove(title+'.txt')
        with open(title+'.txt', mode='w', encoding='utf-8') as f:
            # 写入换行
            f.write('\n')
            # 写入标题
            # f.write(title)
            # 写入小说内容
            f.write(content)
            # 写入换行
            f.write('\n')

    def get_one_novel(self, novel_url, title):
        # 调用请求网页数据函数
        response = get_response(novel_url, 'gbk')
        # 构建解析对象
        soup = BeautifulSoup(response.text, 'lxml')
        # 小说内容 返回的是列表
        content = soup.select('.showtxt#content')[0].get_text().strip()
        self.save(title, content)

    def get_all_url(self, html_url):
        # 调用请求网页函数 获取到的是某个小说的首页
        response = get_response(html_url, 'gbk')
        # 解析我们需要的小说章节标题 章节url内容
        # 构成解析对象
        soup = BeautifulSoup(response.text, 'lxml')
        novel_name = soup.select('div.info>h2')[0].get_text()
        novel_list = soup.select('div.listmain>dl>dd')[12:-10]
        mkdir(novel_name)
        # 首页的所有章节url
        for link in tqdm(novel_list):
            title = link.a.get_text()
            novel_url = 'https://www.bqkan8.com'+link.a['href']
            self.get_one_novel(novel_url, title)
        print('爬取小说成功')


class BeautifulPicture():    # 爬煎蛋网的图片

    def __init__(self):  # 类的初始化操作
        self.web_url = 'https://jandan.net/girl'  # 要访问的网页地址
        self.folder_path = 'spider/picture'  # 设置图片要存放的文件目录

    def get_pic(self, times):
        # 使用selenium通过Chrome来进行网络请求
        driver = webdriver.Chrome()
        driver.get(self.web_url)
        all_a = []
        for i in range(times):
            if i != 0:
                elem = driver.find_element_by_class_name(
                    'previous-comment-page')
                elem.click()
            # 执行JavaScript实现网页下拉倒底部
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(15)  # 等待15秒，页面加载出来再执行下拉操作

            all_a.extend(BeautifulSoup(driver.page_source, 'lxml').find_all(
                'a', class_='view_img_link'))  # 获取网页中的class为cV68d的所有a标签

        is_new_folder = mkdir(self.folder_path)  # 创建文件夹，并判断是否是新创建
        file_names = os.listdir(self.folder_path)  # 获取文件家中的所有文件名，类型是list
        os.chdir(self.folder_path)  # 切换路径至上面创建的文件夹

        for a in all_a:  # 循环每个标签，获取标签中图片的url并且进行网络请求，最后保存图片
            img_str = a['href']  # a标签中完整的style字符串
            img_url = 'https:'+img_str

            # 截取url中参数前面、网址后面的字符串为图片名
            name_start_pos = img_url.index(
                '.cn/') + 10  # 通过找.cn/的位置，来确定它之后的字符位置
            img_name = img_url[name_start_pos:]

            if is_new_folder:
                self.save_img(img_url, img_name)  # 调用save_img方法来保存图片
            else:
                if img_name not in file_names:
                    self.save_img(img_url, img_name)  # 调用save_img方法来保存图片
        driver.close()
        print('爬取成功')

    def save_img(self, url, file_name):  # 保存图片
        img = get_response(url)
        f = open(file_name, 'ab')
        f.write(img.content)
        f.close()


picture = BeautifulPicture()  # 创建类的实例
picture.get_pic(4)  # 执行类中的方法
