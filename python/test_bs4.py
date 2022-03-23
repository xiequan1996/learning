from bs4 import BeautifulSoup  # 导入BeautifulSoup库
import re  # 导入正则模块

html_doc = """
<html><head><title>The Dormouse's story</title></head>
    <body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

soup = BeautifulSoup(html_doc, 'html.parser')
head_tag = soup.head
title_tag = head_tag.contents[0]
html_tag = soup.html
link_a = soup.a
last_a_tag = soup.find("a", id="link3")
sibling_soup = BeautifulSoup("<a><b>text1</b><c>text2</c></a>")  # 兄弟节点

print(soup.head)  # 以.tag_name 获取名为name的标签
print(head_tag.contents)  # 以.contents 属性将tag的子节点以列表的方式输出
print(head_tag.string)  # 仅有一个子节点，可以用.string，输出结果和唯一子节点的.string结果相同
print(soup.find_all('a'))  # 查找所有<a>的标签
for child in title_tag.children:
    print(child)  # 以.children生成器对tag的子节点进行循环
for child in head_tag.descendants:
    print(child)  # 以.descendants对tag的子孙节点进行递归循环（先序遍历）
print(len(list(soup.children)))
print(len(list(soup.descendants)))
for string in soup.stripped_strings:  # 以.strings循环获取tag中的所有字符串,.stripped_strings可以去除多余的空白内容
    print(repr(string))  # 以.repr 将对象转换为供解释器读取的形式，返回string格式
print(title_tag.parent)  # 以.parent获取某个元素的父节点
print(type(html_tag.parent))
print(soup.parent)  # BeautifulSoup对象的.parent是None
for parent in link_a.parents:  # 以.parents遍历tag到根节点的所有节点
    if parent is None:
        print(parent)
    else:
        print(parent.name)
print(sibling_soup.prettify())  # 以.prettify使bs4更好的显示
print(sibling_soup.b.next_sibling)  # 以.next_sibling访问下一个兄弟节点
print(sibling_soup.c.previous_sibling)  # 以.previous_sibling访问上一个兄弟节点
for sibling in soup.a.next_siblings:  # 以.next_siblings对当前节点的后面兄弟节点迭代输出
    print(repr(sibling))  # previous_siblings同理
print(last_a_tag.next_element)  # 以.next_element指向下一个被解析的对象(字符串或者tag),previous_element同理
for element in last_a_tag.next_elements:  # 以.next_elements的迭代器向后访问文档的解析内容
    print(repr(element))  # previous_elements同理
for tag in soup.find_all(re.compile("^b")):  # 通过正则表达式查找标签
    print(tag.name)
print(soup.find_all(["a", "b"]))  # 匹配列表中任意一元素，并返回列表形式
for tag in soup.find_all(True):  # True可以匹配任何值,但不会返回字符串节点
    print(tag.name)


def has_class_but_no_id(tag0):  # 校验包含class属性却不包含id的属性
    return tag0.has_attr('class') and not tag0.has_attr('id')


print(soup.find_all(has_class_but_no_id))  # 过滤tag,tag为参数


def not_lacie(href):
    return href and not re.compile("lacie").search(href)


print(soup.find_all(href=not_lacie))  # 过滤标签属性，属性为参数
