from selenium import webdriver  # 导入Selenium的webdriver
from selenium.webdriver.common.keys import Keys   # 导入Keys
from selenium.webdriver.support.ui import Select  

driver = webdriver.Chrome()  # 指定使用的浏览器，初始化webdriver
driver.get("http://www.python.org")  # 请求网页地址
assert "Python" in driver.title  # 看看Python关键字是否在网页title中，如果在则继续，如果不在，程序跳出。
elem = driver.find_element_by_name("q")  # 找到name为q的元素，这里是个搜索框
elem.clear()  # 清空搜索框中的内容
elem.send_keys("pycon")  # 在搜索框中输入pycon
elem.send_keys(Keys.RETURN)  # 相当于回车键，提交
elem.click()  # 点击操作
assert "No results found." not in driver.page_source  # 如果当前页面文本中有“No results found.”则程序跳出

element0 = driver.find_element_by_id("passwd-id")  # 通过id获取元素
element1 = driver.find_element_by_name("passwd")  # 通过name获取元素
element2 = driver.find_element_by_xpath("//input[@id='passwd-id']")  # 通过使用xpath匹配获取元素

'''
<select name="cars">
  <option value ="volvo">沃尔沃</option>
  <option value ="bmw">宝马</option>
  <option value="benz">奔驰</option>
  <option value="audi">奥迪</option>
</select>
'''
select = Select(driver.find_element_by_name('cars'))  # 找到name为cars的select标签
select.select_by_index(1)  # 下拉框选中沃尔沃
select.select_by_visible_text("宝马")  # 下拉框选中宝马
select.select_by_value("benz")  # 下拉框选中奥迪

# 比如登录需要cookie
driver.get("http://www.example.com")  # 先请求一个网页
cookie = {'name': 'foo', 'value': 'bar'}  # 设置cookie内容
driver.add_cookie(cookie)  # 添加cookie

driver.close()  # 关闭webdriver
