#!/usr/bin/env python
# coding: utf-8

# # 1.创建数组

# In[2]:


import numpy as np # 导入numpy模块


# 引例：求数列a(n) = n^3 + n^2的前n项

# In[4]:


def sum(n):
    a = np.arange(1,n+1) ** 3
    b = np.arange(1,n+1) ** 2
    return a+b


# In[7]:


n = int(input("enter a number:"))
sum(n)


# ## 1.1 创建数组的方法

# In[19]:


#创建数组的方法有3种
a = np.array([1,2,3,4,5])
b = np.array(range(1,6)) 
c = np.arange(1,6)

d = [1,2,3,4,5]
e = np.array(d)


# ```Python
# array：将输入数据(元组、数组、列表以及其他序列)转换为ndrray,如不显示指明数据类型，将自动推断，默认赋值所有输入数据
# arange:Python内置函数range的数组版本，返回一个数据
# 其中：    
# arange创建数字序列
# 使用arange创建数字序列：np.arange(开始，结束，步长，dtype = None)
# ```

# In[35]:


print(np.arange(5))
print(np.arange(0,11,2))


# In[21]:


print(a)
print(b)
print(c)
print(d)


# In[13]:


print(a.dtype)
print(type(a))


# In[26]:


#给数据指定数据类型
f = np.array(range(1,8),dtype = float)#修改数据类型
f2 = np.array(range(1,8),dtype = 'float32')#修改数据类型和位数
print(f,f2)
print(f.dtype,f2.dtype)


# ## 1.2 array的属性

# ```python
# shape:返回一个元组，表示array的维度，[形状,几行几列]
# ndim:返回一个数字，表示array的维度的数目
# size:返回一个数字，表示array所有数据元素的数目
# dtype:返回array中元素的数据类型
# ```
# 

# In[33]:


print(f2.shape, f.ndim, a.size, c.dtype)


# ## 1.3 使用ones创建全是1的数组
# ```
# np.ones(shape,dtype = None)
# ```

# In[50]:


a1 = np.ones(3)
a2 = np.ones((2,3))#两行三列的数组,往里传一个（2，3）的元组
a3 = np.ones((5,),dtype = np.int64)#指定元素类型为int64
print(a1)
print(a2)
print(a3)


# ## 1.4 使用zeros创建全是0的数组
# ```
# np.zeros(shape,dtype = None)
# ```

# In[53]:


np.zeros((2,3),dtype = np.int32)


# ## 1.5 使用full创建全是指定值的数组
# ```
# np.full(shape,fill_value,dtype = None)
# ```

# In[66]:


love = np.full((2,2),520, dtype = np.int32)
love


# ## 1.6 多维数组

# In[67]:


b1 = np.array([[1,2,3],
               [4,5,6]])
print(b1)


# In[72]:


print(b1.shape)
print(b1.ndim)
print(b1.size)


# In[78]:


b3 = np.full((2,3,4),6,dtype = np.int32)#创建一个全是6三维数组
print(b3)
print(b3.ndim)#查看维度
print(b3.shape)


# ## 1.7 ones_like; zeros_like; full_like

# 使用```ones_like```返回形状相同的全1数组

# In[79]:


b4 = np.ones_like(b3)
print(b4)


# 类似的，有```zeros_like```返回形状相同的全0数组,以及```full_like```

# In[86]:


b5 = np.zeros_like(b4)
print(b5)

print('----------------------',end='\n')
b6 = np.full_like(b5,3,dtype = np.int32)
print(b6)


# ## 1.8 使用random模块生成随机数组

# In[89]:


v1 = np.random.randn()#1个随机数
v2 = np.random.randn(3)#3个随机数
v3 = np.random.randn(2,3)#两行三列的随机数
print(v1)
print(v2)
print(v3)


# In[91]:


#四舍五入：
np.round(v3,2)#变量a保留2位小数点


# # 2.数组的计算

# ## 2.1 ```reshape```不改值修改数组形状

# In[104]:


c1 = np.arange(10)
print(c1)

print('----------------------',end='\n')

c2 = c1.reshape(2,5)
print(c2)

print('----------------------',end='\n')

c3 = c2.reshape(10,1)
print(c3)

print('----------------------',end='\n')

#不知道数组中有多少个数，但是仍然像将其转换为一维
c4 = c2.flatten()#将一个多维数组转换为一个一维数组
print(c4)


# ## 2.2 数组具体的计算

# 相同形状的数组进行运算<==>对应元素进行指定的四则运算

# In[116]:


res = np.arange(1,10)
res = res.reshape(3,3)
res += 1#每一个元素都加一
print(res)

print('-'*30,end='\n')

sig = np.arange(1,19,2)
sig = sig.reshape(3,3)
print(sig)

print('-'*30,end='\n')

print(sig * res)#每个元素对应进行四则运算


# 形状不同的：（1）维度不同：列数一样，行数不一样

# In[123]:


#eg1:a是一个2行5列的数组，b是一个1行5列的数组
a = np.arange(1,11)
a = a.reshape(2,5)
b = np.arange(1,6)

print(a)
print('-'*30,end='\n')
print(b)

print('-'*30,end='\n')

c = a + b#二维数组的每一维与一维数组对应相加；也可以把该一维数组视作为两行一样的二维数组，再与原二维数组进行四则运算
print(c)



# 形状不同的：（2）维度相同：列数不一样，行数一样

# In[130]:


a1 = a.reshape(5,2)
b1 = b.reshape(5,1)#这样竖着的一列也是二维的！！一维是横着的
print(a1)

print('-'*30,end='\n')

print(b1)

print('-'*30,end='\n')

c1 = a1 + b1#把该数组视作为两列一样的二维数组，再与原二维数组进行四则运算
print(c1)


# In[135]:


repo = np.arange(1,11).reshape(5,2)
reqe = np.arange(1,16).reshape(5,3)
print(repo)
print('-'*30,end='\n')
print(reqe)
reqe + repo


# ```显然，由于形状不同的原因:一个是(5,3),一个是 (5,2) ==>repo和reqe不能相加```

# # 3. 索引与切片

# In[5]:


a = np.arange(10)
print(a)
print(a[3:9])
print(a[5])


# ## 3.1 二维数组索引用行列坐标

# In[22]:


A = np.arange(20).reshape(4,5)
print(A)

print('-'*30,end='\n')

print("A[0,0] =",A[0,0])

print('-'*30,end='\n')

print("A[-1,2] =", A[-1,2])#最后一行的第二列

print('-'*30,end='\n')

print("A[2] =", A[2])#第二行

print('-'*30,end='\n')

print("第二列A[:,2]",A[:,2])

print('-'*30,end='\n')

print("最后一行A[-1] =", A[-1])#最后一行

print('-'*30,end='\n')

print("除了最后一行之外的所有行A[0:-1]:",'\n', A[0:-1])#除了最后一行之外的所有行

print('-'*30,end='\n')

print("第0行和第1行的第二列和第三列A[0:2,2:4]:",'\n',A[0:2,2:4])#第0行和第1行的第二列和第三列



# ## 3.2布尔索引

# In[31]:


#实例1：对一维数组进行01处理
a = np.arange(1,11)
print(a)
cond = a > 5
print(cond)
print(a[a>5])
a[a<=5] = 0
a[a>5] = 1
print(a)


# In[49]:


#实例2：多条件组合
a = np.arange(1,11)

print(a)

f = np.arange(1,21).reshape(5,4)
f_condict = (a % 2==0) | (a < 7)
print(f_condict)
print(a[(a % 2==0) | (a < 7)])


# ## 3.3神奇索引：使用整数数组进行数据索引

# In[64]:


a = np.array([3,6,7,9,5,2,7])
a[[2,3,5]]#返回对应下标的数组
numarr = np.arange(1,37).reshape(9,4)
print(numarr)

print('-'*30,end='\n')

print(numarr[[0,5,2,3]])#输出原数组的第0，5，2，3行

print('-'*30,end='\n')

print(numarr[[0,5,2,3]])#输出原数组的第0，5，2，3行

print('-'*30,end='\n')

print(numarr[[5,6,7,8],[0,1,2,3]])#取出[5,0],[6,1],[7,2],[8,3]，两两对应

print('-'*30,end='\n')

print(numarr[:,[1,3]])#取出第1列和第三列


# 实例：通过利用```argsort()```函数返回排序后的下标的功能获取数组中最大的前n个数字

# In[71]:


subs = np.random.randint(1,100,10)#1~100以内的10个随机整数
print(subs)


# In[85]:


min_to_max = subs.argsort()#正序：从小到大
max_to_min = subs.argsort()[::-1]#倒序：从大到小
print(subs[min_to_max])#正序输出
print(subs[max_to_min])#逆序输出

下标 = subs.argsort()[-3:]#最大的三个数
print("最大的三个数是:",subs[下标])


# 
# ## 3.4Numpy的轴

# ```
# 几个维度的数组就有几个轴
# shape得到的元组，从第一个数字开始是0轴，第二个是1轴，第三个是2轴...以此类推
# 相当于一个数组A(n) = n(n从0开始)
# ```

# ## 3.5求数组的转置

# In[92]:


print(numarr)

print('-'*30,end='\n')

print(numarr.reshape(4,9))

print('-'*30,end='\n')

print(numarr.T)#求转置

print('-'*30,end='\n')

print(numarr.transpose())#行列转置

print('-'*30,end='\n')

print(numarr.swapaxes(1,0))#0轴和1轴进行交换


# # 使用seed向随机数生成器传递随即状态种子

# In[97]:


import random
random.seed(10)#这里不一定写10，只要是一个整数就行，但是不能为空，为空相当于未设置seed
#seed仅限于这一台电脑上，换电脑的话该种子就会发生变化
print(random.random())
print(random.random())
print(random.random())


# 因为设定了随机数种子，所以生成过的随机数不会改变

# rand返回(0,1)之间的数，从均匀分布中抽取样本

# In[105]:


一维 = np.random.rand(3)
print(一维)

print('-'*30,end='\n')

二维 = np.random.rand(2,3)
print(二维)

print('-'*30,end='\n')

三维 = np.random.rand(2,3,4)
print(三维)


# In[132]:


import matplotlib.pyplot as plt#导入画图模块
#绘制正弦曲线
axisX = np.linspace(-10,10,100)#在[-10,10]的闭区间中，数量为100
axisY = np.sin(axisX)
plt.plot(axisX,axisY)
plt.show()


# In[131]:


#加入噪声
X = np.linspace(-10,10,100)
Y = np.sin(X) + np.random.rand(len(X))
plt.plot(X,Y)
plt.show()


# randn返回从标准正态分布N(0,1)得到的随机标量

# In[135]:


一维 = np.random.rand(3)
二维 = np.random.rand(2,3)
三维 = np.random.rand(2,3,4)
print(一维)
print(二维)
print(三维)


# ```randint```随机整数

# In[214]:


a = np.random.randint(4)#0到3之间的随机整数
b = np.random.randint(1,11)#1到10之间的随机整数
c = np.random.randint(1,11,size = (5,3))#1到10之间随机取5*3个数字组成5行3列的二维矩阵
c


# random生成0.0到1.0的随机数:仅把randint改成random，别的用法都一样

# In[220]:


a = np.random.randint(3)#0到3之间的随机整数
b = np.random.random(size = (2,3))#生成2行3列的二维随机数矩阵
c = np.random.random(size = (2,3,4))#生成shape = (2,3,4)的三维随机数矩阵
c


# 使用choice从一维数组中生成随机数

# In[7]:


a = np.random.choice(5,(2,3))#从0-4之间随机取数字，组成2*3的二维矩阵
print(a)
b = np.random.choice([1,2,4,5,10,28,31],(4,3))#从1,2,4,5,10,28,31中随机取数字组成2*3的矩阵
print(b)


# 使用shuffle把一个数组进行随机排列(shuffle:洗牌==>改变原数组)

# In[25]:


一维数组 = np.arange(10)
print("未进行随机排列前的一维数组:",一维数组)
np.random.shuffle(一维数组)
print("进行随机排列后的一维数组:",一维数组)
二维数组 = np.arange(20).reshape(4,5)
print("未进行随机排列前的二维数组:\n",二维数组)#按行随机排（先把各行随机排序，再把排序之后的每行随机组合）多维数组也是
np.random.shuffle(二维数组)
print("进行随机排列后的二维数组:\n",二维数组)


# normal生成正态分布数字，又叫常态分布，高斯分布。
# ```normal[平均值，方差，size]```

# In[26]:


数组 = np.random.normal(1,10,(3,4))
数组


# 通用函数
# 也称ufunc,是一种在np.array数据中进行逐元素操作的函数。

# In[48]:


arr = np.arange(1,13).reshape(3,4)
wil = np.random.choice(16,(4,3))
print("arr =\n",arr)

print('-'*30,end='\n')

print("wil =\n",wil)

print('-'*30,end='\n')
#一元通用函数

print(np.sqrt(arr))#开平方根

print('-'*30,end='\n')

print(np.exp(arr))#e的x次方


# In[47]:


#二元通用函数
#实际函数比在此示例的更多，可以去查阅numpy函数手册，不在此做过多的赘述
x = arr.reshape(1,12)
y = wil.reshape(1,12)
print(np.maximum(x,y))#对位比较大小，取较大值，生成新的数组返回，逐个元素的将两个数组中的较大值计算出来
print(np.dot(arr,wil))#矩阵的点乘


# 数学方法

# In[61]:


一维数组
a = np.array([1,3,4,2,5,3,1,6,9,3])
print("数组a为:",a)

print(np.sum(a))#求数组内元素的累加
print(np.prod(a))#求数组内元素连乘

print(np.max(a))#求数组内元素最大值
print(np.argmax(a))#返回最大值（第一次出现）下标

print(np.min(a))#求数组内元素最小值
print(np.argmin(a))#返回最下值（第一次出现）下标

print(np.mean(a))#求平均值

print(np.median(a))#中位数

print(np.average(a))#加权平均

counts = np.bincount(a)#统计非负整数的个数，不能统计浮点数
print(np.argmax(counts))#众数


# Numpy中axis参数的用途(axis = 0代表行，axis = 1代表列),所有的数学和统计函数都有这个参数，都可以使用，当我们想按行或者按列计算的时候都可以使用这个参数。

# In[66]:


a = np.array([[1,3,6],[9,3,2],[1,4,3]])
print("原数组为\n",a)

print('-'*30,end='\n')

print(np.sum(a,axis = 0))#不同行元素相加<==>列元素求和
print('-'*30,end='\n')
print(np.sum(a,axis = 1))#不同列元素相加<==>行元素求和


# # 条件计算

# ## 7.1将条件逻辑转换为数组操作

# In[69]:


print(a > 3)
print(np.where(a>3,520,1314))#如果a大于3，改为540；否则改为1314


# ## 7.2 布尔值数组方法 any 和 all

# In[73]:


print((a>3).sum())#数组中a大于3的个数<==大于3的都为1，小于等于的都为0。3个1相加为3<==>有3个大于3的数字


# ```
# 对于布尔值数组，有两个常用方法：any和all
# any:检查数组中是否至少有一个True
# all:检查是否每个值都是True
# ```

# In[76]:


print((a>3).any())
print((a>3).all())


# 按照值的大小进行排序sort

# In[89]:


#一维数组
a = np.array([13,6,7,9,2,1,8,5,4])
a.sort()#修改了原始数组
print(a)

print('-'*30,end='\n')

#二维数组
b = np.array([[0,12,48],[4,18,14],[7,1,99]])
b.sort()#默认按照最后一个轴来排序
print(b)


# ## 唯一值与其它集合逻辑;unique和in1d

# In[90]:


姓名 = np.array(['孙悟空','猪八戒','孙悟空','沙和尚','孙悟空','唐僧'])
print(np.unique(姓名))
数组 = np.array([1,3,1,3,5,3,1,3,7,3,5,6])
print(np.unique(数组))
#检查一个数组中的值是否在另外一个数组中，并返回一个布尔数组：
a = np.array([6,0,0,3,2,5,6])
print(np.in1d(a,[2,3,6]))

