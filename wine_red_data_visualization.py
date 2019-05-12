import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
# 颜色
color = sns.color_palette()
# 数据print精度
pd.set_option('precision',3)
df = pd.read_csv('./data/winequality-red.csv',sep = ';')
'''
['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality'],
'''


'''单变量分析'''

#箱线子图
plt.style.use('ggplot')
colmn=df.columns.tolist()
fig=plt.figure(figsize=(10,6))#画布尺寸
for i in range(12):
	plt.subplot(2,6,i+1)#子图创建
	sns.boxplot(df[colmn[i]],orient='v',width=0.5,color=color[0])
	#做箱线图方法，参数表[数据列，竖/横型，宽度]
	plt.ylabel(colmn[i],fontsize=12)

plt.tight_layout()


#柱状图（频次直方图）
##单属性区间数量统计1
plt.style.use('ggplot')
colmn=df.columns.tolist()

fig=plt.figure(figsize=(10,8))

for i in range(12):
	plt.subplot(4,3,i+1)
	plt.hist(df[colmn[i]],bins=100,color=color[0])
	#柱状图方法，参数表[数据列，步长，颜色]
	#df[colmn[i]].hist(bins=100,color=color[0])
	plt.xlabel(colmn[i],fontsize=12)
	plt.ylabel('Frequency')

plt.tight_layout()

##单属性区间数量统计2（属性值参与运算后对比）
acidityFeat = ['fixed acidity', 'volatile acidity', 'citric acid',
               'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']
plt.figure(figsize=(10,6))

for i in range(6):
	ax=plt.subplot(3,2,i+1)
	v=np.clip(df[acidityFeat[i]].values,a_min=0.001,a_max=None)
	plt.hist(np.log10(v),bins=50,color=color[0])
	plt.xlabel('log(' + acidityFeat[i] + ')',fontsize = 12)
	plt.ylabel('Frequency')
plt.tight_layout()

##单属性区间数量统计2（三种酸度频次对比）
plt.figure(figsize=(10,6))
bins=10**(np.linspace(-2,2))
plt.hist(df['fixed acidity'],bins=bins,edgecolor='k',label='fixed acidity')
plt.hist(df['volatile acidity'],bins=bins,edgecolor='k',label='volatile acidity')
plt.hist(df['citric acid'],bins=bins,edgecolor='k',label='citric acid',alpha=0.8)
plt.xscale('log')
plt.xlabel('Acid Concentration (g/dm^3)')
plt.ylabel('Frequency')
plt.title('Histogram of Acid Concentration')
plt.legend()#显示不同数据的特征
plt.tight_layout()

##单属性区间数量统计3（总算度频次）
df['total acid']=df['fixed acidity']+df['volatile acidity']+df['citric acid']
plt.figure(figsize = (8,3))

plt.subplot(1,2,1)
plt.hist(df['total acid'], bins = 50, color = color[0])
plt.xlabel('total acid')
plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(np.log(df['total acid']), bins = 50 , color = color[0])
plt.xlabel('log(total acid)')
plt.ylabel('Frequency')
plt.tight_layout()

#单属性区间数量统计4（单一属性值分情况频次统计）
df['sweetness']=pd.cut(df['residual sugar'],bins=[0,4,12,45],
						labels=['dry','medium dry','semi-sweet'])
#返回一个Series，包括index(原始的index)和一个由labels内值构成的数组。bins的区间值与labels值对应
plt.figure(figsize=(10,6))
v_c=df['sweetness'].value_counts()
plt.bar(v_c.index,v_c[:],color=color[0])#参数表[x轴数组，y轴数组]
plt.xlabel('sweetness', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
#plt.xticks(rotation=0)#对x坐标轴进行旋转
plt.tight_layout()

'''双变量分析'''
#箱线图(对于任一属性，其他属性的箱线分布情况)
sns.set_style('ticks')
sns.set_context('notebook',font_scale=1.1)

colnm=df.columns.tolist()[:11]+['total acid']
plt.figure(figsize=(10,6))
for i in range(12):
	plt.subplot(4,3,i+1)
	sns.boxplot(x='quality',y=colnm[i],data=df,color=color[0],width=0.6)
	plt.ylabel(colnm[i],fontsize=12)
plt.tight_layout()
#散点图（x：密度y:酒精度数）
sns.set_style('ticks')
sns.set_context("notebook", font_scale= 1.4)
# plot figure
plt.figure(figsize = (10,6))
sns.regplot(x='density',y = 'alcohol', data = df, 
	scatter_kws = {'s':10}, color = color[0])
#散点图加线性回归方法
plt.xlim(0.989, 1.005)
plt.ylim(7,16)

#散点图(酸性物质含量与pH)
acidity_related = ['fixed acidity', 'volatile acidity', 'total sulfur dioxide', 
                   'sulphates', 'total acid']

plt.figure(figsize = (10,6))

for i in range(5):
    plt.subplot(2,3,i+1)
    sns.regplot(x='pH', y = acidity_related[i], data = df, scatter_kws = {'s':5}, color = color[0])
plt.tight_layout()

'''多变量分析'''
#散点图
plt.style.use('ggplot')

sns.lmplot(x = 'alcohol', y = 'volatile acidity',
			hue = 'quality', #按酒等级来构造关于“酒精度数”和“挥发酸”的散点
           data = df, fit_reg = False, 
           scatter_kws={'s':10}, size = 8)
plt.figure(figsize=(10,6))
sns.lmplot(x = 'alcohol', y = 'volatile acidity',
			 col='quality', #是不同“等级”分散到子图中
			 hue = 'quality', 
           data = df,fit_reg = False, size = 3,  aspect = 0.9, col_wrap=3,
           scatter_kws={'s':20})# style
#散点图
sns.set_style('ticks')
sns.set_context("notebook", font_scale= 1.4)

plt.figure(figsize=(10,6))
cm = plt.cm.get_cmap('RdBu')
sc = plt.scatter(df['fixed acidity'], df['citric acid'], 
	c=df['pH']#用色域来表示第三属性
	,s=15#点的size
	, vmin=2.6, vmax=4,  cmap=cm#色域的值与颜色的映射关系)
bar = plt.colorbar(sc)
bar.set_label('pH',rotation = 0)
plt.xlabel('fixed acidity')
plt.ylabel('citric acid')
plt.xlim(4,18)
plt.ylim(0,1)
