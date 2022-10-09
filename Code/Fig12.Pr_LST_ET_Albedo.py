#--coding:utf-8--
# Zhang Chao 2022/4/18
# Plot correlation between precipitation and LST/ET/Albedo

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import scipy
from sklearn.linear_model import LinearRegression

Dir  = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root = Dir + '\Actual'
root2= Dir + '\Precipitation'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname = outdir + r'\Fig12.svg'


def getDayLST(df):
    df['LSTDay'] = df[[ 'LST_Day_03_x', 'LST_Day_04_x',
              'LST_Day_05_x', 'LST_Day_06_x', 'LST_Day_07_x', 'LST_Day_08_x',
              'LST_Day_09_x', 'LST_Day_10_x', 'LST_Day_11_x']]\
                         .apply(lambda x: x.sum(), axis=1) / 9
    return df['LSTDay']

def getNightLST(df):
    df['LSTNight'] = df[[ 'LST_Night_03_x', 'LST_Night_04_x',
              'LST_Night_05_x', 'LST_Night_06_x', 'LST_Night_07_x', 'LST_Night_08_x',
              'LST_Night_09_x', 'LST_Night_10_x', 'LST_Night_11_x']]\
                         .apply(lambda x: x.sum(), axis=1) / 9
    return df['LSTNight']

def getET(df):
    df['ET'] = df[[ 'ET_03_x', 'ET_04_x',
              'ET_05_x', 'ET_06_x', 'ET_07_x', 'ET_08_x',
              'ET_09_x', 'ET_10_x', 'ET_11_x']]\
                         .apply(lambda x: x.sum(), axis=1)/9
    return df['ET']

def getAlbedo(df):
    df['Albedo'] = df[[ 'Albedo_03_x', 'Albedo_04_x',
              'Albedo_05_x', 'Albedo_06_x', 'Albedo_07_x', 'Albedo_08_x',
              'Albedo_09_x', 'Albedo_10_x', 'Albedo_11_x']]\
                         .apply(lambda x: x.sum(), axis=1) / 9

    return df['Albedo']

def getPr(df):
    df['Yr2000'] = df[['20000101_pr','20000201_pr','20000301_pr','20000401_pr','20000501_pr','20000601_pr',
                       '20000701_pr','20000801_pr','20000901_pr','20001001_pr','20001101_pr','20001201_pr',]]\
                      .apply(lambda x: x.sum(), axis=1)
    df['Yr2020'] = df[['20200101_pr', '20200201_pr', '20200301_pr', '20200401_pr', '20200501_pr', '20200601_pr',
                       '20200701_pr', '20200801_pr', '20200901_pr', '20201001_pr', '20201101_pr', '20201201_pr', ]] \
                      .apply(lambda x: x.sum(), axis=1)
    df['Mean00_20']  = (df['Yr2000'] + df['Yr2020'])/2
    df['Minus20_00'] =  df['Yr2020'] - df['Yr2000']
    return df[['Id','Mean00_20','Minus20_00']]
def getGroupValue(df,varname,Labels):
    elist = []
    for label in Labels:
        list = df[df['Printerval']== label][varname].tolist()
        elist.append(list)
    return elist

def LinearRegressionFunc(X, Y):
    X_train = np.array(X).reshape((len(X), 1))
    Y_train = np.array(Y).reshape((len(Y), 1))
    lineModel = LinearRegression()
    lineModel.fit(X_train, Y_train)
    Y_predict = lineModel.predict(X_train)

    a = lineModel.coef_[0][0]
    b = lineModel.intercept_[0]
    R2 = lineModel.score(X_train, Y_train)
    return a, b, Y_predict,R2

def drawCaptionPlot(ax,title,pos):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc='#D9D9D9', edgecolor='k'))  # ,alpha=0.6)
    ax.text(pos[0],pos[1],title,transform=ax.transAxes,fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
####################################################################
#LST, ET, Albedo
dff1 = pd.read_csv(root + r'\02forest2c\forestGroup.csv')
dfg1 = pd.read_csv(root +  r'\03grass2c\grassGroup.csv')
dfu1 = pd.read_csv(root +  r'\04unuse2c\unuseGroup.csv')
#Precipitation
dff2 = pd.read_csv(root2 +  r'\Pr_forest.csv')
dfg2 = pd.read_csv(root2 +  r'\Pr_grass.csv')
dfu2 = pd.read_csv(root2 +  r'\Pr_unuse.csv')

dff2 = getPr(dff2)
dfg2 = getPr(dfg2)
dfu2 = getPr(dfu2)

dff = pd.merge(dff1,dff2[['Id','Mean00_20','Minus20_00']],on='Id')
dfg = pd.merge(dfg1,dfg2[['Id','Mean00_20','Minus20_00']],on='Id')
dfu = pd.merge(dfu1,dfu2[['Id','Mean00_20','Minus20_00']],on='Id')

DayLST = [getDayLST(dff),getDayLST(dfg),getDayLST(dfu)]
NightLST = [getNightLST(dff),getNightLST(dfg),getNightLST(dfu)]
ET = [getET(dff),getET(dfg),getET(dfu)]
Albedo= [getAlbedo(dff),getAlbedo(dfg),getAlbedo(dfu)]
Pr = [dff['Mean00_20'],dfg['Mean00_20'],dfu['Mean00_20']]


intval = [0,100,150,200,250,600]
Labels= [100,150,200,250,600]
dff['Printerval'] = pd.cut(dff['Mean00_20'],intval,labels=Labels)
dfg['Printerval'] = pd.cut(dfg['Mean00_20'],intval,labels=Labels)
dfu['Printerval'] = pd.cut(dfu['Mean00_20'],intval,labels=Labels)

def getPflag(sig):
    if sig < 0.05:
        return '*'
    elif sig <0.1:
        return '$^\circ$'
    else:
        return ''
def DrawLstPlots(ax,df,var1,var2,No,title,Ylim,xtickabel,ytick,yticklabel,xlabel,ylabel,legendFlag):
    width = 1.0
    x1 = np.arange(1, 1 + width * 2 * 5, width * 2)
    x2 = [i + width * 0.5 for i in x1]
    group1 = getGroupValue(df,var1,Labels)
    group2 = getGroupValue(df, var2, Labels)
    colors = ['#2A557F', '#45BC9C', '#F05073']

    plt.sca(ax)
    bp1 = plt.boxplot(x=group1,
                positions=x1,
                patch_artist=True,
                showmeans=True,
                boxprops={'color': 'blue', 'facecolor': 'white'},
                flierprops={'marker': 'o', 'markeredgecolor': 'b','markerfacecolor': 'b', 'markersize': 0.5},
                meanprops={'marker': 'D', 'markeredgecolor': 'b', 'markerfacecolor': 'none', 'markersize': 1},
                medianprops={'linestyle': '--', 'color': 'blue'},
                capprops={'color': 'blue'},
                whiskerprops={'color': 'blue'},
                )
    bp2 = plt.boxplot(x=group2,
                positions=x2,
                patch_artist=True,
                # labels=[],
                boxprops={'color': 'r', 'facecolor': 'white'},
                flierprops={'marker': 'o', 'markeredgecolor': 'r','markerfacecolor': 'r', 'markersize': 0.5},
                meanprops={'marker': 'D', 'markeredgecolor': 'r', 'markerfacecolor': 'none', 'markersize': 1},
                medianprops={'linestyle': '--', 'color': 'r'},
                capprops={'color': 'r'},
                whiskerprops={'color': 'r'},
                )

    #计算降水量与LSTDay, LSTNight的相关系数
    r1, sig1 = scipy.stats.pearsonr(df['Mean00_20'], df[var1])
    r2, sig2 = scipy.stats.pearsonr(df['Mean00_20'], df[var2])

    plt.text(0.38, 0.05, "r=%.2f%s" % (r1, getPflag(sig1)), transform=ax.transAxes, fontsize=12,c='b')
    plt.text(0.68, 0.05, "r=%.2f%s" % (r2, getPflag(sig2)), transform=ax.transAxes, fontsize=12,c='r')

    plt.axhline(0, 0, 20, color="k",linestyle='--',linewidth=0.7)
    plt.ylim(Ylim)
    plt.xlim([0, 10])
    plt.ylabel(ylabel, fontsize=12)  #, labelpad=0.5
    plt.xlabel(xlabel, fontsize=12)
    # plt.title(title)
    plt.text(0.02, 0.85, No, transform=ax.transAxes, fontsize=12)  # , style='italic'
    plt.yticks(ytick,yticklabel, fontsize=12)
    plt.xticks([], [])
    if (xtickabel==False):
        plt.xticks(x1,[])
    else:
        plt.xticks(x1, ['<1','1~1.5','1.5~2','2~2.5','>2.5'])
    if (legendFlag == True):
        plt.legend([bp1["boxes"][0], bp2["boxes"][0]],['Daytime','Nighttime'],frameon = False,
                   loc='upper right', bbox_to_anchor=(1.05, 1.08))


def DrawETAlbedoPlots(ax,df,var1,No,title,Ylim,xtickabel,ytick,yticklabel,xlabel,ylabel):
    width = 1.0
    x1 = np.arange(1, 1 + width * 2 * 5, width * 2)
    group1 = getGroupValue(df,var1,Labels)
    colors = ['#2A557F', '#45BC9C', '#F05073']

    plt.sca(ax)
    plt.boxplot(x=group1,
                positions=x1,
                patch_artist=True,
                # labels=[],  # 添加x轴的刻度标签
                boxprops={'color': 'k', 'facecolor': 'white'},
                flierprops={'marker': 'o', 'markerfacecolor': 'k', 'markersize': 0.5},
                meanprops={'marker': 'D', 'markeredgecolor': 'k', 'markerfacecolor': 'none', 'markersize': 1},
                medianprops={'linestyle': '--', 'color': 'k'},
                capprops={'color': 'k'},
                whiskerprops={'color': 'k'},
                )
    # 计算降水量与ET/Albedo的相关系数
    r1, sig1 = scipy.stats.pearsonr(df['Mean00_20'], df[var1])

    plt.text(0.68, 0.05, "r=%.2f%s" % (r1, getPflag(sig1)), transform=ax.transAxes, fontsize=12, c='k')

    plt.axhline(0, 0, 20, color="k", linestyle='--', linewidth=0.7)
    # plt.xlim([0.2,5.2])
    plt.ylim(Ylim)
    plt.xlim([0, 10])
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    # plt.title(title)
    plt.text(0.02, 0.85, No, transform=ax.transAxes, fontsize=12)  # , style='italic'
    plt.yticks(ytick,yticklabel, fontsize=12)
    if (xtickabel==False):
        plt.xticks(x1,[])
    else:
        plt.xticks(x1, ['<1','1~1.5','1.5~2','2~2.5','>2.5'], fontsize=11)



# plt.style.use('ggplot')

fig = plt.figure(figsize=(8.0, 5.0))
fig.subplots_adjust(left=0.07, right=0.995,
                    bottom=0.1, top=0.95,
                    wspace=0.05, hspace=0.02)

ax1 = plt.subplot(331)
ax2 = plt.subplot(332)
ax3 = plt.subplot(333)
ax4 = plt.subplot(334)
ax5 = plt.subplot(335)
ax6 = plt.subplot(336)
ax7 = plt.subplot(337)
ax8 = plt.subplot(338)
ax9 = plt.subplot(339)

ax1_1= fig.add_axes([0.070,0.95,0.299,0.05])
ax2_1= fig.add_axes([0.384,0.95,0.297,0.05])
ax3_1= fig.add_axes([0.696,0.95,0.299,0.05])
drawCaptionPlot(ax1_1,'forest2crop',[0.3,0.30])
drawCaptionPlot(ax2_1,'grass2crop', [0.3,0.30])
drawCaptionPlot(ax3_1,'unuse2crop', [0.3,0.30])

Ylim1 = [-4.5, 4.5]
Ylim2 = [-0.045,0.045]
Ylim3 = [-3,5.5]
Ytick1 = [-4,-2,0,2,4]
Ytick2 = [-0.04,-0.02,0,0.02,0.04]
Yticklabel = [-4,-2,0,2,4]
Ytick3 = [-2,0,2,4]
Xticklabel = ['<-10','-10~0','0~10','10~30','>30']
# Yticklabel = [-1,-0.5,0]
Xlabel = 'Precipitation(100 mm)'
plt.rc('font',size = 12)#, family='Times New Roman'

DrawLstPlots(ax1,dff,'LSTDay','LSTNight','(a)','forest2crop',Ylim1,False,Ytick1,Ytick1 ,'','$\Delta$LST (K)',True)#$^\circ$C
DrawLstPlots(ax2,dfg,'LSTDay','LSTNight','(b)','grass2crop',Ylim1,False,Ytick1,'','','',False)
DrawLstPlots(ax3,dfu,'LSTDay','LSTNight','(c)','unuse2crop',Ylim1,False,Ytick1,'','','',False)

DrawETAlbedoPlots(ax4,dff,'Albedo','(d)','',Ylim2,False,Ytick2,Yticklabel ,'','$\Delta$Albedo(%)')
DrawETAlbedoPlots(ax5,dfg,'Albedo','(e)','',Ylim2,False,Ytick2,'','','')
DrawETAlbedoPlots(ax6,dfu,'Albedo','(f)','',Ylim2,False,Ytick2,'','','')

DrawETAlbedoPlots(ax7,dff,'ET','(g)','',Ylim3,True,Ytick3,Ytick3 ,'','$\Delta$ET (mm)')
DrawETAlbedoPlots(ax8,dfg,'ET','(h)','',Ylim3,True,Ytick3,'',Xlabel,'')
DrawETAlbedoPlots(ax9,dfu,'ET','(i)','',Ylim3,True,Ytick3,'','','')

# plt.savefig(outname)

plt.show()
