#--coding:utf-8--
# Zhang Chao 2022/4/8
# Correlation analysis of LST and ET in daytime and nighttime in three regions

import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import os
from scipy import stats
import scipy
from sklearn.linear_model import LinearRegression


Dir  = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root = Dir  + r'\Actual'
root2 = Dir + r'\FenquGridID'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname1 = outdir + r'\FigS9.svg'
outname2 = outdir + r'\FigS10.svg'

def getDF(dff,dfg,dfu,roiBJ,roiNJ,roiHX):
    dff_BJ = pd.merge(dff,
                      pd.read_csv(roiBJ).dropna(), on='Id')
    dff_NJ = pd.merge(dff,
                      pd.read_csv(roiNJ).dropna(), on='Id')
    dff_HX = pd.merge(dff,
                      pd.read_csv(roiHX).dropna(), on='Id')

    dfg_BJ = pd.merge(dfg,
                      pd.read_csv(roiBJ).dropna(), on='Id')
    dfg_NJ = pd.merge(dfg,
                      pd.read_csv(roiNJ).dropna(), on='Id')
    dfg_HX = pd.merge(dfg,
                      pd.read_csv(roiHX).dropna(), on='Id')

    dfu_BJ = pd.merge(dfu,
                      pd.read_csv(roiBJ).dropna(), on='Id')
    dfu_NJ = pd.merge(dfu,
                      pd.read_csv(roiNJ).dropna(), on='Id')
    dfu_HX = pd.merge(dfu,
                      pd.read_csv(roiHX).dropna(), on='Id')
    return dff_BJ,dff_NJ,dff_HX,dfg_BJ,dfg_NJ,dfg_HX,dfu_BJ,dfu_NJ,dfu_HX

def getSE(data):
    count = data.shape[0]*data.shape[1] #len(df): row; df.shape[1]: columns
    return np.std(data.values.flatten())/math.sqrt(count)*1.96
def getDayLST(df):
    df['LSTDay'] = df[[ 'LST_Day_03_x', 'LST_Day_04_x',
              'LST_Day_05_x', 'LST_Day_06_x', 'LST_Day_07_x', 'LST_Day_08_x',
              'LST_Day_09_x', 'LST_Day_10_x', 'LST_Day_11_x']]\
                         .apply(lambda x: x.sum(), axis=1) / 9
    return df[['LSTDay']]

def getNightLST(df):
    df['LSTNight'] = df[[ 'LST_Night_03_x', 'LST_Night_04_x',
              'LST_Night_05_x', 'LST_Night_06_x', 'LST_Night_07_x', 'LST_Night_08_x',
              'LST_Night_09_x', 'LST_Night_10_x', 'LST_Night_11_x']]\
                         .apply(lambda x: x.sum(), axis=1) / 9
    return df[['LSTNight']]

def getET(df):
    df['ET'] = df[[ 'ET_03_x', 'ET_04_x',
              'ET_05_x', 'ET_06_x', 'ET_07_x', 'ET_08_x',
              'ET_09_x', 'ET_10_x', 'ET_11_x']]\
                         .apply(lambda x: x.sum(), axis=1)/9
    return df[['ET']]

def getAlbedo(df):
    df['Albedo'] = df[[ 'Albedo_03_x', 'Albedo_04_x',
              'Albedo_05_x', 'Albedo_06_x', 'Albedo_07_x', 'Albedo_08_x',
              'Albedo_09_x', 'Albedo_10_x', 'Albedo_11_x']]\
                         .apply(lambda x: x.sum(), axis=1) / 9

    return df[['Albedo']]

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

roiBJ = os.path.join(root2,'Beijiang.csv')
roiNJ = os.path.join(root2,'Nanjiang.csv')
roiHX = os.path.join(root2,'Hexi.csv')

####################################################################################
####################################################################################
#LST
file_f1 = root + r'\02forest2c\LST.csv'
file_g1 = root + r'\03grass2c\LST.csv'
file_u1 = root + r'\04unuse2c\LST.csv'

file_f2 = root + r'\02forest2c\ET.csv'
file_g2 = root + r'\03grass2c\ET.csv'
file_u2 = root + r'\04unuse2c\ET.csv'

file_f3 = root + r'\02forest2c\Albedo.csv'
file_g3 = root +  r'\03grass2c\Albedo.csv'
file_u3 = root +  r'\04unuse2c\Albedo.csv'

dff = pd.merge(pd.merge(pd.read_csv(file_f1).dropna(),
                      pd.read_csv(file_f2).dropna(), on='Id'),
                  pd.read_csv(file_f3).dropna(), on='Id')
# print('dff(merge)=\n',dff)
dfg = pd.merge(pd.merge(pd.read_csv(file_g1).dropna(),
                      pd.read_csv(file_g2).dropna(), on='Id'),
                  pd.read_csv(file_g3).dropna(), on='Id')

dfu = pd.merge(pd.merge(pd.read_csv(file_u1).dropna(),
                      pd.read_csv(file_u2).dropna(), on='Id'),
                  pd.read_csv(file_u3).dropna(), on='Id')

dff_BJ,dff_NJ,dff_HX,dfg_BJ,dfg_NJ,dfg_HX,dfu_BJ,dfu_NJ,dfu_HX\
    = getDF(dff,dfg,dfu,roiBJ,roiNJ,roiHX)

varLSTday  = ['LST_Day_01_x','LST_Day_02_x','LST_Day_03_x','LST_Day_04_x',
              'LST_Day_05_x','LST_Day_06_x','LST_Day_07_x','LST_Day_08_x',
              'LST_Day_09_x','LST_Day_10_x','LST_Day_11_x','LST_Day_12_x']
varLSTnight = ['LST_Night_01_x','LST_Night_02_x','LST_Night_03_x','LST_Night_04_x',
              'LST_Night_05_x','LST_Night_06_x','LST_Night_07_x','LST_Night_08_x',
              'LST_Night_09_x','LST_Night_10_x','LST_Night_11_x','LST_Night_12_x']

# print('dff_BJ(LST)=\n',dff_BJ)

LSTDay_f2c =   [getDayLST(dff_BJ),      getDayLST(dff_NJ),  getDayLST(dff_HX)]
LSTNight_f2c = [getNightLST(dff_BJ),  getNightLST(dff_NJ),getNightLST(dff_HX)]

LSTDay_g2c   =   [getDayLST(dfg_BJ),    getDayLST(dfg_NJ),  getDayLST(dfg_HX)]
LSTNight_g2c = [getNightLST(dfg_BJ),  getNightLST(dfg_NJ),getNightLST(dfg_HX)]

LSTDay_u2c   =   [getDayLST(dfu_BJ),    getDayLST(dfu_NJ),  getDayLST(dfu_HX)]
LSTNight_u2c = [getNightLST(dfu_BJ),  getNightLST(dfu_NJ),getNightLST(dfu_HX)]


############################################################


######################################################################
######################################################################
#######################################################################
#ET
varET  = ['ET_03_x','ET_04_x',
          'ET_05_x','ET_06_x','ET_07_x','ET_08_x',
          'ET_09_x','ET_10_x','ET_11_x']

ET_f2c   =   [getET(dff_BJ),      getET(dff_NJ),  getET(dff_HX)]
ET_g2c   =   [getET(dfg_BJ),      getET(dfg_NJ),  getET(dfg_HX)]
ET_u2c   =   [getET(dfu_BJ),      getET(dfu_NJ),  getET(dfu_HX)]



#################################################################
#Albedo

varAlbedo = ['Albedo_01_x','Albedo_02_x','Albedo_03_x','Albedo_04_x',
             'Albedo_05_x','Albedo_06_x','Albedo_07_x','Albedo_08_x',
             'Albedo_09_x','Albedo_10_x','Albedo_11_x','Albedo_12_x',]




Albedo_f2c = [getAlbedo(dff_BJ),  getAlbedo(dff_NJ),  getAlbedo(dff_HX)]
Albedo_g2c = [getAlbedo(dfg_BJ),  getAlbedo(dfg_NJ),  getAlbedo(dfg_HX)]
Albedo_u2c = [getAlbedo(dfu_BJ),  getAlbedo(dfu_NJ),  getAlbedo(dfu_HX)]


def DrawsubPlots(ax,X,Y,No,title,color,xlabel,ylabel,ylim,fenqu):
    plt.sca(ax)
    plt.scatter(X,Y, s=10, c=color, marker='.', alpha=0.95)

    a, b, Yp, R2 = LinearRegressionFunc(X, Y)
    r, sig = scipy.stats.pearsonr(X, Y)
    # correlation = np.corrcoef(X, Y)
    parameter = np.polyfit(X, Y, 1)
    f = np.poly1d(parameter)
    plt.plot(X, f(X), ls='-', c='r',lw=1)
    plt.text(0.03,0.88,No,transform=ax.transAxes)
    plt.text(0.28, 1.04,title, transform=ax.transAxes)
    if (b < 0):
        plt.text(0.05, 0.20, "y=%.2fx%.2f" % (a, b),
                 transform=ax.transAxes, c='k')
    else:
        plt.text(0.05, 0.20, "y=%.2fx+%.2f" % (a, b),
                 transform=ax.transAxes, c='k')
    plt.text(0.05, 0.04, "r=%.2f, p=%.3f" % (r, sig),
             transform=ax.transAxes, c='k')
    plt.text(1.03, 0.0, fenqu, transform=ax.transAxes, rotation=270,)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

colors = ['#003f5c', '#bc5090', '#ffa600']
plt.rc('font',size = 12)#, family='Times New Roman'
#########################################################################
#########################################################################
#plot the scatterplot of LSTDay vs. ET
fig = plt.figure(figsize=(8.0, 5.0))
fig.subplots_adjust(left=0.1, right=0.965,
                    bottom=0.1, top=0.95,
                    wspace=0.25, hspace=0.22)

ax1 = plt.subplot(331)
ax2 = plt.subplot(332)
ax3 = plt.subplot(333)
ax4 = plt.subplot(334)
ax5 = plt.subplot(335)
ax6 = plt.subplot(336)
ax7 = plt.subplot(337)
ax8 = plt.subplot(338)
ax9 = plt.subplot(339)


colors = ['#003f5c', '#bc5090', '#ffa600']  # 蓝紫黄
DrawsubPlots(ax1,ET_f2c[0].iloc[:,0].to_list(),
             LSTDay_f2c[0].iloc[:,0].to_list(),'(a)','forest2crop',colors[0],'','',[-4,4],'')
DrawsubPlots(ax4,ET_f2c[1].iloc[:,0].to_list(),
             LSTDay_f2c[1].iloc[:,0].to_list(),'(d)','',colors[1],'','$\Delta$LST (K)',[-4,4],'')
DrawsubPlots(ax7,ET_f2c[2].iloc[:,0].to_list(),
             LSTDay_f2c[2].iloc[:,0].to_list(),'(g)','',colors[2],'','',[-2,2],'')
DrawsubPlots(ax2,ET_g2c[0].iloc[:,0].to_list(),
             LSTDay_g2c[0].iloc[:,0].to_list(),'(b)','grass2crop',colors[0],'','',[-8,4],'')
DrawsubPlots(ax5,ET_g2c[1].iloc[:,0].to_list(),
             LSTDay_g2c[1].iloc[:,0].to_list(),'(e)','',colors[1],'','',[-7,4],'')
DrawsubPlots(ax8,ET_g2c[2].iloc[:,0].to_list(),
             LSTDay_g2c[2].iloc[:,0].to_list(),'(h)','',colors[2],'$\Delta$ET (mm)','',[-5,3],'')
DrawsubPlots(ax3,ET_u2c[0].iloc[:,0].to_list(),
             LSTDay_u2c[0].iloc[:,0].to_list(),'(c)','unuse2crop',colors[0],'','',[-6,1],'North Xinjiang')
DrawsubPlots(ax6,ET_u2c[1].iloc[:,0].to_list(),
             LSTDay_u2c[1].iloc[:,0].to_list(),'(f)','',colors[1],'','',[-6,3],'South Xinjiang')
DrawsubPlots(ax9,ET_u2c[2].iloc[:,0].to_list(),
             LSTDay_u2c[2].iloc[:,0].to_list(),'(i)','',colors[2],'','',[-6,3],'Hexi Corridor')

# plt.savefig(outname1)
#########################################################################
#########################################################################
#plot the scatterplot of LSTNight vs. ET
fig = plt.figure(figsize=(8.0, 5.0))
fig.subplots_adjust(left=0.08, right=0.965,
                    bottom=0.1, top=0.95,
                    wspace=0.25, hspace=0.22)
plt.rc('font',size = 12)#, family='Times New Roman'

ax1 = plt.subplot(331)
ax2 = plt.subplot(332)
ax3 = plt.subplot(333)
ax4 = plt.subplot(334)
ax5 = plt.subplot(335)
ax6 = plt.subplot(336)
ax7 = plt.subplot(337)
ax8 = plt.subplot(338)
ax9 = plt.subplot(339)



DrawsubPlots(ax1,ET_f2c[0].iloc[:,0].to_list(),LSTNight_f2c[0].iloc[:,0].to_list(),'(a)',
             'forest2crop',colors[0],'','',[-2,1],'')
DrawsubPlots(ax4,ET_f2c[1].iloc[:,0].to_list(),LSTNight_f2c[1].iloc[:,0].to_list(),'(d)',
             '',colors[1],'','$\Delta$LST (K)',[-2,1.5],'')
DrawsubPlots(ax7,ET_f2c[2].iloc[:,0].to_list(),LSTNight_f2c[2].iloc[:,0].to_list(),'(g)',
             '',colors[2],'','',[-0.7,0.5],'')
DrawsubPlots(ax2,ET_g2c[0].iloc[:,0].to_list(),LSTNight_g2c[0].iloc[:,0].to_list(),'(b)',
             'grass2crop',colors[0],'','',[-3,1.5],'')
DrawsubPlots(ax5,ET_g2c[1].iloc[:,0].to_list(),LSTNight_g2c[1].iloc[:,0].to_list(),'(e)',
             '',colors[1],'','',[-2,2],'')
DrawsubPlots(ax8,ET_g2c[2].iloc[:,0].to_list(),LSTNight_g2c[2].iloc[:,0].to_list(),'(h)',
             '',colors[2],'$\Delta$ET (mm)','',[-1.8,1.1],'')
DrawsubPlots(ax3,ET_u2c[0].iloc[:,0].to_list(),LSTNight_u2c[0].iloc[:,0].to_list(),'(c)',
             'unuse2crop',colors[0],'','',[-2,1.2],'North Xinjiang')
DrawsubPlots(ax6,ET_u2c[1].iloc[:,0].to_list(),LSTNight_u2c[1].iloc[:,0].to_list(),'(f)',
             '',colors[1],'','',[-2.5,2],'South Xinjiang')
DrawsubPlots(ax9,ET_u2c[2].iloc[:,0].to_list(),LSTNight_u2c[2].iloc[:,0].to_list(),'(i)',
             '',colors[2],'','',[-2.5,2],'Hexi Corridor')


# plt.savefig(outname2)

plt.show()
