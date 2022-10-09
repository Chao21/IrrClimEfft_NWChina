#--coding:utf-8--
# Zhang Chao 2022/4/8
# Plot actual yearly changes in LST, ET, Albedo

import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

Dir     = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root = Dir + r'\Actual'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname = outdir + r'\Fig5.svg'

dff_LST    = pd.read_csv(root +  r'\02forest2c\LST.csv')
dfg_LST    = pd.read_csv(root +  r'\03grass2c\LST.csv')
dfu_LST    = pd.read_csv(root +  r'\04unuse2c\LST.csv')
dff_ET     = pd.read_csv(root +  r'\02forest2c\ET.csv')
dfg_ET     = pd.read_csv(root +  r'\03grass2c\ET.csv')
dfu_ET     = pd.read_csv(root +  r'\04unuse2c\ET.csv')
dff_Albedo = pd.read_csv(root +  r'\02forest2c\Albedo.csv')
dfg_Albedo = pd.read_csv(root +  r'\03grass2c\Albedo.csv')
dfu_Albedo = pd.read_csv(root +  r'\04unuse2c\Albedo.csv')

dff = pd.merge((pd.merge(dff_LST.dropna(),dff_ET.dropna(),on='Id'))
               ,dff_Albedo.dropna(),on='Id')
dfg = pd.merge((pd.merge(dfg_LST.dropna(),dfg_ET.dropna(),on='Id'))
               ,dfg_Albedo.dropna(),on='Id')
dfu = pd.merge((pd.merge(dfu_LST.dropna(),dfu_ET.dropna(),on='Id'))
               ,dfu_Albedo.dropna(),on='Id')


def getDayLST(df):
    LST = df[['LST_Day_01_x', 'LST_Day_02_x', 'LST_Day_03_x', 'LST_Day_04_x',
              'LST_Day_05_x', 'LST_Day_06_x', 'LST_Day_07_x', 'LST_Day_08_x',
              'LST_Day_09_x', 'LST_Day_10_x', 'LST_Day_11_x', 'LST_Day_12_x']]
    return LST

def getNightLST(df):
    LST = df[['LST_Night_01_x', 'LST_Night_02_x', 'LST_Night_03_x', 'LST_Night_04_x',
              'LST_Night_05_x', 'LST_Night_06_x', 'LST_Night_07_x', 'LST_Night_08_x',
              'LST_Night_09_x', 'LST_Night_10_x', 'LST_Night_11_x', 'LST_Night_12_x']]
    return LST

def getDailyLST(dayLST, nightLST):
    nightLST.columns = dayLST.columns
    return (dayLST + nightLST)/2

def getDtrLST(dayLST, nightLST):
    nightLST.columns = dayLST.columns
    return (dayLST - nightLST)

# discard the negligible winter ET data because of many empty MODIS ET values
def getET(df):
    ET = df[[ 'ET_03_x', 'ET_04_x',
              'ET_05_x', 'ET_06_x', 'ET_07_x', 'ET_08_x',
              'ET_09_x', 'ET_10_x', 'ET_11_x']]
    return ET

def getAlbedo(df):
    Albedo = df[['Albedo_01_x', 'Albedo_02_x', 'Albedo_03_x', 'Albedo_04_x',
              'Albedo_05_x', 'Albedo_06_x', 'Albedo_07_x', 'Albedo_08_x',
              'Albedo_09_x', 'Albedo_10_x', 'Albedo_11_x', 'Albedo_12_x']]
    return Albedo

DayLST   = [getDayLST(dff),getDayLST(dfg),getDayLST(dfu)]

NightLST = [getNightLST(dff),getNightLST(dfg),getNightLST(dfu)]

DailyLST = [getDailyLST(DayLST[0],NightLST[0]),
            getDailyLST(DayLST[1],NightLST[1]),
            getDailyLST(DayLST[2],NightLST[2]),]

DtrLST   = [getDtrLST(DayLST[0],NightLST[0]),
            getDtrLST(DayLST[1],NightLST[1]),
            getDtrLST(DayLST[2],NightLST[2]),]


### 95% confidence interval of yearly data
def getSE(data):
    count = data.shape[0]*data.shape[1] #len(df): row; df.shape[1]: columns
    return np.std(data.values.flatten())/math.sqrt(count)*1.96


###################################################################
DayLSTyear       = [np.average(list(DayLST[0].mean())),
                     np.average(list(DayLST[1].mean())),
                     np.average(list(DayLST[2].mean()))]

DayLSTyearstd    = [getSE(DayLST[0]),
                     getSE(DayLST[1]),
                     getSE(DayLST[2])]

NightLSTyear       = [np.average(list(NightLST[0].mean())),
                       np.average(list(NightLST[1].mean())),
                       np.average(list(NightLST[2].mean()))]
NightLSTyearstd    = [getSE(NightLST[0]),
                       getSE(NightLST[1]),
                       getSE(NightLST[2])]
#
DailyLSTyear     =  [np.average(list(DailyLST[0].mean())),
                      np.average(list(DailyLST[1].mean())),
                      np.average(list(DailyLST[2].mean()))]
DailyLSTyearstd  = [getSE(DailyLST[0]),
                     getSE(DailyLST[1]),
                     getSE(DailyLST[2])]

DtrLSTyear      =  [np.average(list(DtrLST[0].mean())),
                      np.average(list(DtrLST[1].mean())),
                      np.average(list(DtrLST[2].mean()))]
DtrLSTyearstd    = [getSE(DtrLST[0]),
                     getSE(DtrLST[1]),
                     getSE(DtrLST[2])]

YearLST    = [DayLSTyear,NightLSTyear]  #DailyLSTyear,
YearLSTstd = [DayLSTyearstd,NightLSTyearstd]  #DailyLSTyearstd,

####################################################################
ETyear       = [np.average(list(getET(dff).mean())),
                np.average(list(getET(dfg).mean())),
                np.average(list(getET(dfu).mean()))]

ETyearstd  = [getSE(getET(dff)),
              getSE(getET(dfg)),
              getSE(getET(dfu))]



Albedoyear   = [np.average(list(getAlbedo(dff).mean())),
                np.average(list(getAlbedo(dfg).mean())),
                np.average(list(getAlbedo(dfu).mean()))]

Albedoyearstd  = [getSE(getAlbedo(dff)),
                  getSE(getAlbedo(dfg)),
                  getSE(getAlbedo(dfu))]




def DrawsubPlots(data,se,ax,title,ylabel):
    plt.sca(ax)
    x1 = [1, 5]
    x2 = [i + 1 for i in x1]
    x3 = [i + 2 for i in x1]
    y1 = [i[0] for i in data]
    y2 = [i[1] for i in data]
    y3 = [i[2] for i in data]
    SE1 = [i[0] for i in se]
    SE2 = [i[1] for i in se]
    SE3 = [i[2] for i in se]

    labelx = ['Daytime', 'Nighttime']
    err_attri = dict(elinewidth=.8, ecolor='gray', capsize=1)
    colors = ['#2A557F', '#45BC9C', '#F05073']
    plt.bar(x1, y1, alpha=1, width=1, color=colors[0], label="forest2crop", yerr=SE1, error_kw=err_attri,zorder=100)
    plt.bar(x2, y2, alpha=1, width=1, color=colors[1], label="grass2crop",  yerr=SE2, error_kw=err_attri,zorder=100)
    plt.bar(x3, y3, alpha=1, width=1, color=colors[2], label="unuse2crop",  yerr=SE3, error_kw=err_attri,zorder=100)
    plt.text(0.02, 0.92, title, transform = ax.transAxes)
    plt.ylabel(ylabel,labelpad=0.05)
    plt.xticks(x2, labelx, rotation=0)
    plt.grid(zorder=0, linestyle='--', axis='y')

def DrawETAlbedoPlots(data,se,ax,title,ylabel,yticks,legendFlag):
    plt.sca(ax)
    x = [1,3,5]
    y = data
    labels = ["forest2crop","grass2crop","unuse2crop"]
    err_attri = dict(elinewidth=.8, ecolor='gray', capsize=1)
    plt.bar(x, y, alpha=1, width=1, color=colors,yerr=se, error_kw = err_attri,zorder=100)
    plt.bar(x, [0, 0, 0], alpha=1, width=1, color=colors[0], label = labels[0], error_kw=err_attri,zorder=100)
    plt.bar(x, [0, 0, 0], alpha=1, width=1, color=colors[1], label = labels[1], error_kw=err_attri,zorder=100)
    plt.bar(x, [0, 0, 0], alpha=1, width=1, color=colors[2], label = labels[2], error_kw=err_attri,zorder=100)
    plt.xticks(x,[])#'f2c','g2c','u2c'
    plt.text(0.02, 0.92, title, transform=ax.transAxes)
    plt.ylabel(ylabel,labelpad=0.01)
    plt.yticks(yticks[0],yticks[1])
    if (legendFlag==True):
        plt.legend(frameon=False,ncol=3, loc='right', bbox_to_anchor=(1.0, -0.08))

    plt.grid(zorder=0,linestyle='--',axis='y')

######################################################################################
#plot figure

plt.rc('font',size = 12)#, family='Times New Roman'
fig = plt.figure(figsize=(8,3))#dpi=100, default dpi
fig.subplots_adjust(left = 0.09, right = 0.99,
                    bottom = 0.10, top = 0.99,
                    wspace= 0.35, hspace=0.05)
colors = ['#2A557F', '#45BC9C', '#F05073']


ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

yticks_label_et = [[0.0,0.2,0.4,0.6,0.8,1.0],[0.0,0.2,0.4,0.6,0.8,1.0]]

yticks_label_al = [[-0.008,-0.006,-0.004,-0.002,0.000,0.002,0.004],
                   [-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4]]

DrawsubPlots(YearLST,YearLSTstd,ax1,'(a) $\Delta$LST','$\Delta$LST (K)')
DrawETAlbedoPlots(ETyear,ETyearstd,ax2,'(b) $\Delta$ET','$\Delta$ET (mm)',
                  yticks_label_et,False)
DrawETAlbedoPlots(Albedoyear,Albedoyearstd,ax3,'(c) $\Delta$Albedo','$\Delta$Albedo (%)',
                  yticks_label_al,True)

# plt.savefig(outname)

plt.show()