#--coding:utf-8--
# Zhang Chao 2022/4/11
# Sensitivity test of potential LST changes
# 10, 20, 30 km

import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

Dir  = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname = outdir + r'\FigS3.svg'

root1 = Dir + r'\Potential\10km'
root2 = Dir + r'\Potential\15km'
root3 = Dir + r'\Potential\20km'


def getLST(df):
    crop =    df[[  'LST_Day_01_x',   'LST_Day_02_x',   'LST_Day_03_x',   'LST_Day_04_x',
                    'LST_Day_05_x',   'LST_Day_06_x',   'LST_Day_07_x',   'LST_Day_08_x',
                    'LST_Day_09_x',   'LST_Day_10_x',   'LST_Day_11_x',   'LST_Day_12_x',
                  'LST_Night_01_x', 'LST_Night_02_x', 'LST_Night_03_x', 'LST_Night_04_x',
                  'LST_Night_05_x', 'LST_Night_06_x', 'LST_Night_07_x', 'LST_Night_08_x',
                  'LST_Night_09_x', 'LST_Night_10_x', 'LST_Night_11_x', 'LST_Night_12_x',
                            'ele_x'
                  ]]
    ref =     df[[  'LST_Day_01_y',   'LST_Day_02_y',   'LST_Day_03_y',   'LST_Day_04_y',
                    'LST_Day_05_y',   'LST_Day_06_y',   'LST_Day_07_y',   'LST_Day_08_y',
                    'LST_Day_09_y',   'LST_Day_10_y',   'LST_Day_11_y',   'LST_Day_12_y',
                  'LST_Night_01_y', 'LST_Night_02_y', 'LST_Night_03_y', 'LST_Night_04_y',
                  'LST_Night_05_y', 'LST_Night_06_y', 'LST_Night_07_y', 'LST_Night_08_y',
                  'LST_Night_09_y', 'LST_Night_10_y', 'LST_Night_11_y', 'LST_Night_12_y',
                           'ele_y'
                  ]]
    ref.columns = crop.columns
    delta = crop - ref
    delta1 = delta[(delta['ele_x'] < 50) & (delta['ele_x'] > -50)]
    return delta1


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



def getSE(data):
    count = data.shape[0]*data.shape[1] #len(df): row; df.shape[1]: columns
    return np.std(data.values.flatten())/math.sqrt(count)*1.96


###############################################################################
def getYearLST(root):
    file_f = root + r'\forest2c.csv'
    file_g = root + r'\grass2c.csv'
    file_u = root + r'\unuse2c.csv'

    file_fr = root + r'\forest2c_ref.csv'
    file_gr = root + r'\grass2c_ref.csv'
    file_ur = root + r'\unuse2c_ref.csv'

    dff = pd.merge(pd.read_csv(file_f).dropna(),
                   pd.read_csv(file_fr).dropna(), on='Id')
    dfg = pd.merge(pd.read_csv(file_g).dropna(),
                   pd.read_csv(file_gr).dropna(), on='Id')
    dfu = pd.merge(pd.read_csv(file_u).dropna(),
                   pd.read_csv(file_ur).dropna(), on='Id')


    dff = getLST(dff)
    dfg = getLST(dfg)
    dfu = getLST(dfu)

    DayLST = [getDayLST(dff), getDayLST(dfg), getDayLST(dfu)]

    NightLST = [getNightLST(dff), getNightLST(dfg), getNightLST(dfu)]

    ###################################################################
    DayLSTyear = [np.average(list(DayLST[0].mean())),
                  np.average(list(DayLST[1].mean())),
                  np.average(list(DayLST[2].mean()))]

    DayLSTyearstd = [getSE(DayLST[0]),
                     getSE(DayLST[1]),
                     getSE(DayLST[2])]

    NightLSTyear = [np.average(list(NightLST[0].mean())),
                    np.average(list(NightLST[1].mean())),
                    np.average(list(NightLST[2].mean()))]
    NightLSTyearstd = [getSE(NightLST[0]),
                       getSE(NightLST[1]),
                       getSE(NightLST[2])]

    ##******************************************************************
    Yearmean = [DayLSTyear, NightLSTyear]  # ,DtrLSTyear
    Yearmeanstd = [DayLSTyearstd, NightLSTyearstd]  # ,DtrLSTyearstd
    return Yearmean, Yearmeanstd


###&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def DrawLSTPlots(data,se,ax,title,ylabflag):
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
    # print(y1,y2,y3)

    labelx = ['Daytime', 'Nighttime']
    err_attri = dict(elinewidth=.8, ecolor='gray', capsize=1)
    colors = ['#2A557F', '#45BC9C', '#F05073']
    plt.bar(x1, y1, alpha=1.0, width=1, color=colors[0], label="forest2crop", yerr=SE1, error_kw=err_attri,zorder=100)
    plt.bar(x2, y2, alpha=1.0, width=1, color=colors[1], label="grass2crop",  yerr=SE2, error_kw=err_attri,zorder=100)
    plt.bar(x3, y3, alpha=1.0, width=1, color=colors[2], label="unuse2crop",  yerr=SE3, error_kw=err_attri,zorder=100)
    plt.xticks(x2, labelx)
    plt.text(0.30, 1.02, title, transform=ax.transAxes)
    plt.ylim([-6, 1])
    if (ylabflag == True):
        plt.yticks([-6, -4, -2,0],[-6, -4, -2,0])
        plt.ylabel('$\Delta$LST (K)')
    else:
        plt.yticks([-6, -4, -2,0],[])
    if (ylabflag == True):
        plt.legend(loc='right',bbox_to_anchor=(1.0,0.14),frameon=False)
    plt.xticks(x2, labelx, rotation=0)
    plt.grid(zorder=0, linestyle='--', axis='y')  #

#######################################################################

Yearmean_10, Yearmeanstd_10    = getYearLST(root1)

Yearmean_15, Yearmeanstd_15    = getYearLST(root2)

Yearmean_20, Yearmeanstd_20    = getYearLST(root3)
##******************************************************************

#plot figure
# plt.style.use('ggplot')
plt.rc('font',size = 12)#, family='Times New Roman'
fig = plt.figure(figsize=(8,3))#dpi=100, default dpi
fig.subplots_adjust(left = 0.07, right = 0.99,
                    bottom = 0.10, top = 0.92,
                    wspace= 0.10, hspace=0.05)
colors = ['#2A557F', '#45BC9C', '#F05073']

ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

DrawLSTPlots(Yearmean_10,Yearmeanstd_10,ax1,'(a) 10 x 10',True)
DrawLSTPlots(Yearmean_15,Yearmeanstd_15,ax2,'(b) 15 x 15',False)
DrawLSTPlots(Yearmean_20,Yearmeanstd_20,ax3,'(c) 20 x 20',False)

# plt.savefig(outname)

plt.show()
