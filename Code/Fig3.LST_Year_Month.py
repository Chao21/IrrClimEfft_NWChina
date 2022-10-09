#--coding:utf-8--
# Zhang Chao 2022/4/10
# Yearly and monthly potential changes with 95% confidence interval

import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np


Dir     = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root    = Dir + r'\Potential\10km'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname = outdir + r'\Fig3.svg'

file_f   =  root  +  r'\forest2c.csv'
file_g   =  root  +  r'\grass2c.csv'
file_u   =  root  +  r'\unuse2c.csv'

file_fr  =  root  +  r'\forest2c_ref.csv'
file_gr  =  root  +  r'\grass2c_ref.csv'
file_ur  =  root  +  r'\unuse2c_ref.csv'

dff = pd.merge(pd.read_csv(file_f).dropna(),
               pd.read_csv(file_fr).dropna(),on='Id')
dfg = pd.merge(pd.read_csv(file_g).dropna(),
               pd.read_csv(file_gr).dropna(),on='Id')
dfu = pd.merge(pd.read_csv(file_u).dropna(),
               pd.read_csv(file_ur).dropna(),on='Id')

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
    # eliminate the topographic effect on LST changes
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

# monthly data 95% confidence interval
def getSE(data):
    count = data.shape[0] #len(df): row; df.shape[1]: columns
    return data.std()/math.sqrt(count)*1.96

# yearly data 95% confidence interval
def getSE2(data):
    count = data.shape[0]*data.shape[1] #len(df): row; df.shape[1]: columns
    return np.std(data.values.flatten())/math.sqrt(count)*1.96



dff = getLST(dff)
dfg = getLST(dfg)
dfu = getLST(dfu)

DayLST   = [getDayLST(dff),getDayLST(dfg),getDayLST(dfu)]
NightLST = [getNightLST(dff),getNightLST(dfg),getNightLST(dfu)]

####################################################################
# get the monthly value
DayLSTmonth       = [list(DayLST[0].mean()),
                     list(DayLST[1].mean()),
                     list(DayLST[2].mean())]
DayLSTmonthstd    = [getSE(DayLST[0]),
                     getSE(DayLST[1]),
                     getSE(DayLST[2])]

NightLSTmonth       = [list(NightLST[0].mean()),
                       list(NightLST[1].mean()),
                       list(NightLST[2].mean())]
NightLSTmonthstd    = [getSE(NightLST[0]),
                       getSE(NightLST[1]),
                       getSE(NightLST[2])]

###################################################################
# get the yearly value
DayLSTyear       = [np.average(list(DayLST[0].mean())),
                     np.average(list(DayLST[1].mean())),
                     np.average(list(DayLST[2].mean()))]

DayLSTyearstd    =  [getSE2(DayLST[0]),
                     getSE2(DayLST[1]),
                     getSE2(DayLST[2])]

NightLSTyear       =  [np.average(list(NightLST[0].mean())),
                       np.average(list(NightLST[1].mean())),
                       np.average(list(NightLST[2].mean()))]
NightLSTyearstd    =  [getSE2(NightLST[0]),
                       getSE2(NightLST[1]),
                       getSE2(NightLST[2])]
#


Yearmean    = [DayLSTyear,NightLSTyear]#,DtrLSTyear
Yearmeanstd = [DayLSTyearstd,NightLSTyearstd]#,DtrLSTyearstd


#######################################################################
#plot figure function

def DrawYearLSTPlots(ax,data,se,title):
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
    colors = ['#2A557F', '#45BC9C', '#F05073']  # blue green red
    plt.bar(x1, y1, alpha=1.0, width=1, color=colors[0], label="forest2crop", yerr=SE1, error_kw=err_attri,zorder=100)
    plt.bar(x2, y2, alpha=1.0, width=1, color=colors[1], label="grass2crop",  yerr=SE2, error_kw=err_attri,zorder=100)
    plt.bar(x3, y3, alpha=1.0, width=1, color=colors[2], label="unuse2crop",  yerr=SE3, error_kw=err_attri,zorder=100)
    plt.xticks(x2, labelx)
    plt.text(0.02, 0.94, title, transform=ax.transAxes)
    plt.legend(frameon=False)#ncol=3, loc='right',bbox_to_anchor=(0.66,0.14),
    plt.ylabel('$\Delta$LST (K)',labelpad=0.01)#$^\circ$C
    plt.xticks(x2, labelx, rotation=0)
    plt.grid(zorder=0, linestyle='--', axis='y')  # show the grid line


def DrawMonthLSTPlots(ax,data,se,title,ylim,ylabel,xlabflag,legendFlag):
    x = range(1, 13)
    plt.sca(ax)
    plt.plot(x, data[0], c=colors[0], label='forest2crop')  # c= 'k',
    plt.fill_between(x, data[0] - se[0], data[0] + se[0], color=colors[0], alpha=0.2)

    plt.plot(x, data[1], c=colors[1], label='grass2crop')
    plt.fill_between(x, data[1] - se[1], data[1] + se[1], color=colors[1], alpha=0.2)

    plt.plot(x, data[2], c=colors[2], label='unuse2crop')
    plt.fill_between(x, data[2] - se[2], data[2] + se[2], color=colors[2], alpha=0.2)

    plt.ylim(ylim)
    if (xlabflag == True):
        plt.xticks(x,['Jan','Feb','Mar','Apr','May','Jun',
                      'Jul','Aug','Sep','Oct','Nov','Dec'],rotation = 90)
    else:
        plt.xticks(x, [])
    plt.text(0.03,0.08,title, transform = ax.transAxes)
    if(legendFlag==True):
        plt.legend(loc='right',bbox_to_anchor=(1.0,0.3),frameon = False)
    plt.ylabel(ylabel)


#######################################################################
#plot figure
# plt.style.use('ggplot')
plt.rc('font',size = 12)#, family='Times New Roman'
fig = plt.figure(figsize=(8,3))#dpi=100, default dpi
fig.subplots_adjust(left = 0.15, right = 0.99,
                    bottom = 0.10, top = 0.98,
                    wspace= 0.35, hspace=0.05)
colors = ['#2A557F', '#45BC9C', '#F05073']  # blue green red


ax1 = fig.add_axes([0.08,0.10,0.40,0.9])
ax2 = fig.add_axes([0.54,0.58,0.45,0.42])
ax3 = fig.add_axes([0.54,0.15,0.45,0.42])

yticks_label_et = [[0.0,0.5,1.0,1.5,2.0],
                   [0.0,0.5,1.0,1.5,2.0]]

yticks_label_al = [[-0.020,-0.010,0.000,0.010],
                   [-2,-1,0,1]]

ylim1 = [-10,1]
ylim2 = [-2.5,1]

DrawYearLSTPlots(ax1,Yearmean,Yearmeanstd,'(a) Annual')
DrawMonthLSTPlots(ax2,DayLSTmonth,DayLSTmonthstd,'(b) Daytime',ylim1,'',False,False)
DrawMonthLSTPlots(ax3,NightLSTmonth,NightLSTmonthstd,'(c) Nighttime',ylim2,'',True,False)


# plt.savefig(outname)

plt.show()
