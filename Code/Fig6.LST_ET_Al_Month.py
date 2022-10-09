#--coding:utf-8--
# Zhang Chao 2022/4/8
# Plot actual monthly changes in LST, ET

import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np


Dir     = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root = Dir + r'\Actual'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname = outdir + r'\Fig6.svg'

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


Daymeanf = dff[['LST_Day_01_x','LST_Day_02_x','LST_Day_03_x','LST_Day_04_x',
             'LST_Day_05_x','LST_Day_06_x','LST_Day_07_x','LST_Day_08_x',
             'LST_Day_09_x','LST_Day_10_x','LST_Day_11_x','LST_Day_12_x']]


Daymeang = dfg[['LST_Day_01_x','LST_Day_02_x','LST_Day_03_x','LST_Day_04_x',
             'LST_Day_05_x','LST_Day_06_x','LST_Day_07_x','LST_Day_08_x',
             'LST_Day_09_x','LST_Day_10_x','LST_Day_11_x','LST_Day_12_x']]


Daymeanu = dfu[['LST_Day_01_x','LST_Day_02_x','LST_Day_03_x','LST_Day_04_x',
             'LST_Day_05_x','LST_Day_06_x','LST_Day_07_x','LST_Day_08_x',
             'LST_Day_09_x','LST_Day_10_x','LST_Day_11_x','LST_Day_12_x']]

Daymean = [Daymeanf,Daymeang,Daymeanu]

Nightmeanf = dff[['LST_Night_01_x','LST_Night_02_x','LST_Night_03_x','LST_Night_04_x',
             'LST_Night_05_x','LST_Night_06_x','LST_Night_07_x','LST_Night_08_x',
             'LST_Night_09_x','LST_Night_10_x','LST_Night_11_x','LST_Night_12_x']]


Nightmeang = dfg[['LST_Night_01_x','LST_Night_02_x','LST_Night_03_x','LST_Night_04_x',
             'LST_Night_05_x','LST_Night_06_x','LST_Night_07_x','LST_Night_08_x',
             'LST_Night_09_x','LST_Night_10_x','LST_Night_11_x','LST_Night_12_x']]


Nightmeanu = dfu[['LST_Night_01_x','LST_Night_02_x','LST_Night_03_x','LST_Night_04_x',
             'LST_Night_05_x','LST_Night_06_x','LST_Night_07_x','LST_Night_08_x',
             'LST_Night_09_x','LST_Night_10_x','LST_Night_11_x','LST_Night_12_x']]



Nightmean = [Nightmeanf,Nightmeang,Nightmeanu]


Nightmeanf.columns = Daymeanf.columns
Nightmeang.columns = Daymeang.columns
Nightmeanu.columns = Daymeanu.columns

Dailymeanf = (Daymeanf + Nightmeanf)/2
Dailymeang = (Daymeang + Nightmeang)/2
Dailymeanu = (Daymeanu + Nightmeanu)/2

Dailymean = [Dailymeanf,Dailymeang,Dailymeanu]


def getSE(data):
    count = data.shape[0] #len(df): row; df.shape[1]: columns
    return data.std()/math.sqrt(count)*1.96

def DrawsubPlots(ax1,mean,title,ylabflag,legendFlag):
    plt.sca(ax1)
    width = 1.0
    x1 = np.arange(1,1+width*4*12,width*4)
    x2 = [i + width*1 for i in x1]
    x3 = [i + width*2 for i in x1]
    y1 = list(mean[0].mean())
    y2 = list(mean[1].mean())
    y3 = list(mean[2].mean())
    SE = [list(getSE(i)) for i in mean]

    labelx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    err_attri = dict(elinewidth=.8, ecolor='gray', capsize=1)
    colors = ['#2A557F', '#45BC9C', '#F05073']
    plt.bar(x1, y1, alpha=1.0, width=1, color=colors[0], label="forest2crop", yerr=SE[0], error_kw=err_attri,zorder=100)
    plt.bar(x2, y2, alpha=1.0, width=1, color=colors[1], label="grass2crop",  yerr=SE[1], error_kw=err_attri,zorder=100)
    plt.bar(x3, y3, alpha=1.0, width=1, color=colors[2], label="unuse2crop",  yerr=SE[2], error_kw=err_attri,zorder=100)
    plt.xticks(x2, labelx,rotation = 90)
    plt.grid(zorder=0, linestyle='--', axis='y')



    plt.text(0.01,0.90,title,transform=ax1.transAxes)#,style='italic'
    plt.xticks(x2,labelx)
    plt.ylabel(ylabflag,labelpad=0.01)
    if(legendFlag==True):
        plt.legend(loc='center left',bbox_to_anchor=(-0.02,0.20),frameon=False)#


##############################################################################################
# ET Albedo
ETmeanf = dff[['ET_01_x','ET_02_x','ET_03_x','ET_04_x',
             'ET_05_x','ET_06_x','ET_07_x','ET_08_x',
             'ET_09_x','ET_10_x','ET_11_x','ET_12_x']]

ETmeang = dfg[['ET_01_x','ET_02_x','ET_03_x','ET_04_x',
             'ET_05_x','ET_06_x','ET_07_x','ET_08_x',
             'ET_09_x','ET_10_x','ET_11_x','ET_12_x']]

ETmeanu = dfu[['ET_01_x','ET_02_x','ET_03_x','ET_04_x',
             'ET_05_x','ET_06_x','ET_07_x','ET_08_x',
             'ET_09_x','ET_10_x','ET_11_x','ET_12_x']]

ETmean = [ETmeanf,ETmeang,ETmeanu]

#################################################################################
Albedomeanf = dff[['Albedo_01_x','Albedo_02_x','Albedo_03_x','Albedo_04_x',
             'Albedo_05_x','Albedo_06_x','Albedo_07_x','Albedo_08_x',
             'Albedo_09_x','Albedo_10_x','Albedo_11_x','Albedo_12_x']]


Albedomeang = dfg[['Albedo_01_x','Albedo_02_x','Albedo_03_x','Albedo_04_x',
             'Albedo_05_x','Albedo_06_x','Albedo_07_x','Albedo_08_x',
             'Albedo_09_x','Albedo_10_x','Albedo_11_x','Albedo_12_x']]


Albedomeanu = dfu[['Albedo_01_x','Albedo_02_x','Albedo_03_x','Albedo_04_x',
             'Albedo_05_x','Albedo_06_x','Albedo_07_x','Albedo_08_x',
             'Albedo_09_x','Albedo_10_x','Albedo_11_x','Albedo_12_x']]

Albedomean = [Albedomeanf,Albedomeang,Albedomeanu]

#####################################################################################
# plt.style.use('ggplot')
plt.rc('font',size = 12)#, family='Times New Roman'
fig = plt.figure(figsize=(8,4))#dpi=100, default dpi
fig.subplots_adjust(left = 0.07, right=0.999,
                    bottom = 0.12, top = 0.99,
                    wspace= 0.25, hspace=0.05)
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

DrawsubPlots(ax1,Daymean,'(a) Daytime $\Delta$LST','$\Delta$LST (K)',True)#$^\circ$C
DrawsubPlots(ax2,Nightmean,'(b) Nighttime $\Delta$LST','$\Delta$LST (K)',False)
DrawsubPlots(ax3,ETmean,'(c) $\Delta$ET','$\Delta$ET (mm)',False)
DrawsubPlots(ax4,Albedomean,'(d) $\Delta$Albedo','$\Delta$Albedo',False)

plt.savefig(outname)
plt.show()
