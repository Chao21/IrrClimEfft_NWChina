#--coding:utf-8--
# Zhang Chao 2022/4/17
# Yearly LST_ET_Albedo in North Xinjiang, South Xinjiang, and Hexi Corridor
#

import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np


Dir     = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root   = Dir + r'\ActualFenqu'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname = outdir + r'\Fig7.svg'

# 95% confidence interval of yearly data
def getSE(data):#
    count = data.shape[0]*data.shape[1] #len(df): row; df.shape[1]: columns
    return np.std(data.values.flatten())/math.sqrt(count)*1.96

def drawCaptionPlot(ax,title,pos):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc='#D9D9D9', edgecolor='k'))  # ,alpha=0.6)
    ax.text(pos[0],pos[1],title,transform=ax.transAxes,fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

####################################################################################
####################################################################################
#LST
filef = root + r'\forestGroup.csv'
fileg = root + r'\grassGroup.csv'
fileu = root + r'\unuseGroup.csv'

dff = pd.read_csv(filef)
dfg = pd.read_csv(fileg)
dfu = pd.read_csv(fileu)

dff_BJ = dff[dff['flag']==1]
dff_NJ = dff[dff['flag']==2]
dff_HX = dff[dff['flag']==3]

dfg_BJ = dfg[dfg['flag']==1]
dfg_NJ = dfg[dfg['flag']==2]
dfg_HX = dfg[dfg['flag']==3]

dfu_BJ = dfu[dfu['flag']==1]
dfu_NJ = dfu[dfu['flag']==2]
dfu_HX = dfu[dfu['flag']==3]

varLSTday  = ['LST_Day_01_x','LST_Day_02_x','LST_Day_03_x','LST_Day_04_x',
              'LST_Day_05_x','LST_Day_06_x','LST_Day_07_x','LST_Day_08_x',
              'LST_Day_09_x','LST_Day_10_x','LST_Day_11_x','LST_Day_12_x']
varLSTnight = ['LST_Night_01_x','LST_Night_02_x','LST_Night_03_x','LST_Night_04_x',
              'LST_Night_05_x','LST_Night_06_x','LST_Night_07_x','LST_Night_08_x',
              'LST_Night_09_x','LST_Night_10_x','LST_Night_11_x','LST_Night_12_x']



LSTDay_f2c =   [dff_BJ[varLSTday],  dff_NJ[varLSTday],  dff_HX[varLSTday]]
LSTNight_f2c = [dff_BJ[varLSTnight],dff_NJ[varLSTnight],dff_HX[varLSTnight]]

LSTDay_g2c =   [dfg_BJ[varLSTday],  dfg_NJ[varLSTday],  dfg_HX[varLSTday]]
LSTNight_g2c = [dfg_BJ[varLSTnight],dfg_NJ[varLSTnight],dfg_HX[varLSTnight]]

LSTDay_u2c =   [dfu_BJ[varLSTday],  dfu_NJ[varLSTday],  dfu_HX[varLSTday]]
LSTNight_u2c = [dfu_BJ[varLSTnight],dfu_NJ[varLSTnight],dfu_HX[varLSTnight]]

############################################################
YearLSTDay_f2c    = [np.average(list(LSTDay_f2c[0].mean())),
                     np.average(list(LSTDay_f2c[1].mean())),
                     np.average(list(LSTDay_f2c[2].mean()))]

YearLSTNight_f2c  = [np.average(list(LSTNight_f2c[0].mean())),
                     np.average(list(LSTNight_f2c[1].mean())),
                     np.average(list(LSTNight_f2c[2].mean()))]

YearLSTDay_g2c = [np.average(list(LSTDay_g2c[0].mean())),
                  np.average(list(LSTDay_g2c[1].mean())),
                  np.average(list(LSTDay_g2c[2].mean()))]

YearLSTNight_g2c = [np.average(list(LSTNight_g2c[0].mean())),
                    np.average(list(LSTNight_g2c[1].mean())),
                    np.average(list(LSTNight_g2c[2].mean()))]

YearLSTDay_u2c = [np.average(list(LSTDay_u2c[0].mean())),
                  np.average(list(LSTDay_u2c[1].mean())),
                  np.average(list(LSTDay_u2c[2].mean()))]

YearLSTNight_u2c = [np.average(list(LSTNight_u2c[0].mean())),
                    np.average(list(LSTNight_u2c[1].mean())),
                    np.average(list(LSTNight_u2c[2].mean()))]
############################################################
#95% confidence interval: std()/sqrt(n)*1.96
YearLSTDaySE_f2c    = [getSE(LSTDay_f2c[0]),
                       getSE(LSTDay_f2c[1]),
                       getSE(LSTDay_f2c[2])]

YearLSTDaySE_g2c    = [getSE(LSTDay_g2c[0]),
                       getSE(LSTDay_g2c[1]),
                       getSE(LSTDay_g2c[2])]

YearLSTDaySE_u2c    = [getSE(LSTDay_u2c[0]),
                       getSE(LSTDay_u2c[1]),
                       getSE(LSTDay_u2c[2])]

YearLSTNightSE_f2c  = [getSE(LSTNight_f2c[0]),
                       getSE(LSTNight_f2c[1]),
                       getSE(LSTNight_f2c[2])]

YearLSTNightSE_g2c  = [getSE(LSTNight_g2c[0]),
                       getSE(LSTNight_g2c[1]),
                       getSE(LSTNight_g2c[2])]

YearLSTNightSE_u2c  = [getSE(LSTNight_u2c[0]),
                       getSE(LSTNight_u2c[1]),
                       getSE(LSTNight_u2c[2])]
#

YearLST_f2c = [YearLSTDay_f2c, YearLSTNight_f2c]
YearLSTSE_f2c = [YearLSTDaySE_f2c, YearLSTNightSE_f2c]

YearLST_g2c = [YearLSTDay_g2c, YearLSTNight_g2c]
YearLSTSE_g2c = [YearLSTDaySE_g2c, YearLSTNightSE_g2c]

YearLST_u2c = [YearLSTDay_u2c, YearLSTNight_u2c]
YearLSTSE_u2c = [YearLSTDaySE_u2c, YearLSTNightSE_u2c]
######################################################################
######################################################################
#######################################################################
#ET

varET  = ['ET_03_x','ET_04_x',
          'ET_05_x','ET_06_x','ET_07_x','ET_08_x',
          'ET_09_x','ET_10_x','ET_11_x']

ET_f2c   =   [dff_BJ[varET],  dff_NJ[varET],  dff_HX[varET]]
ET_g2c   =   [dfg_BJ[varET],  dfg_NJ[varET],  dfg_HX[varET]]
ET_u2c   =   [dfu_BJ[varET],  dfu_NJ[varET],  dfu_HX[varET]]


YearET_f2c = [np.average(list(ET_f2c[0].mean())),
              np.average(list(ET_f2c[1].mean())),
              np.average(list(ET_f2c[2].mean()))]

YearET_g2c = [np.average(list(ET_g2c[0].mean())),
              np.average(list(ET_g2c[1].mean())),
              np.average(list(ET_g2c[2].mean()))]

YearET_u2c = [np.average(list(ET_u2c[0].mean())),
              np.average(list(ET_u2c[1].mean())),
              np.average(list(ET_u2c[2].mean()))]


YearETSE_f2c = [getSE(ET_f2c[0]),
                getSE(ET_f2c[1]),
                getSE(ET_f2c[2])]

YearETSE_g2c = [getSE(ET_g2c[0]),
                getSE(ET_g2c[1]),
                getSE(ET_g2c[2])]

YearETSE_u2c = [getSE(ET_u2c[0]),
                getSE(ET_u2c[1]),
                getSE(ET_u2c[2])]

#################################################################
#Albedo
varAlbedo = ['Albedo_01_x','Albedo_02_x','Albedo_03_x','Albedo_04_x',
             'Albedo_05_x','Albedo_06_x','Albedo_07_x','Albedo_08_x',
             'Albedo_09_x','Albedo_10_x','Albedo_11_x','Albedo_12_x',]

Albedo_f2c = [dff_BJ[varAlbedo],dff_NJ[varAlbedo],dff_HX[varAlbedo]]
Albedo_g2c = [dfg_BJ[varAlbedo],dfg_NJ[varAlbedo],dfg_HX[varAlbedo]]
Albedo_u2c = [dfu_BJ[varAlbedo],dfu_NJ[varAlbedo],dfu_HX[varAlbedo]]


YearAlbedo_f2c = [np.average(list(Albedo_f2c[0].mean())),
                  np.average(list(Albedo_f2c[1].mean())),
                  np.average(list(Albedo_f2c[2].mean()))]

YearAlbedo_g2c = [np.average(list(Albedo_g2c[0].mean())),
                  np.average(list(Albedo_g2c[1].mean())),
                  np.average(list(Albedo_g2c[2].mean()))]

YearAlbedo_u2c = [np.average(list(Albedo_u2c[0].mean())),
                  np.average(list(Albedo_u2c[1].mean())),
                  np.average(list(Albedo_u2c[2].mean()))]


YearAlbedoSE_f2c = [getSE(Albedo_f2c[0]),
                    getSE(Albedo_f2c[1]),
                    getSE(Albedo_f2c[2])]

YearAlbedoSE_g2c = [getSE(Albedo_g2c[0]),
                    getSE(Albedo_g2c[1]),
                    getSE(Albedo_g2c[2])]

YearAlbedoSE_u2c = [getSE(Albedo_u2c[0]),
                    getSE(Albedo_u2c[1]),
                    getSE(Albedo_u2c[2])]


#########################################################################
#########################################################################

def DrawLSTsubPlots(data,se,ax,No,title,ylabel,legendFlag):
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


    labelx = ['Daytime', 'Nighttime']  #'Daily',
    labels = ["North Xinjiang","South Xinjiang","Hexi Corridor"]
    err_attri = dict(elinewidth=.8, ecolor='gray', capsize=1)
    colors = ['#003f5c', '#bc5090', '#ffa600']

    plt.bar(x1, y1, alpha=1, width=1, color=colors[0], label=labels[0], yerr=SE1, error_kw=err_attri,zorder=100)
    plt.bar(x2, y2, alpha=1, width=1, color=colors[1], label=labels[1], yerr=SE2, error_kw=err_attri,zorder=100)
    plt.bar(x3, y3, alpha=1, width=1, color=colors[2], label=labels[2], yerr=SE3, error_kw=err_attri,zorder=100)


    plt.text(0.02, 0.90, No, transform=ax.transAxes)
    plt.text(0.25, 1.03, title, transform=ax.transAxes,fontsize=14)
    plt.ylabel(ylabel,labelpad=0.05)
    if (legendFlag==True):
        plt.legend(frameon=False)
    plt.xticks(x2, labelx, rotation=0)
    plt.ylim([-1.0,0.25])
    plt.yticks([-0.8,-0.4,0],
               [-0.8,-0.4,0])
    plt.grid(zorder=0, linestyle='--', axis='y')

def DrawETAlbedoPlots(data,se,ax,No,title,ylabel,ylim,yticks,convertType,textBoxColor,legendFlag):
    plt.sca(ax)
    x = [1,3,5]
    y = data
    labels = ["North Xinjiang","South Xinjiang","Hexi Corridor"]
    err_attri = dict(elinewidth=.8, ecolor='gray', capsize=1)
    colors = ['#003f5c', '#bc5090', '#ffa600']
    plt.bar(x, y, alpha=1, width=1, color=colors,yerr=se, error_kw = err_attri,zorder=100)
    plt.bar(x, [0, 0, 0], alpha=1, width=1, color=colors[0], label = labels[0], error_kw = err_attri,zorder=100)
    plt.bar(x, [0, 0, 0], alpha=1, width=1, color=colors[1], label = labels[1], error_kw = err_attri,zorder=100)
    plt.bar(x, [0, 0, 0], alpha=1, width=1, color=colors[2], label = labels[2], error_kw = err_attri,zorder=100)
    plt.xticks(x,['','',''])
    plt.text(0.02, 0.90, No, transform=ax.transAxes)
    plt.text(0.22, 1.03, title, transform=ax.transAxes,fontsize=14)
    plt.ylabel(ylabel,labelpad=0.01)
    plt.ylim(ylim)
    plt.yticks(yticks[0],yticks[1])
    plt.text(1.02, 0.17, convertType, transform=ax.transAxes, rotation=270,
             bbox=dict(facecolor=textBoxColor, alpha=0.5))  # ,style='italic'
    if (legendFlag==True):
        plt.legend(ncol=3,loc='center left', bbox_to_anchor=(-1.0, -0.20),frameon=False)
    plt.grid(zorder=0, linestyle='--', axis='y')

#plot figure
# plt.style.use('ggplot')
plt.rc('font',size = 12)#, family='Times New Roman'
fig = plt.figure(figsize=(8,3))#dpi=100, default dpi
fig.subplots_adjust(left = 0.08, right = 0.99,
                    bottom = 0.10, top = 0.88,
                    wspace= 0.25, hspace=0.05)



ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

ax1_1= fig.add_axes([0.080,0.88,0.26, 0.12])
ax2_1= fig.add_axes([0.405,0.88,0.26, 0.12])
ax3_1= fig.add_axes([0.730,0.88,0.26, 0.12])
drawCaptionPlot(ax1_1,'forest2crop',[0.28,0.30])
drawCaptionPlot(ax2_1,'grass2crop', [0.28,0.30])
drawCaptionPlot(ax3_1,'unuse2crop', [0.28,0.30])

ylim_et = [0,1.8]
ylim_al = [-0.012,0.01]

yticks_label_et = [[0.0,0.4,0.8,1.2,1.6],[0.0,0.4,0.8,1.2,1.6]]

yticks_label_al = [[-0.012,-0.008,-0.004,0, 0.004, 0.008],
                   [-1.2,-0.8,-0.4,0, 0.4,0.8]]

DrawLSTsubPlots(YearLST_f2c,YearLSTSE_f2c,ax1,'(a)','','$\Delta$LST (K)',True)#$^\circ$C
DrawLSTsubPlots( YearLST_g2c,YearLSTSE_g2c,ax2,'(b)','','',False)
DrawLSTsubPlots( YearLST_u2c,YearLSTSE_u2c,ax3,'(c)','','','')


# plt.savefig(outname)

plt.show()
