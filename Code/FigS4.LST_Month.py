#--coding:utf-8--
# Zhang Chao 2022/4/17
# Actual monthly LST for three  regions

import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

Dir  = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root = Dir + r'\ActualFenqu'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname = outdir + r'\FigS4.svg'


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

varET = 'LST_csv_ET'
varAlbedo = 'LST_csv_Albedo_BSA'

def getSE(data):
    count = data.shape[0] #len(df): row; df.shape[1]: columns
    return data.std()/math.sqrt(count)*1.96

LSTDay_f2c =   [dff_BJ[varLSTday],dff_NJ[varLSTday],dff_HX[varLSTday]]
LSTNight_f2c = [dff_BJ[varLSTnight],dff_NJ[varLSTnight],dff_HX[varLSTnight]]

LSTDay_g2c =   [dfg_BJ[varLSTday],dfg_NJ[varLSTday],dfg_HX[varLSTday]]
LSTNight_g2c = [dfg_BJ[varLSTnight],dfg_NJ[varLSTnight],dfg_HX[varLSTnight]]

LSTDay_u2c =   [dfu_BJ[varLSTday],dfu_NJ[varLSTday],dfu_HX[varLSTday]]
LSTNight_u2c = [dfu_BJ[varLSTnight],dfu_NJ[varLSTnight],dfu_HX[varLSTnight]]
#



def DrawsubPlots(data,ax,No,title,ylim,ylabflag,convertType,textBoxColor,legendFlag):
    plt.sca(ax)
    width = 1.0
    x1 = np.arange(1,1+width*4*12,width*4)
    x2 = [i + width*1 for i in x1]
    x3 = [i + width*2 for i in x1]
    y1 = list(data[0].mean())
    y2 = list(data[1].mean())
    y3 = list(data[2].mean())
    SE = [list(getSE(i)) for i in data]

    labelx = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    err_attri = dict(elinewidth=.8, ecolor='gray', capsize=1)

    colors = ['#003f5c', '#bc5090', '#ffa600']

    plt.bar(x1, y1, alpha=1.0, width=1, color=colors[0], label="North XinJiang", yerr=SE[0], error_kw=err_attri,zorder=100)
    plt.bar(x2, y2, alpha=1.0, width=1, color=colors[1], label="South XinJiang",  yerr=SE[1],error_kw=err_attri,zorder=100)
    plt.bar(x3, y3, alpha=1.0, width=1, color=colors[2], label="Hexi Corridor",  yerr=SE[2], error_kw=err_attri,zorder=100)
    plt.xticks(x2, [])
    plt.grid(zorder=0, linestyle='--', axis='y')
    plt.ylim(ylim)

    #
    plt.text(0.35, 1.06, title, transform=ax.transAxes)
    plt.text(0.015, 0.88, No, transform=ax.transAxes)
    plt.text(1.02, 0.18,convertType,transform=ax.transAxes,rotation=270,
             bbox = dict(facecolor = textBoxColor, alpha = 0.5))#,style='italic'
    plt.xticks(x2,labelx,rotation=90)
    if(ylabflag == True):
        plt.ylabel('$\Delta$LST (K)')
    if(legendFlag==True):
        plt.legend(ncol=3,loc='center left', bbox_to_anchor=(0.2, -0.47),frameon=False)

# plt.style.use('ggplot')
plt.rc('font',size = 12)#, family='Times New Roman'
fig = plt.figure(figsize=(8,5))#dpi=100, default dpi
fig.subplots_adjust(left = 0.08, right=0.96,
                    bottom = 0.14, top = 0.95,
                    wspace= 0.18, hspace=0.05)

ax1 = plt.subplot(321)
ax2 = plt.subplot(322)
ax3 = plt.subplot(323)
ax4 = plt.subplot(324)
ax5 = plt.subplot(325)
ax6 = plt.subplot(326)


ylimace=[-3.3,1]
ylimbdf=[-0.7,0.7]
DrawsubPlots(LSTDay_f2c,ax1,'(a)','Daytime $\Delta$LST',ylimace,False,'','r',False)
DrawsubPlots(LSTNight_f2c,ax2,'(b)','Nighttime $\Delta$LST',ylimbdf,False,'forest2crop','#2A557F',False)
DrawsubPlots(LSTDay_g2c,ax3,'(c)','',ylimace,True,'','r',False)
DrawsubPlots(LSTNight_g2c,ax4,'(d)','',ylimbdf,False,'grass2crop','#45BC9C',False)
DrawsubPlots(LSTDay_u2c,ax5,'(e)','',ylimace,False,'','r',True)
DrawsubPlots(LSTNight_u2c,ax6,'(f)','',ylimbdf,False,'unuse2crop','#F05073',False)

# plt.savefig(outname)

plt.show()
