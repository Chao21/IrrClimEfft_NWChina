#--coding:utf-8--
# Zhang Chao 2022/4/17
# Regional monthly ET and Albedo changes

import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np



Dir  = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root = Dir + r'\ActualFenqu'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname = outdir + r'\FigS6.svg'

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

def getSE(data):
    count = data.shape[0] #len(df): row; df.shape[1]: columns
    return data.std()/math.sqrt(count)*1.96

varET  = ['ET_01_x','ET_02_x','ET_03_x','ET_04_x',
              'ET_05_x','ET_06_x','ET_07_x','ET_08_x',
              'ET_09_x','ET_10_x','ET_11_x','ET_12_x']


ET_f2c   =   [dff_BJ[varET],  dff_NJ[varET],  dff_HX[varET]]
ET_g2c   =   [dfg_BJ[varET],  dfg_NJ[varET],  dfg_HX[varET]]
ET_u2c   =   [dfu_BJ[varET],  dfu_NJ[varET],  dfu_HX[varET]]
#################################################################
#Albedo

varAlbedo = ['Albedo_01_x','Albedo_02_x','Albedo_03_x','Albedo_04_x',
              'Albedo_05_x','Albedo_06_x','Albedo_07_x','Albedo_08_x',
              'Albedo_09_x','Albedo_10_x','Albedo_11_x','Albedo_12_x',]

Albedo_f2c = [dff_BJ[varAlbedo],dff_NJ[varAlbedo],dff_HX[varAlbedo]]
Albedo_g2c = [dfg_BJ[varAlbedo],dfg_NJ[varAlbedo],dfg_HX[varAlbedo]]
Albedo_u2c = [dfu_BJ[varAlbedo],dfu_NJ[varAlbedo],dfu_HX[varAlbedo]]


def DrawsubPlots(data,ax,No,title,ylab,ylim,convertType,textBoxColor,legendFlag,
                 Ytick, Yticklabel):
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

    plt.bar(x1, y1, alpha=1, width=1, color=colors[0], label="North XinJiang", yerr=SE[0], error_kw=err_attri,zorder=100)
    plt.bar(x2, y2, alpha=1, width=1, color=colors[1], label="South XinJiang",  yerr=SE[1],error_kw=err_attri,zorder=100)
    plt.bar(x3, y3, alpha=1, width=1, color=colors[2], label="Hexi Corridor",  yerr=SE[2], error_kw=err_attri,zorder=100)
    plt.xticks(x2, [])
    plt.grid(zorder=0, linestyle='--', axis='y')
    plt.ylim(ylim)

    #
    plt.text(0.40, 1.06, title, transform=ax.transAxes)
    plt.text(0.015, 0.88, No, transform=ax.transAxes)
    plt.text(1.02, 0.18,convertType,transform=ax.transAxes,rotation=270,
             bbox = dict(facecolor = textBoxColor, alpha = 0.5))#,style='italic'
    plt.xticks(x2,labelx,rotation=90)
    plt.yticks(Ytick,Yticklabel)
    plt.ylabel(ylab,labelpad=0.5)
    if(legendFlag==True):
        plt.legend(ncol=3,loc='center left', bbox_to_anchor=(0.2, -0.47),frameon=False)

# plt.style.use('ggplot')
plt.rc('font',size = 12)#, family='Times New Roman'
fig = plt.figure(figsize=(8,5))#dpi=100, default dpi
fig.subplots_adjust(left = 0.06, right=0.96,
                    bottom = 0.14, top = 0.95,
                    wspace= 0.17, hspace=0.05)

ax1 = plt.subplot(321)
ax2 = plt.subplot(322)
ax3 = plt.subplot(323)
ax4 = plt.subplot(324)
ax5 = plt.subplot(325)
ax6 = plt.subplot(326)

ylimace=[-0.5,6.5]
ylimbdf=[-0.025,0.025]

DrawsubPlots(ET_f2c,ax1,'(a)','$\Delta$ET','',ylimace,'','r',False,[0,3,6],[0,3,6])
DrawsubPlots(Albedo_f2c,ax2,'(b)','$\Delta$Albedo','',ylimbdf,'forest2crop',
             '#2A557F',False,[-0.02,0,0.02],[-2,0,2])
DrawsubPlots(ET_g2c,ax3,'(c)','','$\Delta$ET (mm)',ylimace,'','r',False,[0,3,6],[0,3,6])
DrawsubPlots(Albedo_g2c,ax4,'(d)','','$\Delta$Albedo (%)',ylimbdf,'grass2crop',
             '#45BC9C',False,[-0.02,0,0.02],[-2,0,2])
DrawsubPlots(ET_u2c,ax5,'(e)','','',ylimace,'','r',True,[0,3,6],[0,3,6])
DrawsubPlots(Albedo_u2c,ax6,'(f)','','',ylimbdf,'unuse2crop',
             '#F05073',False,[-0.02,0,0.02],[-2,0,2])

# plt.savefig(outname)

plt.show()
