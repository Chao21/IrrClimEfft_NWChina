# 2022/9/23 Zhang Chao
# Energy decomposition for three regions


import matplotlib.pyplot as plt
import pandas as pd
import math
import os

Dir  = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root = Dir + r'\EnergyEq_DTM'
root2 = Dir + r'\FenquGridID'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname = outdir + r'\FigS8.svg'

infilef = root + r'\Eneq_f2c_daily.csv'
infileg = root + r'\Eneq_g2c_daily.csv'
infileu = root + r'\Eneq_u2c_daily.csv'

dff = pd.read_csv(infilef)
dfg = pd.read_csv(infileg)
dfu = pd.read_csv(infileu)

roiBJ = os.path.join(root2,'Beijiang.csv')
roiNJ = os.path.join(root2,'Nanjiang.csv')
roiHX = os.path.join(root2,'Hexi.csv')

dff_BJ = pd.merge(pd.read_csv(infilef).dropna(),
                  pd.read_csv(roiBJ).dropna(),on='Id')
dff_NJ = pd.merge(pd.read_csv(infilef).dropna(),
                  pd.read_csv(roiNJ).dropna(),on='Id')
dff_HX = pd.merge(pd.read_csv(infilef).dropna(),
                  pd.read_csv(roiHX).dropna(),on='Id')

dfg_BJ = pd.merge(pd.read_csv(infileg).dropna(),
                  pd.read_csv(roiBJ).dropna(),on='Id')
dfg_NJ = pd.merge(pd.read_csv(infileg).dropna(),
                  pd.read_csv(roiNJ).dropna(),on='Id')
dfg_HX = pd.merge(pd.read_csv(infileg).dropna(),
                  pd.read_csv(roiHX).dropna(),on='Id')

dfu_BJ = pd.merge(pd.read_csv(infileu).dropna(),
                  pd.read_csv(roiBJ).dropna(),on='Id')
dfu_NJ = pd.merge(pd.read_csv(infileu).dropna(),
                  pd.read_csv(roiNJ).dropna(),on='Id')
dfu_HX = pd.merge(pd.read_csv(infileu).dropna(),
                  pd.read_csv(roiHX).dropna(),on='Id')

def getSE(data):
    count = data.shape[0] #len(df): row; df.shape[1]: columns
    return data.std()/math.sqrt(count)*1.96

def drawCaptionPlot(ax,title,pos,rot):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc='#D9D9D9', edgecolor='k'))  # ,alpha=0.6)
    ax.text(pos[0],pos[1],title,transform=ax.transAxes,fontsize=14,rotation=rot)
    ax.set_xticks([])
    ax.set_yticks([])



def plotBars(ax,df,ylabel,ylim,No):
    plt.sca(ax)
    x = [1,3,5,7,9]
    data = df[['dTo_GS','dTc_GS','dT_Al_GS','dT_LE_GS','dT_H_GS']]
    count = data.shape[0]
    err_attri = dict(elinewidth=.8, ecolor='k', capsize=1)

    colors = ['#A9A9A9', '#696969', '#E36F9A', '#4D78EA', '#B9AF33']
    plt.bar(x, data.mean(), alpha=1.0, width=1, yerr = getSE(data), error_kw = err_attri,color=colors)
    plt.xticks(x,['$\Delta$T_obs','$\Delta$T_cal','$\Delta$T(a)',
                  '$\Delta$T(LE)','$\Delta$T(H)'],rotation=90)
    plt.axhline(y=0.0, c="k",   alpha = 0.4,ls="-", lw=1)
    plt.ylim(ylim)
    plt.ylabel(ylabel)


####################################################################
#Figure
plt.rc('font',size=12)
fig = plt.figure(figsize=(8,5))
fig.subplots_adjust(left = 0.08, right = 0.95,
                    bottom = 0.14, top = 0.92,
                    wspace = 0.20, hspace=0.05)

ax1 = plt.subplot(331)
ax2 = plt.subplot(332)
ax3 = plt.subplot(333)
ax4 = plt.subplot(334)
ax5 = plt.subplot(335)
ax6 = plt.subplot(336)
ax7 = plt.subplot(337)
ax8 = plt.subplot(338)
ax9 = plt.subplot(339)


h = 0.075
w = 0.045
ax_c1= fig.add_axes([ax1.get_position().x0,
                     ax1.get_position().y1,
                     ax1.get_position().x1-ax1.get_position().x0,
                     h])
ax_c2= fig.add_axes([ax2.get_position().x0,
                     ax2.get_position().y1,
                     ax2.get_position().x1-ax2.get_position().x0,
                     h])
ax_c3= fig.add_axes([ax3.get_position().x0,
                     ax3.get_position().y1,
                     ax3.get_position().x1-ax3.get_position().x0,
                     h])
drawCaptionPlot(ax_c1,'North XJ', [0.25,0.30],0)
drawCaptionPlot(ax_c2,'South XJ',  [0.25,0.30],0)
drawCaptionPlot(ax_c3,'HX Corridor',  [0.25,0.30],0)

ax_r1= fig.add_axes([ax3.get_position().x1,
                     ax3.get_position().y0,
                     w,
                     ax3.get_position().y1-ax3.get_position().y0])
ax_r2= fig.add_axes([ax6.get_position().x1,
                     ax6.get_position().y0,
                     w,
                     ax6.get_position().y1-ax6.get_position().y0])
ax_r3= fig.add_axes([ax9.get_position().x1,
                     ax9.get_position().y0,
                     w,
                     ax9.get_position().y1-ax9.get_position().y0])
drawCaptionPlot(ax_r1,'forest2crop',  [0.20,0.08],270)
drawCaptionPlot(ax_r2,'grass2crop',   [0.20,0.08],270)
drawCaptionPlot(ax_r3,'unuse2crop',   [0.20,0.08],270)

ylim1 = [-2.2,2.2]
ylim2 = [-1.5,1.2]
ylim3 = [-2.2,1.0]

plotBars(ax1,dff_BJ,'$\Delta$T (K)',ylim1,'(a)')
plotBars(ax2,dff_NJ,'',ylim1,'(b)')
plotBars(ax3,dff_HX,'',ylim1,'(c)')
plotBars(ax4,dfg_BJ,'$\Delta$T (K)',ylim2,'(d)')
plotBars(ax5,dfg_NJ,'',ylim2,'(e)')
plotBars(ax6,dfg_HX,'',ylim2,'(f)')
plotBars(ax7,dfu_BJ,'$\Delta$T (K)',ylim3,'(d)')
plotBars(ax8,dfu_NJ,'',ylim3,'(e)')
plotBars(ax9,dfu_HX,'',ylim3,'(f)')

# plt.savefig(outname)

plt.show()