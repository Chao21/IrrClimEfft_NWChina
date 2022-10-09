# 2022/9/22 Zhang Chao
# Energy decomposition for the each month in the growing season

import matplotlib.pyplot as plt
import pandas as pd
import math

Dir  = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root = Dir + r'\EnergyEq_DTM'
outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname = outdir + r'\FigS7.svg'

infilef = root + r'\Eneq_f2c_daily.csv'
infileg = root + r'\Eneq_g2c_daily.csv'
infileu = root + r'\Eneq_u2c_daily.csv'

dff = pd.read_csv(infilef)
dfg = pd.read_csv(infileg)
dfu = pd.read_csv(infileu)


dataf5 = dff[['dTo_05','dTc_05','dT_Al_05','dT_LE_05','dT_H_05']]
dataf6 = dff[['dTo_06','dTc_06','dT_Al_06','dT_LE_06','dT_H_06']]
dataf7 = dff[['dTo_07','dTc_07','dT_Al_07','dT_LE_07','dT_H_07']]
dataf8 = dff[['dTo_08','dTc_08','dT_Al_08','dT_LE_08','dT_H_08']]
dataf9 = dff[['dTo_09','dTc_09','dT_Al_09','dT_LE_09','dT_H_09']]

datag5 = dfg[['dTo_05','dTc_05','dT_Al_05','dT_LE_05','dT_H_05']]
datag6 = dfg[['dTo_06','dTc_06','dT_Al_06','dT_LE_06','dT_H_06']]
datag7 = dfg[['dTo_07','dTc_07','dT_Al_07','dT_LE_07','dT_H_07']]
datag8 = dfg[['dTo_08','dTc_08','dT_Al_08','dT_LE_08','dT_H_08']]
datag9 = dfg[['dTo_09','dTc_09','dT_Al_09','dT_LE_09','dT_H_09']]

datau5 = dfu[['dTo_05','dTc_05','dT_Al_05','dT_LE_05','dT_H_05']]
datau6 = dfu[['dTo_06','dTc_06','dT_Al_06','dT_LE_06','dT_H_06']]
datau7 = dfu[['dTo_07','dTc_07','dT_Al_07','dT_LE_07','dT_H_07']]
datau8 = dfu[['dTo_08','dTc_08','dT_Al_08','dT_LE_08','dT_H_08']]
datau9 = dfu[['dTo_09','dTc_09','dT_Al_09','dT_LE_09','dT_H_09']]

def getSE(data):
    count = data.shape[0] #len(df): row; df.shape[1]: columns
    return data.std()/math.sqrt(count)*1.96

def plotBars(ax,data,ylim,yticks,ylabel,xlabel,legflag,text,No):
    plt.sca(ax)
    x = [1,3,5,7,9]
    count = data.shape[0]
    err_attri = dict(elinewidth=.8, ecolor='k', capsize=1)

    colors = ['#A9A9A9', '#696969', '#E36F9A', '#4D78EA', '#B9AF33']
    # '#4D78EA' 潜热, '#B9AF33' 感热, '#E36F9A' 反照率,
    plt.bar([1], data.iloc[:,0].mean(), alpha=1.0, width=1, yerr = getSE(data.iloc[:,0]),
            error_kw = err_attri, label= '$\Delta$T_obs',color=colors[0])
    plt.bar([3], data.iloc[:, 1].mean(), alpha=1.0, width=1, yerr=getSE(data.iloc[:, 1]),
            error_kw=err_attri, label='$\Delta$T_cal', color=colors[1])
    plt.bar([5], data.iloc[:, 2].mean(), alpha=1.0, width=1, yerr=getSE(data.iloc[:, 2]),
            error_kw=err_attri, label='$\Delta$T(a)', color=colors[2])
    plt.bar([7], data.iloc[:, 3].mean(), alpha=1.0, width=1, yerr=getSE(data.iloc[:, 3]),
            error_kw=err_attri, label='$\Delta$T(LE)', color=colors[3])
    plt.bar([9], data.iloc[:, 4].mean(), alpha=1.0, width=1, yerr=getSE(data.iloc[:, 4]),
            error_kw=err_attri, label='$\Delta$T(H)', color=colors[4])
    plt.text(0.1,0.1,text,transform=ax.transAxes,style='italic')
    plt.axhline(y=0.0, c="k",   alpha = 0.4,ls="-", lw=1)
    plt.xticks([])
    plt.ylim(ylim)
    plt.yticks(yticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(0.05, 0.92, No, transform=ax.transAxes)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legflag == 1:
        plt.legend(loc='lower left',bbox_to_anchor=(-2.5, -0.54),
                   ncol=5, frameon=False)


####################################################################
#Figure
plt.rc('font',size=12)
fig = plt.figure(figsize=(8,5))
fig.subplots_adjust(left = 0.08, right = 0.99,
                    bottom = 0.12, top = 0.99,
                    wspace = 0.35, hspace=0.15)

ax1  = plt.subplot(3,5,1)
ax2  = plt.subplot(3,5,2)
ax3  = plt.subplot(3,5,3)
ax4  = plt.subplot(3,5,4)
ax5  = plt.subplot(3,5,5)
ax6  = plt.subplot(3,5,6)
ax7  = plt.subplot(3,5,7)
ax8  = plt.subplot(3,5,8)
ax9  = plt.subplot(3,5,9)
ax10 = plt.subplot(3,5,10)
ax11 = plt.subplot(3,5,11)
ax12 = plt.subplot(3,5,12)
ax13 = plt.subplot(3,5,13)
ax14 = plt.subplot(3,5,14)
ax15 = plt.subplot(3,5,15)

ytick1 = [-1,0,1]
ytick2 = [-1,0]
ytick3 = [-2,-1,0]
ylim1  = [-1.9,1.7]
ylim2  = [-2,1.0]
ylim3  = [-2,0.8]
plotBars(ax1,dataf5,ylim1,ytick1,'$\Delta$T (K)','',0,'forest2crop','(a)') #'$\Delta$T (K)'
plotBars(ax2,dataf6,ylim1,ytick1,'','',0,'','(b)')
plotBars(ax3,dataf7,ylim1,ytick1,'','',0,'','(c)')
plotBars(ax4,dataf8,ylim1,ytick1,'','',0,'','(d)')
plotBars(ax5,dataf9,ylim1,ytick1,'','',0,'','(e)')

plotBars(ax6, datag5,ylim2,ytick2,'$\Delta$T (K)','',0,'grass2crop','(f)') #'$\Delta$T (K)'
plotBars(ax7, datag6,ylim2,ytick2,'','',0,'','(g)')
plotBars(ax8, datag7,ylim2,ytick2,'','',0,'','(h)')
plotBars(ax9, datag8,ylim2,ytick2,'','',0,'','(i)')
plotBars(ax10,datag9,ylim2,ytick2,'','',0,'','(j)')

plotBars(ax11, datau5,ylim3,ytick3,'$\Delta$T (K)','May',0,'unuse2crop','(k)')
plotBars(ax12, datau6,ylim3,ytick3,'','June',0,     '','(l)')
plotBars(ax13, datau7,ylim3,ytick3,'','July',1,     '','(m)')
plotBars(ax14, datau8,ylim3,ytick3,'','August',0,   '','(n)')
plotBars(ax15, datau9,ylim3,ytick3,'','September',0,'','o')

# plt.savefig(outname)

plt.show()