#--coding:utf-8--
# Zhang Chao 2022/4/8
# correlation analysis between ET and LST in daytime and nighttime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import scipy
from sklearn.linear_model import LinearRegression



Dir  = r'F:\00myWork\06-ClimEffectInNWChina\10-paperData'
root = Dir + r'\Actual'
file_forest =  root +  r'\02forest2c\LST.csv'
file_grass  =  root +  r'\03grass2c\LST.csv'
file_unuse  =  root +  r'\04unuse2c\LST.csv'

outdir = r'F:\00myWork\06-ClimEffectInNWChina\11paperGraphs'
outname = outdir + r'\Fig9.svg'

dff = pd.read_csv(file_forest)
dfg = pd.read_csv(file_grass)
dfu = pd.read_csv(file_unuse)

def getDayLST(df):
    df['LSTDay'] = df[[ 'LST_Day_03_x', 'LST_Day_04_x',
              'LST_Day_05_x', 'LST_Day_06_x', 'LST_Day_07_x', 'LST_Day_08_x',
              'LST_Day_09_x', 'LST_Day_10_x', 'LST_Day_11_x']]\
                         .apply(lambda x: x.sum(), axis=1) / 9
    return df[['LSTDay', 'Id']]

def getNightLST(df):
    df['LSTNight'] = df[[ 'LST_Night_03_x', 'LST_Night_04_x',
              'LST_Night_05_x', 'LST_Night_06_x', 'LST_Night_07_x', 'LST_Night_08_x',
              'LST_Night_09_x', 'LST_Night_10_x', 'LST_Night_11_x']]\
                         .apply(lambda x: x.sum(), axis=1) / 9
    return df[['LSTNight', 'Id']]


def getET(df):
    df['ET'] = df[[ 'ET_03_x', 'ET_04_x',
              'ET_05_x', 'ET_06_x', 'ET_07_x', 'ET_08_x',
              'ET_09_x', 'ET_10_x', 'ET_11_x']]\
                         .apply(lambda x: x.sum(), axis=1)/9
    return df[['ET','Id']]

def getAlbedo(df):
    df['Albedo'] = df[[ 'Albedo_03_x', 'Albedo_04_x',
              'Albedo_05_x', 'Albedo_06_x', 'Albedo_07_x', 'Albedo_08_x',
              'Albedo_09_x', 'Albedo_10_x', 'Albedo_11_x']]\
                         .apply(lambda x: x.sum(), axis=1) / 9

    return df[['Albedo', 'Id']]

def getMeanSe(df,var):
    Mean = list(df.groupby(['ETinterval'])[var].mean())
    count = list(df.groupby(['ETinterval'])[var].count())
    std = list(df.groupby(['ETinterval'])[var].std())
    se = list(np.true_divide(std,np.sqrt(count))*1.96)
    return [Mean,se]


def drawCaptionPlot(ax,title,pos):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc='#D9D9D9', edgecolor='k'))  # ,alpha=0.6)
    ax.text(pos[0],pos[1],title,transform=ax.transAxes,fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

DayLSTf, DayLSTg, DayLSTu = getDayLST(dff),getDayLST(dfg),getDayLST(dfu)
NightLSTf, NightLSTg, NightLSTu = getNightLST(dff),getNightLST(dfg),getNightLST(dfu)


####################################################################
##################################################################
file_forest =  root + r'\02forest2c\ET.csv'
file_grass  =  root + r'\03grass2c\ET.csv'
file_unuse  =  root + r'\04unuse2c\ET.csv'

dff = pd.read_csv(file_forest)
dfg = pd.read_csv(file_grass)
dfu = pd.read_csv(file_unuse)

ETf, ETg, ETu = getET(dff),getET(dfg),getET(dfu)


#####################################################################
file_forest =  root + r'\02forest2c\Albedo.csv'
file_grass  =  root +  r'\03grass2c\Albedo.csv'
file_unuse  =  root +  r'\04unuse2c\Albedo.csv'

dff = pd.read_csv(file_forest)
dfg = pd.read_csv(file_grass)
dfu = pd.read_csv(file_unuse)

Albedof, Albedog, Albedou = getAlbedo(dff),getAlbedo(dfg),getAlbedo(dfu)

Groupf = pd.merge(pd.merge(pd.merge(DayLSTf.dropna(),NightLSTf.dropna(),on='Id'),
                  ETf,on='Id'),
                  Albedof,on='Id')

Groupg = pd.merge(pd.merge(pd.merge(DayLSTg.dropna(),NightLSTg.dropna(),on='Id'),
                  ETg,on='Id'),
                  Albedog,on='Id')

Groupu = pd.merge(pd.merge(pd.merge(DayLSTu.dropna(),NightLSTu.dropna(),on='Id'),
                  ETu,on='Id'),
                  Albedou,on='Id')

dff = Groupf
dfg = Groupg
dfu = Groupu



dff['ETinterval'] = pd.cut(dff['ET'],[-10,0.1,0.5,10],
                           labels=[0.1,0.5,10])
dfg['ETinterval'] = pd.cut(dfg['ET'],[-10,0.1,0.5,10],
                           labels=[0.1,0.5,10])
dfu['ETinterval'] = pd.cut(dfu['ET'],[-10,0.1,0.5,10],
                           labels=[0.1,0.5,10])

LSTDayf = getMeanSe(dff,'LSTDay')
LSTNightf = getMeanSe(dff,'LSTNight')

LSTDayg = getMeanSe(dfg,'LSTDay')
LSTNightg = getMeanSe(dfg,'LSTNight')

LSTDayu = getMeanSe(dfu,'LSTDay')
LSTNightu = getMeanSe(dfu,'LSTNight')

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

def DrawsubPlots(ax,group,df,varX,varY,No,title,Ylim,xtickabel,
                 yticks,yticklabel,xlabel,ylabel,insetYtick,insetYlim):
    x = [1, 3, 5]
    y = group[0]
    ye = group[1]
    colors = ['#2A557F', '#45BC9C', '#F05073']
    err_attri = dict(elinewidth=.8, ecolor='k', capsize=1)

    plt.sca(ax)
    plt.bar(x, y, alpha=.9, width=1, color=colors[0], label="forest2crop", yerr=ye, error_kw=err_attri,zorder=100)
    plt.ylim(Ylim)
    plt.xticks([1,3,5],xtickabel,fontsize=12)
    plt.yticks(yticks,yticklabel,fontsize=12)
    plt.xlabel(xlabel,fontsize=12)  #, labelpad=2.5
    plt.ylabel(ylabel, labelpad=0.5,fontsize=12)  #
    plt.title(title,fontsize=12)#
    plt.text(0.02, 0.02, No, transform=ax.transAxes,fontsize=12)  # , style='italic'
    plt.grid(zorder=0,linestyle='--',axis='y')

    #############################################
    axins = ax.inset_axes((0.2, 0.2, 0.55, 0.4))
    Y = df[varY]
    X = df[varX]
    a, b, Yp, R2 = LinearRegressionFunc(X, Y)
    r, sig = scipy.stats.pearsonr(X, Y)
    parameter = np.polyfit(X, Y, 1)
    f = np.poly1d(parameter)


    axins.scatter(X, Y, s=10, c='#2A557F', marker='.', alpha=0.8)
    axins.plot(X, f(X), 'r-', lw=1)
    if (b < 0):
        axins.text(0.05, 0.20, "y=%.2fx%.2f" % (a, b),
                   transform=axins.transAxes, c='k')
    else:
        axins.text(0.05, 0.20, "y=%.2fx+%.2f" % (a, b),
                   transform=axins.transAxes, c='k')
    axins.text(0.05, 0.04, "r=%.2f, p=%.3f" % (r,sig),
             transform=axins.transAxes, c = 'k')
    axins.set_yticks(insetYtick)
    axins.set_ylim(insetYlim)
    axins.set_xlabel('$\Delta$ET (mm)',labelpad = 0.5 ,fontsize = 10)#
    axins.set_ylabel('$\Delta$LST(K)', labelpad=0.5, fontsize=10)  #$^\circ$C
    axins.grid(color='r', linestyle='--', linewidth=1, alpha=0)

    axins.spines['right'].set_color('k')
    axins.spines['left'].set_color('k')
    axins.spines['top'].set_color('k')
    axins.spines['bottom'].set_color('k')



# plt.style.use('ggplot')
fig = plt.figure(figsize=(8.0, 5.0))
fig.subplots_adjust(left=0.08, right=0.995,
                    bottom=0.1, top=0.95,
                    wspace=0.05, hspace=0.02)
# plt.rc('font',size = 12)#, family='Times New Roman'
ax1 = plt.subplot(231)
ax2 = plt.subplot(232)
ax3 = plt.subplot(233)
ax4 = plt.subplot(234)
ax5 = plt.subplot(235)
ax6 = plt.subplot(236)

ax1_1= fig.add_axes([0.080,0.95,0.295,0.05])
ax2_1= fig.add_axes([0.390,0.95,0.295,0.05])
ax3_1= fig.add_axes([0.700,0.95,0.2951,0.05])
drawCaptionPlot(ax1_1,'forest2crop',[0.3,0.20])
drawCaptionPlot(ax2_1,'grass2crop', [0.3,0.20])
drawCaptionPlot(ax3_1,'unuse2crop', [0.3,0.20])


Ylim1 = [-2.3,0.3]
Ylim2 = [-0.8,0.2]
Xticklabel = ['<0.1','0.1~0.5','>0.5']
Yticks1 = [-2.0,-1.5,-1.0,-0.5,0]
Yticklabel1 = [-2.0,-1.5,-1.0,-0.5,0]
Yticks2 = [-0.5,0]
Yticklabel2 = [-0.5,0]
Xlabel = '$\Delta$ET (mm)'
Ylabel1 = 'Daytime $\Delta$LST (K)'#$^\circ$C
Ylabel2 = 'Nighttime $\Delta$LST (K)'#$^\circ$C


DrawsubPlots(ax1,LSTDayf,dff,'ET','LSTDay','(a)','forest2crop',Ylim1,'',Yticks1,Yticklabel1,'',Ylabel1,[-3,0,3],[-6,3.5])
DrawsubPlots(ax4,LSTNightf,dff,'ET','LSTNight','(d)','',Ylim2,Xticklabel,Yticks2,Yticklabel2,'',Ylabel2,[-1,0,1],[-3,2])

DrawsubPlots(ax2,LSTDayg,dfg,'ET','LSTDay','(b)','grass2crop',Ylim1,'',Yticks1,'','','',[-5,0],[-9,4])
DrawsubPlots(ax5,LSTNightg,dfg,'ET','LSTNight','(e)','',Ylim2,Xticklabel,Yticks2,'',Xlabel,'',[-1,0,1],[-3,2])

DrawsubPlots(ax3,LSTDayu,dfu,'ET','LSTDay','(c)','unuse2crop',Ylim1,'',Yticks1,'','','',[-5,0],[-9,4])
DrawsubPlots(ax6,LSTNightu,dfu,'ET','LSTNight','(f)','',Ylim2,Xticklabel,Yticks2,'','','',[-1,0,1],[-3,2])
#
# plt.savefig(outname)

plt.show()