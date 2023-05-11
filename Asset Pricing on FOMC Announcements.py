#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
This Notebook provides the code which is needed to replicate the empirical results in the working paper "Asset Pricing on FOMC announcements".
The cells must be executed sequentially. The code is commented out where necessary for understanding.
'''


# In[1]:


#Import all the relevant packages

import numpy as np
import pandas as pd
import pyreadstat 
import calendar
import datetime as dt
from datetime import date
from datetime import datetime
from calendar import monthrange
import matplotlib.pyplot as plt
import pickle
import yfinance as yf
import pandas_datareader as pdr
from matplotlib.dates import DateFormatter
from scipy.stats.mstats import winsorize
import warnings
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from scipy.stats import t
from statsmodels import api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.io
import math 
from yahoo_finance import Share
from scipy import stats
from matplotlib.dates import YearLocator, DateFormatter
from datetime import timedelta
from pandas.tseries.offsets import BDay
from pandas.tseries.offsets import *
pd.set_option('display.float_format', lambda x: '%.9f' % x)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')


# In[2]:


'''
Import the dataset which consists of TAQ data of S&P 500 firms at the 5-minute frequency over a horizon from 1998-2022. 
The dataset is pre-cleaned by following Barndorff-Nielsen et al.(2009), i.e., we:

1.) Delete entries with corrected trades (trades with a correction indicator, CORR φ 0). 
2.) Delete entries with abnormal sale condition (trades where COND has a letter code, except for Έ' and 'F'). 
'''

Data_=pd.read_stata(r"Data_1998_2022_all_clean.dta") 
Data_['Dates'] = pd.to_datetime(Data_['date_time']).dt.date
Data_['Weekday']=pd.to_datetime(Data_['Dates']).dt.dayofweek
Data_=Data_[Data_['Dates'].astype(str)>='2001-01-01'] #Due to low trading volume before 2001, we restrict the sample to 2001-2022
Data_=Data_[Data_['Dates']!='2022-01-02'] #Due to missing vaues, we drop observations on this day (it's no announcement day)
Data_['Dates']=Data_['Dates'].astype(str)
Data_['Time'] = pd.to_datetime(Data_['date_time']).dt.time
Data_=Data_.rename(columns={'date_time':'Date_Time'})
Data_=Data_.rename(columns={'price':'Price'})
Data_=Data_.rename(columns={'permno':'Permno'})
Data_=Data_.set_index('Date_Time')


# In[3]:


#Import FOMC Announcement Dates (the ones at 12:30 are excluded) from the Fed's website and the restrict dataset to FOMC dates and Wednesdays (NA days)

FOMC_Dates=pd.read_csv(r"FOMC Dates.csv",sep=';') 
FOMC_Dates=FOMC_Dates[FOMC_Dates['Announcement Date']>'2001-01-01'] 
FOMC_Dates['Announcement Date']=FOMC_Dates['Announcement Date'].astype(str)
FOMC_Dates=FOMC_Dates.sort_values(by='Announcement Date')
FOMC_Dates_List=FOMC_Dates['Announcement Date']
Wednesdays=Data_[Data_['Weekday']==2]
Wednesdays=Wednesdays['Dates'].drop_duplicates()
FOMC_Wednesdays_List=FOMC_Dates_List.append(Wednesdays)
Data=Data_[Data_['Dates'].isin(FOMC_Wednesdays_List)]
Data=Data.sort_values(by=['Permno','Dates','Time'])
print('The number of announcements is:',len(FOMC_Dates))


# In[4]:


#Import data on shares outstanding (source:CRSP)

datafile = 'Shrouts.dta'
Shrouts, meta = pyreadstat.read_dta(datafile)
Shrouts['Dates']=Shrouts['DlyCalDt']
Shrouts=Shrouts.drop('DlyCalDt',1)
Shrouts=Shrouts.rename(columns={'PERMNO':'Permno'})
Shrouts['Permno']=Shrouts['Permno'].astype(str)
Shrouts['Permno']=Shrouts['Permno'].str[:5]
Shrouts['Permno']=Shrouts['Permno'].astype(int)
Shrouts['Dates']=Shrouts['Dates'].astype(str)
List_Permnos=Data['Permno'].drop_duplicates() #Only consider the stocks in the sample
Shrouts=Shrouts[Shrouts["Permno"].isin(List_Permnos)]
Data['Date_Time']=Data.index
Data_1= pd.merge(Data, Shrouts,  how='left', left_on=['Permno','Dates'], right_on = ['Permno','Dates'])
Data_1=Data_1.drop_duplicates(subset =['Permno','Date_Time'], keep = 'last') #Drop duplicates
Data_1=Data_1.rename(columns={'ShrOut':'SHROUT'})
Data_1


# In[5]:


#Drop Permno-Dates with no variation in prices over the day

NAs=Data_1.groupby(['Dates','Permno'])['Price'].nunique()
NAs=pd.DataFrame(NAs)
NAs1=NAs
NAs1['Dates']=NAs.index.get_level_values(0)
NAs1['Permno']=NAs.index.get_level_values(1)
NAs2=NAs1.reset_index(drop=True)
NAs3=NAs2[NAs2['Price']<=5]
NAs3=NAs3[['Dates','Permno']]
NAs3['indicator']=1

Data_3= pd.merge(Data_1,NAs3,  how='left', left_on=['Permno','Dates'], right_on = ['Permno','Dates'])
Data_3=Data_3[Data_3['indicator'].isnull()]
Data_3=Data_3.drop('indicator',1)
Data_3


# In[6]:


#Caluclate market cap (in millions)

Data_3['M']=(Data_3['SHROUT']*Data_3['Price'])/1000


#Calculate market portfolio weights (value-weighting)

market=Data_3
market['Returns']=market.groupby(['Permno','Dates'])['Price'].pct_change()
#We winsorize the returns at 0.01%. We choose the treshold low on purpose, as we are particularly interested in the jumps at the announcements
market['Returns']=winsorize(market['Returns'],(0.0001,0.0001)) 
Total_MV=market.groupby('Date_Time').sum()
Total_MV=Total_MV[['M']]
Total_MV=Total_MV.rename(columns={'M':'Total_MV'})
MV= pd.merge(market, Total_MV,  how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
MV['weight']=MV['M']/MV['Total_MV']
MV=MV[['Date_Time','Permno','weight']]
market= pd.merge(market, MV,  how='left', left_on=['Permno','Date_Time'], right_on = ['Permno','Date_Time'])
market['weight_l1']=market.groupby(['Permno','Dates'])['weight'].shift(1)
market['weighted_return']=market['weight_l1']*market['Returns']
Market_Returns=market.groupby('Date_Time').sum()
Market_Returns=Market_Returns[['weighted_return']]
Market_Returns=Market_Returns.rename(columns={'weighted_return':'Market_Return'})
Market_Returns['Date_Time1']=Market_Returns.index
Market_Returns['Time']=pd.to_datetime(Market_Returns['Date_Time1']).dt.time
Market_Returns['Dates']=pd.to_datetime(Market_Returns['Date_Time1']).dt.date
market=Market_Returns
market=market[market['Time'].astype(str)>'09:34:00']
market['Dates']=market['Dates'].astype(str)
market['Market_Return_discrete']=market['Market_Return']
market['Market_Return']=np.log(market['Market_Return_discrete']+1) #calculate log returns
market=market.drop(['Date_Time1','Market_Return_discrete'],1)
market=market.sort_values(by=['Dates','Time'])


# In[7]:


market_A=market[market['Dates'].isin(FOMC_Dates_List)] #Dataset that consists of announcement days (A-days)
market_A=market_A.rename(columns={'Announcement Date':'Dates'})
market_NA=market[~market['Dates'].isin(FOMC_Dates_List)] #Dataset that consists of non-announcement days (NA days)


# In[8]:


#create plot of announcement returns over 2 years rolling windows (plot is not shown in paper, but in presentations

Daily_Returns=pd.DataFrame(market.groupby('Dates')['Market_Return'].sum())
Daily_Returns_A1=Daily_Returns[Daily_Returns.index.isin(FOMC_Dates['Announcement Date'])]
Daily_Returns_A=pd.DataFrame(market_A.groupby('Dates')['Market_Return'].sum())
Evolution_of_A_Returns=Daily_Returns_A
Evolution_of_A_Returns['Dates']=Evolution_of_A_Returns.index
Evolution_of_A_Returns['Dates']=pd.to_datetime(Evolution_of_A_Returns['Dates'])
Evolution_of_A_Returns=Evolution_of_A_Returns.set_index('Dates')
Evolution_of_A_Returns_Mean=Evolution_of_A_Returns.rolling(16).mean()*10000 #returns are reported in basis points (bps)
Evolution_of_A_Returns_Mean=Evolution_of_A_Returns_Mean.rename(columns={'Market_Return':'Announcement Premium'})
ax=Evolution_of_A_Returns_Mean.plot(color='black',linestyle='solid', title='Announcement Returns over Time')
ax.figure.savefig("FOMC Returns.pdf")
Evolution_of_A_Returns_Mean['Date1']=Evolution_of_A_Returns_Mean.index
Evolution_of_A_Returns_Mean['year']=pd.to_datetime(Evolution_of_A_Returns_Mean['Date1']).dt.year
Evolution_of_A_Returns_Mean['month']=pd.to_datetime(Evolution_of_A_Returns_Mean['Date1']).dt.month
Evolution_of_A_Returns_Mean['yearm']=Evolution_of_A_Returns_Mean['year'].astype(str)+':'+Evolution_of_A_Returns_Mean['month'].astype(str)
Evolution_of_A_Returns_Mean=Evolution_of_A_Returns_Mean.set_index('yearm')
Evolution_of_A_Returns_Mean=Evolution_of_A_Returns_Mean[['Announcement Premium']]
Evolution_of_A_Returns_Mean=Evolution_of_A_Returns_Mean.dropna()
Evolution_of_A_Returns_Mean=Evolution_of_A_Returns_Mean
#Evolution_of_A_Returns_Mean.to_csv('A-Premium.csv')


# In[9]:


#Calculate cumulative market returns over each day

market_cumsum=market
market_cumsum=market_cumsum.groupby('Dates').cumsum()
market_cumsum['Time']=market_cumsum.index
market_cumsum['Time']=pd.to_datetime(market_cumsum['Time']).dt.time
market_cumsum['Date']=market_cumsum.index
market_cumsum['Date']=pd.to_datetime(market_cumsum['Date']).dt.date
market_cumsum['Date']=market_cumsum['Date'].astype(str)
market_cumsum_A=market_cumsum[market_cumsum['Date'].isin(FOMC_Dates_List)]
market_cumsum_NA=market_cumsum[~market_cumsum['Date'].isin(FOMC_Dates_List)]


# In[10]:


#Calculate pre-announcement returns on announcement days

#We calculate the timeline relative to the announcement time

market_cumsum_A1=market_cumsum_A
market_cumsum_A1=pd.merge(market_cumsum_A1,FOMC_Dates, how='left',left_on=['Date'],right_on=['Announcement Date'])
market_cumsum_A1['Ann_Date_Time']=market_cumsum_A1['Announcement Date'] + ' ' + market_cumsum_A1['Announcement Time'].astype(str)
market_cumsum_A1['Date_Time']=market_cumsum_A1['Announcement Date'] + ' ' + market_cumsum_A1['Time'].astype(str)
market_cumsum_A1['Ann_Date_Time']=pd.to_datetime(market_cumsum_A1['Ann_Date_Time'])
market_cumsum_A1['Date_Time']=pd.to_datetime(market_cumsum_A1['Date_Time'])
market_cumsum_A1['Min before announcement']=(market_cumsum_A1['Date_Time']- market_cumsum_A1['Ann_Date_Time']).dt.total_seconds()/(60*5) 
market_cumsum_A1['Min before announcement']=market_cumsum_A1['Min before announcement'].apply(np.floor)

#Pre-announcement return is the cumulative up to t-1

Pre_A_drift=market_cumsum_A1[market_cumsum_A1['Min before announcement']==-1]
Pre_A_drift=Pre_A_drift[['Announcement Date','Time', 'Market_Return']]
Pre_A_drift=Pre_A_drift.rename(columns={'Market_Return':'Pre_A'})

Pre_A_drift['Pre_A']=Pre_A_drift['Pre_A']*10000


# In[11]:


#Calculate pre-announcement returns on announcement days

#We calculate the timeline relative to the announcement time

FOMC_Dates['Announcement Date']=FOMC_Dates['Announcement Date'].astype(str)
market_cumsum_NA1=market_cumsum_NA
market_cumsum_NA1['Announcement Time']=str('14:00:00')
market_cumsum_NA1['Ann_Date_Time']=market_cumsum_NA1['Date'] + ' ' + market_cumsum_NA1['Announcement Time'].astype(str)
market_cumsum_NA1['Date_Time']=market_cumsum_NA1['Date'] + ' ' + market_cumsum_NA1['Time'].astype(str)
market_cumsum_NA1['Ann_Date_Time']=pd.to_datetime(market_cumsum_NA1['Ann_Date_Time'])
market_cumsum_NA1['Date_Time']=pd.to_datetime(market_cumsum_NA1['Date_Time'])
market_cumsum_NA1['Min before announcement']=(market_cumsum_NA1['Date_Time']- market_cumsum_NA1['Ann_Date_Time']).dt.total_seconds()/(60*5)
market_cumsum_NA1['Min before announcement']=market_cumsum_NA1['Min before announcement'].apply(np.floor)

#Pre-announcement return is the cumulative up to t-1

Pre_A_drift_NA=market_cumsum_NA1[market_cumsum_NA1['Min before announcement']==-1]
Pre_A_drift_NA=Pre_A_drift_NA[['Date','Time', 'Market_Return']]
Pre_A_drift_NA=Pre_A_drift_NA.rename(columns={'Market_Return':'Pre_A'})
Pre_A_drift_NA['Pre_A']=Pre_A_drift_NA['Pre_A']*10000


# In[12]:


#Plot that shows pre-announcementdrift and announcement returns (not in paper, but in presentation)

Pre_A_drift1=Pre_A_drift.set_index('Announcement Date')
drift=pd.DataFrame(Pre_A_drift1['Pre_A'].rolling(16).mean())
drift1=drift.dropna()
drift1=drift1
drift1['Date1']=drift1.index
drift1['year']=pd.to_datetime(drift1['Date1']).dt.year
drift1['month']=pd.to_datetime(drift1['Date1']).dt.month
drift1['yearm']=drift1['year'].astype(str)+':'+drift1['month'].astype(str)
drift1=drift1.set_index('yearm')
drift1=drift1.rename(columns={'Pre_A':'Pre-A Drift'})
drift1=drift1[['Pre-A Drift']]
drift1.to_csv('Drift.csv')

#Merge drift and premium

Drift_Premium=pd.merge(Evolution_of_A_Returns_Mean,drift1,left_index=True, right_index=True)
Drift_Premium['date_index']=range(1,1+len(Drift_Premium))
Drift_Premium['date_index']=Drift_Premium['date_index'].astype(str)+':0'
Drift_Premium['yearm']=Drift_Premium.index
Drift_Premium=Drift_Premium.set_index('date_index')
Drift_Premium.plot()
#Drift_Premium.to_csv('Premium_Drift.csv')


# In[13]:


#Import all the announcement dates with press conferences (PCs)

PCs=pd.read_csv('PC_Announcement_Dates.csv', sep=';')
PCs=PCs['Announcement Date']
non_PCs=FOMC_Dates[~FOMC_Dates['Announcement Date'].isin(PCs)]
non_PCs=non_PCs[non_PCs['Announcement Date']>'2001-01-01']
non_PCs=non_PCs.reset_index(drop=True)
non_PCs=non_PCs['Announcement Date'].astype(str)

print('number of PC days is:',len(PCs))
print('number of non-PC days is:',len(non_PCs))


# In[14]:


#Create a list of dates before and after the announcements (needed for later calculations)

FOMC_Dates_bef=FOMC_Dates
FOMC_Dates_bef['A_Dates_bef']=pd.to_datetime(FOMC_Dates_bef['Announcement Date'])-BDay(1)
FOMC_Dates_bef=FOMC_Dates_bef['A_Dates_bef'].astype(str)
FOMC_Dates_bef

FOMC_Dates_after=FOMC_Dates
FOMC_Dates_after['A_Dates_bef']=pd.to_datetime(FOMC_Dates_after['Announcement Date'])+BDay(1)
FOMC_Dates_after=FOMC_Dates_after['A_Dates_bef'].astype(str)
FOMC_Dates_after


# In[15]:


'''
The following code calculates categorize returns in a jump and a diffuse component. 
A return is considered as a jump if it exceeds a certain threshold V that is based on the bipower variation (Barndorff-Nielsen
and Shephard (2004)) which estimates the diffusive part of the quadratic variation of the
market portfolio m on day t.
'''

#Bipower Variation BV

n=len(market['Time'].drop_duplicates())
Pi=math.pi
market_jump=market
market_jump['Market_Return_lagged']=market_jump.groupby(['Dates'])['Market_Return'].shift(1)
market_jump['BV_min_market']=(market_jump['Market_Return'].abs())*(market_jump['Market_Return_lagged'].abs())
BV_market=market_jump.groupby(['Dates'])['BV_min_market'].sum()*(Pi/2)
BV_market=pd.DataFrame(BV_market)

#V corresponds to the treshold that differentiates between diffusion an jump

V=3*((BV_market**(1/2))*(n**(-0.47)))
V_market=pd.DataFrame(V)
V_market=V_market.rename(columns={'BV_min_market':'V_market'})
market1= pd.merge(market_jump, V_market,  how='left', left_on=['Dates'], right_on = ['Dates'])
market1['diffusion_dummy_market']=np.where(market1['Market_Return'].abs()>=market1['V_market'],1,0) # if dummy assumes value 1, there is a jump
market1['Date_Time']=market1['Dates'].astype(str) + ' ' + market1['Time'].astype(str)
market1=market1.set_index('Date_Time')
market1=market1.drop(['Market_Return_lagged','BV_min_market','V_market'],1)
Jump_market=market1[['Dates','Time','diffusion_dummy_market']]
Jump_market


# In[16]:


#Create a histogram to show when jumps in the market occur (FIGURE 10 in PAPER)

Jump_market_A=Jump_market[Jump_market['Dates'].isin(FOMC_Dates_List)]
Jump_market_A=pd.merge(Jump_market_A,FOMC_Dates,how='left',left_on='Dates',right_on='Announcement Date')
Jump_market_A['Ann_Date_Time']=Jump_market_A['Announcement Date'].astype(str) + ' ' +Jump_market_A['Announcement Time'].astype(str)
Jump_market_A['Date_Time']=Jump_market_A['Dates'].astype(str) + ' ' + Jump_market_A['Time'].astype(str)
Jump_market_A['Ann_Date_Time']=pd.to_datetime(Jump_market_A['Ann_Date_Time'])
Jump_market_A['Date_Time']=pd.to_datetime(Jump_market_A['Date_Time'])

Jump_market_A['Min before announcement']=(Jump_market_A['Date_Time']- Jump_market_A['Ann_Date_Time']).dt.total_seconds()/(60*5)
Jump_market_A['Min before announcement']=Jump_market_A['Min before announcement'].apply(np.floor)
Jump_market_A=Jump_market_A.rename(columns={'Min before announcement':'Rel Time'})
#Jump_market_A=Jump_market_A[Jump_market_A['Dates']>'2011-01-01']
ax=Jump_market_A.groupby('Rel Time')['diffusion_dummy_market'].sum().plot()


# In[17]:


Jump_market_A_plot=pd.DataFrame(Jump_market_A.groupby('Rel Time')['diffusion_dummy_market'].sum())

Jump_market_A_plot['Rel Time']=Jump_market_A_plot.index
Jump_market_A_plot['Rel Time']=Jump_market_A_plot['Rel Time']*5 #convert 5-min units to 1-min units
Jump_market_A_plot['Rel Time']=Jump_market_A_plot['Rel Time']
Jump_market_A_plot['Rel Time']=Jump_market_A_plot['Rel Time'].astype(str)+':0'
Jump_market_A_plot=Jump_market_A_plot.set_index('Rel Time')
Jump_market_A_plot.to_csv('Jump_market_A_plot.csv')

Jump_market_A_plot=Jump_market_A_plot.rename(columns={'diffusion_dummy_market':'Jumps'})
#Jump_market_A_plot.to_csv('Jump_market.csv')


# In[18]:


x=Jump_market_A[Jump_market_A['diffusion_dummy_market']==1]
x=x[x['Rel Time']>=0]
x=x[x['Rel Time']<=6]
Jump_Dates=x['Dates'].drop_duplicates()
Jump_Dates

jump_locator=x
jump_locator=jump_locator[['Dates','Rel Time','diffusion_dummy_market']]
jump_locator


# In[19]:


#Estimate noise in the announcement

#Synchronize the trading minutes relative to the announcement
market_A_rel=pd.merge(market_A,FOMC_Dates,how='left',left_on='Dates',right_on='Announcement Date')
market_A_rel['Ann_Date_Time']=market_A_rel['Announcement Date'].astype(str) + ' ' + market_A_rel['Announcement Time'].astype(str)
market_A_rel['Date_Time']=market_A_rel['Dates'].astype(str) + ' ' + market_A_rel['Time'].astype(str)
market_A_rel['Ann_Date_Time']=pd.to_datetime(market_A_rel['Ann_Date_Time'])
market_A_rel['Date_Time']=pd.to_datetime(market_A_rel['Date_Time'])
market_A_rel['Min before announcement']=(market_A_rel['Date_Time']- market_A_rel['Ann_Date_Time']).dt.total_seconds()/(60*5)
market_A_rel['Min before announcement']=market_A_rel['Min before announcement'].apply(np.floor)
market_A_rel=market_A_rel.rename(columns={'Min before announcement':'Rel Time'})
market_A_rel


# In[20]:


'''
In the following cells we calculate the noise in the market reaction to the announcement.
'''


# In[21]:


'''
In the following cells we calculate the noise in the market reaction to the announcement.

Since we consider the announcement return AND the return the next day for the calculation of noise, 
we import daily Market returns from CRSP
'''
Market_ER= pd.read_stata('Market_ER_daily.dta') 
Market_ER['mktrf']=np.log(1+Market_ER['mktrf'])
Market_ER['Post_A']=Market_ER['mktrf'].rolling(1).sum().shift(-1)
Market_ER['date']=Market_ER['date'].astype(str)
Market_ER=Market_ER.dropna()
Market_ER=Market_ER[['date','Post_A']]


# In[22]:


#We create a date index which we need for the loop
date_index=pd.DataFrame(FOMC_Dates_List)
date_index=date_index.rename(columns={'Announcement Date':'Dates'})
date_index['date_index']=range(1,1+len(date_index))
date_index=date_index.sort_values(by='Dates')


# In[23]:


'''
We need the total return over the announcement day and the following day
and the cumulative return over the announcement day
'''

Excess_R2_2=market_A_rel
full_day_ret=Excess_R2_2
full_day_ret=full_day_ret[full_day_ret['Rel Time']>=-51]
full_day_ret=pd.DataFrame(full_day_ret.groupby('Dates')['Market_Return'].sum())
full_day_ret=full_day_ret.rename(columns={'Market_Return':'Full_Day_Ret'})

Excess_R2_3=pd.merge(Excess_R2_2,full_day_ret,how='left',left_on='Dates', right_on='Dates')
Excess_R2_3=Excess_R2_3[Excess_R2_3['Rel Time']>=-51]
Excess_R2_3=Excess_R2_3[Excess_R2_3['Rel Time']<=19]
Excess_R2_3['cum_ret']=Excess_R2_3.groupby('Dates')['Market_Return'].cumsum()

Excess_R2_4=pd.merge(Excess_R2_3,Market_ER,how='left',left_on='Dates',right_on='date')
Excess_R2_4['total ret']=Excess_R2_4['Full_Day_Ret']+Excess_R2_4['Post_A']
Excess_R2_4=pd.merge(Excess_R2_4,date_index,how='left',left_on='Dates',right_on='Dates')
Excess_R2_4


# In[24]:


'''
We run regressions over 3-year rolling windows (j=24 announcements)
The dependent variable is the total return over the announcement plus the followin day
and the independent variable is the cumulative return up to each point in time over the announcement day.
Market noise is (-)* the change of R^2 of these regressions around the announcement.
'''

j=24
R2_all_all=pd.DataFrame()

for i in range(1,(len(date_index)-j+1)):
    Test=Excess_R2_4[Excess_R2_4['date_index']>=i]
    Test=Test[Test['date_index']<=(i+j)]
    
    R2_all=[]
    
    for z in range(-51,20):
        Test1=Test[Test['Rel Time']==z]
        x=Test1['cum_ret']
        y=Test1['total ret']
        x = sm.add_constant(x) # adding a constant
        model = sm.OLS(y, x).fit(cov_type='HC3')
        R_2=model.rsquared
        R2_all.append(R_2)
        
    R2_all=pd.DataFrame(R2_all)
    R2_all['Rel Time']=range(-51,20)
    R2_all=R2_all.rename(columns={0:'R2'})
    R2_all['date_index']=(i+j)
    R2_all_1=pd.merge(R2_all,date_index,how='left',left_on='date_index',right_on='date_index')
    R2_all_all=pd.concat([R2_all_all,R2_all_1])


# In[25]:


Excess_R2=R2_all_all
Excess_R2_1=Excess_R2

#We use a window of 30 minutes arund the announcement 
Excess_R2_1['delta_R2']=Excess_R2_1.groupby('Dates')['R2'].diff(7)
Excess_R2_1=Excess_R2_1[Excess_R2_1['Rel Time']==6]
Excess_R2_1=Excess_R2_1.set_index('Dates')
Excess_R2_1['noise']=-Excess_R2_1['delta_R2']
ax=Excess_R2_1['noise'].plot()
for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
    
ax.legend()


# In[26]:


#Export the data to create the plot above in Latex

Market_Noise=Excess_R2_1[['noise']]
Market_Noise['Dates']=Market_Noise.index
Market_Noise['Year']=pd.to_datetime(Market_Noise['Dates']).dt.year
Market_Noise['Month']=pd.to_datetime(Market_Noise['Dates']).dt.month
Market_Noise['Date']=Market_Noise['Year'].astype(str)+':'+Market_Noise['Month'].astype(str)
Market_Noise=Market_Noise.set_index('Date')
Market_Noise=Market_Noise[['noise']]
Market_Noise['date_index']=range(1,1+len(Market_Noise))
Market_Noise['date_index']=Market_Noise['date_index'].astype(str)+':0'
Market_Noise['yearm']=Market_Noise.index
Market_Noise=Market_Noise.set_index('date_index')
Market_Noise.to_csv('Market_Noise.csv')


# In[27]:


#Our empirical counterparts of equilibrium 1 and 2 (E1 and E2) are PC and non-PC days, respectively

E1_Dates=PCs
E2_Dates=non_PCs


# In[28]:


'''
Specify the announcement days with returns that are higher/lower than 75% of daily returns (--> PN/NN days)
In addition, keep the corresponding date before the announcement, which are needed for further tests later
'''

Daily_Returns['Dates']=Daily_Returns.index

#E1
daily_E1=Daily_Returns[Daily_Returns['Dates'].isin(E1_Dates)]
E1_PN=daily_E1[daily_E1['Market_Return']>daily_E1['Market_Return'].quantile(0.75)]
E1_PN=E1_PN['Dates']
E1_PN_bef=pd.DataFrame(E1_PN)
E1_PN_bef['Dates']=(pd.to_datetime(E1_PN_bef['Dates'])-BDay(1)).astype(str)
E1_PN_bef=E1_PN_bef['Dates']
E1_NN=daily_E1[daily_E1['Market_Return']<daily_E1['Market_Return'].quantile(0.25)]
E1_NN=E1_NN['Dates']
E1_NN_bef=pd.DataFrame(E1_NN)
E1_NN_bef['Dates']=(pd.to_datetime(E1_NN_bef['Dates'])-BDay(1)).astype(str)
E1_NN_bef=E1_NN_bef['Dates']

#E2
daily_E2=Daily_Returns[Daily_Returns['Dates'].isin(E2_Dates)]
E2_PN=daily_E2[daily_E2['Market_Return']>daily_E2['Market_Return'].quantile(0.75)]
E2_PN=E2_PN['Dates']
E2_PN_bef=pd.DataFrame(E2_PN)
E2_PN_bef['Dates']=(pd.to_datetime(E2_PN_bef['Dates'])-BDay(1)).astype(str)
E2_PN_bef=E2_PN_bef['Dates']

E2_NN=daily_E2[daily_E2['Market_Return']<daily_E2['Market_Return'].quantile(0.25)]
E2_NN=E2_NN['Dates']
E2_NN_bef=pd.DataFrame(E2_NN)
E2_NN_bef['Dates']=(pd.to_datetime(E2_NN_bef['Dates'])-BDay(1)).astype(str)
E2_NN_bef=E2_NN_bef['Dates']

#All dates together
NN_all_bef=E1_NN_bef.append(E2_NN_bef)
PN_all_bef=E1_PN_bef.append(E2_PN_bef)

NN_all=E1_NN.append(E2_NN)
PN_all=E1_PN.append(E2_PN)


# In[29]:


market=market[['Market_Return','Time','Dates']]
market_A=market_A[['Market_Return','Time','Dates']]
market_NA=market_NA[['Market_Return','Time','Dates']]


# In[30]:


#Produce cumulative returns on all A days (not in paper, but presentation)

market_A_whole_Cumsum=market_A.groupby('Dates').cumsum()
market_A_whole_Cumsum['Time']=market_A_whole_Cumsum.index
market_A_whole_Cumsum['Time']=pd.to_datetime(market_A_whole_Cumsum['Time']).dt.time
market_A_whole_Cumsum=market_A_whole_Cumsum.groupby('Time').mean()
market_A_whole_Cumsum=market_A_whole_Cumsum.rename(columns={'Market_Return':'A Days'})
market_A_whole_Cumsum=market_A_whole_Cumsum*10000
market_A_whole_Cumsum.plot()

market_NA_whole_Cumsum=market_NA.groupby('Dates').cumsum()
market_NA_whole_Cumsum['Time']=market_NA_whole_Cumsum.index
market_NA_whole_Cumsum['Time']=pd.to_datetime(market_NA_whole_Cumsum['Time']).dt.time
market_NA_whole_Cumsum=market_NA_whole_Cumsum.groupby('Time').mean()
market_NA_whole_Cumsum=market_NA_whole_Cumsum.rename(columns={'Market_Return':'NA Days'})
market_NA_whole_Cumsum=market_NA_whole_Cumsum*10000
market_NA_whole_Cumsum

Cum_Returns=pd.merge(market_A_whole_Cumsum,market_NA_whole_Cumsum,left_index=True, right_index=True)
Cum_Returns['Time']=Cum_Returns.index
Cum_Returns['Time']=pd.to_datetime(Cum_Returns['Time'].astype(str)).dt.strftime("%H:%M")


# In[31]:


#Adjust and save the data to reproduce the above plot of cumulative returns in Latex

Cum_Returns=Cum_Returns.replace(['09:34'],'9:34')
Cum_Returns=Cum_Returns.replace(['09:39'],'9:39')
Cum_Returns=Cum_Returns.replace(['09:44'],'9:44')
Cum_Returns=Cum_Returns.replace(['09:49'],'9:49')
Cum_Returns=Cum_Returns.replace(['09:54'],'9:54')
Cum_Returns=Cum_Returns.replace(['09:59'],'9:59')
Cum_Returns=Cum_Returns.replace(['10:04'],'10:4')
Cum_Returns=Cum_Returns.replace(['10:09'],'10:9')
Cum_Returns=Cum_Returns.replace(['11:04'],'11:4')
Cum_Returns=Cum_Returns.replace(['11:09'],'11:9')
Cum_Returns=Cum_Returns.replace(['12:04'],'12:4')
Cum_Returns=Cum_Returns.replace(['12:09'],'12:9')
Cum_Returns=Cum_Returns.replace(['13:04'],'13:4')
Cum_Returns=Cum_Returns.replace(['13:09'],'13:9')
Cum_Returns=Cum_Returns.replace(['14:04'],'14:4')
Cum_Returns=Cum_Returns.replace(['14:09'],'14:9')
Cum_Returns=Cum_Returns.replace(['15:04'],'15:4')
Cum_Returns=Cum_Returns.replace(['15:09'],'15:9')
Cum_Returns=Cum_Returns.set_index('Time')
#Cum_Returns.to_csv('Cum_Returns.csv')


# In[32]:


#Calculate cumulative returns, conditional on PC days (E1) and good/bad news (PN/NN)

market_NA=market_NA
market_NA_cumsum=market_NA.groupby('Dates').cumsum()
market_NA_cumsum['Time']=market_NA_cumsum.index
market_NA_cumsum['Time']=pd.to_datetime(market_NA_cumsum['Time']).dt.time
market_NA_cumsum=market_NA_cumsum.groupby('Time').mean()
market_NA_cumsum=market_NA_cumsum.rename(columns={'Market_Return':'NA days'})
market_NA_cumsum=market_NA_cumsum*10000

market_full_E1=market_A[market_A['Dates'].isin(E1_Dates)]
market_full_cumsum=market_full_E1.groupby('Dates').cumsum()
market_full_cumsum['Time']=market_full_cumsum.index
market_full_cumsum['Time']=pd.to_datetime(market_full_cumsum['Time']).dt.time
market_full_cumsum=market_full_cumsum.groupby('Time').mean()
market_full_cumsum=market_full_cumsum.rename(columns={'Market_Return':'A Days'})
market_full_cumsum=market_full_cumsum*10000

market_PN_E1=market_A[market_A['Dates'].isin(E1_PN)]
market_PN_cumsum=market_PN_E1.groupby('Dates').cumsum()
market_PN_cumsum['Time']=market_PN_cumsum.index
market_PN_cumsum['Time']=pd.to_datetime(market_PN_cumsum['Time']).dt.time
market_PN_cumsum=market_PN_cumsum.groupby('Time').mean()
market_PN_cumsum=market_PN_cumsum.rename(columns={'Market_Return':'PN Days'})
market_PN_cumsum=market_PN_cumsum*10000

market_NN_E1=market_A[market_A['Dates'].isin(E1_NN)]
market_NN_cumsum=market_NN_E1.groupby('Dates').cumsum()
market_NN_cumsum['Time']=market_NN_cumsum.index
market_NN_cumsum['Time']=pd.to_datetime(market_NN_cumsum['Time']).dt.time
market_NN_cumsum=market_NN_cumsum.groupby('Time').mean()
market_NN_cumsum=market_NN_cumsum.rename(columns={'Market_Return':'NN Days'})
market_NN_cumsum=market_NN_cumsum*10000


ax=market_full_cumsum.plot(color='peru',linestyle='solid', title='E1')
market_PN_cumsum.plot(ax=ax, color="black", linestyle='solid')
market_NN_cumsum.plot(ax=ax, color="firebrick", linestyle='solid')

                                 
ax.set_xticks([36000, 45000,50400, 57600])

ax.axvspan(50400, 51540, alpha=0.55, color='grey')

#ax.axvline(50400, color='grey',linestyle='dashed')

plt.legend(loc=2, prop={'size': 8})

plt.ylabel('Cum Returns (bps)')

plt.ylim(-130,150)

#ax.figure.savefig("Cumulative Returns_2005.pdf")


# In[33]:


#Calculate cumulative returns, conditional on non-PC days (E2) and good/bad news (PN/NN)

market_NA=market_NA
market_NA_cumsum=market_NA.groupby('Dates').cumsum()
market_NA_cumsum['Time']=market_NA_cumsum.index
market_NA_cumsum['Time']=pd.to_datetime(market_NA_cumsum['Time']).dt.time
market_NA_cumsum=market_NA_cumsum.groupby('Time').mean()
market_NA_cumsum=market_NA_cumsum.rename(columns={'Market_Return':'NA days'})
market_NA_cumsum=market_NA_cumsum*10000

market_full_E2=market_A[market_A['Dates'].isin(E2_Dates)]
market_full_cumsum=market_full_E2.groupby('Dates').cumsum()
market_full_cumsum['Time']=market_full_cumsum.index
market_full_cumsum['Time']=pd.to_datetime(market_full_cumsum['Time']).dt.time
market_full_cumsum=market_full_cumsum.groupby('Time').mean()
market_full_cumsum=market_full_cumsum.rename(columns={'Market_Return':'A Days'})
market_full_cumsum=market_full_cumsum*10000

market_PN_E2=market_A[market_A['Dates'].isin(E2_PN)]
market_PN_cumsum=market_PN_E2.groupby('Dates').cumsum()
market_PN_cumsum['Time']=market_PN_cumsum.index
market_PN_cumsum['Time']=pd.to_datetime(market_PN_cumsum['Time']).dt.time
market_PN_cumsum=market_PN_cumsum.groupby('Time').mean()
market_PN_cumsum=market_PN_cumsum.rename(columns={'Market_Return':'PN Days'})
market_PN_cumsum=market_PN_cumsum*10000

market_NN_E2=market_A[market_A['Dates'].isin(E2_NN)]
market_NN_cumsum=market_NN_E2.groupby('Dates').cumsum()
market_NN_cumsum['Time']=market_NN_cumsum.index
market_NN_cumsum['Time']=pd.to_datetime(market_NN_cumsum['Time']).dt.time
market_NN_cumsum=market_NN_cumsum.groupby('Time').mean()
market_NN_cumsum=market_NN_cumsum.rename(columns={'Market_Return':'NN Days'})
market_NN_cumsum=market_NN_cumsum*10000


ax=market_full_cumsum.plot(color='peru',linestyle='solid', title='E2')
market_PN_cumsum.plot(ax=ax, color="black", linestyle='solid')
market_NN_cumsum.plot(ax=ax, color="firebrick", linestyle='solid')
                                 
ax.set_xticks([36000, 45000,50400, 57600])

ax.axvspan(50400, 51540, alpha=0.55, color='grey')


plt.legend(loc=2, prop={'size': 8})

plt.ylabel('Cum Returns (bps)')

#plt.ylim(-130,180)

#ax.figure.savefig("Cumulative Returns_2005.pdf")


# In[34]:


'''
Concat Dataframes to generate figure 7 in the paper
Note: the code below generates the dataset that is needed to produce figure 7 in
the paper. But it is the same code for PC and non PC days, so the respective cell above must
'''

Cum_Returns_SA=pd.merge(market_full_cumsum,market_NA_cumsum, how='left', left_on=['Time'], right_on=['Time'])
Cum_Returns_SA=pd.merge(Cum_Returns_SA,market_PN_cumsum, how='left', left_on=['Time'], right_on=['Time'])
Cum_Returns_SA=pd.merge(Cum_Returns_SA,market_NN_cumsum, how='left', left_on=['Time'], right_on=['Time'])

Cum_Returns_SA['Time']=Cum_Returns_SA.index
Cum_Returns_SA['Time']=pd.to_datetime(Cum_Returns_SA['Time'].astype(str)).dt.strftime("%H:%M")

Cum_Returns_SA=Cum_Returns_SA.replace(['09:34'],'9:34')
Cum_Returns_SA=Cum_Returns_SA.replace(['09:39'],'9:39')
Cum_Returns_SA=Cum_Returns_SA.replace(['09:44'],'9:44')
Cum_Returns_SA=Cum_Returns_SA.replace(['09:49'],'9:49')
Cum_Returns_SA=Cum_Returns_SA.replace(['09:54'],'9:54')
Cum_Returns_SA=Cum_Returns_SA.replace(['09:59'],'9:59')
Cum_Returns_SA=Cum_Returns_SA.replace(['10:04'],'10:4')
Cum_Returns_SA=Cum_Returns_SA.replace(['10:09'],'10:9')
Cum_Returns_SA=Cum_Returns_SA.replace(['11:04'],'11:4')
Cum_Returns_SA=Cum_Returns_SA.replace(['11:09'],'11:9')
Cum_Returns_SA=Cum_Returns_SA.replace(['12:04'],'12:4')
Cum_Returns_SA=Cum_Returns_SA.replace(['12:09'],'12:9')
Cum_Returns_SA=Cum_Returns_SA.replace(['13:04'],'13:4')
Cum_Returns_SA=Cum_Returns_SA.replace(['13:09'],'13:9')
Cum_Returns_SA=Cum_Returns_SA.replace(['14:04'],'14:4')
Cum_Returns_SA=Cum_Returns_SA.replace(['14:09'],'14:9')
Cum_Returns_SA=Cum_Returns_SA.replace(['15:04'],'15:4')
Cum_Returns_SA=Cum_Returns_SA.replace(['15:09'],'15:9')
Cum_Returns_SA=Cum_Returns_SA.set_index('Time')

Cum_Returns_SA.to_csv('Cum_Returns_PC.csv')


# In[35]:


'''
The following cells consist of the code that produces the summary stats (table 1 in the paper).
Note: "E1" and "E2" needs to be adjusted if one wants to produce the results for E1 or E2, respectively.
'''


# In[36]:


#All A Days

FOMC_Dates['Announcement Date']=FOMC_Dates['Announcement Date'].astype(str)
market_A1=market_full_E2
market_A=market_A1.rename(columns={'Dates':'Announcement Date'})
market_A_1=pd.merge(market_A,FOMC_Dates, how='left',left_on=['Announcement Date'],right_on=['Announcement Date'])
market_A_1['Ann_Date_Time']=market_A_1['Announcement Date'] + ' ' + market_A_1['Announcement Time'].astype(str)
market_A_1['Date_Time']=market_A_1['Announcement Date'] + ' ' + market_A_1['Time'].astype(str)
market_A_1['Ann_Date_Time']=pd.to_datetime(market_A_1['Ann_Date_Time'])
market_A_1['Date_Time']=pd.to_datetime(market_A_1['Date_Time'])

market_A_1['Min before announcement']=(market_A_1['Date_Time']- market_A_1['Ann_Date_Time']).dt.total_seconds()/(60*5)
market_A_1['Min before announcement']=market_A_1['Min before announcement'].apply(np.floor)

A_1_pre_drift=market_A_1[market_A_1['Min before announcement']<=-1]
A_1_pre_drift=A_1_pre_drift[['Announcement Date','Time', 'Market_Return']]
A_1_pre_drift=A_1_pre_drift.groupby('Announcement Date').sum()
A_1_pre_drift=A_1_pre_drift.rename(columns={'Market_Return':'Pre_A'})
average_A_1_pre_drift=A_1_pre_drift.mean()
p_value_Drift_A=stats.ttest_1samp(A_1_pre_drift, popmean=0)
print('t-test Drift A:',p_value_Drift_A)
print('pre-A Drift A (bps):',average_A_1_pre_drift*10000)

Post_A_Return_A=market_A_1[market_A_1['Min before announcement']>-1]
Post_A_Return_A=Post_A_Return_A[['Announcement Date','Time', 'Market_Return']]
Post_A_Return_A=Post_A_Return_A.groupby('Announcement Date').sum()  
Post_A_Return_A=Post_A_Return_A.rename(columns={'Market_Return':'Post_A'})

A_PI=pd.merge(A_1_pre_drift,Post_A_Return_A, how='left', left_on=['Announcement Date'],right_on=['Announcement Date'])
A_PI['Return']=A_PI['Pre_A']+A_PI['Post_A']
Average_Return_A=A_PI['Return'].mean()
p_value_Return_A=stats.ttest_1samp(A_PI['Return'], popmean=0)
print('p-value return A:',p_value_Return_A)
print('Average Return A (bps):',Average_Return_A*10000)

A_PI['PI']=A_PI['Pre_A']/A_PI['Post_A']
A_PI['informative']=np.where(A_PI['PI']>0,1,0)
informative_A=A_PI[A_PI['informative']==1]
A_PI_1=len(informative_A)/len(A_PI)

print('PI A:',A_PI_1)


# In[37]:


#PN (in order to produce the results for E1/E2, market_PN_E1(E2, respectively) must be adjusted in the code


FOMC_Dates['Announcement Date']=FOMC_Dates['Announcement Date'].astype(str)

market_PN=market_PN_E1.rename(columns={'Dates':'Announcement Date'})
market_PN_1=pd.merge(market_PN,FOMC_Dates, how='left',left_on=['Announcement Date'],right_on=['Announcement Date'])
market_PN_1['Ann_Date_Time']=market_PN_1['Announcement Date'] + ' ' + market_PN_1['Announcement Time'].astype(str)
market_PN_1['Date_Time']=market_PN_1['Announcement Date'] + ' ' + market_PN_1['Time'].astype(str)
market_PN_1['Ann_Date_Time']=pd.to_datetime(market_PN_1['Ann_Date_Time'])
market_PN_1['Date_Time']=pd.to_datetime(market_PN_1['Date_Time'])

market_PN_1['Min before announcement']=(market_PN_1['Date_Time']- market_PN_1['Ann_Date_Time']).dt.total_seconds()/(60*5)
market_PN_1['Min before announcement']=market_PN_1['Min before announcement'].apply(np.floor)


PN_1_pre_drift=market_PN_1[market_PN_1['Min before announcement']<=-1]
PN_1_pre_drift=PN_1_pre_drift[['Announcement Date','Time', 'Market_Return']]
PN_1_pre_drift=PN_1_pre_drift.groupby('Announcement Date').sum()
PN_1_pre_drift=PN_1_pre_drift.rename(columns={'Market_Return':'Pre_A'})
average_PN_1_pre_drift=PN_1_pre_drift.mean()
p_value_Drift_PN=stats.ttest_1samp(PN_1_pre_drift, popmean=0)
print('t-test Drift PN:',p_value_Drift_PN)
print('pre-A Drift PN (bps):',average_PN_1_pre_drift*10000)

Post_A_Return_PN=market_PN_1[market_PN_1['Min before announcement']>-1]
Post_A_Return_PN=Post_A_Return_PN[['Announcement Date','Time', 'Market_Return']]
Post_A_Return_PN=Post_A_Return_PN.groupby('Announcement Date').sum()  
Post_A_Return_PN=Post_A_Return_PN.rename(columns={'Market_Return':'Post_A'})

PN_PI=pd.merge(PN_1_pre_drift,Post_A_Return_PN, how='left', left_on=['Announcement Date'],right_on=['Announcement Date'])
PN_PI['Return']=PN_PI['Pre_A']+PN_PI['Post_A']
Average_Return_PN=PN_PI['Return'].mean()
p_value_Return_PN=stats.ttest_1samp(PN_PI['Return'], popmean=0)
print('p-value return PN:',p_value_Return_PN)
print('Average Return PN (bps):',Average_Return_PN*10000)

PN_PI['PI']=PN_PI['Pre_A']/PN_PI['Post_A']
PN_PI['informative']=np.where(PN_PI['PI']>0,1,0)
informative_PN=PN_PI[PN_PI['informative']==1]
PN_PI_1=len(informative_PN)/len(PN_PI)

print('PI PN:',PN_PI_1)


# In[38]:


#NN (in order to produce the results for E1/E2, market_PN_E1(E2, respectively) must be adjusted in the code

FOMC_Dates['Announcement Date']=FOMC_Dates['Announcement Date'].astype(str)

market_NN=market_NN_E2.rename(columns={'Dates':'Announcement Date'})
market_NN_1=pd.merge(market_NN,FOMC_Dates, how='left',left_on=['Announcement Date'],right_on=['Announcement Date'])
market_NN_1['Ann_Date_Time']=market_NN_1['Announcement Date'] + ' ' + market_NN_1['Announcement Time'].astype(str)
market_NN_1['Date_Time']=market_NN_1['Announcement Date'] + ' ' + market_NN_1['Time'].astype(str)
market_NN_1['Ann_Date_Time']=pd.to_datetime(market_NN_1['Ann_Date_Time'])
market_NN_1['Date_Time']=pd.to_datetime(market_NN_1['Date_Time'])

market_NN_1['Min before announcement']=(market_NN_1['Date_Time']- market_NN_1['Ann_Date_Time']).dt.total_seconds()/(60*5)
market_NN_1['Min before announcement']=market_NN_1['Min before announcement'].apply(np.floor)


NN_1_pre_drift=market_NN_1[market_NN_1['Min before announcement']<=-1]
NN_1_pre_drift=NN_1_pre_drift[['Announcement Date','Time', 'Market_Return']]
NN_1_pre_drift=NN_1_pre_drift.groupby('Announcement Date').sum()
NN_1_pre_drift=NN_1_pre_drift.rename(columns={'Market_Return':'Pre_A'})
average_NN_1_pre_drift=NN_1_pre_drift.mean()
p_value_Drift_NN=stats.ttest_1samp(NN_1_pre_drift, popmean=0)
print('t-test Drift NN:',p_value_Drift_NN)
print('pre-A Drift NN (bps):',average_NN_1_pre_drift*10000)

Post_A_Return_NN=market_NN_1[market_NN_1['Min before announcement']>-1]
Post_A_Return_NN=Post_A_Return_NN[['Announcement Date','Time', 'Market_Return']]
Post_A_Return_NN=Post_A_Return_NN.groupby('Announcement Date').sum()  
Post_A_Return_NN=Post_A_Return_NN.rename(columns={'Market_Return':'Post_A'})

NN_PI=pd.merge(NN_1_pre_drift,Post_A_Return_NN, how='left', left_on=['Announcement Date'],right_on=['Announcement Date'])
NN_PI['Return']=NN_PI['Pre_A']+NN_PI['Post_A']
Average_Return_NN=NN_PI['Return'].mean()
p_value_Return_NN=stats.ttest_1samp(NN_PI['Return'], popmean=0)
print('p-value return NN:',p_value_Return_NN)
print('Average Return NN (bps):',Average_Return_NN*10000)


NN_PI['PI']=NN_PI['Pre_A']/NN_PI['Post_A']
NN_PI['informative']=np.where(NN_PI['PI']>0,1,0)
informative_NN=NN_PI[NN_PI['informative']==1]
NN_PI_1=len(informative_NN)/len(NN_PI)

print('PI NN:',NN_PI_1)


# In[39]:


#Produce Latex code for table 1 in paper

d = {'A-Days': ['%.2f'%(Average_Return_A*10000),'%.2f'%p_value_Return_A.statistic, '%.2f'%(average_A_1_pre_drift*10000), '%.2f'% np.float(p_value_Drift_A.statistic),'%.2f'%(A_PI_1*100),'%.0f'%len(A_PI)], 'PN Days': ['%.2f'%(Average_Return_PN*10000),'%.2f'%np.float(p_value_Return_PN.statistic), '%.2f'%np.float(average_PN_1_pre_drift*10000),'%.2f'%np.float(p_value_Drift_PN.statistic),'%.2f'%(PN_PI_1*100),'%.0f'%len(PN_PI)], 'NN Days': ['%.2f'%(Average_Return_NN*10000), '%.2f'% p_value_Return_NN.statistic, '%.2f'% np.float(average_NN_1_pre_drift*10000), '%.2f'%np.float(p_value_Drift_NN.statistic), '%.2f'%(NN_PI_1*100),'%.0f'%len(NN_PI)]}
Table_descriptive=pd.DataFrame(data=d, index=["Return (bps)"," ", "Pre-A Drift (bps)","", "PI", "Observations"])
Table_descriptive=round(Table_descriptive,4)
Table_descriptive
print(Table_descriptive.to_latex())


# In[40]:


'''
The following cells produce the results for tables 2 and 3.
'''


# In[41]:


'''
Import 7-10 year treasury ETF (Ticker:IEF) and 1-3 year treasury bond (Ticker: SHY) and create an equally-weighted portfolio of these two ETFs
Data is retrieved from CRSP
'''
datafile = 'Gov_Bond_Returns.dta'
Bonds, meta = pyreadstat.read_dta(datafile)
Bonds=Bonds.rename(columns={'DlyCalDt':'Dates'})
Bonds['Dates']=Bonds['Dates'].astype(str)
SP500 = pd.read_stata('Market_ER_daily.dta')
Bond_Returns=pd.DataFrame(Bonds.groupby('Dates')['DlyRet'].mean())
Bond_Returns=Bond_Returns.rename(columns={'DlyRet':'Bond return'})
Bond_Returns['Dates']=Bond_Returns.index
Bond_Returns=Bond_Returns.reset_index(drop=True)
Bonds=Bond_Returns


# In[42]:


#Import Treasury Yields (for 2001-mid2002, as ETFs only available as of mid 2002)

Treasury_Yields = pd.read_csv('Yield Curve.csv', parse_dates=True)
Treasury_Yields['Date']=pd.to_datetime(Treasury_Yields['Date'])
Treasury_Yields=Treasury_Yields[['Date','1 Yr','2 Yr','3 Yr','7 Yr','10 Yr']]
Treasury_Yields=Treasury_Yields.sort_values(by='Date', ascending=True)
Treasury_Yields['Date']=Treasury_Yields['Date'].astype(str)
Treasury_Yields=Treasury_Yields.set_index('Date')
Treasury_Yields_change=Treasury_Yields.diff()

'''
We multiply with -1 so that a DECREASE in the yields correspond to an INCREASE in bond prices
Note that the direction of interest rate changes is important for the analysis, therefore it doesn't matter
that changes in treasury yields and return on bond ETFs do not have the same unit.
'''

Treasury_Yields_change=Treasury_Yields_change*-1
Treasury_Yields_change['Bond return']=1/4*Treasury_Yields_change['1 Yr']+1/4*Treasury_Yields_change['3 Yr']+1/4*Treasury_Yields_change['7 Yr']+1/4*Treasury_Yields_change['10 Yr']
Treasury_Yields_change['Bond return']=Treasury_Yields_change['Bond return']/100
Treasury_Yields_change=Treasury_Yields_change[['Bond return']]
Treasury_Yields_change['Dates']=Treasury_Yields_change.index
Treasury_Yields_change=Treasury_Yields_change[Treasury_Yields_change['Dates']>='2001-01-01']
Treasury_Yields_change=Treasury_Yields_change[Treasury_Yields_change['Dates']<'2002-07-29']
Treasury_Yields_change=Treasury_Yields_change.reset_index(drop=True)
Bonds=pd.concat([Treasury_Yields_change,Bonds])
Bonds


# In[43]:


#Import Fed Funds Target Rate (FFTR) from FRED

FFTR_1 = pd.read_csv('FTR_2008.csv', sep=',')
FFTR_2 = pd.read_csv('FTR2008_.csv', sep=',')
FFTR_1=FFTR_1.rename(columns={'DFEDTAR':'FFTR'})
FFTR_2=FFTR_2.rename(columns={'DFEDTARL':'FFTR'})
FFTR=pd.concat([FFTR_1,FFTR_2])
FFTR=FFTR.rename(columns={'DATE':'Date'})
FFTR['Date']=FFTR['Date'].astype(str)
FFTR=FFTR[FFTR['FFTR']!='#NV']
FFTR['FFTR']=pd.to_numeric(FFTR['FFTR'])
FFTR['FFTR change (bps)']=FFTR['FFTR'].diff()*100
FFTR


# In[44]:


'''
Import 30-day FF futures obtained from Thomson Reuters
To obtain the surprise changes in the Fed funds rate, we follow Kuttner (2001)

'''

FF_Futures = pd.read_csv('FedFundsFutures.csv')
FF_Futures=FF_Futures.drop('Unnamed: 0',1)
FF_Futures['FFF_diff']=FF_Futures['Settlement Price'].diff()
FF_Futures['Date']=pd.to_datetime(FF_Futures['Date'])
FF_Futures['days in month']=pd.to_datetime(FF_Futures['Date']).dt.days_in_month
FF_Futures['day of month']=pd.to_datetime(FF_Futures['Date']).dt.day
FF_Futures['remaining days']=FF_Futures['days in month']-FF_Futures['day of month']
FF_Futures['scaling factor']=FF_Futures['days in month']/(FF_Futures['days in month']-FF_Futures['day of month'])
FF_Futures['unexp FF change (bps)']=(FF_Futures['FFF_diff']*FF_Futures['scaling factor'])*100*-1
FF_Futures


# In[45]:


'''
We import data of "individual" futures (as opposed to "contiuous" like above).
Individual futures are the futures that expire in the respective month.
We need to make slight adjustments at the end/the beginning of each
month when the "old" month's contract expires and liquidity moves to the new month's contract.
And the "individual" futures provide more detailed information which is needed to do this.
'''

FF_Futures_ind = pd.read_csv('FFF_individual.csv', sep=';')
#FF_Futures_ind=FF_Futures_ind[FF_Futures_ind['YearMonth'].isin(Required_Futures['Required Futures'])]
FF_Futures_ind['Date']=pd.to_datetime(FF_Futures_ind['Date'],dayfirst=True)
FF_Futures_ind['maturity month']=FF_Futures_ind['Name'].str[28:31]
FF_Futures_ind['maturity year']=FF_Futures_ind['Name'].str[31:36]
FF_Futures_ind['maturity month'] = FF_Futures_ind['maturity month'].replace(['JAN'],'01')
FF_Futures_ind['maturity month'] = FF_Futures_ind['maturity month'].replace(['FEB'],'02')
FF_Futures_ind['maturity month'] = FF_Futures_ind['maturity month'].replace(['MAR'],'03')
FF_Futures_ind['maturity month'] = FF_Futures_ind['maturity month'].replace(['APR'],'04')
FF_Futures_ind['maturity month'] = FF_Futures_ind['maturity month'].replace(['MAY'],'05')
FF_Futures_ind['maturity month'] = FF_Futures_ind['maturity month'].replace(['JUN'],'06')
FF_Futures_ind['maturity month'] = FF_Futures_ind['maturity month'].replace(['JUL'],'07')
FF_Futures_ind['maturity month'] = FF_Futures_ind['maturity month'].replace(['AUG'],'08')
FF_Futures_ind['maturity month'] = FF_Futures_ind['maturity month'].replace(['SEP'],'09')
FF_Futures_ind['maturity month'] = FF_Futures_ind['maturity month'].replace(['OCT'],'10')
FF_Futures_ind['maturity month'] = FF_Futures_ind['maturity month'].replace(['NOV'],'11')
FF_Futures_ind['maturity month'] = FF_Futures_ind['maturity month'].replace(['DEC'],'12')

FF_Futures_ind['Maturity']=FF_Futures_ind['maturity year'].astype(str)+'-'+FF_Futures_ind['maturity month'].astype(str)
FF_Futures_ind['Maturity']=FF_Futures_ind['Maturity'].astype(str)
FF_Futures_ind=FF_Futures_ind[['Date','Name','Settlement Price','Maturity']]
FF_Futures_ind['Maturity']=FF_Futures_ind['Maturity'].str.strip()

FF_Futures_ind


# In[46]:


#Calculate FF rate surprise if FOMC on the first of the month

List_First_BD_of_month=pd.date_range('1/1/2000', '12/31/2022', freq = 'BMS')
List_First_BD_of_month=pd.DataFrame(List_First_BD_of_month)
List_First_BD_of_month=List_First_BD_of_month.rename(columns={0:'First BDay of Month'})
List_First_BD_of_month['Year']=List_First_BD_of_month['First BDay of Month'].dt.year
List_First_BD_of_month['Month']=List_First_BD_of_month['First BDay of Month'].dt.month
List_First_BD_of_month['Month'] = List_First_BD_of_month['Month'].astype(str).replace(['1'],'01')
List_First_BD_of_month['Month'] = List_First_BD_of_month['Month'].astype(str).replace(['2'],'02')
List_First_BD_of_month['Month'] = List_First_BD_of_month['Month'].astype(str).replace(['3'],'03')
List_First_BD_of_month['Month'] = List_First_BD_of_month['Month'].astype(str).replace(['4'],'04')
List_First_BD_of_month['Month'] = List_First_BD_of_month['Month'].astype(str).replace(['5'],'05')
List_First_BD_of_month['Month'] = List_First_BD_of_month['Month'].astype(str).replace(['6'],'06')
List_First_BD_of_month['Month'] = List_First_BD_of_month['Month'].astype(str).replace(['7'],'07')
List_First_BD_of_month['Month'] = List_First_BD_of_month['Month'].astype(str).replace(['8'],'08')
List_First_BD_of_month['Month'] = List_First_BD_of_month['Month'].astype(str).replace(['9'],'09')
List_First_BD_of_month['YearMonth']=List_First_BD_of_month['Year'].astype(str)+'-'+List_First_BD_of_month['Month'].astype(str)
List_First_BD_of_month=List_First_BD_of_month.drop(['Year','Month'],1)
Beginning_of_month_A=List_First_BD_of_month[List_First_BD_of_month['First BDay of Month'].isin(FOMC_Dates_List)]
FF_Futures_beg_month_A=FF_Futures[FF_Futures['Date'].isin(Beginning_of_month_A['First BDay of Month'])]
FF_Futures_beg_month_A['Maturity']=FF_Futures_beg_month_A['YearMonth'].astype(str)                            
FF_Futures_beg_month_A1=pd.merge(FF_Futures_ind,FF_Futures_beg_month_A,how='left', left_on=['Maturity'], right_on=['Maturity'])
FF_Futures_beg_month_A1=FF_Futures_beg_month_A1.dropna()
FF_Futures_beg_month_A1=FF_Futures_beg_month_A1[['Date_x','Date_y','Settlement Price_x','Settlement Price_y','Maturity']]
FF_Futures_beg_month_A1['Date_x']=pd.to_datetime(FF_Futures_beg_month_A1['Date_x'],dayfirst=True)
Last_Business_Days_of_M=pd.DataFrame(pd.date_range('1/1/2000','12/31/2020',  freq='BM'))
Last_Business_Days_of_M=Last_Business_Days_of_M.rename(columns={0:'Date_x'})
Last_Business_Days_of_M['first BD of next month']=Last_Business_Days_of_M['Date_x']+BusinessDay()
Last_Business_Days_of_M['Year']=Last_Business_Days_of_M['first BD of next month'].dt.year
Last_Business_Days_of_M['Month']=Last_Business_Days_of_M['first BD of next month'].dt.month
Last_Business_Days_of_M['Month'] = Last_Business_Days_of_M['Month'].astype(str).replace(['1'],'01')
Last_Business_Days_of_M['Month'] = Last_Business_Days_of_M['Month'].astype(str).replace(['2'],'02')
Last_Business_Days_of_M['Month'] = Last_Business_Days_of_M['Month'].astype(str).replace(['3'],'03')
Last_Business_Days_of_M['Month'] = Last_Business_Days_of_M['Month'].astype(str).replace(['4'],'04')
Last_Business_Days_of_M['Month'] = Last_Business_Days_of_M['Month'].astype(str).replace(['5'],'05')
Last_Business_Days_of_M['Month'] = Last_Business_Days_of_M['Month'].astype(str).replace(['6'],'06')
Last_Business_Days_of_M['Month'] = Last_Business_Days_of_M['Month'].astype(str).replace(['7'],'07')
Last_Business_Days_of_M['Month'] = Last_Business_Days_of_M['Month'].astype(str).replace(['8'],'08')
Last_Business_Days_of_M['Month'] = Last_Business_Days_of_M['Month'].astype(str).replace(['9'],'09')
Last_Business_Days_of_M['Maturity']=Last_Business_Days_of_M['Year'].astype(str)+'-'+Last_Business_Days_of_M['Month'].astype(str)
Last_Business_Days_of_M['Maturity']=Last_Business_Days_of_M['Maturity'].str.strip()
Last_Business_Days_of_M['Maturity']=Last_Business_Days_of_M['Maturity'].astype(str)
Beginning_of_Month_Futures=pd.merge(Last_Business_Days_of_M,FF_Futures_beg_month_A1,how='left',left_on=['Maturity','Date_x'], right_on=['Maturity','Date_x'])
Beginning_of_Month_Futures=Beginning_of_Month_Futures.dropna()
Beginning_of_Month_Futures=Beginning_of_Month_Futures[['Date_y','Settlement Price_x','Settlement Price_y']]
Beginning_of_Month_Futures['days in month']=pd.to_datetime(Beginning_of_Month_Futures['Date_y']).dt.days_in_month
Beginning_of_Month_Futures['day of month']=pd.to_datetime(Beginning_of_Month_Futures['Date_y']).dt.day
Beginning_of_Month_Futures['remaining days']=Beginning_of_Month_Futures['days in month']-Beginning_of_Month_Futures['day of month']
Beginning_of_Month_Futures['scaling factor']=Beginning_of_Month_Futures['days in month']/(Beginning_of_Month_Futures['remaining days']-Beginning_of_Month_Futures['day of month'])
Beginning_of_Month_Futures['unexp FF change (bps)']=(Beginning_of_Month_Futures['scaling factor']*(Beginning_of_Month_Futures['Settlement Price_y']-Beginning_of_Month_Futures['Settlement Price_x']))*-100
Beginning_of_Month_Futures=Beginning_of_Month_Futures[['Date_y','unexp FF change (bps)']]
Beginning_of_Month_Futures=Beginning_of_Month_Futures.rename(columns={'Date_y':'Announcement Date'})
Beginning_of_Month_Futures


# In[47]:


'''
The only complication to this calculation arises near ends of month. With liquidity moving to next
months’ contracts, small amounts of noise in the underlying price can cause severe measurement
noise as the denominator D−d becomes small (see Kuttner (2001)). We again follow Kuttner (2001) and use the change in
the next month futures price to calculate the change during the last three trading days of a month
'''

FOMC_Dates1=pd.DataFrame(FOMC_Dates_List)
FOMC_Dates1=FOMC_Dates1[FOMC_Dates1['Announcement Date']>'2001-01-01']
FOMC_Dates1['days in month']=pd.to_datetime(FOMC_Dates1['Announcement Date']).dt.days_in_month
FOMC_Dates1['day of month']=pd.to_datetime(FOMC_Dates1['Announcement Date']).dt.day
FOMC_Dates1['remaining days']=FOMC_Dates1['days in month']-FOMC_Dates1['day of month']


#FOMC dates that fall on last 3 days of month
FOMC_Dates1_last_3=FOMC_Dates1[FOMC_Dates1['remaining days'].isin([0,1,2,3])]
FOMC_Dates1_last_3['Announcement Date']=pd.to_datetime(FOMC_Dates1_last_3['Announcement Date'])
FOMC_Dates1_last_3['next month date']=pd.to_datetime(FOMC_Dates1_last_3['Announcement Date'])+BusinessDay(4)
FOMC_Dates1_last_3['next month']=FOMC_Dates1_last_3['next month date'].dt.month
FOMC_Dates1_last_3['next month year']=FOMC_Dates1_last_3['next month date'].dt.year
FOMC_Dates1_last_3['next month'] = FOMC_Dates1_last_3['next month'].astype(str).replace(['1'],'01')
FOMC_Dates1_last_3['next month'] = FOMC_Dates1_last_3['next month'].astype(str).replace(['2'],'02')
FOMC_Dates1_last_3['next month'] = FOMC_Dates1_last_3['next month'].astype(str).replace(['3'],'03')
FOMC_Dates1_last_3['next month'] = FOMC_Dates1_last_3['next month'].astype(str).replace(['4'],'04')
FOMC_Dates1_last_3['next month'] = FOMC_Dates1_last_3['next month'].astype(str).replace(['5'],'05')
FOMC_Dates1_last_3['next month'] = FOMC_Dates1_last_3['next month'].astype(str).replace(['6'],'06')
FOMC_Dates1_last_3['next month'] = FOMC_Dates1_last_3['next month'].astype(str).replace(['7'],'07')
FOMC_Dates1_last_3['next month'] = FOMC_Dates1_last_3['next month'].astype(str).replace(['8'],'08')
FOMC_Dates1_last_3['next month'] = FOMC_Dates1_last_3['next month'].astype(str).replace(['9'],'09')
FOMC_Dates1_last_3['Maturity']=FOMC_Dates1_last_3['next month year'].astype(str)+'-'+FOMC_Dates1_last_3['next month'].astype(str)
FOMC_Dates1_last_3['Announcement Date']=FOMC_Dates1_last_3['Announcement Date'].astype(str)
FF_Futures_ind['Date Maturity']=FF_Futures_ind['Date'].astype(str)+' '+FF_Futures_ind['Maturity'].astype(str)
FF_Futures_ind1=FF_Futures_ind.set_index('Date Maturity')
FF_Futures_ind2=pd.DataFrame(FF_Futures_ind1.groupby('Maturity')['Settlement Price'].diff())
FF_Futures_ind2['Date Maturity']=FF_Futures_ind2.index.astype(str)
FF_Futures_ind2['Maturity']=FF_Futures_ind2['Date Maturity'].str[11:]
FF_Futures_ind2['Announcement Date']=FF_Futures_ind2['Date Maturity'].str[:10]
FF_Futures_ind3=pd.merge(FOMC_Dates1_last_3,FF_Futures_ind2, how='left', left_on=['Maturity','Announcement Date'],right_on=['Maturity','Announcement Date'])              
FF_Futures_ind3['unexp FF change (bps)']=FF_Futures_ind3['Settlement Price']*-100
FF_Surprise_end_of_month=FF_Futures_ind3[['Announcement Date','unexp FF change (bps)']]


# In[48]:


#Concatenate all the above generated datasets to obtain the final datasets that consists of all FF surprises

A_Days_List_EoM=FOMC_Dates1_last_3['Announcement Date']
Beginning_of_Month_Futures['Announcement Date']=Beginning_of_Month_Futures['Announcement Date'].astype(str)
A_Days_List_BoM=Beginning_of_Month_Futures['Announcement Date']
List=pd.concat([A_Days_List_EoM,A_Days_List_BoM])
FF_Futures1=FF_Futures[FF_Futures['Date'].isin(FOMC_Dates_List)]
FF_Futures2=FF_Futures1[~FF_Futures1['Date'].isin(List)]        
FF_Futures3=FF_Futures2[FF_Futures2['Date']>='2001-01-01']  
FF_Futures4=FF_Futures3[['Date','unexp FF change (bps)']]
FF_Futures4=FF_Futures4.rename(columns={'Date':'Announcement Date'})
FF_Futures4['Announcement Date']=FF_Futures4['Announcement Date'].astype(str)
FF_Surprises=pd.concat([FF_Futures4,Beginning_of_Month_Futures,FF_Surprise_end_of_month])
FF_Surprises=FF_Surprises.drop_duplicates(subset=['Announcement Date'])
FF_Surprises=FF_Surprises.sort_values(by='Announcement Date')
FF_Surprises


# In[49]:



#For the proxy of "FedPutSignal" and "MP channel", we need market excess returns which we can download from CRSP.

Market_daily= pd.read_stata('Market_ER_daily.dta') 
Market_daily['date']= Market_daily['date'].astype(str)

#Merge all the datesets with the data we need to creeate the proxies in subsection 5.2.3 in the paper

Federal_Funds_Rate=pd.merge(FF_Surprises,FFTR,how='left',left_on=['Announcement Date'], right_on=['Date'])
Bonds_FF=pd.merge(Federal_Funds_Rate,Bonds,how='left',left_on=['Announcement Date'], right_on=['Dates'])
Bonds_FF=Bonds_FF.sort_values(by='Announcement Date', ascending=True)
Bonds_FF_Market=pd.merge(Bonds_FF,Market_daily,how='left',left_on=['Announcement Date'], right_on=['date'])
Bonds_FF_Market=Bonds_FF_Market[['Announcement Date','FFTR change (bps)','unexp FF change (bps)','Bond return','mktrf']]
Bonds_FF_Market


# In[50]:


#For the proxy FedPutSignal, we need the market excess returns between announcements (cycles)

Market_ER= pd.read_stata('Market_ER_daily.dta') 
Market_ER['market']=np.log(1+Market_ER['mktrf'])
Market_ER['Market_cum']=Market_ER['market'].cumsum()
Market_ER=Market_ER[['Market_cum','date']]
Market_ER['Announcement Date before']=Market_ER['date']+BDay(2)
Market_cum_before=Market_ER[['Announcement Date before','date','Market_cum']]
Market_cum_before['Announcement Date']=Market_cum_before['Announcement Date before'].astype(str)
Market_cum_before=Market_cum_before[['Announcement Date','date','Market_cum']]
Market_cum_before=Market_cum_before.rename(columns={'Market_cum':'Market_cum_pre'})

FOMC_Dates_all=pd.read_csv('FOMC Dates all.csv', sep=';')
Market_cum_after=Market_cum_before
Market_cum_after=Market_cum_after[['date','Market_cum_pre']]
Market_cum_after=Market_cum_after.rename(columns={'Market_cum_pre':'Market_cum_post'})
Market_cum_after=Market_cum_after.rename(columns={'date':'Announcement Date'})
Market_cum_after['Announcement Date']=Market_cum_after['Announcement Date'].astype(str)

FOMC_cycle_Returns=pd.merge(FOMC_Dates_all,Market_cum_before, how='left', left_on=['Announcement Date'], right_on=['Announcement Date'])
FOMC_cycle_Returns_1=pd.merge(FOMC_cycle_Returns,Market_cum_after, how='left', left_on=['Announcement Date'], right_on=['Announcement Date'])
FOMC_cycle_Returns_1=FOMC_cycle_Returns_1.dropna()
FOMC_cycle_Returns_1=FOMC_cycle_Returns_1.drop('date',1)
FOMC_cycle_Returns_1['previous_cycle_return']=FOMC_cycle_Returns_1['Market_cum_pre']-FOMC_cycle_Returns_1['Market_cum_post'].shift(1)
FOMC_cycle_Returns_2=FOMC_cycle_Returns_1[['Announcement Date','previous_cycle_return']]

Bonds_FF_Market_prevcycle_ret=pd.merge(Bonds_FF_Market,FOMC_cycle_Returns_2,how='left',left_on=['Announcement Date'], right_on=['Announcement Date'])
Bonds_FF_Market_prevcycle_ret['previous_cycle_return']=Bonds_FF_Market_prevcycle_ret['previous_cycle_return']*10000
Bonds_FF_Market_prevcycle_ret=Bonds_FF_Market_prevcycle_ret.rename(columns={'previous_cycle_return':'prev_cycl_ret (bps)'})

#Create "MP" dummy variable
Bonds_FF_Market_prevcycle_ret['Bond_Market']=Bonds_FF_Market_prevcycle_ret['Bond return']*Bonds_FF_Market_prevcycle_ret['mktrf']
Bonds_FF_Market_prevcycle_ret['MP']=np.where(Bonds_FF_Market_prevcycle_ret['Bond_Market']>0,1,0)
Bonds_FF_Market_prevcycle_ret


# In[51]:


'''
Evidence on short-selling constraints

Atmaz and Basak (2019): Higher shorting-fees lead to: higher bid-ask spreads (put call),
higher put option implied vol, and higher put call parity violations
'''

#Import options data on SPX, downloaded from OptionMetrics. Note that data is only available until December 2021

Options_SPX = pd.read_csv('Options_SPX1.csv') 
Options_SPX['date']=Options_SPX['date'].astype(str).str[:4]+'-'+Options_SPX['date'].astype(str).str[4:6]+'-'+Options_SPX['date'].astype(str).str[6:8]
Options_SPX['exdate']=Options_SPX['exdate'].astype(str).str[:4]+'-'+Options_SPX['exdate'].astype(str).str[4:6]+'-'+Options_SPX['exdate'].astype(str).str[6:8]
Options_SPX


# In[52]:


#Import data on USD 3-month LIBOR, which is used as a proxy for the risk-free rate

LIBOR_3M = pd.read_csv('USD 3M LIBOR.csv', sep=';')

LIBOR_3M['Date']=pd.to_datetime(LIBOR_3M['Date'])
LIBOR_3M=LIBOR_3M.sort_values(by='Date', ascending=True)
LIBOR_3M['Date']=LIBOR_3M['Date'].astype(str)
LIBOR_3M=LIBOR_3M[LIBOR_3M['Date']>'2001-01-01']


LIBOR_3M=LIBOR_3M.rename(columns={'PX_ASK':'LIBOR_3M'})
LIBOR_3M   

# We focus on the day before the announcement

FOMC_Dates_1=FOMC_Dates
A_Dates_bef=FOMC_Dates_1
A_Dates_bef['A Dates bef']=pd.to_datetime(A_Dates_bef['Announcement Date'])+BDay(-1)
A_Dates_bef=A_Dates_bef['A Dates bef'].astype(str)


# In[53]:


#Import data on S&P500 index
SP500 = pd.read_stata('SP500.dta') 
SP500['caldt']=SP500['DlyCalDt'].astype(str)
Options_SPX_1=pd.merge(Options_SPX,SP500, how='left', left_on='date', right_on='caldt')
Options_SPX_1=Options_SPX_1[Options_SPX_1['date'].isin(A_Dates_bef)]
Options_SPX_1['Time to Maturity']=pd.to_datetime(Options_SPX_1['exdate'])-pd.to_datetime(Options_SPX_1['date'])

Options_SPX_1=Options_SPX_1[Options_SPX_1['Time to Maturity']!='0 days']
Options_SPX_1=Options_SPX_1[Options_SPX_1['Time to Maturity']!='1 days']


Options_SPX_1_Calls=Options_SPX_1[Options_SPX_1['cp_flag']=='C']
Options_SPX_1_Puts=Options_SPX_1[Options_SPX_1['cp_flag']=='P']

Options_SPX_1_Calls=Options_SPX_1_Calls.rename(columns={'best_bid':'C_bid','best_offer':'C_ask','volume':'C_vol','open_interest':'C_open_interest','impl_volatility':'C_impl_vol'})
Options_SPX_1_Puts=Options_SPX_1_Puts.rename(columns={'best_bid':'P_bid','best_offer':'P_ask','volume':'P_vol','open_interest':'P_open_interest','impl_volatility':'P_impl_vol'})
Options_SPX_1_Puts


# In[54]:


#In order to create a synthetic short position, we need at-the-money put and call options on the same underlying (S&P500) with the same time to maturity 

Options_SPX_2=pd.merge(Options_SPX_1_Calls,Options_SPX_1_Puts,how='left',left_on=['date','exdate','strike_price'],right_on=['date','exdate','strike_price'])
Options_SPX_2=Options_SPX_2.dropna()
Options_SPX_3=Options_SPX_2.drop(['cp_flag_y','cp_flag_x','optionid_x','optionid_y','exercise_style_x','exercise_style_y','caldt_x','caldt_y','Time to Maturity_y','spindx_y','issuer_x','issuer_y','index_flag_x','index_flag_y'],1)

Options_SPX_3=Options_SPX_3.set_index('date')
Options_SPX_3['date1']=Options_SPX_3.index
Options_SPX_3=Options_SPX_3.rename(columns={'spindx_x':'S&P'})
Options_SPX_3=Options_SPX_3.rename(columns={'Time to Maturity_x':'TTM'})
Options_SPX_3['diff-value-strike']=Options_SPX_3['strike_price']/1000-Options_SPX_3['S&P']
Options_SPX_3['diff-value-strike']=Options_SPX_3['diff-value-strike'].abs()
diff_min=pd.DataFrame(Options_SPX_3.groupby('date')['diff-value-strike'].min())
Options_SPX_4=pd.merge(Options_SPX_3,diff_min, how='left', left_on='date', right_on='date')
Options_SPX_4['ATM_ind']=np.where((Options_SPX_4['diff-value-strike_y']-Options_SPX_4['diff-value-strike_x'])==0,1,0)
Options_SPX_ATM=Options_SPX_4[Options_SPX_4['ATM_ind']==1]
Options_SPX_ATM['min TTM']=Options_SPX_ATM.groupby('date')['TTM'].min()
Options_SPX_ATM['indicator TTM']=Options_SPX_ATM['min TTM']-Options_SPX_ATM['TTM']
Options_SPX_ATM=Options_SPX_ATM.drop_duplicates(subset=['exdate','date1'], keep='first')


# In[55]:


'''
At each point in time, we have ATM options with different maturities, and from each we can derive a synthetic short
For each synthetic short position we calculate the implied costs and weight them by the relative open interest
'''

Options_SPX_ATM['mean_open_interest']=(Options_SPX_ATM['C_open_interest']+Options_SPX_ATM['P_open_interest'])*0.5
Options_SPX_ATM['total open int put']=Options_SPX_ATM.groupby('date1')['P_open_interest'].sum()
Options_SPX_ATM['open int put weight']=Options_SPX_ATM['P_open_interest']/Options_SPX_ATM['total open int put']
Options_SPX_ATM['total open int']=Options_SPX_ATM.groupby('date1')['mean_open_interest'].sum()
Options_SPX_ATM['total open int weight']=Options_SPX_ATM['mean_open_interest']/Options_SPX_ATM['total open int']

Options_SPX_ATM=pd.merge(Options_SPX_ATM,LIBOR_3M ,how='left',left_on='date1',right_on='Date')
Options_SPX_ATM=Options_SPX_ATM.set_index('date1')



Options_SPX_ATM['TTM int']=Options_SPX_ATM['TTM'].dt.days
Options_SPX_ATM['TTM int']=Options_SPX_ATM['TTM int'].astype(int)
Options_SPX_ATM['PV of K']=Options_SPX_ATM['strike_price']/1000*np.exp(-Options_SPX_ATM['TTM int']*(np.log(1+(Options_SPX_ATM['LIBOR_3M']/36500))))
Options_SPX_ATM['synthetic short']=(Options_SPX_ATM['C_bid']+Options_SPX_ATM['PV of K']-Options_SPX_ATM['P_ask'])
Options_SPX_ATM['implied costs']=((Options_SPX_ATM['synthetic short']/Options_SPX_ATM['S&P'])-1)*-1
Options_SPX_ATM['weighted implied costs']=Options_SPX_ATM['implied costs']*Options_SPX_ATM['total open int weight']
Options_SPX_ATM['total weighted implied costs']=Options_SPX_ATM.groupby('date1')['weighted implied costs'].sum()
Options_SPX_ATM

#in bps
Short_selling_costs=Options_SPX_ATM[['total weighted implied costs']]*10000
Short_selling_costs['date']=Short_selling_costs.index
Short_selling_costs=Short_selling_costs.drop_duplicates(subset=['date'])
Short_selling_costs=Short_selling_costs.rename(columns={'total weighted implied costs':'SS costs'})


# In[56]:


#Since we are interested in the short selling costs 1 day before the announcement, we lag the respective values

Short_selling_costs['date_bef']=pd.to_datetime(Short_selling_costs['date'])+BDay(1)
Short_selling_costs['date_bef']=Short_selling_costs['date_bef'].astype(str)
Short_selling_costs_1=Short_selling_costs[['SS costs','date_bef']]
Short_selling_costs_1


# In[57]:


#Import VIX from CBOE

VIX_AA = pd.read_stata('VIX.dta') 
VIX_AA=VIX_AA[VIX_AA['Date'].isin(FOMC_Dates_List)]
VIX_AA=VIX_AA[VIX_AA['Date']>'2001-01-01']
VIX_AA=VIX_AA.drop_duplicates(subset='Date', keep='last')
VIX_AA=VIX_AA[['Date','vixo']] #We want to have the level of the VIX at the open
VIX_AA=VIX_AA.dropna()


# In[58]:


#Merge the data on all proxies we need for the regression in table 2 and 3

#A Days
Drivers_A=pd.merge(Pre_A_drift,Bonds_FF_Market_prevcycle_ret,how='left', left_on=['Announcement Date'], right_on=['Announcement Date'])

Drivers_A['Pre_A']=Drivers_A['Pre_A']
Drivers_A=Drivers_A.rename(columns={'Pre_A':'Pre_A_Drift'})
VIX_AA['Date']=VIX_AA['Date'].astype(str)
Drivers_A1=pd.merge(Drivers_A,VIX_AA,how='left',left_on='Announcement Date',right_on='Date')

#PN Days
Drivers_PN=pd.merge(Pre_A_drift,Bonds_FF_Market_prevcycle_ret,how='left', left_on=['Announcement Date'], right_on=['Announcement Date'])
Drivers_PN=Drivers_PN[Drivers_PN['Announcement Date'].isin(PN_all)]
Drivers_PN['Pre_A']=Drivers_PN['Pre_A']
Drivers_PN=Drivers_PN.rename(columns={'Pre_A':'Pre_A_Drift'})
VIX_AA['Date']=VIX_AA['Date'].astype(str)
Drivers_PN1=pd.merge(Drivers_PN,VIX_AA,how='left',left_on='Announcement Date',right_on='Date')

#NN Days
Drivers_NN=pd.merge(Pre_A_drift,Bonds_FF_Market_prevcycle_ret,how='left', left_on=['Announcement Date'], right_on=['Announcement Date'])
Drivers_NN=Drivers_NN[Drivers_NN['Announcement Date'].isin(NN_all)]
Drivers_NN['Pre_A']=Drivers_NN['Pre_A']
Drivers_NN=Drivers_NN.rename(columns={'Pre_A':'Pre_A_Drift'})
VIX_AA['Date']=VIX_AA['Date'].astype(str)
Drivers_NN1=pd.merge(Drivers_NN,VIX_AA,how='left',left_on='Announcement Date',right_on='Date')


# In[59]:


#Import Risk shift from Kroencke (data available until end of 2019)

Risk_Shift=pd.read_csv('Risk_Shift.csv', sep=';',parse_dates=True)
Risk_Shift=Risk_Shift[['date','RS',]]
Risk_Shift['date']=pd.to_datetime(Risk_Shift['date'])
Risk_Shift['date']=Risk_Shift['date'].astype(str)
Risk_Shift_full=Risk_Shift[Risk_Shift['date'].isin(FOMC_Dates_List)]
Risk_Shift_PN=Risk_Shift[Risk_Shift['date'].isin(PN_all)]
Risk_Shift_NN=Risk_Shift[Risk_Shift['date'].isin(NN_all)]

Risk_Shift_full


# In[60]:


#We run the regressions in subsection 5.2.3  for PN an NN days seperately

#PN days

Drivers_PN_WS=Drivers_PN1

#Add risk shift and short selling costs (SS) to the dataset
Drivers_PN_WS1=pd.merge(Drivers_PN_WS,Short_selling_costs,how='left', left_on='Announcement Date', right_on='date_bef')
Drivers_PN_WS2=pd.merge(Drivers_PN_WS1,Risk_Shift,how='left', left_on='Announcement Date', right_on='date')

#Define dummy FedPutSignal
Drivers_PN_WS2['Neg_prev_cycle_Ret']=np.where((Drivers_PN_WS2['prev_cycl_ret (bps)']<0),1,0)

Drivers_PN_WS2_ex2019=Drivers_PN_WS2[Drivers_PN_WS2['Date']<'2019-12-31']
Drivers_PN_WS2_ex2021=Drivers_PN_WS2[Drivers_PN_WS2['Date']<'2021-12-31']

#Run the resgressions. Note that data on SS and RiskShift only last until 2021 and 2019, respectively

x_PN_WS1 = Drivers_PN_WS2_ex2021[['SS costs']]
x_PN_WS1 = sm.add_constant(x_PN_WS1) # adding a constant

x_PN_WS2 = Drivers_PN_WS2[['Neg_prev_cycle_Ret']]
x_PN_WS2 = sm.add_constant(x_PN_WS2) # adding a constant


x_PN_WS3 = Drivers_PN_WS2[['vixo']]
x_PN_WS3 = sm.add_constant(x_PN_WS3) # adding a constant

x_PN_WS4 = Drivers_PN_WS2[['unexp FF change (bps)']]
x_PN_WS4 = sm.add_constant(x_PN_WS4) # adding a constant

x_PN_WS5 = Drivers_PN_WS2[['MP']]
x_PN_WS5 = sm.add_constant(x_PN_WS5) # adding a constant

x_PN_WS6 = Drivers_PN_WS2_ex2019[['RS']]
x_PN_WS6 = sm.add_constant(x_PN_WS6) # adding a constant


x_PN_WS7 = Drivers_PN_WS2_ex2019[['SS costs','Neg_prev_cycle_Ret','vixo','unexp FF change (bps)','MP','RS']]
x_PN_WS7 = sm.add_constant(x_PN_WS7) # adding a constant

y_PN_WS = Drivers_PN_WS2['Pre_A_Drift']
y_PN_WS_ex2019=Drivers_PN_WS2_ex2019['Pre_A_Drift']
y_PN_WS_ex2021=Drivers_PN_WS2_ex2021['Pre_A_Drift']

from statsmodels.iolib.summary2 import summary_col

reg1 = sm.OLS(y_PN_WS_ex2021,x_PN_WS1).fit(cov_type='HC3')
reg2 = sm.OLS(y_PN_WS,x_PN_WS2).fit(cov_type='HC3')
reg3 = sm.OLS(y_PN_WS,x_PN_WS3).fit(cov_type='HC3')
reg4 = sm.OLS(y_PN_WS,x_PN_WS4).fit(cov_type='HC3')
reg5 = sm.OLS(y_PN_WS,x_PN_WS5).fit(cov_type='HC3')
reg6 = sm.OLS(y_PN_WS_ex2019,x_PN_WS6).fit(cov_type='HC3')
reg7 = sm.OLS(y_PN_WS_ex2019,x_PN_WS7).fit(cov_type='HC3')


print(summary_col([reg1,reg2,reg3,reg4, reg5, reg6,reg7],stars=True,float_format='%0.2f').as_latex())

#summary_col([reg1,reg2,reg3,reg4, reg5],stars=True,float_format='%0.2f')


# In[61]:


#NN days

Drivers_NN_WS=Drivers_NN1

#Add risk shift and short selling costs (SS) to the dataset
Drivers_NN_WS1=pd.merge(Drivers_NN_WS,Short_selling_costs,how='left', left_on='Announcement Date', right_on='date_bef')
Drivers_NN_WS2=pd.merge(Drivers_NN_WS1,Risk_Shift,how='left', left_on='Announcement Date', right_on='date')

Drivers_NN_WS2['Neg_prev_cycle_Ret']=np.where((Drivers_NN_WS2['prev_cycl_ret (bps)']<0),1,0)

Drivers_NN_WS2_ex2019=Drivers_NN_WS2[Drivers_NN_WS2['Date']<'2019-12-31']
Drivers_NN_WS2_ex2021=Drivers_NN_WS2[Drivers_NN_WS2['Date']<'2021-12-31']

#Run the resgressions. Note that data on SS and RiskShift only last until 2021 and 2019, respectively

x_NN_WS1 = Drivers_NN_WS2_ex2021[['SS costs']]
x_NN_WS1 = sm.add_constant(x_NN_WS1) # adding a constant

x_NN_WS2 = Drivers_NN_WS2[['Neg_prev_cycle_Ret']]
x_NN_WS2 = sm.add_constant(x_NN_WS2) # adding a constant


x_NN_WS3 = Drivers_NN_WS2[['vixo']]
x_NN_WS3 = sm.add_constant(x_NN_WS3) # adding a constant

x_NN_WS4 = Drivers_NN_WS2[['unexp FF change (bps)']]
x_NN_WS4 = sm.add_constant(x_NN_WS4) # adding a constant

x_NN_WS5 = Drivers_NN_WS2[['MP']]
x_NN_WS5 = sm.add_constant(x_NN_WS5) # adding a constant

x_NN_WS6 = Drivers_NN_WS2_ex2019[['RS']]
x_NN_WS6 = sm.add_constant(x_NN_WS6) # adding a constant


x_NN_WS7 = Drivers_NN_WS2_ex2019[['SS costs','Neg_prev_cycle_Ret','vixo','unexp FF change (bps)','MP','RS']]
x_NN_WS7 = sm.add_constant(x_NN_WS7) # adding a constant

y_NN_WS = Drivers_NN_WS2['Pre_A_Drift']
y_NN_WS_ex2019=Drivers_NN_WS2_ex2019['Pre_A_Drift']
y_NN_WS_ex2021=Drivers_NN_WS2_ex2021['Pre_A_Drift']

#from statsmodels.iolib.summary2 import summary_col

reg1 = sm.OLS(y_NN_WS_ex2021,x_NN_WS1).fit(cov_type='HC3')
reg2 = sm.OLS(y_NN_WS,x_NN_WS2).fit(cov_type='HC3')
reg3 = sm.OLS(y_NN_WS,x_NN_WS3).fit(cov_type='HC3')
reg4 = sm.OLS(y_NN_WS,x_NN_WS4).fit(cov_type='HC3')
reg5 = sm.OLS(y_NN_WS,x_NN_WS5).fit(cov_type='HC3')
reg6 = sm.OLS(y_NN_WS_ex2019,x_NN_WS6).fit(cov_type='HC3')
reg7 = sm.OLS(y_NN_WS_ex2019,x_NN_WS7).fit(cov_type='HC3')


print(summary_col([reg1,reg2,reg3,reg4, reg5, reg6,reg7],stars=True,float_format='%0.2f').as_latex())

#summary_col([reg1,reg2,reg3,reg4, reg5],stars=True,float_format='%0.2f')


# In[62]:


'''
In the next part of the code, we move on to prediction 3 and 4 in the paper, i.e.,
Beta dispersion and CAPM testing
'''


# In[63]:


Data_3_2=Data_3

#5 minutes lagged market cap for the calculation of portfolio returns
Data_3_2['M_l1']=Data_3_2.groupby(['Dates','Permno'])['M'].shift()

#Dates and Time in string format so that we can merge
market['Dates']=market['Dates'].astype(str)
market['Time']=market['Time'].astype(str)
Data_3_2['Dates']=Data_3_2['Dates'].astype(str)
Data_3_2['Time']=Data_3_2['Time'].astype(str)
Data_4= pd.merge(Data_3_2, market,  how='left', left_on=['Dates','Time'], right_on = ['Dates','Time'])
Data_4=Data_4.set_index('Date_Time')
Data_4


# In[64]:


#Calculate stock returns

Data_4['Returns']=Data_4.groupby(['Permno','Dates'])['Price'].pct_change()
Data_4['Returns']=winsorize(Data_4['Returns'],(0.0001,0.0001)) #winsorize at 0.01%
Data_4['Returns_discrete']=Data_4['Returns']
Data_4['Returns']=np.log(Data_4['Returns_discrete']+1) #log returns
Data_4=Data_4[Data_4['Time'].astype(str)>'09:34:00'] #First return is on 09:39
Data_4


# In[65]:


'''
In order to calculate betas at the stock level, we follow Andersen et al.(2021) and Bodilsen et al (2021)
and use a 10-minute frequency for the covariance part and a 5 minute frequency for the variance part of the beta.
'''

#
Market_Return_10min=market.groupby('Dates')['Market_Return'].rolling(2).sum()
Market_Return_10min=pd.DataFrame(Market_Return_10min)
Market_Return_10min['Date_Time']=Market_Return_10min.index.get_level_values('Date_Time')
Market_Return_10min=Market_Return_10min.set_index('Date_Time')
Market_Return_10min=Market_Return_10min.rename(columns={'Market_Returns':'Market_Returns_10min'})
Market_Return_10min['Date_Time']=Market_Return_10min.index
Market_Return_10min['Time']=pd.to_datetime(Market_Return_10min['Date_Time']).dt.time
Market_Return_10min['Dates']=pd.to_datetime(Market_Return_10min['Date_Time']).dt.date
Market_Return_10min['Date_Time']=pd.to_datetime(Market_Return_10min['Date_Time'])
Market_Return_10min=Market_Return_10min.set_index('Date_Time',1)

Market_Return_10min['Time']=Market_Return_10min['Time'].astype(str)
Market_Return_10min['Dates']=Market_Return_10min['Dates'].astype(str)

Market_Return_10min=Market_Return_10min[Market_Return_10min['Time'].astype(str)>'09:39:00']
Market_Return_10min=Market_Return_10min[Market_Return_10min.index.minute.isin([ 9, 19,29, 39, 49,59])]
Market_Return_10min['Date_Time']=Market_Return_10min.index
Market_Return_10min


# In[66]:


'''
In a first step, I need to estimate betas at the stock level and rank them to create the beta-sorted portfolios. 
'''

Data_Pre_Rank_cov=Data_4
Data_Pre_Rank_cov['Date_Time']=Data_Pre_Rank_cov.index
Data_Pre_Rank_cov['Date_Time']=pd.to_datetime(Data_Pre_Rank_cov['Date_Time'])
Data_Pre_Rank_cov=Data_Pre_Rank_cov.set_index('Date_Time')
Data_Pre_Rank_cov=Data_Pre_Rank_cov[['Permno','Dates','Time','Returns']]
Data_Pre_Rank_cov=Data_Pre_Rank_cov.groupby(['Dates','Permno'])['Returns'].rolling(2).sum()
Data_Pre_Rank_cov=pd.DataFrame(Data_Pre_Rank_cov)
Data_Pre_Rank_cov['Permno']=Data_Pre_Rank_cov.index.get_level_values(1)
Data_Pre_Rank_cov['Date_Time']=Data_Pre_Rank_cov.index.get_level_values(2)
Data_Pre_Rank_cov=Data_Pre_Rank_cov.reset_index(drop=True)
Data_Pre_Rank_cov=Data_Pre_Rank_cov.set_index('Date_Time')
Data_Pre_Rank_cov=Data_Pre_Rank_cov[Data_Pre_Rank_cov.index.minute.isin([ 9, 19,29, 39, 49,59])]
Data_Pre_Rank_cov['Time']=Data_Pre_Rank_cov.index
Data_Pre_Rank_cov['Time']=pd.to_datetime(Data_Pre_Rank_cov['Time']).dt.time
Data_Pre_Rank_cov['Time']=Data_Pre_Rank_cov['Time'].astype(str)
Data_Pre_Rank_cov['Dates']=Data_Pre_Rank_cov.index
Data_Pre_Rank_cov['Dates']=pd.to_datetime(Data_Pre_Rank_cov['Dates']).dt.date
Data_Pre_Rank_cov['Dates']=Data_Pre_Rank_cov['Dates'].astype(str)
Data_Pre_Rank_cov=Data_Pre_Rank_cov[Data_Pre_Rank_cov['Time']>'09:39:00']
Data_Pre_Rank_cov=pd.merge(Data_Pre_Rank_cov,Market_Return_10min, how='left', left_on=['Dates','Time'], right_on = ['Dates','Time'])
Data_Pre_Rank_cov


# In[67]:


#Covariance beetween stocks and market (numerator) --> 10-min frequency

Data_Pre_Rank_cov1=Data_Pre_Rank_cov
Data_Pre_Rank_cov1['Covariance']=Data_Pre_Rank_cov['Returns']*Data_Pre_Rank_cov['Market_Return']
Data_Pre_Rank_cov1=Data_Pre_Rank_cov1.set_index('Date_Time')
Data_Pre_Rank_cov1=Data_Pre_Rank_cov1[['Dates','Time','Permno','Covariance']]
Data_Pre_Rank_cov1_intraday=Data_Pre_Rank_cov1
Data_Pre_Rank_cov1=Data_Pre_Rank_cov1.groupby(['Dates','Permno']).sum()
Data_Pre_Rank_cov1['Permno']=Data_Pre_Rank_cov1.index.get_level_values('Permno')
Data_Pre_Rank_cov1['Dates']=Data_Pre_Rank_cov1.index.get_level_values('Dates')
Data_Pre_Rank_cov1=Data_Pre_Rank_cov1.set_index('Dates')
Data_Pre_Rank_cov1

#Variance of the market (denominator)--> 5 min frequency

Data_Pre_Rank_var=market
Data_Pre_Rank_var=Data_Pre_Rank_var[Data_Pre_Rank_var['Time'].astype(str)>='09:49:00']
Data_Pre_Rank_var['Market_Var']=Data_Pre_Rank_var['Market_Return']**2
Data_Pre_Rank_var1=Data_Pre_Rank_var[['Dates','Time','Market_Var']]
Data_Pre_Rank_var1_intraday=Data_Pre_Rank_var1
Data_Pre_Rank_var1=Data_Pre_Rank_var1.groupby(['Dates']).sum()
Data_Pre_Rank_var1

#Daily betas at the stock level
Betas=pd.merge(Data_Pre_Rank_cov1,Data_Pre_Rank_var1, how='left', left_on=['Dates'], right_on = ['Dates'])
Betas['Beta']=Betas['Covariance']/Betas['Market_Var']
Betas['Date']=Betas.index
Betas


# In[68]:


#Each day, we rank stocks based on their intraday beta

Betas['q1']=Betas.groupby('Date')['Beta'].quantile(0.1)
Betas['q2']=Betas.groupby('Date')['Beta'].quantile(0.2)
Betas['q3']=Betas.groupby('Date')['Beta'].quantile(0.3)
Betas['q4']=Betas.groupby('Date')['Beta'].quantile(0.4)
Betas['q5']=Betas.groupby('Date')['Beta'].quantile(0.5)
Betas['q6']=Betas.groupby('Date')['Beta'].quantile(0.6)
Betas['q7']=Betas.groupby('Date')['Beta'].quantile(0.7)
Betas['q8']=Betas.groupby('Date')['Beta'].quantile(0.8)
Betas['q9']=Betas.groupby('Date')['Beta'].quantile(0.9)

Betas['P1']=np.where((Betas['Beta']<=Betas['q1']),1,0)
Betas['P2']=np.where((Betas['Beta']<=Betas['q2'])&(Betas['Beta']>Betas['q1']),1,0)
Betas['P3']=np.where((Betas['Beta']<=Betas['q3'])&(Betas['Beta']>Betas['q2']),1,0)
Betas['P4']=np.where((Betas['Beta']<=Betas['q4'])&(Betas['Beta']>Betas['q3']),1,0)
Betas['P5']=np.where((Betas['Beta']<=Betas['q5'])&(Betas['Beta']>Betas['q4']),1,0)
Betas['P6']=np.where((Betas['Beta']<=Betas['q6'])&(Betas['Beta']>Betas['q5']),1,0)
Betas['P7']=np.where((Betas['Beta']<=Betas['q7'])&(Betas['Beta']>Betas['q6']),1,0)
Betas['P8']=np.where((Betas['Beta']<=Betas['q8'])&(Betas['Beta']>Betas['q7']),1,0)
Betas['P9']=np.where((Betas['Beta']<=Betas['q9'])&(Betas['Beta']>Betas['q8']),1,0)
Betas['P10']=np.where((Betas['Beta']>Betas['q9']),1,0)

Betas=Betas[['Permno','Date','Beta','P1','P2','P3','P4','P5','P6','P7', 'P8','P9','P10']]

Beta_indic=Betas


# In[69]:


#Merge dataset with daily betas

Data_4=Data_4[['Permno','Dates','Time','Price', 'M','M_l1', 'Returns','Market_Return']]
Data_7=pd.merge(Data_4,Betas, how='left', left_on=['Dates','Permno'], right_on = ['Dates','Permno'])
Data_7['Date_Time']=Data_7['Dates'] + ' ' + Data_7['Time']
Data_7=Data_7.set_index('Date_Time')
Data_7


# In[70]:


'''
Portfolio Betas can either be estimated with "pure" returns, or with the diffusive part only. In the latter case
returns are "truncated". In the paper, we use pure returns as we want to keep the jump part. For the sake of completeness, 
we also calculate betas for the diffusive part only ("diffusive betas").
'''


# In[71]:


#Bipower Variation BV

n=len(market['Time'].drop_duplicates())
Pi=math.pi
market['Market_Return_lagged']=market.groupby(['Dates'])['Market_Return'].shift(1)
market['BV_min_market']=(market['Market_Return'].abs())*(market['Market_Return_lagged'].abs())
BV_market=market.groupby(['Dates'])['BV_min_market'].sum()*(Pi/2)
BV_market=pd.DataFrame(BV_market)

#V corresponds to the treshold that differentiates between diffusion an jump

V=4*((BV_market**(1/2))*(n**(-0.49)))
V_market=pd.DataFrame(V)

V_market=V_market.rename(columns={'BV_min_market':'V_market'})

V_market
market1= pd.merge(market, V_market,  how='left', left_on=['Dates'], right_on = ['Dates'])



market1['diffusion_dummy_market']=np.where(market1['Market_Return'].abs()>=market1['V_market'],0,1)


market1
market1['Date_Time']=market1['Dates'] + ' ' + market1['Time']
market1=market1.set_index('Date_Time')

market1=market1.drop(['Market_Return_lagged','BV_min_market','V_market'],1)

Jump_market=market1[['Dates','Time','diffusion_dummy_market']]
Jump_market


# In[72]:


#Create value-weighted, beta-sorted portfolios


P1=Data_7[Data_7['P1']==1]
Tot_Market_Cap=pd.DataFrame(P1.groupby('Date_Time')['M_l1'].sum())
P1=pd.merge(P1,Tot_Market_Cap, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
P1['weight_l1']=P1['M_l1_x']/P1['M_l1_y']
P1['Returns_discrete']=np.exp(P1['Returns'])-1
P1['weighted_return']=P1['weight_l1']*P1['Returns_discrete']
P1_Returns=pd.DataFrame(P1.groupby('Date_Time')['weighted_return'].sum())
P1_Returns=P1_Returns.rename(columns={'weighted_return':'P1'})
P1_Returns['Date_Time']=P1_Returns.index
P1_Returns['Dates']=pd.to_datetime(P1_Returns['Date_Time']).dt.date
P1_Returns['Time']=pd.to_datetime(P1_Returns['Date_Time']).dt.time
P1_Returns['P1']=np.log(1+P1_Returns['P1'])



P2=Data_7[Data_7['P2']==1]
Tot_Market_Cap=pd.DataFrame(P2.groupby('Date_Time')['M_l1'].sum())
P2=pd.merge(P2,Tot_Market_Cap, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
P2['weight_l1']=P2['M_l1_x']/P2['M_l1_y']
P2['Returns_discrete']=np.exp(P2['Returns'])-1
P2['weighted_return']=P2['weight_l1']*P2['Returns_discrete']
P2_Returns=pd.DataFrame(P2.groupby('Date_Time')['weighted_return'].sum())
P2_Returns=P2_Returns.rename(columns={'weighted_return':'P2'})
P2_Returns['Date_Time']=P2_Returns.index
P2_Returns['Dates']=pd.to_datetime(P2_Returns['Date_Time']).dt.date
P2_Returns['Time']=pd.to_datetime(P2_Returns['Date_Time']).dt.time
P2_Returns['P2']=np.log(1+P2_Returns['P2'])



P3=Data_7[Data_7['P3']==1]
Tot_Market_Cap=pd.DataFrame(P3.groupby('Date_Time')['M_l1'].sum())
P3=pd.merge(P3,Tot_Market_Cap, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
P3['weight_l1']=P3['M_l1_x']/P3['M_l1_y']
P3['Returns_discrete']=np.exp(P3['Returns'])-1
P3['weighted_return']=P3['weight_l1']*P3['Returns_discrete']
P3_Returns=pd.DataFrame(P3.groupby('Date_Time')['weighted_return'].sum())
P3_Returns=P3_Returns.rename(columns={'weighted_return':'P3'})
P3_Returns['Date_Time']=P3_Returns.index
P3_Returns['Dates']=pd.to_datetime(P3_Returns['Date_Time']).dt.date
P3_Returns['Time']=pd.to_datetime(P3_Returns['Date_Time']).dt.time
P3_Returns['P3']=np.log(1+P3_Returns['P3'])



P4=Data_7[Data_7['P4']==1]
Tot_Market_Cap=pd.DataFrame(P4.groupby('Date_Time')['M_l1'].sum())
P4=pd.merge(P4,Tot_Market_Cap, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
P4['weight_l1']=P4['M_l1_x']/P4['M_l1_y']
P4['Returns_discrete']=np.exp(P4['Returns'])-1
P4['weighted_return']=P4['weight_l1']*P4['Returns_discrete']
P4_Returns=pd.DataFrame(P4.groupby('Date_Time')['weighted_return'].sum())
P4_Returns=P4_Returns.rename(columns={'weighted_return':'P4'})
P4_Returns['Date_Time']=P4_Returns.index
P4_Returns['Dates']=pd.to_datetime(P4_Returns['Date_Time']).dt.date
P4_Returns['Time']=pd.to_datetime(P4_Returns['Date_Time']).dt.time
P4_Returns['P4']=np.log(1+P4_Returns['P4'])


P5=Data_7[Data_7['P5']==1]
Tot_Market_Cap=pd.DataFrame(P5.groupby('Date_Time')['M_l1'].sum())
P5=pd.merge(P5,Tot_Market_Cap, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
P5['weight_l1']=P5['M_l1_x']/P5['M_l1_y']
P5['Returns_discrete']=np.exp(P5['Returns'])-1
P5['weighted_return']=P5['weight_l1']*P5['Returns_discrete']
P5_Returns=pd.DataFrame(P5.groupby('Date_Time')['weighted_return'].sum())
P5_Returns=P5_Returns.rename(columns={'weighted_return':'P5'})
P5_Returns['Date_Time']=P5_Returns.index
P5_Returns['Dates']=pd.to_datetime(P5_Returns['Date_Time']).dt.date
P5_Returns['Time']=pd.to_datetime(P5_Returns['Date_Time']).dt.time
P5_Returns['P5']=np.log(1+P5_Returns['P5'])


P6=Data_7[Data_7['P6']==1]
Tot_Market_Cap=pd.DataFrame(P6.groupby('Date_Time')['M_l1'].sum())
P6=pd.merge(P6,Tot_Market_Cap, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
P6['weight_l1']=P6['M_l1_x']/P6['M_l1_y']
P6['Returns_discrete']=np.exp(P6['Returns'])-1
P6['weighted_return']=P6['weight_l1']*P6['Returns_discrete']
P6_Returns=pd.DataFrame(P6.groupby('Date_Time')['weighted_return'].sum())
P6_Returns=P6_Returns.rename(columns={'weighted_return':'P6'})
P6_Returns['Date_Time']=P6_Returns.index
P6_Returns['Dates']=pd.to_datetime(P6_Returns['Date_Time']).dt.date
P6_Returns['Time']=pd.to_datetime(P6_Returns['Date_Time']).dt.time
P6_Returns['P6']=np.log(1+P6_Returns['P6'])


P7=Data_7[Data_7['P7']==1]
Tot_Market_Cap=pd.DataFrame(P7.groupby('Date_Time')['M_l1'].sum())
P7=pd.merge(P7,Tot_Market_Cap, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
P7['weight_l1']=P7['M_l1_x']/P7['M_l1_y']
P7['Returns_discrete']=np.exp(P7['Returns'])-1
P7['weighted_return']=P7['weight_l1']*P7['Returns_discrete']
P7_Returns=pd.DataFrame(P7.groupby('Date_Time')['weighted_return'].sum())
P7_Returns=P7_Returns.rename(columns={'weighted_return':'P7'})
P7_Returns['Date_Time']=P7_Returns.index
P7_Returns['Dates']=pd.to_datetime(P7_Returns['Date_Time']).dt.date
P7_Returns['Time']=pd.to_datetime(P7_Returns['Date_Time']).dt.time
P7_Returns['P7']=np.log(1+P7_Returns['P7'])


P8=Data_7[Data_7['P8']==1]
Tot_Market_Cap=pd.DataFrame(P8.groupby('Date_Time')['M_l1'].sum())
P8=pd.merge(P8,Tot_Market_Cap, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
P8['weight_l1']=P8['M_l1_x']/P8['M_l1_y']
P8['Returns_discrete']=np.exp(P8['Returns'])-1
P8['weighted_return']=P8['weight_l1']*P8['Returns_discrete']
P8_Returns=pd.DataFrame(P8.groupby('Date_Time')['weighted_return'].sum())
P8_Returns=P8_Returns.rename(columns={'weighted_return':'P8'})
P8_Returns['Date_Time']=P8_Returns.index
P8_Returns['Dates']=pd.to_datetime(P8_Returns['Date_Time']).dt.date
P8_Returns['Time']=pd.to_datetime(P8_Returns['Date_Time']).dt.time
P8_Returns['P8']=np.log(1+P8_Returns['P8'])


P9=Data_7[Data_7['P9']==1]
Tot_Market_Cap=pd.DataFrame(P9.groupby('Date_Time')['M_l1'].sum())
P9=pd.merge(P9,Tot_Market_Cap, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
P9['weight_l1']=P9['M_l1_x']/P9['M_l1_y']
P9['Returns_discrete']=np.exp(P9['Returns'])-1
P9['weighted_return']=P9['weight_l1']*P9['Returns_discrete']
P9_Returns=pd.DataFrame(P9.groupby('Date_Time')['weighted_return'].sum())
P9_Returns=P9_Returns.rename(columns={'weighted_return':'P9'})
P9_Returns['Date_Time']=P9_Returns.index
P9_Returns['Dates']=pd.to_datetime(P9_Returns['Date_Time']).dt.date
P9_Returns['Time']=pd.to_datetime(P9_Returns['Date_Time']).dt.time
P9_Returns['P9']=np.log(1+P9_Returns['P9'])



P10=Data_7[Data_7['P10']==1]
Tot_Market_Cap=pd.DataFrame(P10.groupby('Date_Time')['M_l1'].sum())
P10=pd.merge(P10,Tot_Market_Cap, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
P10['weight_l1']=P10['M_l1_x']/P10['M_l1_y']
P10['Returns_discrete']=np.exp(P10['Returns'])-1
P10['weighted_return']=P10['weight_l1']*P10['Returns_discrete']
P10_Returns=pd.DataFrame(P10.groupby('Date_Time')['weighted_return'].sum())
P10_Returns=P10_Returns.rename(columns={'weighted_return':'P10'})
P10_Returns['Date_Time']=P10_Returns.index
P10_Returns['Dates']=pd.to_datetime(P10_Returns['Date_Time']).dt.date
P10_Returns['Time']=pd.to_datetime(P10_Returns['Date_Time']).dt.time
P10_Returns['P10']=np.log(1+P10_Returns['P10'])


#Truncation portfolio level

#P1
#Bipower Variation BV
n=len(P1_Returns['Time'].drop_duplicates())
Pi=math.pi
P1_Returns['Returns_lagged']=P1_Returns.groupby(['Dates'])['P1'].shift(1)
P1_Returns['BV_min']=(P1_Returns['P1'].abs())*(P1_Returns['Returns_lagged'].abs())
BV=P1_Returns.groupby(['Dates'])['BV_min'].sum()*(Pi/2)
#V corresponds to the treshold that differentiates between diffusion an jump
V=4*((BV**(1/2))*(n**(-0.49)))
V=pd.DataFrame(V)
V=V.rename(columns={'BV_min':'V_stock'})
P1_Returns= pd.merge(P1_Returns, V,  how='left', left_on=['Dates'], right_on = ['Dates'])
P1_Returns['diffusion_dummy_P1']=np.where(P1_Returns['P1'].abs()>=P1_Returns['V_stock'],0,1)
P1_Returns=P1_Returns[['Date_Time','P1','diffusion_dummy_P1']]

#P2
#Bipower Variation BV
n=len(P2_Returns['Time'].drop_duplicates())
Pi=math.pi
P2_Returns['Returns_lagged']=P2_Returns.groupby(['Dates'])['P2'].shift(1)
P2_Returns['BV_min']=(P2_Returns['P2'].abs())*(P2_Returns['Returns_lagged'].abs())
BV=P2_Returns.groupby(['Dates'])['BV_min'].sum()*(Pi/2)
#V corresponds to the treshold that differentiates between diffusion an jump
V=4*((BV**(1/2))*(n**(-0.49)))
V=pd.DataFrame(V)
V=V.rename(columns={'BV_min':'V_stock'})
P2_Returns= pd.merge(P2_Returns, V,  how='left', left_on=['Dates'], right_on = ['Dates'])
P2_Returns['diffusion_dummy_P2']=np.where(P2_Returns['P2'].abs()>=P2_Returns['V_stock'],0,1)
P2_Returns=P2_Returns[['Date_Time','P2','diffusion_dummy_P2']]

#P3
#Bipower Variation BV
n=len(P3_Returns['Time'].drop_duplicates())
Pi=math.pi
P3_Returns['Returns_lagged']=P3_Returns.groupby(['Dates'])['P3'].shift(1)
P3_Returns['BV_min']=(P3_Returns['P3'].abs())*(P3_Returns['Returns_lagged'].abs())
BV=P3_Returns.groupby(['Dates'])['BV_min'].sum()*(Pi/2)
#V corresponds to the treshold that differentiates between diffusion an jump
V=4*((BV**(1/2))*(n**(-0.49)))
V=pd.DataFrame(V)
V=V.rename(columns={'BV_min':'V_stock'})
P3_Returns= pd.merge(P3_Returns, V,  how='left', left_on=['Dates'], right_on = ['Dates'])
P3_Returns['diffusion_dummy_P3']=np.where(P3_Returns['P3'].abs()>=P3_Returns['V_stock'],0,1)
P3_Returns=P3_Returns[['Date_Time','P3','diffusion_dummy_P3']]

#P4
#Bipower Variation BV
n=len(P4_Returns['Time'].drop_duplicates())
Pi=math.pi
P4_Returns['Returns_lagged']=P4_Returns.groupby(['Dates'])['P4'].shift(1)
P4_Returns['BV_min']=(P4_Returns['P4'].abs())*(P4_Returns['Returns_lagged'].abs())
BV=P4_Returns.groupby(['Dates'])['BV_min'].sum()*(Pi/2)
#V corresponds to the treshold that differentiates between diffusion an jump
V=4*((BV**(1/2))*(n**(-0.49)))
V=pd.DataFrame(V)
V=V.rename(columns={'BV_min':'V_stock'})
P4_Returns= pd.merge(P4_Returns, V,  how='left', left_on=['Dates'], right_on = ['Dates'])
P4_Returns['diffusion_dummy_P4']=np.where(P4_Returns['P4'].abs()>=P4_Returns['V_stock'],0,1)
P4_Returns=P4_Returns[['Date_Time','P4','diffusion_dummy_P4']]

#P5
#Bipower Variation BV
n=len(P5_Returns['Time'].drop_duplicates())
Pi=math.pi
P5_Returns['Returns_lagged']=P5_Returns.groupby(['Dates'])['P5'].shift(1)
P5_Returns['BV_min']=(P5_Returns['P5'].abs())*(P5_Returns['Returns_lagged'].abs())
BV=P5_Returns.groupby(['Dates'])['BV_min'].sum()*(Pi/2)
#V corresponds to the treshold that differentiates between diffusion an jump
V=4*((BV**(1/2))*(n**(-0.49)))
V=pd.DataFrame(V)
V=V.rename(columns={'BV_min':'V_stock'})
P5_Returns= pd.merge(P5_Returns, V,  how='left', left_on=['Dates'], right_on = ['Dates'])
P5_Returns['diffusion_dummy_P5']=np.where(P5_Returns['P5'].abs()>=P5_Returns['V_stock'],0,1)
P5_Returns=P5_Returns[['Date_Time','P5','diffusion_dummy_P5']]

#P6
#Bipower Variation BV
n=len(P6_Returns['Time'].drop_duplicates())
Pi=math.pi
P6_Returns['Returns_lagged']=P6_Returns.groupby(['Dates'])['P6'].shift(1)
P6_Returns['BV_min']=(P6_Returns['P6'].abs())*(P6_Returns['Returns_lagged'].abs())
BV=P6_Returns.groupby(['Dates'])['BV_min'].sum()*(Pi/2)
#V corresponds to the treshold that differentiates between diffusion an jump
V=4*((BV**(1/2))*(n**(-0.49)))
V=pd.DataFrame(V)
V=V.rename(columns={'BV_min':'V_stock'})
P6_Returns= pd.merge(P6_Returns, V,  how='left', left_on=['Dates'], right_on = ['Dates'])
P6_Returns['diffusion_dummy_P6']=np.where(P6_Returns['P6'].abs()>=P6_Returns['V_stock'],0,1)
P6_Returns=P6_Returns[['Date_Time','P6','diffusion_dummy_P6']]

#P7
#Bipower Variation BV
n=len(P7_Returns['Time'].drop_duplicates())
Pi=math.pi
P7_Returns['Returns_lagged']=P7_Returns.groupby(['Dates'])['P7'].shift(1)
P7_Returns['BV_min']=(P7_Returns['P7'].abs())*(P7_Returns['Returns_lagged'].abs())
BV=P7_Returns.groupby(['Dates'])['BV_min'].sum()*(Pi/2)
#V corresponds to the treshold that differentiates between diffusion an jump
V=4*((BV**(1/2))*(n**(-0.49)))
V=pd.DataFrame(V)
V=V.rename(columns={'BV_min':'V_stock'})
P7_Returns= pd.merge(P7_Returns, V,  how='left', left_on=['Dates'], right_on = ['Dates'])
P7_Returns['diffusion_dummy_P7']=np.where(P7_Returns['P7'].abs()>=P7_Returns['V_stock'],0,1)
P7_Returns=P7_Returns[['Date_Time','P7','diffusion_dummy_P7']]

#P8
#Bipower Variation BV
n=len(P8_Returns['Time'].drop_duplicates())
Pi=math.pi
P8_Returns['Returns_lagged']=P8_Returns.groupby(['Dates'])['P8'].shift(1)
P8_Returns['BV_min']=(P8_Returns['P8'].abs())*(P8_Returns['Returns_lagged'].abs())
BV=P8_Returns.groupby(['Dates'])['BV_min'].sum()*(Pi/2)
#V corresponds to the treshold that differentiates between diffusion an jump
V=4*((BV**(1/2))*(n**(-0.49)))
V=pd.DataFrame(V)
V=V.rename(columns={'BV_min':'V_stock'})
P8_Returns= pd.merge(P8_Returns, V,  how='left', left_on=['Dates'], right_on = ['Dates'])
P8_Returns['diffusion_dummy_P8']=np.where(P8_Returns['P8'].abs()>=P8_Returns['V_stock'],0,1)
P8_Returns=P8_Returns[['Date_Time','P8','diffusion_dummy_P8']]

#P9
#Bipower Variation BV
n=len(P9_Returns['Time'].drop_duplicates())
Pi=math.pi
P9_Returns['Returns_lagged']=P9_Returns.groupby(['Dates'])['P9'].shift(1)
P9_Returns['BV_min']=(P9_Returns['P9'].abs())*(P9_Returns['Returns_lagged'].abs())
BV=P9_Returns.groupby(['Dates'])['BV_min'].sum()*(Pi/2)
#V corresponds to the treshold that differentiates between diffusion an jump
V=4*((BV**(1/2))*(n**(-0.49)))
V=pd.DataFrame(V)
V=V.rename(columns={'BV_min':'V_stock'})
P9_Returns= pd.merge(P9_Returns, V,  how='left', left_on=['Dates'], right_on = ['Dates'])
P9_Returns['diffusion_dummy_P9']=np.where(P9_Returns['P9'].abs()>=P9_Returns['V_stock'],0,1)
P9_Returns=P9_Returns[['Date_Time','P9','diffusion_dummy_P9']]

#P10
#Bipower Variation BV
n=len(P10_Returns['Time'].drop_duplicates())
Pi=math.pi
P10_Returns['Returns_lagged']=P10_Returns.groupby(['Dates'])['P10'].shift(1)
P10_Returns['BV_min']=(P10_Returns['P10'].abs())*(P10_Returns['Returns_lagged'].abs())
BV=P10_Returns.groupby(['Dates'])['BV_min'].sum()*(Pi/2)
#V corresponds to the treshold that differentiates between diffusion an jump
V=4*((BV**(1/2))*(n**(-0.49)))
V=pd.DataFrame(V)
V=V.rename(columns={'BV_min':'V_stock'})
P10_Returns= pd.merge(P10_Returns, V,  how='left', left_on=['Dates'], right_on = ['Dates'])
P10_Returns['diffusion_dummy_P10']=np.where(P10_Returns['P10'].abs()>=P10_Returns['V_stock'],0,1)
P10_Returns=P10_Returns[['Date_Time','P10','diffusion_dummy_P10']]


# In[73]:


#Merge all the returns of the beta-sorted portfolios

Portfolio_Returns=pd.merge(P1_Returns,P2_Returns, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
Portfolio_Returns=pd.merge(Portfolio_Returns,P3_Returns, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
Portfolio_Returns=pd.merge(Portfolio_Returns,P4_Returns, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
Portfolio_Returns=pd.merge(Portfolio_Returns,P5_Returns, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
Portfolio_Returns=pd.merge(Portfolio_Returns,P6_Returns, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
Portfolio_Returns=pd.merge(Portfolio_Returns,P7_Returns, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
Portfolio_Returns=pd.merge(Portfolio_Returns,P8_Returns, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
Portfolio_Returns=pd.merge(Portfolio_Returns,P9_Returns, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
Portfolio_Returns=pd.merge(Portfolio_Returns,P10_Returns, how='left', left_on=['Date_Time'], right_on = ['Date_Time'])
Portfolio_Returns



# In[74]:


#Add market returns to the dataset

Portfolio_Returns=Portfolio_Returns.set_index('Date_Time')
Portfolio_Returns['Date_Time']=Portfolio_Returns.index
Portfolio_Returns['Dates']=pd.to_datetime(Portfolio_Returns['Date_Time']).dt.date
Portfolio_Returns['Dates']=Portfolio_Returns['Dates'].astype(str)
Portfolio_Returns['Time']=pd.to_datetime(Portfolio_Returns['Date_Time']).dt.time
Portfolio_Returns['Time']=Portfolio_Returns['Time'].astype(str)
market['Dates']=market['Dates'].astype(str)
market['Time']=market['Time'].astype(str)
Portfolio_Returns=pd.merge(Portfolio_Returns,market1, how='left', left_on=['Dates','Time'], right_on = ['Dates','Time'])
Portfolio_Returns=Portfolio_Returns.rename(columns={'Date_Time_x':'Date_Time'})
Portfolio_Returns




# In[75]:


#For the "diffusive" betas, we only consider observations where the jump component of BOTH the market and the portfolio is zero.


Portfolio_Returns['joint_diffusion_P1']=np.where((Portfolio_Returns['diffusion_dummy_P1']==0)&(Portfolio_Returns['diffusion_dummy_market']==0),0,1)
Portfolio_Returns['P1_diff']=Portfolio_Returns['joint_diffusion_P1']*Portfolio_Returns['P1']
Portfolio_Returns['M_P1_diff']=Portfolio_Returns['joint_diffusion_P1']*Portfolio_Returns['Market_Return']

Portfolio_Returns['joint_diffusion_P2']=np.where((Portfolio_Returns['diffusion_dummy_P2']==0)&(Portfolio_Returns['diffusion_dummy_market']==0),0,1)
Portfolio_Returns['P2_diff']=Portfolio_Returns['joint_diffusion_P2']*Portfolio_Returns['P2']
Portfolio_Returns['M_P2_diff']=Portfolio_Returns['joint_diffusion_P2']*Portfolio_Returns['Market_Return']

Portfolio_Returns['joint_diffusion_P3']=np.where((Portfolio_Returns['diffusion_dummy_P3']==0)&(Portfolio_Returns['diffusion_dummy_market']==0),0,1)
Portfolio_Returns['P3_diff']=Portfolio_Returns['joint_diffusion_P3']*Portfolio_Returns['P3']
Portfolio_Returns['M_P3_diff']=Portfolio_Returns['joint_diffusion_P3']*Portfolio_Returns['Market_Return']

Portfolio_Returns['joint_diffusion_P4']=np.where((Portfolio_Returns['diffusion_dummy_P4']==0)&(Portfolio_Returns['diffusion_dummy_market']==0),0,1)
Portfolio_Returns['P4_diff']=Portfolio_Returns['joint_diffusion_P4']*Portfolio_Returns['P4']
Portfolio_Returns['M_P4_diff']=Portfolio_Returns['joint_diffusion_P4']*Portfolio_Returns['Market_Return']

Portfolio_Returns['joint_diffusion_P5']=np.where((Portfolio_Returns['diffusion_dummy_P5']==0)&(Portfolio_Returns['diffusion_dummy_market']==0),0,1)
Portfolio_Returns['P5_diff']=Portfolio_Returns['joint_diffusion_P5']*Portfolio_Returns['P5']
Portfolio_Returns['M_P5_diff']=Portfolio_Returns['joint_diffusion_P5']*Portfolio_Returns['Market_Return']

Portfolio_Returns['joint_diffusion_P6']=np.where((Portfolio_Returns['diffusion_dummy_P6']==0)&(Portfolio_Returns['diffusion_dummy_market']==0),0,1)
Portfolio_Returns['P6_diff']=Portfolio_Returns['joint_diffusion_P6']*Portfolio_Returns['P6']
Portfolio_Returns['M_P6_diff']=Portfolio_Returns['joint_diffusion_P6']*Portfolio_Returns['Market_Return']

Portfolio_Returns['joint_diffusion_P7']=np.where((Portfolio_Returns['diffusion_dummy_P7']==0)&(Portfolio_Returns['diffusion_dummy_market']==0),0,1)
Portfolio_Returns['P7_diff']=Portfolio_Returns['joint_diffusion_P7']*Portfolio_Returns['P7']
Portfolio_Returns['M_P7_diff']=Portfolio_Returns['joint_diffusion_P7']*Portfolio_Returns['Market_Return']

Portfolio_Returns['joint_diffusion_P8']=np.where((Portfolio_Returns['diffusion_dummy_P8']==0)&(Portfolio_Returns['diffusion_dummy_market']==0),0,1)
Portfolio_Returns['P8_diff']=Portfolio_Returns['joint_diffusion_P8']*Portfolio_Returns['P8']
Portfolio_Returns['M_P8_diff']=Portfolio_Returns['joint_diffusion_P8']*Portfolio_Returns['Market_Return']

Portfolio_Returns['joint_diffusion_P9']=np.where((Portfolio_Returns['diffusion_dummy_P9']==0)&(Portfolio_Returns['diffusion_dummy_market']==0),0,1)
Portfolio_Returns['P9_diff']=Portfolio_Returns['joint_diffusion_P9']*Portfolio_Returns['P9']
Portfolio_Returns['M_P9_diff']=Portfolio_Returns['joint_diffusion_P9']*Portfolio_Returns['Market_Return']

Portfolio_Returns['joint_diffusion_P10']=np.where((Portfolio_Returns['diffusion_dummy_P10']==0)&(Portfolio_Returns['diffusion_dummy_market']==0),0,1)
Portfolio_Returns['P10_diff']=Portfolio_Returns['joint_diffusion_P10']*Portfolio_Returns['P10']
Portfolio_Returns['M_P10_diff']=Portfolio_Returns['joint_diffusion_P10']*Portfolio_Returns['Market_Return']
Portfolio_Returns=Portfolio_Returns[['Date_Time','Dates','Time','P1','P1_diff','M_P1_diff','P2','P2_diff','M_P2_diff','P3','P3_diff','M_P3_diff','P4','P4_diff','M_P4_diff','P5','P5_diff','M_P5_diff','P6','P6_diff','M_P6_diff','P7','P7_diff','M_P7_diff','P8','P8_diff','M_P8_diff','P9','P9_diff','M_P9_diff','P10','P10_diff','M_P10_diff','Market_Return']]


# In[77]:


#Intraday Portfolio Beta Compression WITH TRUNCATION. Only execute if we want to calculate "diffusive betas"!

Portfolio_Returns['P1_cov']=Portfolio_Returns['P1_diff']*Portfolio_Returns['M_P1_diff']
Portfolio_Returns['P2_cov']=Portfolio_Returns['P2_diff']*Portfolio_Returns['M_P2_diff']
Portfolio_Returns['P3_cov']=Portfolio_Returns['P3_diff']*Portfolio_Returns['M_P3_diff']
Portfolio_Returns['P4_cov']=Portfolio_Returns['P4_diff']*Portfolio_Returns['M_P4_diff']
Portfolio_Returns['P5_cov']=Portfolio_Returns['P5_diff']*Portfolio_Returns['M_P5_diff']
Portfolio_Returns['P6_cov']=Portfolio_Returns['P6_diff']*Portfolio_Returns['M_P6_diff']
Portfolio_Returns['P7_cov']=Portfolio_Returns['P7_diff']*Portfolio_Returns['M_P7_diff']
Portfolio_Returns['P8_cov']=Portfolio_Returns['P8_diff']*Portfolio_Returns['M_P8_diff']
Portfolio_Returns['P9_cov']=Portfolio_Returns['P9_diff']*Portfolio_Returns['M_P9_diff']
Portfolio_Returns['P10_cov']=Portfolio_Returns['P10_diff']*Portfolio_Returns['M_P10_diff']

Portfolio_Returns['Market_Var1']=Portfolio_Returns['M_P1_diff']**2
Portfolio_Returns['Market_Var2']=Portfolio_Returns['M_P2_diff']**2
Portfolio_Returns['Market_Var3']=Portfolio_Returns['M_P3_diff']**2
Portfolio_Returns['Market_Var4']=Portfolio_Returns['M_P4_diff']**2
Portfolio_Returns['Market_Var5']=Portfolio_Returns['M_P5_diff']**2
Portfolio_Returns['Market_Var6']=Portfolio_Returns['M_P6_diff']**2
Portfolio_Returns['Market_Var7']=Portfolio_Returns['M_P7_diff']**2
Portfolio_Returns['Market_Var8']=Portfolio_Returns['M_P8_diff']**2
Portfolio_Returns['Market_Var9']=Portfolio_Returns['M_P9_diff']**2
Portfolio_Returns['Market_Var10']=Portfolio_Returns['M_P10_diff']**2

Portfolio_Returns=Portfolio_Returns.set_index('Date_Time')
Rolling_Betas=Portfolio_Returns
Rolling_Betas=Rolling_Betas.groupby('Dates').rolling(19).sum()
Rolling_Betas=Rolling_Betas.dropna()

Rolling_Betas['P1_Rolling_Beta']=Rolling_Betas['P1_cov']/Rolling_Betas['Market_Var1']
Rolling_Betas['P2_Rolling_Beta']=Rolling_Betas['P2_cov']/Rolling_Betas['Market_Var2']
Rolling_Betas['P3_Rolling_Beta']=Rolling_Betas['P3_cov']/Rolling_Betas['Market_Var3']
Rolling_Betas['P4_Rolling_Beta']=Rolling_Betas['P4_cov']/Rolling_Betas['Market_Var4']
Rolling_Betas['P5_Rolling_Beta']=Rolling_Betas['P5_cov']/Rolling_Betas['Market_Var5']
Rolling_Betas['P6_Rolling_Beta']=Rolling_Betas['P6_cov']/Rolling_Betas['Market_Var6']
Rolling_Betas['P7_Rolling_Beta']=Rolling_Betas['P7_cov']/Rolling_Betas['Market_Var7']
Rolling_Betas['P8_Rolling_Beta']=Rolling_Betas['P8_cov']/Rolling_Betas['Market_Var8']
Rolling_Betas['P9_Rolling_Beta']=Rolling_Betas['P9_cov']/Rolling_Betas['Market_Var9']
Rolling_Betas['P10_Rolling_Beta']=Rolling_Betas['P10_cov']/Rolling_Betas['Market_Var10']


Rolling_Betas['Beta_Dispersion']=(((Rolling_Betas['P1_Rolling_Beta']-1)**2)+((Rolling_Betas['P2_Rolling_Beta']-1)**2)+((Rolling_Betas['P3_Rolling_Beta']-1)**2)+((Rolling_Betas['P4_Rolling_Beta']-1)**2)+((Rolling_Betas['P5_Rolling_Beta']-1)**2)+((Rolling_Betas['P6_Rolling_Beta']-1)**2)+((Rolling_Betas['P7_Rolling_Beta']-1)**2)+((Rolling_Betas['P8_Rolling_Beta']-1)**2)+((Rolling_Betas['P9_Rolling_Beta']-1)**2)+((Rolling_Betas['P10_Rolling_Beta']-1)**2))/10
Rolling_Betas['Beta_Dispersion']=Rolling_Betas['Beta_Dispersion']**(1/2)
Beta_Dispersion=Rolling_Betas[['Beta_Dispersion']]

#Calculate Daily Portfolio Betas


Portfolio_Betas=Portfolio_Returns
Portfolio_Betas=Portfolio_Betas.groupby('Dates').sum()

Portfolio_Betas['P1_Beta']=Portfolio_Betas['P1_cov']/Portfolio_Betas['Market_Var1']
Portfolio_Betas['P2_Beta']=Portfolio_Betas['P2_cov']/Portfolio_Betas['Market_Var2']
Portfolio_Betas['P3_Beta']=Portfolio_Betas['P3_cov']/Portfolio_Betas['Market_Var3']
Portfolio_Betas['P4_Beta']=Portfolio_Betas['P4_cov']/Portfolio_Betas['Market_Var4']
Portfolio_Betas['P5_Beta']=Portfolio_Betas['P5_cov']/Portfolio_Betas['Market_Var5']
Portfolio_Betas['P6_Beta']=Portfolio_Betas['P6_cov']/Portfolio_Betas['Market_Var6']
Portfolio_Betas['P7_Beta']=Portfolio_Betas['P7_cov']/Portfolio_Betas['Market_Var7']
Portfolio_Betas['P8_Beta']=Portfolio_Betas['P8_cov']/Portfolio_Betas['Market_Var8']
Portfolio_Betas['P9_Beta']=Portfolio_Betas['P9_cov']/Portfolio_Betas['Market_Var9']
Portfolio_Betas['P10_Beta']=Portfolio_Betas['P10_cov']/Portfolio_Betas['Market_Var10']


Portfolio_Betas_Trunc=Portfolio_Betas[['P1_Beta','P2_Beta','P3_Beta','P4_Beta','P5_Beta','P6_Beta','P7_Beta','P8_Beta','P9_Beta','P10_Beta']]


# In[76]:


'''
Intraday Portfolio Beta Compression WITHOUT TRUNCATION. Only execute if we want to calculate "pure betas"! 
Execute this for the replication of the results in the paper.
'''

#Calculate covariances between portfolios and the market plus market variances to obtain betas over rolling windows
Portfolio_Returns['P1_cov']=Portfolio_Returns['P1']*Portfolio_Returns['Market_Return']
Portfolio_Returns['P2_cov']=Portfolio_Returns['P2']*Portfolio_Returns['Market_Return']
Portfolio_Returns['P3_cov']=Portfolio_Returns['P3']*Portfolio_Returns['Market_Return']
Portfolio_Returns['P4_cov']=Portfolio_Returns['P4']*Portfolio_Returns['Market_Return']
Portfolio_Returns['P5_cov']=Portfolio_Returns['P5']*Portfolio_Returns['Market_Return']
Portfolio_Returns['P6_cov']=Portfolio_Returns['P6']*Portfolio_Returns['Market_Return']
Portfolio_Returns['P7_cov']=Portfolio_Returns['P7']*Portfolio_Returns['Market_Return']
Portfolio_Returns['P8_cov']=Portfolio_Returns['P8']*Portfolio_Returns['Market_Return']
Portfolio_Returns['P9_cov']=Portfolio_Returns['P9']*Portfolio_Returns['Market_Return']
Portfolio_Returns['P10_cov']=Portfolio_Returns['P10']*Portfolio_Returns['Market_Return']
Portfolio_Returns['Market_Var']=Portfolio_Returns['Market_Return']**2
Portfolio_Returns=Portfolio_Returns.rename(columns={'Date_Time_x':'Date_Time'})
Portfolio_Returns_help=Portfolio_Returns.set_index('Date_Time')

Rolling_Betas=Portfolio_Returns_help
Rolling_Betas=Rolling_Betas.groupby('Dates').rolling(19).sum() #use 1.5 hour rolling windows
Rolling_Betas=Rolling_Betas[['P1_cov','P2_cov', 'P3_cov', 'P4_cov', 'P5_cov', 'P6_cov', 'P7_cov', 'P8_cov', 'P9_cov', 'P10_cov', 'Market_Var']]
Rolling_Betas['Time']=Rolling_Betas.index.get_level_values(1)
Rolling_Betas['Time']=pd.to_datetime(Rolling_Betas['Time']).dt.time
Rolling_Betas['Time']=Rolling_Betas['Time'].astype(str)
Rolling_Betas=Rolling_Betas[Rolling_Betas['Time']>='11:19:00']
Rolling_Betas['Date']=Rolling_Betas.index.get_level_values(0)

Rolling_Betas['P1_Rolling_Beta']=Rolling_Betas['P1_cov']/Rolling_Betas['Market_Var']
Rolling_Betas['P2_Rolling_Beta']=Rolling_Betas['P2_cov']/Rolling_Betas['Market_Var']
Rolling_Betas['P3_Rolling_Beta']=Rolling_Betas['P3_cov']/Rolling_Betas['Market_Var']
Rolling_Betas['P4_Rolling_Beta']=Rolling_Betas['P4_cov']/Rolling_Betas['Market_Var']
Rolling_Betas['P5_Rolling_Beta']=Rolling_Betas['P5_cov']/Rolling_Betas['Market_Var']
Rolling_Betas['P6_Rolling_Beta']=Rolling_Betas['P6_cov']/Rolling_Betas['Market_Var']
Rolling_Betas['P7_Rolling_Beta']=Rolling_Betas['P7_cov']/Rolling_Betas['Market_Var']
Rolling_Betas['P8_Rolling_Beta']=Rolling_Betas['P8_cov']/Rolling_Betas['Market_Var']
Rolling_Betas['P9_Rolling_Beta']=Rolling_Betas['P9_cov']/Rolling_Betas['Market_Var']
Rolling_Betas['P10_Rolling_Beta']=Rolling_Betas['P10_cov']/Rolling_Betas['Market_Var']

#Calculate beta dispersion based on Andersen et al. (2021)
Rolling_Betas['Beta_Dispersion']=(((Rolling_Betas['P1_Rolling_Beta']-1)**2)+((Rolling_Betas['P2_Rolling_Beta']-1)**2)+((Rolling_Betas['P3_Rolling_Beta']-1)**2)+((Rolling_Betas['P4_Rolling_Beta']-1)**2)+((Rolling_Betas['P5_Rolling_Beta']-1)**2)+((Rolling_Betas['P6_Rolling_Beta']-1)**2)+((Rolling_Betas['P7_Rolling_Beta']-1)**2)+((Rolling_Betas['P8_Rolling_Beta']-1)**2)+((Rolling_Betas['P9_Rolling_Beta']-1)**2)+((Rolling_Betas['P10_Rolling_Beta']-1)**2))/10
Rolling_Betas['Beta_Dispersion']=Rolling_Betas['Beta_Dispersion']**(1/2)
Beta_Dispersion=Rolling_Betas[['Beta_Dispersion']]
Rolling_Betas=Rolling_Betas.reset_index(drop=True)


#Calculate Daily Portfolio Betas

Portfolio_Betas=Portfolio_Returns
Portfolio_Betas=Portfolio_Betas.groupby('Dates').sum()

Portfolio_Betas['P1_Beta']=Portfolio_Betas['P1_cov']/Portfolio_Betas['Market_Var']
Portfolio_Betas['P2_Beta']=Portfolio_Betas['P2_cov']/Portfolio_Betas['Market_Var']
Portfolio_Betas['P3_Beta']=Portfolio_Betas['P3_cov']/Portfolio_Betas['Market_Var']
Portfolio_Betas['P4_Beta']=Portfolio_Betas['P4_cov']/Portfolio_Betas['Market_Var']
Portfolio_Betas['P5_Beta']=Portfolio_Betas['P5_cov']/Portfolio_Betas['Market_Var']
Portfolio_Betas['P6_Beta']=Portfolio_Betas['P6_cov']/Portfolio_Betas['Market_Var']
Portfolio_Betas['P7_Beta']=Portfolio_Betas['P7_cov']/Portfolio_Betas['Market_Var']
Portfolio_Betas['P8_Beta']=Portfolio_Betas['P8_cov']/Portfolio_Betas['Market_Var']
Portfolio_Betas['P9_Beta']=Portfolio_Betas['P9_cov']/Portfolio_Betas['Market_Var']
Portfolio_Betas['P10_Beta']=Portfolio_Betas['P10_cov']/Portfolio_Betas['Market_Var']

Portfolio_Betas=Portfolio_Betas[['P1_Beta','P2_Beta','P3_Beta','P4_Beta','P5_Beta','P6_Beta','P7_Beta','P8_Beta','P9_Beta','P10_Beta']]
Portfolio_Betas



# In[77]:


#Calculate average daily beta dispersion:
Beta_Dispersion=Portfolio_Betas
Beta_Dispersion['Beta_Dispersion']=((((Beta_Dispersion['P1_Beta']-1)**2)+((Beta_Dispersion['P2_Beta']-1)**2)+((Beta_Dispersion['P3_Beta']-1)**2)+((Beta_Dispersion['P4_Beta']-1)**2)+((Beta_Dispersion['P5_Beta']-1)**2)+((Beta_Dispersion['P6_Beta']-1)**2)+((Beta_Dispersion['P7_Beta']-1)**2)+((Beta_Dispersion['P8_Beta']-1)**2)+((Beta_Dispersion['P9_Beta']-1)**2)+((Beta_Dispersion['P10_Beta']-1)**2))/10)**0.5
Beta_Dispersion=Beta_Dispersion[['Beta_Dispersion']]
Beta_Dispersion_A=Beta_Dispersion[Beta_Dispersion.index.isin(FOMC_Dates_List)]
Beta_Dispersion_NA=Beta_Dispersion[~Beta_Dispersion.index.isin(FOMC_Dates_List)]
Beta_Dispersion_E1=Beta_Dispersion[Beta_Dispersion.index.isin(E1_Dates)]
Beta_Dispersion_E2=Beta_Dispersion[Beta_Dispersion.index.isin(E2_Dates)]


# In[78]:


#Create different subsamples for E1 and E2 (PC/non-PC)

Portfolio_Returns_A=Portfolio_Returns[Portfolio_Returns['Dates'].isin(FOMC_Dates_List)]
Portfolio_Returns_NA=Portfolio_Returns[~Portfolio_Returns['Dates'].isin(FOMC_Dates_List)]
Portfolio_Returns_E1=Portfolio_Returns_A[Portfolio_Returns_A['Dates'].isin(E1_Dates)]
Portfolio_Returns_E2=Portfolio_Returns[Portfolio_Returns['Dates'].isin(E2_Dates)]                                     


# In[79]:


#Full sample, day specific portfolio betas NOT TRUNCATED--> execute the correct cell above (not truncated)!!


#Day-specific (A,NA,PN,NN)

#A-Days
Full_sample_Betas_A=Portfolio_Returns_A
Full_sample_Betas_A=Full_sample_Betas_A.sum()
Full_sample_Betas_A['P1_Beta']=Full_sample_Betas_A['P1_cov']/Full_sample_Betas_A['Market_Var']
Full_sample_Betas_A['P2_Beta']=Full_sample_Betas_A['P2_cov']/Full_sample_Betas_A['Market_Var']
Full_sample_Betas_A['P3_Beta']=Full_sample_Betas_A['P3_cov']/Full_sample_Betas_A['Market_Var']
Full_sample_Betas_A['P4_Beta']=Full_sample_Betas_A['P4_cov']/Full_sample_Betas_A['Market_Var']
Full_sample_Betas_A['P5_Beta']=Full_sample_Betas_A['P5_cov']/Full_sample_Betas_A['Market_Var']
Full_sample_Betas_A['P6_Beta']=Full_sample_Betas_A['P6_cov']/Full_sample_Betas_A['Market_Var']
Full_sample_Betas_A['P7_Beta']=Full_sample_Betas_A['P7_cov']/Full_sample_Betas_A['Market_Var']
Full_sample_Betas_A['P8_Beta']=Full_sample_Betas_A['P8_cov']/Full_sample_Betas_A['Market_Var']
Full_sample_Betas_A['P9_Beta']=Full_sample_Betas_A['P9_cov']/Full_sample_Betas_A['Market_Var']
Full_sample_Betas_A['P10_Beta']=Full_sample_Betas_A['P10_cov']/Full_sample_Betas_A['Market_Var']

Full_sample_Betas_A=Full_sample_Betas_A[['P1_Beta','P2_Beta','P3_Beta','P4_Beta','P5_Beta','P6_Beta','P7_Beta','P8_Beta','P9_Beta','P10_Beta']]



#E1
Full_sample_Betas_E1=Portfolio_Returns_E1
Full_sample_Betas_E1=Full_sample_Betas_E1.sum()
Full_sample_Betas_E1['P1_Beta']=Full_sample_Betas_E1['P1_cov']/Full_sample_Betas_E1['Market_Var']
Full_sample_Betas_E1['P2_Beta']=Full_sample_Betas_E1['P2_cov']/Full_sample_Betas_E1['Market_Var']
Full_sample_Betas_E1['P3_Beta']=Full_sample_Betas_E1['P3_cov']/Full_sample_Betas_E1['Market_Var']
Full_sample_Betas_E1['P4_Beta']=Full_sample_Betas_E1['P4_cov']/Full_sample_Betas_E1['Market_Var']
Full_sample_Betas_E1['P5_Beta']=Full_sample_Betas_E1['P5_cov']/Full_sample_Betas_E1['Market_Var']
Full_sample_Betas_E1['P6_Beta']=Full_sample_Betas_E1['P6_cov']/Full_sample_Betas_E1['Market_Var']
Full_sample_Betas_E1['P7_Beta']=Full_sample_Betas_E1['P7_cov']/Full_sample_Betas_E1['Market_Var']
Full_sample_Betas_E1['P8_Beta']=Full_sample_Betas_E1['P8_cov']/Full_sample_Betas_E1['Market_Var']
Full_sample_Betas_E1['P9_Beta']=Full_sample_Betas_E1['P9_cov']/Full_sample_Betas_E1['Market_Var']
Full_sample_Betas_E1['P10_Beta']=Full_sample_Betas_E1['P10_cov']/Full_sample_Betas_E1['Market_Var']

Full_sample_Betas_E1=Full_sample_Betas_E1[['P1_Beta','P2_Beta','P3_Beta','P4_Beta','P5_Beta','P6_Beta','P7_Beta','P8_Beta','P9_Beta','P10_Beta']]

#E2
Full_sample_Betas_E2=Portfolio_Returns_E2
Full_sample_Betas_E2=Full_sample_Betas_E2.sum()
Full_sample_Betas_E2['P1_Beta']=Full_sample_Betas_E2['P1_cov']/Full_sample_Betas_E2['Market_Var']
Full_sample_Betas_E2['P2_Beta']=Full_sample_Betas_E2['P2_cov']/Full_sample_Betas_E2['Market_Var']
Full_sample_Betas_E2['P3_Beta']=Full_sample_Betas_E2['P3_cov']/Full_sample_Betas_E2['Market_Var']
Full_sample_Betas_E2['P4_Beta']=Full_sample_Betas_E2['P4_cov']/Full_sample_Betas_E2['Market_Var']
Full_sample_Betas_E2['P5_Beta']=Full_sample_Betas_E2['P5_cov']/Full_sample_Betas_E2['Market_Var']
Full_sample_Betas_E2['P6_Beta']=Full_sample_Betas_E2['P6_cov']/Full_sample_Betas_E2['Market_Var']
Full_sample_Betas_E2['P7_Beta']=Full_sample_Betas_E2['P7_cov']/Full_sample_Betas_E2['Market_Var']
Full_sample_Betas_E2['P8_Beta']=Full_sample_Betas_E2['P8_cov']/Full_sample_Betas_E2['Market_Var']
Full_sample_Betas_E2['P9_Beta']=Full_sample_Betas_E2['P9_cov']/Full_sample_Betas_E2['Market_Var']
Full_sample_Betas_E2['P10_Beta']=Full_sample_Betas_E2['P10_cov']/Full_sample_Betas_E2['Market_Var']

Full_sample_Betas_E2=Full_sample_Betas_E2[['P1_Beta','P2_Beta','P3_Beta','P4_Beta','P5_Beta','P6_Beta','P7_Beta','P8_Beta','P9_Beta','P10_Beta']]

Full_sample_Betas_NA=Portfolio_Returns_NA
Full_sample_Betas_NA=Full_sample_Betas_NA.sum()
Full_sample_Betas_NA['P1_Beta']=Full_sample_Betas_NA['P1_cov']/Full_sample_Betas_NA['Market_Var']
Full_sample_Betas_NA['P2_Beta']=Full_sample_Betas_NA['P2_cov']/Full_sample_Betas_NA['Market_Var']
Full_sample_Betas_NA['P3_Beta']=Full_sample_Betas_NA['P3_cov']/Full_sample_Betas_NA['Market_Var']
Full_sample_Betas_NA['P4_Beta']=Full_sample_Betas_NA['P4_cov']/Full_sample_Betas_NA['Market_Var']
Full_sample_Betas_NA['P5_Beta']=Full_sample_Betas_NA['P5_cov']/Full_sample_Betas_NA['Market_Var']
Full_sample_Betas_NA['P6_Beta']=Full_sample_Betas_NA['P6_cov']/Full_sample_Betas_NA['Market_Var']
Full_sample_Betas_NA['P7_Beta']=Full_sample_Betas_NA['P7_cov']/Full_sample_Betas_NA['Market_Var']
Full_sample_Betas_NA['P8_Beta']=Full_sample_Betas_NA['P8_cov']/Full_sample_Betas_NA['Market_Var']
Full_sample_Betas_NA['P9_Beta']=Full_sample_Betas_NA['P9_cov']/Full_sample_Betas_NA['Market_Var']
Full_sample_Betas_NA['P10_Beta']=Full_sample_Betas_NA['P10_cov']/Full_sample_Betas_NA['Market_Var']

Full_sample_Betas_NA=Full_sample_Betas_NA[['P1_Beta','P2_Beta','P3_Beta','P4_Beta','P5_Beta','P6_Beta','P7_Beta','P8_Beta','P9_Beta','P10_Beta']]


#Full-sample

Full_sample_Betas=Portfolio_Returns
Full_sample_Betas=Full_sample_Betas.sum()
Full_sample_Betas['P1_Beta']=Full_sample_Betas['P1_cov']/Full_sample_Betas['Market_Var']
Full_sample_Betas['P2_Beta']=Full_sample_Betas['P2_cov']/Full_sample_Betas['Market_Var']
Full_sample_Betas['P3_Beta']=Full_sample_Betas['P3_cov']/Full_sample_Betas['Market_Var']
Full_sample_Betas['P4_Beta']=Full_sample_Betas['P4_cov']/Full_sample_Betas['Market_Var']
Full_sample_Betas['P5_Beta']=Full_sample_Betas['P5_cov']/Full_sample_Betas['Market_Var']
Full_sample_Betas['P6_Beta']=Full_sample_Betas['P6_cov']/Full_sample_Betas['Market_Var']
Full_sample_Betas['P7_Beta']=Full_sample_Betas['P7_cov']/Full_sample_Betas['Market_Var']
Full_sample_Betas['P8_Beta']=Full_sample_Betas['P8_cov']/Full_sample_Betas['Market_Var']
Full_sample_Betas['P9_Beta']=Full_sample_Betas['P9_cov']/Full_sample_Betas['Market_Var']
Full_sample_Betas['P10_Beta']=Full_sample_Betas['P10_cov']/Full_sample_Betas['Market_Var']

Full_sample_Betas=Full_sample_Betas[['P1_Beta','P2_Beta','P3_Beta','P4_Beta','P5_Beta','P6_Beta','P7_Beta','P8_Beta','P9_Beta','P10_Beta']]





# In[80]:


Rolling_Betas_Portfolios=Rolling_Betas[['Date','Time','P1_Rolling_Beta','P2_Rolling_Beta','P3_Rolling_Beta','P4_Rolling_Beta','P5_Rolling_Beta','P6_Rolling_Beta','P7_Rolling_Beta','P8_Rolling_Beta','P9_Rolling_Beta','P10_Rolling_Beta', 'Beta_Dispersion']]
Rolling_Betas_Portfolios_NA=Rolling_Betas_Portfolios[~Rolling_Betas_Portfolios['Date'].isin(FOMC_Dates_List)]
Rolling_Betas_Portfolios_A=Rolling_Betas_Portfolios[Rolling_Betas_Portfolios['Date'].isin(FOMC_Dates_List)]

#Synchronize time relative to the announcement

#A days
Rolling_Beta_Dispersion_A=Rolling_Betas_Portfolios_A
Rolling_Beta_Dispersion_A['Time']=Rolling_Beta_Dispersion_A['Time'].astype(str)
Rolling_Beta_Dispersion_A=pd.merge(Rolling_Beta_Dispersion_A, FOMC_Dates, how='left',left_on=['Date'], right_on = ['Announcement Date'])
Rolling_Beta_Dispersion_A['Rel Time']=pd.to_datetime(Rolling_Beta_Dispersion_A['Time']).astype(int)-pd.to_datetime(Rolling_Beta_Dispersion_A['Announcement Time']).astype(int)
Rolling_Beta_Dispersion_A['Rel Time']=Rolling_Beta_Dispersion_A['Rel Time']/((10**9)*60*5)
Rolling_Beta_Dispersion_A['Rel Time']=Rolling_Beta_Dispersion_A['Rel Time'].apply(np.floor)

#So that we we equal number of observations per relative time, we restrict data to:
Rolling_Beta_Dispersion_A=Rolling_Beta_Dispersion_A[Rolling_Beta_Dispersion_A['Rel Time']>=-33]
Rolling_Beta_Dispersion_A=Rolling_Beta_Dispersion_A[Rolling_Beta_Dispersion_A['Rel Time']<20]

#Create E1 and E2 subsamples:

Rolling_Beta_Dispersion_E1=Rolling_Beta_Dispersion_A[Rolling_Beta_Dispersion_A['Announcement Date'].isin(E1_Dates)]
Rolling_Beta_Dispersion_E2=Rolling_Beta_Dispersion_A[Rolling_Beta_Dispersion_A['Announcement Date'].isin(E2_Dates)]

#NA days

Rolling_Beta_Dispersion_NA=Rolling_Betas_Portfolios_NA
Rolling_Beta_Dispersion_NA['Time']=Rolling_Beta_Dispersion_NA['Time'].astype(str)
Rolling_Beta_Dispersion_NA['Announcement Time']='14:00:00' #on NA days, we set "hypothetical announcement" to 2pm
Rolling_Beta_Dispersion_NA['Rel Time']=pd.to_datetime(Rolling_Beta_Dispersion_NA['Time']).astype(int)-pd.to_datetime(Rolling_Beta_Dispersion_NA['Announcement Time']).astype(int)
Rolling_Beta_Dispersion_NA['Rel Time']=Rolling_Beta_Dispersion_NA['Rel Time']/((10**9)*60*5)
Rolling_Beta_Dispersion_NA['Rel Time']=Rolling_Beta_Dispersion_NA['Rel Time'].apply(np.floor)

#So that we we equal number of observations per relative time, we restrict data to:
Rolling_Beta_Dispersion_NA=Rolling_Beta_Dispersion_NA[Rolling_Beta_Dispersion_NA['Rel Time']>=-33]
Rolling_Beta_Dispersion_NA=Rolling_Beta_Dispersion_NA[Rolling_Beta_Dispersion_NA['Rel Time']<20]


# In[81]:


Average_Beta_Dispersion_Portfolios_E1=pd.DataFrame(Rolling_Beta_Dispersion_E1.groupby('Rel Time').mean())
Average_Beta_Dispersion_E1=Average_Beta_Dispersion_Portfolios_E1.rename(columns={'Beta_Dispersion':'PC Days'})

Average_Beta_Dispersion_Portfolios_E2=pd.DataFrame(Rolling_Beta_Dispersion_E2.groupby('Rel Time').mean())
Average_Beta_Dispersion_E2=Average_Beta_Dispersion_Portfolios_E2.rename(columns={'Beta_Dispersion':'non-PC Days'})

Average_Beta_Dispersion_Portfolios_NA=pd.DataFrame(Rolling_Beta_Dispersion_NA.groupby('Rel Time').mean())
Average_Beta_Dispersion_NA=Average_Beta_Dispersion_Portfolios_NA.rename(columns={'Beta_Dispersion':'NA-Days'})

ax=Average_Beta_Dispersion_E1['PC Days'].plot(color='black',linestyle='solid', title='Beta Dispersion', legend=False)
Average_Beta_Dispersion_E2['non-PC Days'].plot(ax=ax, color="firebrick", linestyle='solid')     
Average_Beta_Dispersion_NA['NA-Days'].plot(ax=ax, color="coral", linestyle='dashed')     

#ax.axvspan(50940, 51540, alpha=0.5, color='grey')

plt.legend(loc=3, prop={'size': 10})
#ax.set_xticks([36000, 45000,50400, 57600])
plt.axvline(x=0, color='grey', linestyle='dashed')
plt.xlabel('Rel Time')
#ax.figure.savefig("Beta_Dispersion.pdf")


# In[82]:


#Concat Dataframes and generate plots in Latex
Average_Beta_Dispersion_E1=Average_Beta_Dispersion_E1[['PC Days']]
Average_Beta_Dispersion_E2=Average_Beta_Dispersion_E2[['non-PC Days']]
Average_Beta_Dispersion_NA=Average_Beta_Dispersion_NA[['NA-Days']]
Beta_Dispersion_SA=pd.merge(Average_Beta_Dispersion_E1,Average_Beta_Dispersion_E2, how='left', left_on=['Rel Time'], right_on=['Rel Time'])
Beta_Dispersion_SA=pd.merge(Beta_Dispersion_SA,Average_Beta_Dispersion_NA, how='left', left_on=['Rel Time'], right_on=['Rel Time'])
Beta_Dispersion_SA['Rel Time']=Beta_Dispersion_SA.index
Beta_Dispersion_SA['Rel Time']=Beta_Dispersion_SA['Rel Time']*5
Beta_Dispersion_SA['Rel Time']=Beta_Dispersion_SA['Rel Time']
Beta_Dispersion_SA['Rel Time']=Beta_Dispersion_SA['Rel Time'].astype(str)+':0'
Beta_Dispersion_SA=Beta_Dispersion_SA.set_index('Rel Time')
#Beta_Dispersion_SA.to_csv('Beta_Dispersion.csv')


# In[83]:


#Concat PN and NN

PN_Dates_all=E1_PN.append(E2_PN)
NN_Dates_all=E1_PN.append(E2_NN)


# In[84]:


#Hereafter, we estimate the SML slope and excess slope following the Fama-McBeth procedure

#Calculate daily portfolio returns
Portfolio_Returns_1=Portfolio_Returns[['Dates','P1','P2','P3','P4','P5','P6','P7','P8','P9','P10']]
Daily_Portfolio_Returns=Portfolio_Returns_1.groupby('Dates').sum()

#Merge with estimated betas and create seperate dataframe for A, NA, E1 and E2
Daily_Portfolio_Returns_1=pd.merge(Daily_Portfolio_Returns,Portfolio_Betas, how='left',left_on='Dates',right_on='Dates')

Daily_Portfolio_Returns_A=Daily_Portfolio_Returns_1[Daily_Portfolio_Returns_1.index.isin(FOMC_Dates_List)]
Daily_Portfolio_Returns_A['date_index']=range(1,1+len(Daily_Portfolio_Returns_A))
date_index_A=Daily_Portfolio_Returns_A[['date_index']]
date_index_A['Dates']=Daily_Portfolio_Returns_A.index
date_index_A=date_index_A.reset_index(drop=True)

Daily_Portfolio_Returns_NA=Daily_Portfolio_Returns_1[~Daily_Portfolio_Returns_1.index.isin(FOMC_Dates_List)]
Daily_Portfolio_Returns_NA['date_index']=range(1,1+len(Daily_Portfolio_Returns_NA))
date_index_NA=Daily_Portfolio_Returns_NA[['date_index']]
date_index_NA['Dates']=Daily_Portfolio_Returns_NA.index
date_index_NA=date_index_NA.reset_index(drop=True)

Daily_Portfolio_Returns_E1=Daily_Portfolio_Returns_1[Daily_Portfolio_Returns_1.index.isin(E1_Dates)]
Daily_Portfolio_Returns_E1['date_index']=range(1,1+len(Daily_Portfolio_Returns_E1))
date_index_E1=Daily_Portfolio_Returns_E1[['date_index']]
date_index_E1['Dates']=Daily_Portfolio_Returns_E1.index
date_index_E1=date_index_E1.reset_index(drop=True)

Daily_Portfolio_Returns_E2=Daily_Portfolio_Returns_1[Daily_Portfolio_Returns_1.index.isin(E2_Dates)]
Daily_Portfolio_Returns_E2['date_index']=range(1,1+len(Daily_Portfolio_Returns_E2))
date_index_E2=Daily_Portfolio_Returns_E2[['date_index']]
date_index_E2['Dates']=Daily_Portfolio_Returns_E2.index
date_index_E2=date_index_E2.reset_index(drop=True)


# In[85]:


# ABSOLUTE Excess Slope at daily level: All A days

# Step 1: Run cross-sectional regression of daily portfolio returns on betas 

slope_all_A=[]
intercept_all_A=[]

Test=Daily_Portfolio_Returns_A
for i in range(1,len(date_index_A)+1):
    Test1=Test[Test['date_index']==i]
    P_Ret=Test1
    P_Ret=pd.DataFrame(P_Ret.iloc[:,0:10])
    P_Ret=P_Ret.T
    Betas=Test1
    Betas=pd.DataFrame(Betas.iloc[:,10:20])
    Betas=Betas.T
    y=P_Ret.iloc[:,0]*10000
    x=Betas.iloc[:,0] 
    #x=Full_sample_Betas #Put Full_sample_Betas here if we want to fix the betas 
    results= linregress(x.astype(float), y.astype(float))
    slope=results.slope
    intercept=results.intercept
    slope_all_A.append(slope)
    intercept_all_A.append(intercept)
    
daily_slope_A=pd.DataFrame(slope_all_A)
daily_slope_A=daily_slope_A.rename(columns={0:'daily slope A'})
daily_slope_A['date_index']=range(1,1+len(date_index_A))
daily_slope_A=pd.merge(daily_slope_A,date_index_A,how='left',left_on='date_index', right_on='date_index')
daily_slope_A=daily_slope_A.drop('date_index',1)
Daily_Returns=Daily_Returns.rename(columns={'Dates':'Dates1'})
daily_slope_A=pd.merge(daily_slope_A,Daily_Returns,how='left',left_on='Dates', right_on='Dates1')
daily_slope_A['Market_Return']=daily_slope_A['Market_Return']*10000

daily_intercept_A=pd.DataFrame(intercept_all_A)
daily_intercept_A=daily_intercept_A.rename(columns={0:'daily intercept A'})
daily_intercept_A['date_index']=range(1,1+len(date_index_A))
daily_intercept_A=pd.merge(daily_intercept_A,date_index_A,how='left',left_on='date_index', right_on='date_index')
daily_intercept_A=daily_intercept_A.drop('date_index',1)

'''
Step 2: Calculate excess slope:
Since we want to focus on those cases where the SML slope and the market have the same sign, we need to create
dummies that indicate whether they have the same sign or not.
'''

daily_slope_A['slope_x_M']=daily_slope_A['daily slope A']*daily_slope_A['Market_Return']
daily_slope_A['same_pos']=np.where((daily_slope_A['slope_x_M']>0)&(daily_slope_A['Market_Return']>0),1,0)
daily_slope_A['ES same_pos']=((daily_slope_A['daily slope A'])-(daily_slope_A['Market_Return']))
daily_slope_A['same_neg']=np.where((daily_slope_A['slope_x_M']>0)&(daily_slope_A['Market_Return']<0),1,0)
daily_slope_A['ES same_neg']=(daily_slope_A['Market_Return'])-(daily_slope_A['daily slope A'])
daily_slope_A['notsame_pos']=np.where((daily_slope_A['slope_x_M']<0)&(daily_slope_A['Market_Return']>0),1,0)
daily_slope_A['ES notsame_pos']=(daily_slope_A['daily slope A']-daily_slope_A['Market_Return'])
daily_slope_A['notsame_neg']=np.where((daily_slope_A['slope_x_M']<0)&(daily_slope_A['Market_Return']<0),1,0)
daily_slope_A['ES notsame_neg']=(daily_slope_A['Market_Return']-daily_slope_A['daily slope A'])

daily_slope_A['notsame']=daily_slope_A['notsame_pos']+daily_slope_A['notsame_neg']
daily_slope_A_all=daily_slope_A
numb_of_obs_same_direction=len(daily_slope_A_all[daily_slope_A_all['notsame']==0])

daily_slope_A_all=daily_slope_A_all[daily_slope_A_all['notsame']==0]

daily_slope_A_all['excess_slope']=daily_slope_A_all['same_pos']*daily_slope_A_all['ES same_pos']+daily_slope_A_all['same_neg']*daily_slope_A_all['ES same_neg']#+daily_slope_A_all['notsame_pos']*daily_slope_A_all['ES notsame_pos']+daily_slope_A_all['notsame_neg']*daily_slope_A_all['ES notsame_neg']#+daily_slope_A_all['close_to_zero']*daily_slope_A_all['ES close_to_zero']
t_stat_ES_A_all=stats.ttest_1samp(daily_slope_A_all['excess_slope'], popmean=0).statistic

#Step 3: Calculate t-stats based on Fama-McBeth regressions:

#Intercept
daily_intercept_A_all=daily_intercept_A
daily_intercept_A_all=daily_intercept_A_all[daily_intercept_A_all['Dates'].isin(daily_slope_A_all['Dates'])]
SE_intercept_all=daily_intercept_A_all['daily intercept A'].var()*(1/len(daily_intercept_A_all['daily intercept A']))
SE_intercept_all=SE_intercept_all**(1/2)
t_stat_intercept_A_all=daily_intercept_A_all['daily intercept A'].mean()/SE_intercept_all
dof_A_all=len(daily_intercept_A_all)
from scipy.stats import t
P_val_intercept_A_all=2*(1 - t.cdf(abs(t_stat_intercept_A_all), dof_A_all))

#Slope
SE_slope_all=daily_slope_A_all['daily slope A'].var()*(1/len(daily_slope_A_all['daily slope A']))
SE_slope_all=SE_slope_all**(1/2)
t_stat_slope_A_all=daily_slope_A_all['daily slope A'].mean()/SE_slope_all
dof_A_all=len(daily_slope_A_all)
from scipy.stats import t
P_val_slope_A_all=2*(1 - t.cdf(abs(t_stat_slope_A_all), dof_A_all))

d0 = {'A-Days': ['%.2f'%daily_intercept_A_all['daily intercept A'].mean(),'%.2f'%(t_stat_intercept_A_all),'%.2f'%daily_slope_A_all['daily slope A'].mean(),'%.2f'%(t_stat_slope_A_all), '%.2f'%daily_slope_A_all['excess_slope'].mean(),'%.2f'%(t_stat_ES_A_all),'%.2f'%Beta_Dispersion_A.mean(),numb_of_obs_same_direction]}
Table_Drivers=pd.DataFrame(data=d0, index=["intercept","","slope","","Excess Slope","","Avrg Beta Disp",'Observations'])
Table_Drivers=round(Table_Drivers,4)
print(Table_Drivers.to_latex())  


# In[ ]:





# In[86]:


# ABSOLUTE Excess Slope at daily level: E1 Dates

# Step 1: Run cross-sectional regression of daily portfolio returns on betas 

slope_all_A=[]
intercept_all_A=[]

Test=Daily_Portfolio_Returns_A
for i in range(1,len(date_index_A)+1):
    Test1=Test[Test['date_index']==i]
    P_Ret=Test1
    P_Ret=pd.DataFrame(P_Ret.iloc[:,0:10])
    P_Ret=P_Ret.T
    Betas=Test1
    Betas=pd.DataFrame(Betas.iloc[:,10:20])
    Betas=Betas.T
    y=P_Ret.iloc[:,0]*10000
    x=Betas.iloc[:,0] 
    #x=Full_sample_Betas #Put Full_sample_Betas here if we want to fix the betas 
    results= linregress(x.astype(float), y.astype(float))
    slope=results.slope
    intercept=results.intercept
    slope_all_A.append(slope)
    intercept_all_A.append(intercept)
    
daily_slope_A=pd.DataFrame(slope_all_A)
daily_slope_A=daily_slope_A.rename(columns={0:'daily slope A'})
daily_slope_A['date_index']=range(1,1+len(date_index_A))
daily_slope_A=pd.merge(daily_slope_A,date_index_A,how='left',left_on='date_index', right_on='date_index')
daily_slope_A=daily_slope_A.drop('date_index',1)
Daily_Returns=Daily_Returns.rename(columns={'Dates':'Dates1'})
daily_slope_A=pd.merge(daily_slope_A,Daily_Returns,how='left',left_on='Dates', right_on='Dates1')
daily_slope_A['Market_Return']=daily_slope_A['Market_Return']*10000

daily_intercept_A=pd.DataFrame(intercept_all_A)
daily_intercept_A=daily_intercept_A.rename(columns={0:'daily intercept A'})
daily_intercept_A['date_index']=range(1,1+len(date_index_A))
daily_intercept_A=pd.merge(daily_intercept_A,date_index_A,how='left',left_on='date_index', right_on='date_index')
daily_intercept_A=daily_intercept_A.drop('date_index',1)

'''
Step 2:Calculate excess slope:
Since we want to focus on those cases where the SML slope and the market have the same sign, we need to create
dummies that indicate whether they have the same sign or not.
'''

daily_slope_A['slope_x_M']=daily_slope_A['daily slope A']*daily_slope_A['Market_Return']
daily_slope_A['same_pos']=np.where((daily_slope_A['slope_x_M']>0)&(daily_slope_A['Market_Return']>0),1,0)
daily_slope_A['ES same_pos']=((daily_slope_A['daily slope A'])-(daily_slope_A['Market_Return']))
daily_slope_A['same_neg']=np.where((daily_slope_A['slope_x_M']>0)&(daily_slope_A['Market_Return']<-0),1,0)
daily_slope_A['ES same_neg']=(daily_slope_A['Market_Return'])-(daily_slope_A['daily slope A'])
daily_slope_A['notsame_pos']=np.where((daily_slope_A['slope_x_M']<0)&(daily_slope_A['Market_Return']>0),1,0)
daily_slope_A['ES notsame_pos']=(daily_slope_A['daily slope A']-daily_slope_A['Market_Return'])
daily_slope_A['notsame_neg']=np.where((daily_slope_A['slope_x_M']<0)&(daily_slope_A['Market_Return']<-0),1,0)
daily_slope_A['ES notsame_neg']=(daily_slope_A['Market_Return']-daily_slope_A['daily slope A'])


daily_slope_A['notsame']=daily_slope_A['notsame_pos']+daily_slope_A['notsame_neg']

#Restrict to E1#

daily_slope_A_E1=daily_slope_A[daily_slope_A['Dates'].isin(E1_Dates)]
daily_intercept_A_E1=daily_intercept_A[daily_intercept_A['Dates'].isin(E1_Dates)]


numb_of_obs_same_direction=len(daily_slope_A_E1[daily_slope_A_E1['notsame']==0])
daily_slope_A_E1=daily_slope_A_E1[daily_slope_A_E1['notsame']==0]
daily_slope_A_E1['excess_slope']=daily_slope_A_E1['same_pos']*daily_slope_A_E1['ES same_pos']+daily_slope_A_E1['same_neg']*daily_slope_A_E1['ES same_neg']+daily_slope_A_E1['notsame_pos']*daily_slope_A_E1['ES notsame_pos']+daily_slope_A_E1['notsame_neg']*daily_slope_A_E1['ES notsame_neg']#+daily_slope_A_E1['close_to_zero']*daily_slope_A_E1['ES close_to_zero']
t_stat_ES_A_E1=stats.ttest_1samp(daily_slope_A_E1['excess_slope'], popmean=0).statistic

#Step 3: Calculate t-stats based on Fama-McBeth regressions:

#Intercept
daily_intercept_A_E1=daily_intercept_A_E1[daily_intercept_A_E1['Dates'].isin(daily_slope_A_E1['Dates'])]
SE_intercept_E1=daily_intercept_A_E1['daily intercept A'].var()*(1/len(daily_intercept_A_E1['daily intercept A']))
SE_intercept_E1=SE_intercept_E1**(1/2)
t_stat_intercept_A_E1=daily_intercept_A_E1['daily intercept A'].mean()/SE_intercept_E1
dof_A_E1=len(daily_intercept_A_E1)
from scipy.stats import t
P_val_intercept_A_E1=2*(1 - t.cdf(abs(t_stat_intercept_A_E1), dof_A_E1))

#Slope
SE_slope_E1=daily_slope_A_E1['daily slope A'].var()*(1/len(daily_slope_A_E1['daily slope A']))
SE_slope_E1=SE_slope_E1**(1/2)
t_stat_slope_A_E1=daily_slope_A_E1['daily slope A'].mean()/SE_slope_E1
dof_A_E1=len(daily_slope_A_E1)
from scipy.stats import t
P_val_slope_A_E1=2*(1 - t.cdf(abs(t_stat_slope_A_E1), dof_A_E1))

d1 = {'E1': ['%.2f'%daily_intercept_A_E1['daily intercept A'].mean(),'%.2f'%(t_stat_intercept_A_E1),'%.2f'%daily_slope_A_E1['daily slope A'].mean(),'%.2f'%(t_stat_slope_A_E1), '%.2f'%daily_slope_A_E1['excess_slope'].mean(),'%.2f'%(t_stat_ES_A_E1),'%.2f'%Beta_Dispersion_E1.mean(),numb_of_obs_same_direction]}
Table_Drivers=pd.DataFrame(data=d1, index=["intercept","","slope","","Excess Slope","","Avrg Beta Disp",'Observations'])
Table_Drivers=round(Table_Drivers,4)
print(Table_Drivers.to_latex())  


# In[87]:


#ABSOLUTE Excess Slope at daily level: E2 Days

# Step 1: Run cross-sectional regression of daily portfolio returns on betas 

slope_all_A=[]
intercept_all_A=[]

Test=Daily_Portfolio_Returns_A
for i in range(1,len(date_index_A)+1):
    Test1=Test[Test['date_index']==i]
    P_Ret=Test1
    P_Ret=pd.DataFrame(P_Ret.iloc[:,0:10])
    P_Ret=P_Ret.T
    Betas=Test1
    Betas=pd.DataFrame(Betas.iloc[:,10:20])
    Betas=Betas.T
    y=P_Ret.iloc[:,0]*10000
    x=Betas.iloc[:,0] 
    #x=Full_sample_Betas #Put Full_sample_Betas here if we want to fix the betas 
    results= linregress(x.astype(float), y.astype(float))
    slope=results.slope
    intercept=results.intercept
    slope_all_A.append(slope)
    intercept_all_A.append(intercept)
    
daily_slope_A=pd.DataFrame(slope_all_A)
daily_slope_A=daily_slope_A.rename(columns={0:'daily slope A'})
daily_slope_A['date_index']=range(1,1+len(date_index_A))
daily_slope_A=pd.merge(daily_slope_A,date_index_A,how='left',left_on='date_index', right_on='date_index')
daily_slope_A=daily_slope_A.drop('date_index',1)
Daily_Returns=Daily_Returns.rename(columns={'Dates':'Dates1'})
daily_slope_A=pd.merge(daily_slope_A,Daily_Returns,how='left',left_on='Dates', right_on='Dates1')
daily_slope_A['Market_Return']=daily_slope_A['Market_Return']*10000

daily_intercept_A=pd.DataFrame(intercept_all_A)
daily_intercept_A=daily_intercept_A.rename(columns={0:'daily intercept A'})
daily_intercept_A['date_index']=range(1,1+len(date_index_A))
daily_intercept_A=pd.merge(daily_intercept_A,date_index_A,how='left',left_on='date_index', right_on='date_index')
daily_intercept_A=daily_intercept_A.drop('date_index',1)

'''
Step 2: Calculate excess slope:
Since we want to focus on those cases where the SML slope and the market have the same sign, we need to create
dummies that indicate whether they have the same sign or not.
'''

daily_slope_A['slope_x_M']=daily_slope_A['daily slope A']*daily_slope_A['Market_Return']
daily_slope_A['same_pos']=np.where((daily_slope_A['slope_x_M']>0)&(daily_slope_A['Market_Return']>0),1,0)
daily_slope_A['ES same_pos']=((daily_slope_A['daily slope A'])-(daily_slope_A['Market_Return']))
daily_slope_A['same_neg']=np.where((daily_slope_A['slope_x_M']>0)&(daily_slope_A['Market_Return']<-0),1,0)
daily_slope_A['ES same_neg']=(daily_slope_A['Market_Return'])-(daily_slope_A['daily slope A'])
daily_slope_A['notsame_pos']=np.where((daily_slope_A['slope_x_M']<0)&(daily_slope_A['Market_Return']>0),1,0)
daily_slope_A['ES notsame_pos']=(daily_slope_A['daily slope A']-daily_slope_A['Market_Return'])
daily_slope_A['notsame_neg']=np.where((daily_slope_A['slope_x_M']<0)&(daily_slope_A['Market_Return']<-0),1,0)
daily_slope_A['ES notsame_neg']=(daily_slope_A['Market_Return']-daily_slope_A['daily slope A'])


daily_slope_A['notsame']=daily_slope_A['notsame_pos']+daily_slope_A['notsame_neg']



#Restrict to E2#
daily_slope_A_E2=daily_slope_A[daily_slope_A['Dates'].isin(E2_Dates)]
daily_intercept_A_E2=daily_intercept_A[daily_intercept_A['Dates'].isin(E2_Dates)]

numb_of_obs_same_direction=len(daily_slope_A_E2[daily_slope_A_E2['notsame']==0])
daily_slope_A_E2=daily_slope_A_E2[daily_slope_A_E2['notsame']==0]

daily_slope_A_E2['excess_slope']=daily_slope_A_E2['same_pos']*daily_slope_A_E2['ES same_pos']+daily_slope_A_E2['same_neg']*daily_slope_A_E2['ES same_neg']#daily_slope_A_E2['notsame_pos']*daily_slope_A_E2['ES notsame_pos']+daily_slope_A_E2['notsame_neg']*daily_slope_A_E2['ES notsame_neg']+daily_slope_A_E2['close_to_zero']*daily_slope_A_E2['ES close_to_zero']
t_stat_ES_A_E2=stats.ttest_1samp(daily_slope_A_E2['excess_slope'], popmean=0).statistic

#Step 3: Calculate t-stats based on Fama-McBeth regressions:

#Intercept
daily_intercept_A_E2=daily_intercept_A_E2[daily_intercept_A_E2['Dates'].isin(daily_slope_A_E2['Dates'])]
SE_intercept_E2=daily_intercept_A_E2['daily intercept A'].var()*(1/len(daily_intercept_A_E2['daily intercept A']))
SE_intercept_E2=SE_intercept_E2**(1/2)
t_stat_intercept_A_E2=daily_intercept_A_E2['daily intercept A'].mean()/SE_intercept_E2
dof_A_E2=len(daily_intercept_A_E2)
from scipy.stats import t
P_val_intercept_A_E2=2*(1 - t.cdf(abs(t_stat_intercept_A_E2), dof_A_E2))

#Slope
SE_slope_E2=daily_slope_A_E2['daily slope A'].var()*(1/len(daily_slope_A_E2['daily slope A']))
SE_slope_E2=SE_slope_E2**(1/2)
t_stat_slope_A_E2=daily_slope_A_E2['daily slope A'].mean()/SE_slope_E2
dof_A_E2=len(daily_slope_A_E2)
from scipy.stats import t
P_val_slope_A_E2=2*(1 - t.cdf(abs(t_stat_slope_A_E2), dof_A_E2))

d2 = {'E2': ['%.2f'%daily_intercept_A_E2['daily intercept A'].mean(),'%.2f'%(t_stat_intercept_A_E2),'%.2f'%daily_slope_A_E2['daily slope A'].mean(),'%.2f'%(t_stat_slope_A_E2), '%.2f'%daily_slope_A_E2['excess_slope'].mean(),'%.2f'%(t_stat_ES_A_E2),'%.2f'%Beta_Dispersion_E2.mean(),numb_of_obs_same_direction]}
Table_Drivers=pd.DataFrame(data=d2, index=["intercept","","slope","","Excess Slope","","Avrg Beta Disp",'Observations'])
Table_Drivers=round(Table_Drivers,4)
print(Table_Drivers.to_latex()) 


# In[88]:


#Table for Latex (table 4 in paper)

d0.update(d1)
d0.update(d2)

Table_Drivers=pd.DataFrame(data=d0, index=["intercept","","slope","","Excess Slope","","Avrg Beta Disp",'Observations'])
Table_Drivers=round(Table_Drivers,4)
print(Table_Drivers.to_latex()) 


# In[89]:


# ABSOLUTE Excess Slope at daily level: All NA days


slope_all_NA=[]
intercept_all_NA=[]

Test=Daily_Portfolio_Returns_NA
for i in range(1,len(date_index_NA)+1):
    Test1=Test[Test['date_index']==i]
    P_Ret=Test1
    P_Ret=pd.DataFrame(P_Ret.iloc[:,0:10])
    P_Ret=P_Ret.T
    Betas=Test1
    Betas=pd.DataFrame(Betas.iloc[:,10:20])
    Betas=Betas.T
    y=P_Ret.iloc[:,0]*10000
    x=Betas.iloc[:,0] 
    #x=Full_sample_Betas #Put Full_sample_Betas here if we want to fix the betas 
    results= linregress(x.astype(float), y.astype(float))
    slope=results.slope
    intercept=results.intercept
    slope_all_NA.append(slope)
    intercept_all_NA.append(intercept)
    
daily_slope_NA=pd.DataFrame(slope_all_NA)
daily_slope_NA=daily_slope_NA.rename(columns={0:'daily slope NA'})
daily_slope_NA['date_index']=range(1,1+len(date_index_NA))
daily_slope_NA=pd.merge(daily_slope_NA,date_index_NA,how='left',left_on='date_index', right_on='date_index')
daily_slope_NA=daily_slope_NA.drop('date_index',1)
Daily_Returns=Daily_Returns.rename(columns={'Dates':'Dates1'})
daily_slope_NA=pd.merge(daily_slope_NA,Daily_Returns,how='left',left_on='Dates', right_on='Dates')
daily_slope_NA['Market_Return']=daily_slope_NA['Market_Return']*10000

daily_intercept_NA=pd.DataFrame(intercept_all_NA)
daily_intercept_NA=daily_intercept_NA.rename(columns={0:'daily intercept NA'})
daily_intercept_NA['date_index']=range(1,1+len(date_index_NA))
daily_intercept_NA=pd.merge(daily_intercept_NA,date_index_NA,how='left',left_on='date_index', right_on='date_index')
daily_intercept_NA=daily_intercept_NA.drop('date_index',1)

'''
Calculate excess slope:
Since we want to focus on those cases where the SML slope and the market have the same sign, we need to create
dummies that indicate whether they have the same sign or not.
'''


daily_slope_NA['slope_x_M']=daily_slope_NA['daily slope NA']*daily_slope_NA['Market_Return']
daily_slope_NA['same_pos']=np.where((daily_slope_NA['slope_x_M']>0)&(daily_slope_NA['Market_Return']>0),1,0)
daily_slope_NA['ES same_pos']=((daily_slope_NA['daily slope NA'])-(daily_slope_NA['Market_Return']))
daily_slope_NA['same_neg']=np.where((daily_slope_NA['slope_x_M']>0)&(daily_slope_NA['Market_Return']<-0),1,0)
daily_slope_NA['ES same_neg']=(daily_slope_NA['Market_Return'])-(daily_slope_NA['daily slope NA'])
daily_slope_NA['notsame_pos']=np.where((daily_slope_NA['slope_x_M']<0)&(daily_slope_NA['Market_Return']>0),1,0)
daily_slope_NA['ES notsame_pos']=(daily_slope_NA['daily slope NA']-daily_slope_NA['Market_Return'])
daily_slope_NA['notsame_neg']=np.where((daily_slope_NA['slope_x_M']<0)&(daily_slope_NA['Market_Return']<-0),1,0)
daily_slope_NA['ES notsame_neg']=(daily_slope_NA['Market_Return']-daily_slope_NA['daily slope NA'])

daily_slope_NA['notsame']=daily_slope_NA['notsame_pos']+daily_slope_NA['notsame_neg']

#All announcements
daily_slope_NA_all=daily_slope_NA

numb_of_obs_same_direction=len(daily_slope_NA_all[daily_slope_NA_all['notsame']==0])

daily_slope_NA_all=daily_slope_NA_all[daily_slope_NA_all['notsame']==0]

daily_slope_NA_all['excess_slope']=daily_slope_NA_all['same_pos']*daily_slope_NA_all['ES same_pos']+daily_slope_NA_all['same_neg']*daily_slope_NA_all['ES same_neg']#+daily_slope_A_all['notsame_pos']*daily_slope_A_all['ES notsame_pos']+daily_slope_A_all['notsame_neg']*daily_slope_A_all['ES notsame_neg']#+daily_slope_A_all['close_to_zero']*daily_slope_A_all['ES close_to_zero']
t_stat_ES_NA_all=stats.ttest_1samp(daily_slope_NA_all['excess_slope'], popmean=0).statistic


#Estimate FB t-stat intercept
daily_intercept_NA_all=daily_intercept_NA
daily_intercept_NA_all=daily_intercept_NA_all[daily_intercept_NA_all['Dates'].isin(daily_slope_NA_all['Dates'])]
SE_intercept_all=daily_intercept_NA_all['daily intercept NA'].var()*(1/len(daily_intercept_NA_all['daily intercept NA']))
SE_intercept_all=SE_intercept_all**(1/2)
t_stat_intercept_NA_all=daily_intercept_NA_all['daily intercept NA'].mean()/SE_intercept_all
dof_NA_all=len(daily_intercept_NA_all)
from scipy.stats import t
P_val_intercept_NA_all=2*(1 - t.cdf(abs(t_stat_intercept_NA_all), dof_NA_all))

#Estimate FB t-stat slope
SE_slope_all=daily_slope_NA_all['daily slope NA'].var()*(1/len(daily_slope_NA_all['daily slope NA']))
SE_slope_all=SE_slope_all**(1/2)
t_stat_slope_NA_all=daily_slope_NA_all['daily slope NA'].mean()/SE_slope_all
dof_NA_all=len(daily_slope_NA_all)
from scipy.stats import t
P_val_slope_NA_all=2*(1 - t.cdf(abs(t_stat_slope_NA_all), dof_NA_all))

d0 = {'NA-Days': ['%.2f'%daily_intercept_NA_all['daily intercept NA'].mean(),'%.2f'%(t_stat_intercept_NA_all),'%.2f'%daily_slope_NA_all['daily slope NA'].mean(),'%.2f'%(t_stat_slope_NA_all), '%.2f'%daily_slope_NA_all['excess_slope'].mean(),'%.2f'%(t_stat_ES_NA_all),'%.2f'%Beta_Dispersion.mean(),numb_of_obs_same_direction]}
Table_Drivers=pd.DataFrame(data=d0, index=["intercept","","slope","","Excess Slope","","Avrg Beta Disp",'Observations'])
Table_Drivers=round(Table_Drivers,4)
print(Table_Drivers.to_latex()) 


# In[90]:


'''
Hereafter, we analyze whether the dspersion in value and/or dispersion in betas is able to explain 
the excess SML slope.
'''


# In[91]:


#Merge datasets

Excess_Slopes_A=daily_slope_A_all[['Dates','excess_slope','Market_Return']]
Excess_Slopes_1=pd.merge(Excess_Slopes_A,Beta_Dispersion,how='left',left_on='Dates',right_on='Dates')
Excess_Slopes_A=Excess_Slopes_1[['Dates','excess_slope','Beta_Dispersion','Market_Return']]

Excess_Slopes_A_E1=daily_slope_A_E1[['Dates','excess_slope','Market_Return']]
Excess_Slopes_1=pd.merge(Excess_Slopes_A_E1,Beta_Dispersion,how='left',left_on='Dates',right_on='Dates')
Excess_Slopes_A_E1=Excess_Slopes_1[['Dates','excess_slope','Beta_Dispersion','Market_Return']]

Excess_Slopes_A_E2=daily_slope_A_E2[['Dates','excess_slope','Market_Return']]
Excess_Slopes_1=pd.merge(Excess_Slopes_A_E2,Beta_Dispersion,how='left',left_on='Dates',right_on='Dates')
Excess_Slopes_A_E2=Excess_Slopes_1[['Dates','excess_slope','Beta_Dispersion','Market_Return']]

Excess_Slopes_NA=daily_slope_NA_all[['Dates','excess_slope']]
Excess_Slopes_1=pd.merge(Excess_Slopes_NA,Beta_Dispersion,how='left',left_on='Dates',right_on='Dates')
Excess_Slopes_NA=Excess_Slopes_1[['Dates','excess_slope','Beta_Dispersion']]
Excess_Slopes_NA


# In[92]:


#Import data from Compustat in order to calculate book values 

book_values=pd.read_stata('Book Values.dta')
CRSP_Compustat_Link=pd.read_stata('CRSP_Compustat_Link.dta')
CRSP_Compustat_Link['LPERMNO']=CRSP_Compustat_Link['LPERMNO'].astype(int)
CRSP_Compustat_Link1=CRSP_Compustat_Link
CRSP_Compustat_Link1=CRSP_Compustat_Link1.drop_duplicates(subset='LPERMNO', keep='last')
CRSP_Compustat_Link1=CRSP_Compustat_Link1[['GVKEY','LPERMNO','LINKDT','LINKENDDT']]
book_values=pd.merge(book_values,CRSP_Compustat_Link1,how='left',left_on='GVKEY', right_on='GVKEY')
book_values=book_values[book_values['LPERMNO'].notna()]
book_values['itcb'] = book_values['itcb'].fillna(0)
book_values['pstkl'] = book_values['pstkl'].fillna(0)
book_values['txdb'] = book_values['txdb'].fillna(0)
book_values


# In[93]:


#calculations based on Bali et al. (2016):

book_values['B']=book_values['seq']+book_values['txdb']+book_values['itcb']-book_values['pstkl']
book_values=book_values[book_values['B']>0]
book_values=book_values[book_values['indfmt']=='INDL']
book_values=book_values[book_values['datafmt']=='STD']
book_values1=book_values
book_values1=book_values1[['LPERMNO','datadate','B']]
book_values1['LPERMNO']=book_values1['LPERMNO'].astype(int)
book_values1['year']=pd.to_datetime(book_values1['datadate']).dt.year

#Make sure that we avoid any look-ahead bias:
book_values1['next_year']=book_values1['year']+1
book_values1['next_year2']=book_values1['year']+2
book_values1['month']=6
book_values1['year-month avail']=book_values1['next_year'].astype(str)+str('-06')+str('-01')
book_values1['key1']=book_values1['next_year'].astype(str)+str('_2')
book_values1['key2']=book_values1['next_year2'].astype(str)+str('_1')
book_values2=book_values1[['LPERMNO','B','key1','key2']]
key1=book_values2[['LPERMNO','B','key1']]
key2=book_values2[['LPERMNO','B','key2']]
key2=key2.rename(columns={'key2':'key1'})
key=pd.concat([key1,key2])
key=key.rename(columns={'LPERMNO':'Permno'})
key


# In[94]:


#Calculate book-to-market ratios:
Market_Caps=pd.read_stata('CRSP_Market_Cap.dta')
Market_Caps['PERMNO']=Market_Caps['PERMNO'].astype(int)
Market_Caps=Market_Caps[Market_Caps['PERMNO'].isin(List_Permnos)]
Market_Caps['year']=pd.to_datetime(Market_Caps['DlyCalDt']).dt.year
Market_Caps['month']=pd.to_datetime(Market_Caps['DlyCalDt']).dt.month
Market_Caps['half_year']=np.where(Market_Caps['month']>6,2,1)
Market_Caps['key1']=Market_Caps['year'].astype(str)+str('_')+Market_Caps['half_year'].astype(str)
Market_Caps['M_prevd']=Market_Caps['DlyPrevCap']/1000 #Market cap in millions
B_M_ratios=pd.merge(Market_Caps,key,how='left',left_on=['PERMNO','key1'], right_on=['Permno','key1'])
B_M_ratios['B_M']=np.log(B_M_ratios['B']/B_M_ratios['M_prevd'])
B_M_ratios['B_M']=winsorize(B_M_ratios['B_M'],(0.005,0.005)) #B/M ratios
B_M_ratios


# In[95]:


#Merge B/M ratios with daily returns data at the stock level

B_M_ratios=B_M_ratios[['DlyCalDt','M_prevd','PERMNO','B_M']]
B_M_ratios['DlyCalDt']=B_M_ratios['DlyCalDt'].astype(str)
B_M_ratios_1=pd.merge(Beta_indic,B_M_ratios,how='left',left_on=['Permno','Date'],right_on=['PERMNO','DlyCalDt'])
B_M_ratios_1              


# In[96]:


#Calculate weighted average book-to-market (BM) ratio of beta-sorted portfolios

BM_P1=B_M_ratios_1[B_M_ratios_1['P1']==1]
M_sum=pd.DataFrame(BM_P1.groupby('Date')['M_prevd'].sum())
BM_P1=pd.merge(BM_P1,M_sum,how='left',left_on='Date',right_on='Date')
BM_P1['weight']=BM_P1['M_prevd_x']/BM_P1['M_prevd_y']
BM_P1=BM_P1.dropna()
BM_P1['weighted_BM']=BM_P1['weight']*BM_P1['B_M']
BM_P1=pd.DataFrame(BM_P1.groupby('Date')['weighted_BM'].sum())
BM_P1=BM_P1.rename(columns={'weighted_BM':'P1_BM'})

BM_P2=B_M_ratios_1[B_M_ratios_1['P2']==1]
M_sum=pd.DataFrame(BM_P2.groupby('Date')['M_prevd'].sum())
BM_P2=pd.merge(BM_P2,M_sum,how='left',left_on='Date',right_on='Date')
BM_P2['weight']=BM_P2['M_prevd_x']/BM_P2['M_prevd_y']
BM_P2=BM_P2.dropna()
BM_P2['weighted_BM']=BM_P2['weight']*BM_P2['B_M']
BM_P2=pd.DataFrame(BM_P2.groupby('Date')['weighted_BM'].sum())
BM_P2=BM_P2.rename(columns={'weighted_BM':'P2_BM'})

BM_P3=B_M_ratios_1[B_M_ratios_1['P3']==1]
M_sum=pd.DataFrame(BM_P3.groupby('Date')['M_prevd'].sum())
BM_P3=pd.merge(BM_P3,M_sum,how='left',left_on='Date',right_on='Date')
BM_P3['weight']=BM_P3['M_prevd_x']/BM_P3['M_prevd_y']
BM_P3=BM_P3.dropna()
BM_P3['weighted_BM']=BM_P3['weight']*BM_P3['B_M']
BM_P3=pd.DataFrame(BM_P3.groupby('Date')['weighted_BM'].sum())
BM_P3=BM_P3.rename(columns={'weighted_BM':'P3_BM'})

BM_P4=B_M_ratios_1[B_M_ratios_1['P4']==1]
M_sum=pd.DataFrame(BM_P4.groupby('Date')['M_prevd'].sum())
BM_P4=pd.merge(BM_P4,M_sum,how='left',left_on='Date',right_on='Date')
BM_P4['weight']=BM_P4['M_prevd_x']/BM_P4['M_prevd_y']
BM_P4=BM_P4.dropna()
BM_P4['weighted_BM']=BM_P4['weight']*BM_P4['B_M']
BM_P4=pd.DataFrame(BM_P4.groupby('Date')['weighted_BM'].sum())
BM_P4=BM_P4.rename(columns={'weighted_BM':'P4_BM'})

BM_P5=B_M_ratios_1[B_M_ratios_1['P5']==1]
M_sum=pd.DataFrame(BM_P5.groupby('Date')['M_prevd'].sum())
BM_P5=pd.merge(BM_P5,M_sum,how='left',left_on='Date',right_on='Date')
BM_P5['weight']=BM_P5['M_prevd_x']/BM_P5['M_prevd_y']
BM_P5=BM_P5.dropna()
BM_P5['weighted_BM']=BM_P5['weight']*BM_P5['B_M']
BM_P5=pd.DataFrame(BM_P5.groupby('Date')['weighted_BM'].sum())
BM_P5=BM_P5.rename(columns={'weighted_BM':'P5_BM'})

BM_P6=B_M_ratios_1[B_M_ratios_1['P6']==1]
M_sum=pd.DataFrame(BM_P6.groupby('Date')['M_prevd'].sum())
BM_P6=pd.merge(BM_P6,M_sum,how='left',left_on='Date',right_on='Date')
BM_P6['weight']=BM_P6['M_prevd_x']/BM_P6['M_prevd_y']
BM_P6=BM_P6.dropna()
BM_P6['weighted_BM']=BM_P6['weight']*BM_P6['B_M']
BM_P6=pd.DataFrame(BM_P6.groupby('Date')['weighted_BM'].sum())
BM_P6=BM_P6.rename(columns={'weighted_BM':'P6_BM'})

BM_P7=B_M_ratios_1[B_M_ratios_1['P7']==1]
M_sum=pd.DataFrame(BM_P7.groupby('Date')['M_prevd'].sum())
BM_P7=pd.merge(BM_P7,M_sum,how='left',left_on='Date',right_on='Date')
BM_P7['weight']=BM_P7['M_prevd_x']/BM_P7['M_prevd_y']
BM_P7=BM_P7.dropna()
BM_P7['weighted_BM']=BM_P7['weight']*BM_P7['B_M']
BM_P7=pd.DataFrame(BM_P7.groupby('Date')['weighted_BM'].sum())
BM_P7=BM_P7.rename(columns={'weighted_BM':'P7_BM'})

BM_P8=B_M_ratios_1[B_M_ratios_1['P8']==1]
M_sum=pd.DataFrame(BM_P8.groupby('Date')['M_prevd'].sum())
BM_P8=pd.merge(BM_P8,M_sum,how='left',left_on='Date',right_on='Date')
BM_P8['weight']=BM_P8['M_prevd_x']/BM_P8['M_prevd_y']
BM_P8=BM_P8.dropna()
BM_P8['weighted_BM']=BM_P8['weight']*BM_P8['B_M']
BM_P8=pd.DataFrame(BM_P8.groupby('Date')['weighted_BM'].sum())
BM_P8=BM_P8.rename(columns={'weighted_BM':'P8_BM'})

BM_P9=B_M_ratios_1[B_M_ratios_1['P9']==1]
M_sum=pd.DataFrame(BM_P9.groupby('Date')['M_prevd'].sum())
BM_P9=pd.merge(BM_P9,M_sum,how='left',left_on='Date',right_on='Date')
BM_P9['weight']=BM_P9['M_prevd_x']/BM_P9['M_prevd_y']
BM_P9=BM_P9.dropna()
BM_P9['weighted_BM']=BM_P9['weight']*BM_P9['B_M']
BM_P9=pd.DataFrame(BM_P9.groupby('Date')['weighted_BM'].sum())
BM_P9=BM_P9.rename(columns={'weighted_BM':'P9_BM'})

BM_P10=B_M_ratios_1[B_M_ratios_1['P10']==1]
M_sum=pd.DataFrame(BM_P10.groupby('Date')['M_prevd'].sum())
BM_P10=pd.merge(BM_P10,M_sum,how='left',left_on='Date',right_on='Date')
BM_P10['weight']=BM_P10['M_prevd_x']/BM_P10['M_prevd_y']
BM_P10=BM_P10.dropna()
BM_P10['weighted_BM']=BM_P10['weight']*BM_P10['B_M']
BM_P10=pd.DataFrame(BM_P10.groupby('Date')['weighted_BM'].sum())
BM_P10=BM_P10.rename(columns={'weighted_BM':'P10_BM'})

BM_ratios=pd.merge(BM_P1,BM_P2, how='left',left_on='Date',right_on='Date')
BM_ratios=pd.merge(BM_ratios,BM_P3, how='left',left_on='Date',right_on='Date')
BM_ratios=pd.merge(BM_ratios,BM_P4, how='left',left_on='Date',right_on='Date')
BM_ratios=pd.merge(BM_ratios,BM_P5, how='left',left_on='Date',right_on='Date')
BM_ratios=pd.merge(BM_ratios,BM_P6, how='left',left_on='Date',right_on='Date')
BM_ratios=pd.merge(BM_ratios,BM_P7, how='left',left_on='Date',right_on='Date')
BM_ratios=pd.merge(BM_ratios,BM_P8, how='left',left_on='Date',right_on='Date')
BM_ratios=pd.merge(BM_ratios,BM_P9, how='left',left_on='Date',right_on='Date')
BM_ratios=pd.merge(BM_ratios,BM_P10, how='left',left_on='Date',right_on='Date')


# In[97]:


#Calculate dispersion in B/M, scaled by the average B/M (at the portfolio level)

Avrg_BM=pd.DataFrame(BM_ratios.T.mean()).abs()+1
Avrg_BM=Avrg_BM.rename(columns={0:'Avrg_B/M'})
Var_BM=pd.DataFrame(BM_ratios.T.std())
Var_BM=Var_BM.rename(columns={0:'Var_B/M'})
Disp_Value=pd.merge(Avrg_BM,Var_BM,how='left',left_on='Date',right_on='Date')
Disp_Value['Disp_Value']=Disp_Value['Var_B/M']/Disp_Value['Avrg_B/M']
Disp_Value_1=Disp_Value
Disp_Value_1.plot()


# In[98]:


#Merge Dispersion in value with excess return data

Expl_Excess_Slope_A=pd.merge(Excess_Slopes_A,Disp_Value,how='left',left_on='Dates',right_on='Date')
Expl_Excess_Slope_E1=pd.merge(Excess_Slopes_A_E1,Disp_Value,how='left',left_on='Dates',right_on='Date')
Expl_Excess_Slope_E2=pd.merge(Excess_Slopes_A_E2,Disp_Value,how='left',left_on='Dates',right_on='Date')

#Calculate ratio between dispersion in value and dispersion in beta

Expl_Excess_Slope_A['Value_Disp/Beta_Disp']=Expl_Excess_Slope_A['Disp_Value']/Expl_Excess_Slope_A['Beta_Dispersion']
Expl_Excess_Slope_E1['Value_Disp/Beta_Disp']=Expl_Excess_Slope_E1['Disp_Value']/Expl_Excess_Slope_E1['Beta_Dispersion']
Expl_Excess_Slope_E2['Value_Disp/Beta_Disp']=Expl_Excess_Slope_E2['Disp_Value']/Expl_Excess_Slope_E2['Beta_Dispersion']


# In[99]:


#Run regression of excess slope on ratio between dispersion in value and dispersion in betas

#A

x=Expl_Excess_Slope_A[['Disp_Value','Beta_Dispersion']]
x1=Expl_Excess_Slope_A['Value_Disp/Beta_Disp']
y=Expl_Excess_Slope_A['excess_slope']

x = sm.add_constant(x) # adding a constant
x1 = sm.add_constant(x1) # adding a constant

model = sm.OLS(y, x).fit(cov_type='HC3')
model_1 = sm.OLS(y, x1).fit(cov_type='HC3')
rsquared=model.rsquared


# In[100]:


#Run regression of excess slope on ratio between dispersion in value and dispersion in betas

#E1

Expl_Excess_Slope_E1['Value_Disp/Beta_Disp']=Expl_Excess_Slope_E1['Disp_Value']/Expl_Excess_Slope_E1['Beta_Dispersion']

x=Expl_Excess_Slope_E1[['Disp_Value','Beta_Dispersion']]
x1=Expl_Excess_Slope_E1['Value_Disp/Beta_Disp']
y=Expl_Excess_Slope_E1['excess_slope']

x = sm.add_constant(x) # adding a constant
x1 = sm.add_constant(x1) # adding a constant

model1 = sm.OLS(y, x).fit(cov_type='HC3')
model1_1 = sm.OLS(y, x1).fit(cov_type='HC3')


# In[101]:


#Run regression of excess slope on ratio between dispersion in value and dispersion in betas

#E2

x=Expl_Excess_Slope_E2[['Disp_Value','Beta_Dispersion']]
x1=Expl_Excess_Slope_E2['Value_Disp/Beta_Disp']
y=Expl_Excess_Slope_E2['excess_slope']

x = sm.add_constant(x) # adding a constant
x1 = sm.add_constant(x1) # adding a constant

model2 = sm.OLS(y, x).fit(cov_type='HC3')
model2_1 = sm.OLS(y, x1).fit(cov_type='HC3')


# In[102]:


#Produce Latex code for table 5 in paper

from statsmodels.iolib.summary2 import summary_col
print(summary_col([model_1,model1_1,model2_1],stars=True,float_format='%0.2f').as_latex())


# In[110]:


'''
END
'''

