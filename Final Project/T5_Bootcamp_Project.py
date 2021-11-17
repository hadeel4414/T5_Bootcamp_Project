#!/usr/bin/env python
# coding: utf-8

# #  Ford Go Bike 2017
# ## by Hadeel Altamimi
# 
# 
# > The data set contains information about bicycles rides provided by Bay Area Motivate from the Bay Wheels bicycle sharing service. The service motivate is committed to supporting bicycling as an alternative transportation option.
# ><br><br>The data are available at https://s3.amazonaws.com/fordgobike-data/index.html
# 
# > The dataset contains the following features:
# Trip Duration (seconds)<br>
# Start Time and Date<br>
# End Time and Date<br>
# Start Station ID<br>
# Start Station Name<br>
# Start Station Latitude<br>
# Start Station Longitude<br>
# End Station ID<br>
# End Station Name<br>
# End Station Latitude<br>
# End Station Longitude<br>
# Bike ID<br>
# User Type (Subscriber or Customer – “Subscriber” = Member or “Customer” = Casual)<br>
# User Birth Year<br>
# User Gender
# 

# In[1]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


import datetime 
import calendar 
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression




get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Gathering

# In[2]:


ford = pd.read_csv('/Users/hadeel/Desktop/Data BootCamp/T5_Bootcamp_Project/Data set/2017-fordgobike-tripdata.csv')


# ## Data Assessing

# In[3]:


ford.head()


# In[4]:


ford.info()


# In[5]:


ford.shape


# In[6]:


ford.describe()


# In[7]:


ford.isnull().sum()


# ## Accessing Summary
# 
# 1. member_birth_year and member_gender columns have missing values
#    
#    1.1. start_station_area and end_station_area columns should be added to label strat and end stations
#    
#    
# 2. start_time and end_time should be data type datetime
# 3. weekday and hours column should be added for analysis
# 4. start_station_id, end_station_id, member_birth_year should be data type int
# 5. member_birth_year should be converted to member's age
# 
#    5.1. member_age outliers should be removed
#    
#    5.2. age_group column added
#    
#    
# 6. user_type, member_gender, month, weekday and hour should be categorical variables
# 
# 7. duration_sec has many large values maybe outlier due to customers forgot to log off after using
# 
# 

# ## Data Cleaning

# In[8]:


df1=ford.copy()


# #### Define 1 :
# 
# member_birth_year and member_gender columns have missing values

# In[9]:


# Drop Null values
df1.dropna(subset = ["member_birth_year","member_gender"], inplace=True)
df1.isnull().sum()


# In[10]:


#Change the birth year from float to int
df1["member_birth_year"]=df1["member_birth_year"].astype(int)
df1.info()


# In[11]:


# Create a column for age which be derived from the birth year
df1['age'] = 2017-df1['member_birth_year']


# In[12]:


df1.describe()


# In[13]:


#There are incorrect age, the maximum age 131 which is not accepted
# I will filter the ages and keeps the ages equal or less than 60
df1=df1[df1['age'] <= 60]
df1.info()


# > Only consider coordinates that are located in Bay Area. Outlier should be removed and not useful for the analysis.
# 
# 

# In[14]:


# Remove all the coordinates outside Bay Area
df1= df1.query('end_station_latitude > 37 and end_station_latitude < 38 and end_station_longitude > -123 and end_station_longitude < -121 and start_station_latitude > 37 and start_station_latitude < 38 and start_station_longitude > -123 and start_station_longitude < -121')


# In[15]:


df1.shape


# In[16]:


# Check for missing values
df1.isna().sum().any()


# In[17]:


# Plotting no missing data df start/end id and station name 
plt.figure(figsize=(10,10))
sb.scatterplot(data = df1[df1.start_station_id.isnull()], 
                x = "start_station_longitude",
                y = "start_station_latitude",
                s = 300, alpha = 0.1)
sb.scatterplot(data = df1.dropna(subset=["start_station_id"]).sample(100000),
                x = "start_station_longitude",
                y = "start_station_latitude",
                s = 300, alpha = 0.1);


# >Above graph shows the 3 main areas of GoBike: SF, SJ and East Bay.
# 
# >start_station_area and start_station_area columns should be added to label each station

# #### Define 1.1: 
# 
# start_station_area and end_station_area columns should be added to label strat and end stations

# In[18]:


# Using sklean module's K-Mean clustering algorithm to label each start/end stations
# Start staion
kmeans_start = KMeans(n_clusters=3, random_state=0).fit(df1[["start_station_latitude", "start_station_longitude"]])
df1['start_station_area'] = kmeans_start.labels_
df1['start_station_area'].replace({0:'East Bay', 1:'San Jose', 2:'San Francisco'}, inplace=True) # Hard coding the labels
                                                                                                    # set random_state = 0


# In[19]:


# End station
kmeans_end = KMeans(n_clusters=3, random_state=0).fit(df1[["end_station_latitude", "end_station_longitude"]])
df1['end_station_area'] = kmeans_end.labels_
df1['end_station_area'].replace({0:'East Bay', 1:'San Francisco', 2:'San Jose'}, inplace=True)   # Hard coding the labels
                                                                                                    # set random_state = 0


# #### Test 1.1

# In[20]:


# Plotting start station coordinates
plt.figure(figsize=(10,10))
plt.title('Location of Start Stations')
sb.scatterplot(data = df1.sample(100000),
                x = "start_station_longitude",
                y = "start_station_latitude",
                hue = "start_station_area",
                s=300, alpha = 0.1);


# In[21]:



# Plotting end station coordinates
plt.figure(figsize=(10,10))
plt.title('Location of End Stations')
sb.scatterplot(data = df1.sample(100000),
                x = "end_station_longitude",
                y = "end_station_latitude",
                hue = "end_station_area",
                s=300, alpha = 0.1);


# #### Define 2
# start_time and end_time should be converted to datetime data type

# In[22]:


# Converted to datetime
df1['start_time']=pd.to_datetime(df1['start_time'])
df1['end_time']=pd.to_datetime(df1['end_time'])


# #### Test 2

# In[23]:


df1.dtypes


# #### Define 3
# month, weekday and hours column should be added for analysis

# In[24]:


# Create month column
df1['month'] = df1['start_time'].dt.month

# Create month column
df1['weekday'] = df1['start_time'].dt.day_name().astype('category')

# Create month column
df1['hour'] = df1['start_time'].dt.hour


# #### Test 3

# In[25]:


df1[['month','weekday','hour']].sample(5)


# #### Define 4
# start_station_id, end_station_id, member_birth_year should be data type int

# In[26]:


df1['start_station_id']=df1['start_station_id'].astype('int')
df1['end_station_id']=df1['end_station_id'].astype('int')
df1['member_birth_year']=df1['member_birth_year'].astype('int')


# #### Test 4

# In[27]:


df1.dtypes


# #### Define 5
# member_birth_year should be converted to member_age

# In[28]:


# Get year when member use the bike
df1['start_time'].dt.year.value_counts()


# In[29]:


# Add member_age column
df1['member_age'] = df1['start_time'].dt.year - df1['member_birth_year']


# #### Test 5

# In[30]:


# Check for null
df1['member_age'].isna().sum()


# In[31]:


# Plot member age distibution
plt.figure(figsize = [20, 2])
base_color = sb.color_palette()[0]
sb.boxplot(data=df1, x='member_age', color=base_color);


# In[32]:


df1.member_age.describe(percentiles = [0.01, 0.05, 0.95, 0.99])


# > From above, the youngest users are 18 which makes sense. However, 95% percentile combined with box plot shows older users above 55 year old seems to be outliers. It is logical that rows above 60 yrs old should be removed

# #### Define 5.1
# member_age outlier should be removed

# In[33]:


# Remove all rows with age <= 60
df1 = df1.query('member_age <= 60')


# #### Test 5.1

# In[34]:


# Plot member age <= 60 yrs distibution
plt.figure(figsize = [20, 2])
base_color = sb.color_palette()[0]
sb.boxplot(data=df1, x='member_age', color=base_color);


# #### Define 5.2
# Add age_group label column

# In[35]:


# Add age_group label column
labels = ["{}s".format(i) for i in range(10,51,10)]
df1['age_group'] = pd.cut(df1['member_age'], range(10, 61, 10), right=True, labels=labels)


# #### Test 5.2
# 
# 

# In[36]:


# Test member age and their age_group
df1[['age_group','member_age']].sample(5)


# In[37]:


df1['age_group'].isna().any()


# #### Define 6
# user_type, member_gender, age_group, month, weekday and hour should be categorical variables

# In[38]:


# Converted to categorical 
df1['user_type']=df1['user_type'].astype('category')
df1['member_gender']=df1['member_gender'].astype('category')
df1['age_group']=df1['age_group'].astype('category')
df1['month']=df1['month'].astype('category')
df1['weekday']=df1['weekday'].astype('category')
df1['hour']=df1['hour'].astype('category')
df1['start_station_area']=df1['start_station_area'].astype('category')
df1['end_station_area']=df1['end_station_area'].astype('category')


# #### Test 6

# In[39]:


df1.dtypes


# In[40]:


df1.member_gender.value_counts()


# #### Define 8
# duration_sec has many large values maybe outlier due to customers forgot to log off after using

# In[41]:


# Plot duration distibution
plt.figure(figsize = [20, 2])
base_color = sb.color_palette()[0]
sb.boxplot(data=df1, x='duration_sec', color=base_color);


# In[42]:


df1.duration_sec.describe(percentiles = [0.01, 0.05, 0.95, 0.99])


# In[43]:


# Remove duration outliers
df1 = df1.query('duration_sec <= 6000')


# #### Test 8
# 
# 

# In[44]:


# Plot duration < 6000 sec distibution
plt.figure(figsize = [20, 2])
base_color = sb.color_palette()[0]
sb.boxplot(data=df1, x='duration_sec', color=base_color);


# #### Define 9
# Set orders of categorical variables: age_group, member_gender, weekday and month

# In[45]:


# Set order of each categorical variable

# age_group
df1['age_group'] = pd.Categorical(df1['age_group'], ['10s', '20s', '30s', '40s', '50s'])
# member_gender
df1['member_gender'] = pd.Categorical(df1['member_gender'], ['Male','Female','Other'])
# weekday
df1['weekday'] = pd.Categorical(df1['weekday'], ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
# month
df1['month'] = pd.Categorical(df1['month'], [6,7,8,9,10,11,12,1,2,3,4,5])


# #### Test 9

# In[46]:


df1.weekday.value_counts().index.categories


# ### Final df1
# 

# In[47]:


df1.info()


# ### What is the structure of your dataset? 
# 
# > The dataset consist of 23 columns and 519699 rows<br>.The data types of the columns are: category(8), datetime64[ns](2), float64(4), int64(7), object(2).
# 
# ### What is/are the main feature(s) of interest in your dataset? 
# 
#   1. I will focus on the features related to the riders charastarastics such as gender , Age  and type.
#   2. location
#   3. Duration
#   4. Month, Weekday, Hour
#   5. Route
# 
# ### What features in the dataset do you think will help support your investigation into your feature(s) of interest?
# 
# > After the cleaning, the features i think it will be helpful are gender, user type, age, member age, age_group, station_area, Month, Weekday, Hour and duaration.

# ## Univariate Exploration

# ### The major gender that uses the service

# In[48]:


# Plotting 
fig, ax = plt.subplots(figsize = (6,6))
#color = sns.color_palette('colorblind')[10]
sb.countplot(x = "member_gender", data = df1, 
              order = df1['member_gender'].value_counts().index,
              palette = sb.color_palette('colorblind'), alpha=0.8)

# Aesthetic wrangling
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), 
                (p.get_x()+p.get_width()/2, p.get_height()), 
                ha='center', va='bottom', color = "black", size=14)
plt.title('Gender Distribution\n', size=20)
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
sb.despine(fig, left = True)
plt.xlabel("");


# >The males represents more than half of the service users, then the female and the others.

# ###  The major user type

# In[49]:


#I am going to use the bar plot since it is suitable for qualtative data

#Chart order and color
base_color = sb.color_palette()[0]
cat_order = df1['user_type'].value_counts().index

#Plot
ax = sb.countplot(x="user_type", data=df1, color = base_color, order = cat_order)



# To change the size of the image and text, and add labeles

sb.set(rc={'figure.figsize': (9,8)})
ax.axes.set_title("User Type Distribution",fontsize=20)
ax.set_xlabel("User Type",fontsize=15)
ax.set_ylabel("Count",fontsize=15)
ax.tick_params(labelsize=15)

# To display the percentage

total=float(len(df1))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(100*height/total),
            ha="center") 


# >More than the half of the service users are subscribers

# ### How the age is distributed?

# In[50]:


# I am going to use the Density graph associated with histogram since they are suitable for quantative data

ax=df1["age"].plot.kde() 
ax=df1["age"].plot.hist(density=True)
ax.axes.set_title("Age Distribution",fontsize=20)
ax.set_xlabel("Ages Range",fontsize=15)
ax.set_ylabel("Density",fontsize=15)

plt.xlim((0,70));


# > It seems that the age is heavely distributed between 25 to 45

# In[51]:


# Plotting 
fig, ax = plt.subplots(figsize = (12,5))
sb.countplot(x = "age_group", data = df1,
              color = sb.color_palette('viridis')[1],
              order = ['10s','20s','30s','40s','50s'], 
              # order by age_group
              alpha=0.8)

# Aesthetic wrangling
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), 
                (p.get_x()+p.get_width()/2, p.get_height()), 
                ha='center', va='bottom', color = "black", size=12)
plt.title('Age Group Distribution\n', size=20)
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
sb.despine(fig, left = True)
plt.xlabel("");


# ### How the durations are distributed?

# In[52]:


# I am going to use the histogram since it is suitable for quantative data

#Bins
bins = 10 ** np.arange(np.log10(df1['duration_sec'].min()), np.log10(df1['duration_sec'].max()) + 0.1, 0.1)

#Plot
ax=plt.hist(data = df1, x = 'duration_sec', bins = bins)
plt.title('Duration Distribution',fontsize=20)
plt.xlabel('Duration in Seconds',fontsize=15);
plt.ylabel('Counts',fontsize=15);

#Scale the graph
plt.xscale('log')
plt.xlim((60,10000))

#Change the ticks
plt.xticks([1e2,3e2,1e3,3e3,1e4], [100,300,'1k', '3k','10k'])
plt.tick_params(labelsize=13)


# > From the graph we can see that the duration rates are heavily distributed between 300 and 1000 seconds.

# ### Month Distribution

# In[54]:


# Plotting 
fig, ax = plt.subplots(figsize = (12,5))
sb.countplot(x = "month", data = df1,
              color = sb.color_palette('viridis')[1], alpha=0.8)

# Aesthetic wrangling
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()+0.5), 
                (p.get_x()+p.get_width()/2, p.get_height()-20000), 
                color="white", size=10, ha='center')
plt.title('Month Distribution\n', size=20)
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
sb.despine(fig, left = True)
plt.xlabel("");


# > From graph above, October have high usage over 92,000 . This probably due to weather reasons. Hot summer season and cold winter season prevent users from riding. 

# ### Weekday Distribution

# In[55]:


# Plotting 
fig, ax = plt.subplots(figsize = (12,5))
sb.countplot(x = "weekday", data = df1,
              color = sb.color_palette('viridis')[1],
              order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], 
              # order by weekday
              alpha=0.8)

# Aesthetic wrangling
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()+0.5), 
                (p.get_x()+p.get_width()/2, p.get_height()-25000), 
                color="white", size=14, ha='center')
plt.title('Weekday Distribution\n', size=20)
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
sb.despine(fig, left = True)
plt.xlabel("");


# > From graph above, weekday usage is higher than weekend usage by almost 2 times.

# ### Hour Distribution

# In[56]:


# Customers could have given the wrong birth year information that cause the super old age data outliers 

# Set bin size equal to 1 yr and color
bin_size = 1
bins = np.arange(0, 24+bin_size, bin_size)
color = sb.color_palette('viridis')[1]

# Plotting 
fig, ax = plt.subplots(figsize = (12,5))
plt.hist(df1.hour, bins = bins, color= color, align="mid", alpha=0.8)

# Aesthetic wrangling
plt.xticks(ticks = [x for x in range(0,25,1)])
plt.title('Hour Distribution\n', size=20)
plt.xlabel('Time of Day (24 Hrs)')
sb.despine(fig)
plt.tight_layout();


# > From graph above, there are two peaks of traffic during the day around 8-9am and 5-6pm. The times correspond to the morning and evening rush hours so I believe people go to and leave from work are using the bike to commute between home, office or Bart(connect to other transportations).

# In[57]:


# Most Popular Starting Stations
df_top_start_stations = df1['start_station_name'].value_counts()[:10]
df_top_start_stations


# In[58]:


# Most Popular Ending Stations
df_top_end_stations = df1['end_station_name'].value_counts()[:10]
df_top_end_stations


# In[59]:


# Plotting the 
fig, ax = plt.subplots(figsize = (12,10))
sb.barplot(df_top_start_stations.index, df_top_start_stations.values, alpha=.8, color = sb.color_palette("viridis")[1])

# Add annotation for each bar
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()),
                (p.get_x()+p.get_width()/2, p.get_height()-4000), 
                color = "white", size=14, ha='center')
# Remove axes
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
sb.despine(fig, left = True)
# Add title
plt.title('Most Popular Starting Stations\n', size=20)
# Rotate x label
plt.xticks(rotation=-85)
plt.tight_layout();


# In[60]:


# Plotting the 
fig, ax = plt.subplots(figsize = (12,10))
sb.barplot(df_top_end_stations.index, df_top_end_stations.values, 
            alpha=0.8, color = sb.color_palette("viridis")[1])

# Add annotation for each bar
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), 
                (p.get_x()+p.get_width()/2, p.get_height()-4000), 
                color = "white", size=14, ha='center')
# Remove axes
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
sb.despine(fig, left = True)
# Add title
plt.title('Most Popular Ending Stations\n', size=20)
# Rotate x label
plt.xticks(rotation=-85)
plt.tight_layout();


# > From graph above, most popular starting stations are in SF area. And the fact that popular GoBike stations are near Bart or Caltrain stations confirms the idea people use bike to commute between public transportation and work/home.

# ###  Results of Univariate Exploration

#  
# > While data cleaning, i dropped the nan values in the gender and the birth year. After that i extracted the age by extracting the birth year from the year of the data set and i noticed unusual age (131) so i decided to keep only the ages that are equal or less than 60. For the duration, i have scaled the graph using the log scale and i also set a limit for the graph to make it more clear.
# 
# >  Gender has NaN values and dropped during data cleaning.
# 
# >  Month, Weekday and Hour are directly get from original data.
# 
# >  Route has no obvious outlier because I removed all locations that too far from Bay Area.
# 
# >  The major users whom use the service are males. Also, most of the users are subscribers.
# 
# >the most rides takes within 5-10 minutes (300-600 secs). Durations dramaticlly higher are outliers probably due to user forgot to log out after finish their ride.
# 
# > weekday usage is higher than weekend usage by almost 2 times.
# 
# 
# > It seems that the age is heavely distributed between 25 to 45
# 
# > the duration rates are heavily distributed between 300 and 1000 seconds.
# 
# >  October have high usage over 92,000 . This probably due to weather reasons. Hot summer season and cold winter season prevent users from riding.
# 
# >  there are two peaks of traffic during the day around 8-9am and 5-6pm. The times correspond to the morning and evening rush hours so I believe people go to and leave from work are using the bike to commute between home, office or Bart(connect to other transportations).
# 
# >  most popular starting stations are in SF area. And the fact that popular GoBike stations are near Bart or Caltrain stations confirms the idea people use bike to commute between public transportation and work/home.
# !
# 

# ## Bivariate Exploration

# ### Duaration vs Gender

# In[64]:


# For quantative and qualtative data the violin and box plot are the most suitable, but i am going to use the bar plot since it is easier in the summary reading

#Plot
sb.barplot(data = df1, x = 'member_gender', y = 'duration_sec', color = base_color, errwidth=0)


# Lables
plt.title('Gender VS Duration',fontsize=20)
plt.xlabel('Gender',fontsize=15)
plt.ylabel('Avg. Duaration in Seconds',fontsize=15)
plt.tick_params(labelsize=13)


# > Even though the male represent the major users, but females and others have a higher duration average than the males.

# ### Duration vs User Type

# In[65]:


# For quantative and qualtative data the violin and box plot are the most suitable, but i am going to use the bar plot since it easier to read the summary

#Plot
sb.barplot(data = df1, x = 'user_type', y = 'duration_sec', color = base_color, errwidth=0)

#lables
plt.title('User Type VS Duration',fontsize=20)
plt.xlabel('User Type',fontsize=15)
plt.ylabel('Avg. Duaration in Seconds',fontsize=15)
plt.tick_params(labelsize=13)


# > From the graph above, we can see that the customers has higher duration average than the subscribers, even though the subscribers represent most of our users.

# ###  User Type Vs Gender Count

# In[66]:


# i am going to use the bar plot since it easier to read the summary

# plot
ax=sb.countplot(data = df1, x = 'member_gender', hue = 'user_type')

# Labels
ax.axes.set_title("User Type & Gender Distribution",fontsize=20)
ax.set_xlabel("Gender",fontsize=15)
ax.set_ylabel("Count",fontsize=15)
ax.tick_params(labelsize=15)

#Percentage
total=float(len(df1))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(100*height/total),
            ha="center") 


# > The subscribers number is always higher than the customers number for all different genders, and the male subscribers represnt more than half of the service users.

# ### Duration vs Age

# In[67]:


#Since we have a large amount of data, ill wil be using the heat map since it has a higher transparancy than the scatter plot

# prepare the bins
bins_x = 10 ** np.arange(np.log10(df1['age'].min()), np.log10(df1['age'].max())+0.1, 0.1)
bins_y = 10 ** np.arange(np.log10(df1['duration_sec'].min()), np.log10(df1['duration_sec'].max())+0.1, 0.1)

# plot
plt.hist2d(data = df1, x = 'age', y = 'duration_sec',
           bins = [bins_x, bins_y], cmin=0.5)
plt.colorbar()


# plot information
plt.title('Age vs Duration',fontsize=20)
plt.xlabel('Age',fontsize=15)
plt.ylabel('Duration in Seconds',fontsize=15)



#Scale and adjust the chart
plt.yscale('log')
plt.xlim((20,60))
plt.ylim((60,10000))

#Change the ticks
plt.yticks([1e2,3e2,1e3,3e3,1e4], [100, 300,'1k','3k', '10k'])


# > From the graph above, we can see the majority of users are between 30 and 35 and their duration is between 500 and 600

# ### User_type VS. Age_group

# In[68]:


fig, ax = plt.subplots(figsize = (14,5))

# Plotting
sb.countplot(x = "age_group", data = df1, 
              palette = "viridis", hue = "user_type", alpha = 0.8)

# Percentage for each age-group
perc_list_customer, perc_list_subscriber, perc_list = [], [], []

# Calculate % for 2 user types for each age_group
type_sum = df1.groupby('age_group')['user_type'].value_counts().sort_index().to_list()
total_sum = df1['age_group'].value_counts().sort_index().to_list()

# arrange the % list in same as annotate loop order
for i in range(0,len(total_sum)):
    perc_customer = int(round(100 * type_sum[2*i] / total_sum[i]))
    perc_list_customer.append(perc_customer)
for i in range(0,len(total_sum)):
    perc_subscriber = int(round(100 * type_sum[2*i+1]/ total_sum[i]))
    perc_list_subscriber.append(perc_subscriber)
perc_list = perc_list_customer + perc_list_subscriber
   
# Annotate each bar
i=0
for p in ax.patches:
    ax.annotate('{:.0f}\n{:.0f}%'.format(p.get_height(), perc_list[i]), 
                (p.get_x()+p.get_width()/2, p.get_height()), 
                ha="center", va="bottom", size=12)
    i+=1

# Aesthetic wrangling
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
sb.despine(fig, left = True)
plt.title("Users Type by Age Group\n\n", fontsize= 20)
plt.xlabel("");


# > From graph above, user's subscription ratio increases as the age increase. Younger user tend to use the service but do not subscribe.
# 

# ### User_type vs. Hour

# In[69]:


fig, ax = plt.subplots(figsize = (14,5))

# Plotting
sb.countplot(x = "hour", data = df1, 
              palette = "viridis", hue = "user_type", alpha = 0.8)

# Percentage for each hour
perc_list_customer, perc_list_subscriber, perc_list = [], [], []

# Calculate % for 2 user types for each hour
type_sum = df1.groupby('hour')['user_type'].value_counts().sort_index().to_list()
total_sum = df1['hour'].value_counts().sort_index().to_list()

# arrange the % list in same as annotate loop order
for i in range(0,len(total_sum)):
    perc_customer = int(round(100 * type_sum[2*i] / total_sum[i]))
    perc_list_customer.append(perc_customer)
for i in range(0,len(total_sum)):
    perc_subscriber = int(round(100 * type_sum[2*i+1]/ total_sum[i]))
    perc_list_subscriber.append(perc_subscriber)
perc_list = perc_list_customer + perc_list_subscriber

# Annotate each bar
i=0
for p in ax.patches:
    ax.annotate('{:.0f}\n{:.0f}%'.format(p.get_height(), perc_list[i]), 
                (p.get_x()+p.get_width()/2, p.get_height()), 
                ha="center", va="bottom", size=8)
    i+=1

# Aesthetic wrangling
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
sb.despine(fig, left = True)
plt.title("Users Type by Hour\n\n", fontsize= 20)
plt.xlabel("")
plt.tight_layout();


# > From graph above, user population in the morning and evening rush hours have the highest subscription ratio. This makes sense because user rides bike as regular commute to work or home tend to subscribe for the service.

# ### User_type vs. Weekday

# In[70]:


fig, ax = plt.subplots(figsize = (14,5))
# Plotting
sb.countplot(x = "weekday", data = df1, 
              palette = "viridis", hue = "user_type", alpha = 0.8)

# Percentage for each day
perc_list_customer, perc_list_subscriber, perc_list = [], [], []

# Calculate % for 2 user types for each day
type_sum = df1.groupby('weekday')['user_type'].value_counts().sort_index().to_list()
total_sum = df1['weekday'].value_counts().sort_index().to_list()

# arrange the % list in same as annotate loop order
for i in range(0,len(total_sum)):
    perc_customer = int(round(100 * type_sum[2*i] / total_sum[i]))
    perc_list_customer.append(perc_customer)
for i in range(0,len(total_sum)):
    perc_subscriber = int(round(100 * type_sum[2*i+1]/ total_sum[i]))
    perc_list_subscriber.append(perc_subscriber)
perc_list = perc_list_customer + perc_list_subscriber

# Annotate each bar
i=0
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), 
                (p.get_x()+p.get_width()/2, p.get_height()), 
                ha="center", va="bottom", size=12)
    ax.annotate('{:.0f}%'.format(perc_list[i]), 
                (p.get_x()+p.get_width()/2, p.get_height()), 
                ha="center", va="top", color='white', size=12)
    i+=1

# Aesthetic wrangling
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
sb.despine(fig, left = True)
plt.title("Users Type by Weekday\n\n", fontsize= 20)
plt.xlabel("")
plt.tight_layout();


# > From graph above, Saturday and Sunday user populations have unusual low subscription ratio and total rides. This is probably due to people use the bike for work mostly.

# ### the relationships that i observed in this part of the investigation. 
# 
# > The females and others have a higher duration average than the males<br>
# The customers have a higher duration rate than the subscribers<br>
# The male subscribers represnt more than half of the service users.<br>
# duration and age has no statistical correlation.<br>
# user_type: registered users have longer duration than casual users per trip. registered users are younger than casual users.<br>
# member_gender: all genders have similar duration. Female user are younger than other genders.<br>
# weekday: weekends' users are younger than weekday users and weekend's trips have longer duration.<br>
# 
# ###  interesting relationships between the other features (not the main feature(s) of interest)?
# 
# > I observed that younger people (25- 45) tend to have a higher duration rate than the older people.<br>
# From user_type vs. weekday graph, Wednesday and Thursday user populations have unusual low subscription ratio but the total rides counts not significantly different from weekday traffic.

# ## Multivariate Exploration

# ### 1-User Type vs Duration Vs Age

# In[71]:


g= sb.FacetGrid(data = df1, col = 'user_type', height=5.2,aspect=1,margin_titles = True)
g.map(plt.scatter, 'age', 'duration_sec',alpha=0.30)

#Title
plt.subplots_adjust(top=0.87)
g.fig.suptitle('User Type Vs Duration Vs Age')

# Axes
g.set_xlabels('Age')
g.set_ylabels('Duration in Seconds')


# >from the graph, we noticed that the Subscribers have a higher duration rates than the customers at different ages.

# ### 2- Gender vs Duration vs  Age

# In[72]:


g= sb.FacetGrid(data = df1, col = 'member_gender', height=5,margin_titles = True)
g.map(plt.scatter, 'age', 'duration_sec',alpha=1/3)

#Title
plt.subplots_adjust(top=0.87)
g.fig.suptitle('Gender Vs Duration Vs Age')

# Axes
g.set_xlabels('Age')
g.set_ylabels('Duration in Seconds')


# >Males have higher duration rates, then the females, and at last the others.

# ### 3-The Average duration spent by differnt user types and genders

# In[73]:


ax=sb.barplot(data = df1, x = 'member_gender', y='duration_sec', hue = 'user_type', errwidth=0)

ax.axes.set_title("Duration vs Gender vs User Type",fontsize=20)
ax.set_xlabel("Gender",fontsize=15)
ax.set_ylabel("Avg. Duration in Seconds",fontsize=15)
ax.tick_params(labelsize=15)


# > Though the number of customers is less than the subscribers, as well as well as the number of females and the others is less than the males. The females customers have the highest duration average.

# ### user_type for age_group by station_area

# In[75]:


# Since there's only three subplots to create, using the full data should be fine.
fig = plt.figure(figsize = [16, 12])

# Subplot 1: San Francisco, ride count for age_group by user_type
##############################################################
ax1 = plt.subplot(3, 1, 1)
ax1.set_title("San Francisco\n", size=18)
sb.countplot(data = df1.query('start_station_area == "San Francisco"'), 
              x = 'age_group', hue = 'user_type', palette = 'Set2', alpha=0.8)
plt.xlabel('')
plt.ylabel('')
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
ax1.legend(loc = 1, ncol = 2)

# Percentage for each age_group
perc_list_customer, perc_list_subscriber, perc_list = [], [], []

# Calculate % for 2 user types for each age_group
type_sum = df1.query('start_station_area == "San Francisco"').groupby('age_group')['user_type'].value_counts().sort_index().to_list()
total_sum = df1.query('start_station_area == "San Francisco"')['age_group'].value_counts().sort_index().to_list()

# Arrange the % list in same as annotate loop order
for i in range(0,len(total_sum)):
    perc_customer = int(round(100 * type_sum[2*i] / total_sum[i]))
    perc_list_customer.append(perc_customer)
for i in range(0,len(total_sum)):
    perc_subscriber = int(round(100 * type_sum[2*i+1]/ total_sum[i]))
    perc_list_subscriber.append(perc_subscriber)
perc_list = perc_list_customer + perc_list_subscriber

# Add annotations
i=0
for p in ax1.patches:
    ax1.annotate('{:.0f}\n{:.0f}%'.format(p.get_height(), perc_list[i]), 
                (p.get_x()+p.get_width()/2, p.get_height()), 
                ha="center", va="bottom", size=12)
    i+=1

# Subplot 2: East Bay, ride count for age_group by user_type
##############################################################
ax2 = plt.subplot(3, 1, 2)
ax2.set_title("East Bay", size=18)
sb.countplot(data = df1.query('start_station_area == "East Bay"'), 
              x = 'age_group', hue = 'user_type', palette = "Set2", alpha=0.8)
plt.xlabel('')
plt.ylabel('')
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
ax2.legend(loc = 1, ncol = 2)

# Percentage for each age_group
perc_list_customer, perc_list_subscriber, perc_list = [], [], []

# Calculate % for 2 user types for each age_group
type_sum = df1.query('start_station_area == "East Bay"').groupby('age_group')['user_type'].value_counts().sort_index().to_list()
total_sum = df1.query('start_station_area == "East Bay"')['age_group'].value_counts().sort_index().to_list()

# Arrange the % list in same as annotate loop order
for i in range(0,len(total_sum)):
    perc_customer = int(round(100 * type_sum[2*i] / total_sum[i]))
    perc_list_customer.append(perc_customer)
for i in range(0,len(total_sum)):
    perc_subscriber = int(round(100 * type_sum[2*i+1]/ total_sum[i]))
    perc_list_subscriber.append(perc_subscriber)
perc_list = perc_list_customer + perc_list_subscriber

# Add annotations
i=0
for p in ax2.patches:
    ax2.annotate('{:.0f}\n{:.0f}%'.format(p.get_height(), perc_list[i]), 
                (p.get_x()+p.get_width()/2, p.get_height()), 
                ha="center", va="bottom", size=12)
    i+=1

# Subplot 3: East Bay, ride count for age_group by user_type
##############################################################
ax3 = plt.subplot(3, 1, 3)
ax3.set_title("San Jose", size=18)
sb.countplot(data = df1.query('start_station_area == "San Jose"'), 
              x = 'age_group', hue = 'user_type', palette = 'Set2', alpha=0.8)
plt.xlabel('')
plt.ylabel('')
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
ax3.legend(loc = 1, ncol = 2)
# Percentage for each age_group
perc_list_customer, perc_list_subscriber, perc_list = [], [], []

# Calculate % for 2 user types for each age_group
type_sum = df1.query('start_station_area == "San Jose"').groupby('age_group')['user_type'].value_counts().sort_index().to_list()
total_sum = df1.query('start_station_area == "San Jose"')['age_group'].value_counts().sort_index().to_list()

# Arrange the % list in same as annotate loop order
for i in range(0,len(total_sum)):
    perc_customer = int(round(100 * type_sum[2*i] / total_sum[i]))
    perc_list_customer.append(perc_customer)
for i in range(0,len(total_sum)):
    perc_subscriber = int(round(100 * type_sum[2*i+1]/ total_sum[i]))
    perc_list_subscriber.append(perc_subscriber)
perc_list = perc_list_customer + perc_list_subscriber

# Add annotations
i=0
for p in ax3.patches:
    ax3.annotate('{:.0f}\n{:.0f}%'.format(p.get_height(), perc_list[i]), 
                (p.get_x()+p.get_width()/2, p.get_height()), 
                ha="center", va="bottom", size=12)
    i+=1

# Aesthetic Wrangling
fig.suptitle('User Type for Age Group by station Area', size=20, y=1.02)
sb.despine(fig, left = True)
plt.tight_layout();


# > From graph above, majority of users is from age groups 20s and 30s. Although San Francisco station area has most rides but the two largest user populations 20s and 30s age groups have the lowest subscription ratio.

# ### user_type for weekday by station_area

# In[76]:


# Since there's only three subplots to create, using the full data should be fine.
fig = plt.figure(figsize = [16, 12])

# Subplot 1: San Francisco, ride count for weekday by user_type
##############################################################
ax1 = plt.subplot(3, 1, 1)
ax1.set_title("San Francisco\n", size=18)
sb.countplot(data = df1.query('start_station_area == "San Francisco"'), 
              x = 'weekday', hue = 'user_type', palette = 'Set2', alpha=0.8) # data start 201806 to 201905
plt.xlabel('')
plt.ylabel('')
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
ax1.legend(ncol = 2)

# Percentage for each day
perc_list_customer, perc_list_subscriber, perc_list = [], [], []

# Calculate % for 2 user types for each day
type_sum = df1.query('start_station_area == "San Francisco"').groupby('weekday')['user_type'].value_counts().sort_index().to_list()
total_sum = df1.query('start_station_area == "San Francisco"')['weekday'].value_counts().sort_index().to_list()

# Arrange the % list in same as annotate loop order
for i in range(0,len(total_sum)):
    perc_customer = int(round(100 * type_sum[2*i] / total_sum[i]))
    perc_list_customer.append(perc_customer)
for i in range(0,len(total_sum)):
    perc_subscriber = int(round(100 * type_sum[2*i+1]/ total_sum[i]))
    perc_list_subscriber.append(perc_subscriber)
perc_list = perc_list_customer + perc_list_subscriber

# Add annotations
i=0
for p in ax1.patches:
    ax1.annotate('{:.0f}\n{:.0f}%'.format(p.get_height(), perc_list[i]), 
                (p.get_x()+p.get_width()/2, p.get_height()), 
                ha="center", va="bottom", size=10)
    i+=1

# Subplot 2: East Bay, ride count for weekday by user_type
##############################################################
ax2 = plt.subplot(3, 1, 2)
ax2.set_title("East Bay\n", size=18)
sb.countplot(data = df1.query('start_station_area == "East Bay"'), 
              x = 'weekday', hue = 'user_type', palette = 'Set2', alpha=0.8)
plt.xlabel('')
plt.ylabel('')
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
ax2.legend(ncol = 2)

# Percentage for each day
perc_list_customer, perc_list_subscriber, perc_list = [], [], []

# Calculate % for 2 user types for each day
type_sum = df1.query('start_station_area == "East Bay"').groupby('weekday')['user_type'].value_counts().sort_index().to_list()
total_sum = df1.query('start_station_area == "East Bay"')['weekday'].value_counts().sort_index().to_list()

# Arrange the % list in same as annotate loop order
for i in range(0,len(total_sum)):
    perc_customer = int(round(100 * type_sum[2*i] / total_sum[i]))
    perc_list_customer.append(perc_customer)
for i in range(0,len(total_sum)):
    perc_subscriber = int(round(100 * type_sum[2*i+1]/ total_sum[i]))
    perc_list_subscriber.append(perc_subscriber)
perc_list = perc_list_customer + perc_list_subscriber

# Add annotations
i=0
for p in ax2.patches:
    ax2.annotate('{:.0f}\n{:.0f}%'.format(p.get_height(), perc_list[i]), 
                (p.get_x()+p.get_width()/2, p.get_height()), 
                ha="center", va="bottom", size=10)
    i+=1

# Subplot 3: East Bay, ride count for weekday by user_type
##############################################################
ax3 = plt.subplot(3, 1, 3)
ax3.set_title("San Jose\n", size=18)
sb.countplot(data = df1.query('start_station_area == "San Jose"'), 
              x = 'weekday', hue = 'user_type', palette = 'Set2', alpha=0.8)
plt.xlabel('')
plt.ylabel('')
cur_axes = plt.gca()
cur_axes.axes.get_yaxis().set_visible(False)
ax3.legend(ncol = 2)
# Percentage for each day
perc_list_customer, perc_list_subscriber, perc_list = [], [], []

# Calculate % for 2 user types for each day
type_sum = df1.query('start_station_area == "San Jose"').groupby('weekday')['user_type'].value_counts().sort_index().to_list()
total_sum = df1.query('start_station_area == "San Jose"')['weekday'].value_counts().sort_index().to_list()

# Arrange the % list in same as annotate loop order
for i in range(0,len(total_sum)):
    perc_customer = int(round(100 * type_sum[2*i] / total_sum[i]))
    perc_list_customer.append(perc_customer)
for i in range(0,len(total_sum)):
    perc_subscriber = int(round(100 * type_sum[2*i+1]/ total_sum[i]))
    perc_list_subscriber.append(perc_subscriber)
perc_list = perc_list_customer + perc_list_subscriber

# Add annotations
i=0
for p in ax3.patches:
    ax3.annotate('{:.0f}\n{:.0f}%'.format(p.get_height(), perc_list[i]), 
                (p.get_x()+p.get_width()/2, p.get_height()), 
                ha="center", va="bottom", size=10)
    i+=1

# Aesthetic Wrangling
fig.suptitle('User Type for Weekday by station Area', size=20, y=1.02)
sb.despine(fig, left = True)
plt.tight_layout();


# > From graph above, Subscribers ride on weekday more often than on weekend. Customers ride on weekend more often than on weekday. This indicate most subscribers use the bike as commute and most of the customers use the bike for leisure.

# ### user_type for hour and weekday

# In[77]:


# Prepare df for subscriber
df_subscriber = df1.query('user_type == "Subscriber"').groupby(['hour','weekday']).agg({'bike_id' : 'count'})
df_subscriber = df_subscriber.pivot_table(index='hour', columns='weekday', values='bike_id')
# Prepare df for customer
df_customer = df1.query('user_type == "Customer"').groupby(['hour','weekday']).agg({'bike_id' : 'count'})
df_customer = df_customer.pivot_table(index='hour', columns='weekday', values='bike_id')


# In[78]:


plt.subplots(figsize=(20,10))

# df_subscriber
fig1 = plt.subplot(1,2,1)
ax1 = sb.heatmap(df_subscriber, annot=True, fmt='d', cmap='Blues')
# Aesthetic Wrangling
plt.title('Subscriber',size=16)
plt.yticks(rotation=360)

# df_customer
fig2 = plt.subplot(1,2,2)
ax2 = sb.heatmap(df_customer, annot=True, fmt='d', cmap='Blues')
# Aesthetic Wrangling
plt.title('Customer',size=16)
plt.yticks(rotation=360)

plt.suptitle("Most ", size=20, y=1.05)
plt.tight_layout();


# >From graph above, Subscriber's most frequently used time is weekday around 7-9am and 4-6pm, which are the commute times. Customer's most frequently used time, beside the commute times, is weekend 12pm-4pm.
# I think Subscribers use the bike mostly for commute to work. Customers use the bike during weekend for leisure.

# ### Results
# 
# >-Males Have higher duration rates, but on average the females and the others has a higher duration.<br> 
# -The females customers has the highest duration average <br>
# -The customers have a higher duration average than the subscribers at different ages.<br>
# -Younger people has higher duration rates compared to older people.
# 
# > #### user_type for age_group by station_area<br>
# Majority of users is from age groups 20s and 30s. Although San Francisco station area has most rides but the two largest user populations 20s and 30s age groups have the lowest subscription ratio.
# 
# > ####user_type for weekday by station_area<br>
# Subscribers ride on weekday more often than on weekend. Customers ride on weekend more often than on weekday. This indicate most subscribers use the bike as commute and most of the customers use the bike for leisure.
# 
# > #### user_type for hour and weekday
# Subscriber's most frequently used time is weekday around 7-9am and 4-6pm, which are the commute times. Customer's most frequently used time, beside the commute times, is weekend 12pm-4pm.<br>
# I think Subscribers use the bike mostly for commute to work. Customers use the bike during weekend for leisure.
# 
# ### Were there any interesting or surprising interactions between features?
# 
# > 
# - Females customers having the highest duration average.

# # Implementing Logestic Regrission 

# #### According to the data set that I have and because I am interested in making a model that predicts the type of customer based on the data: gender, time spent and others
# 
# ### So I will use Logestic Regrission model
# 
# 

# In[219]:


dfp=df1.copy()


# In[220]:


dfp.info()


# In[221]:


dfp.head()


# In[230]:


dummy1 = pd.get_dummies(dfp, columns=['start_station_name','end_station_name','user_type','member_gender', 'start_station_area', 
                            'end_station_area', 'month','weekday','hour' ],drop_first=True)


# In[231]:


dfp = pd.concat([dfp, dummy1], axis=1)


# In[232]:


dfp.head()


# In[238]:


dfp = dfp.drop(['start_station_name','end_station_name','user_type','member_gender', 'start_station_area', 
                            'end_station_area', 'month','weekday','hour'], 1)


# In[241]:


dfp.info()


# In[239]:



X = dfp2.loc[:,'member_birth_year':'age_group']
y = dfp2['user_type']


# In[235]:


X.head()


# In[236]:


y.head()


# In[237]:


logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)


# In[ ]:




