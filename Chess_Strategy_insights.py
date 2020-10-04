#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


games=pd.read_csv(r"C:\Users\singa tharun reddy\Desktop\Github projects\Chess opening recommendation system\games.csv")


# In[3]:


df=games


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# 
# ### <font color=darkblue>**We are going to explore data of 20,000 games and generate strategies for winning a game.**</font>

# In[7]:


#As opening palys a important role in winning,I am ignoring data of less frequent openings. 
s=df["opening_name"].value_counts().gt(20)
df=df.loc[df["opening_name"].isin(s[s].index)]


# In[8]:


len(df.opening_name.value_counts())#We have 220 different openings.


# In[9]:


#I am considering rated players data only.
rated=df[(df["rated"]==True)]


# In[10]:


#we are dropping non essential columns.
rated.drop(["id","rated","created_at", "last_move_at","white_id","black_id","moves"],axis=1,inplace=True)


# In[11]:


#cross checking the data.
rated.head()


# In[12]:


open_win = rated.groupby('opening_name').winner.value_counts()
open_win = open_win.reset_index(name='wins')
open_win = open_win.sort_values(by='wins', ascending=False)


# In[13]:


white_win=open_win[open_win.winner=="white"]
black_win=open_win[open_win.winner=="black"]
draw_win=open_win[open_win.winner=="draw"]


# 
# ### <font color=darkblue>**1.Top 10 openings for the white to win?**</font>

# In[14]:


## Creating a  Bar Plot  ##
plt.figure(figsize=(10,8),dpi=100)
plt.title('Top 10 openings for the white to win.')
white_winning_plot = sns.barplot(y=white_win.opening_name[:10],x=white_win.wins[:10], palette = "magma",edgecolor='black')
plt.ylabel("Opening_name", fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Number of Games won by white", fontsize=10)
plt.title("Top 10 openings for the white to win",fontsize=15)


# #### <font color=darkgreen>**Observation:**</font> <font color=black>**Scandinavian Defense: Mieses-Kotroc Variation Opening</font> <font color=black>**has highest probability of winning for white.</font>

# ### <font color=darkblue>**2.Top 10 openings for the Black to win?**</font>

# In[15]:


## Creating a  Bar Plot  ##
black_win = black_win.sort_values(by='wins', ascending=False)
plt.figure(figsize=(10,8),dpi=100)
plt.title('Least 10 openings for the black to win.')
white_winning_plot = sns.barplot(y=black_win.opening_name[:10],x=black_win.wins[:10], palette = "viridis",edgecolor='black')
plt.ylabel("Opening_name", fontsize=15)
plt.yticks(fontsize=10)
plt.xlabel("Number of Games won by black", fontsize=15)
plt.title("Top 10 openings for the black to win",fontsize=15)


# #### <font color=darkgreen>**Observation:**</font> <font color=black>**Van't Kruijs Opening has highest probability of winning for black.**</font>

# 
# ### <font color=darkblue>**3. Top 10 openings for a draw?**</font>

# In[16]:


## Creating a  Bar Plot  ##
plt.figure(figsize=(10,8))
plt.title('Top 10 openings for a draw.')
white_winning_plot = sns.barplot(y=draw_win.opening_name[:10],x=draw_win.wins[:10], palette = "rocket_r",edgecolor='black')
plt.ylabel("Opening_name", fontsize=15)
plt.yticks(fontsize=10)
plt.xlabel("Number of draw games", fontsize=15)
plt.title("Top 10 openings for a draw",fontsize=15)


# #### <font color=darkgreen>**Observation:**</font> <font color=black>**Van't Kruijs Opening has highest probaility to draw. This opening gives a advantage for black as well as for a draw also.**</font>

# 
# ### <font color=darkblue>**4.Type of winning for white?**</font>

# In[17]:


white=rated[rated["winner"]=="white"]
winning_type=white.victory_status.value_counts()
labels= winning_type.index  # x ticks
sizes= winning_type.values
## Create Pie chart Plot ##
plt.figure(figsize = (7,7))
explode = [0,0,0.1]
colors = ['#EC6B56','#FFC154','#47B39C']
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True,textprops={'fontsize': 14}, startangle=90, colors=colors)
plt.title('Type of winning for white',color = 'Black',fontsize = 20)


# ### <font color=darkgreen>**Observation:**</font> <font color=black>**White is likely to win the game by opponent resiging the game.**</font>

# ### <font color=darkblue>**5.Type of winning for Black?**</font>

# In[18]:


black=rated[rated["winner"]=="black"]
winning_type=black.victory_status.value_counts()
labels= winning_type.index  # x ticks
sizes= winning_type.values

## Create Pie chart Plot ##
plt.figure(figsize = (7,7))
explode = [0,0,0.1]
colors = ['#2D87BB','#64C2A6','#AADEA7']
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True,textprops={'fontsize': 14},pctdistance=0.85,startangle=90, colors=colors)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Type of winning for Black',color = 'Black',fontsize = 20)


# #### <font color=darkgreen>**Observation:**</font> <font color=black>**Black is likely to win the game by oppenent resiging the game.**</font>

# ### <font color=darkblue>**6.How are number of turns related to ratings difference of players?**</font>

# In[19]:


rated["ratings_difference"]=rated["white_rating"]-rated["black_rating"]
plt.figure(figsize=(16, 16))
sns.lmplot( y="turns", x="ratings_difference", data=rated, fit_reg=False, hue='winner',legend=False,markers=["o", "x", "s"],)
# Move the legend to an empty part of the plot
plt.legend(loc='upper right')
plt.show()


# #### <font color=green>**Observation:**</font><font color=black>**Less difference between ratings leads to draw and more number of steps.**</font>

# ### <font color=darkblue>**7.How is difference of ratings related to victory type?**</font>
# 

# In[20]:


plt.figure(figsize=(16, 16))
sns.lmplot( x="turns", y="ratings_difference", data=rated, fit_reg=False, hue='victory_status',legend=False,markers=["o", "x", "s","v"])
# Move the legend to an empty part of the plot
plt.legend(loc='upper right')
plt.show()


# #### <font color=green>**Observation:**</font><font color=black>**Less difference between ratings has led to winning the game by out of time.**</font>

# ### <font color=darkblue>**8.Top 10 openings to play against top rated players for black to win?**</font>

# In[21]:


plt.figure(figsize=(15,10))
rated["ratings_difference"]=rated["ratings_difference"].abs()
df = rated.loc[rated['ratings_difference']>250]
blackWins = df.loc[df['winner']=='black']
mostWins = blackWins['opening_name'].value_counts().nlargest(10)
sns.barplot(y=mostWins.index, x=mostWins.values, palette = "viridis",edgecolor='black')
plt.ylabel("Opening_name", fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Number of Games won by black", fontsize=15)
plt.title("Top 10 openings for the black to win against top rated players",fontsize=15)


# #### <font color=green>**Observation:**</font><font color=black>**For Black, Van't Kruijs Openings is preferrable to beat top rated players.**</font>

# ### <font color=darkblue>**9.Top 10 openings to play against top rated players for white to win?**</font>

# In[22]:


plt.figure(figsize=(15,10))
whiteWins = df.loc[df['winner']=='white']
mostWins = whiteWins['opening_name'].value_counts().nlargest(10)
sns.barplot(y=mostWins.index, x=mostWins.values, palette = "magma",edgecolor='black')
plt.xticks(rotation=90, fontsize=16)
plt.ylabel("Opening_name", fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Number of Games won by white", fontsize=15)
plt.title("Top 10 openings for the white to win against top rated players",fontsize=15)


# #### <font color=green>**Observation:**</font><font color=black>**For White, Scandinavian Defense:Mieses-Kotroc Variation Openings is preferrable to beat top rated players.**</font>

# ### <font color=darkblue>**10.Top 10 openings for white to win game, if white rating is less than black rating?**</font>

# In[23]:


white_win=rated[(rated.winner=="white")&(rated.white_rating<rated.black_rating)]
whitewins=white_win.opening_name.value_counts().nlargest(10)
plt.figure(figsize=(15,10))
sns.barplot(y=whitewins.index, x=whitewins.values, palette = "magma",edgecolor='black')
plt.xticks(fontsize=16)
plt.ylabel("Opening_name", fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Number of Games won by white", fontsize=15)
plt.title("Top 10 openings for the white to win against top rated players",fontsize=15)


# #### <font color=green>**Observation:**</font><font color=black>**For White,Sicilian Defense Opening is preferrable to beat high rated opponents.**</font>

# ### <font color=darkblue>**11.Top 10 openings for black to win game, if black rating is less than white rating?**</font>

# In[24]:


black_win=rated[(rated.winner=="black")&(rated.black_rating<rated.white_rating)]
blackwins=black_win.opening_name.value_counts().nlargest(10)
plt.figure(figsize=(15,10))
sns.barplot(y=blackwins.index, x=blackwins.values, palette = "viridis",edgecolor='black')
plt.xticks(fontsize=16)
plt.ylabel("Opening_name", fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Number of Games won by Black", fontsize=15)
plt.title("Top 10 openings for the Black to win, if black rating is less than white rating",fontsize=15)


# #### <font color=green>**Observation:**</font><font color=black>**For Black,Scotch game Opening is preferrable to beat high rated opponents.**</font>

# In[ ]:




