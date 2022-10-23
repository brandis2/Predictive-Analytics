#!/usr/bin/env python
# coding: utf-8

# # Topic Modeling Exercise
# Presented by Brandi Smith  
# Predictive Analytics Seminar  
# 10/24/2022
# ### About Dataset
# r/VaccineMyths in Reddit: https://www.kaggle.com/datasets/gpreda/reddit-vaccine-myths?resource=download
# ### Import Data using pandas

# In[ ]:


import pandas as pd
reddit=pd.read_csv("reddit_vm.csv")


# In[ ]:


reddit


# In[ ]:


reddit.info()


# ### Clean data
# 1. Create a new column where if `title != Comment`, return the title and if ` title == Comment` return the body
# 2. Lower case the text in the new column
# 3. remove stop words - small dataset
# 4. Remove punctuation

# In[ ]:


#CREATE NEW COLUMN 
def new_column(col):   
    if col.title != "Comment":
        return col.title
    else:
        return col.body

reddit["text"] = reddit.apply(new_column, axis=1)


# In[ ]:


#Lower case text
reddit["text"] = reddit["text"].str.lower()


# In[ ]:


#remove stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')
reddit['text_without_stopwords'] = reddit['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[ ]:


#check that stop words are removed
reddit['text']=reddit['text_without_stopwords']


# In[ ]:


reddit


# In[ ]:


#remove punctuation
reddit["text"] = reddit["text"].str.replace(r'[^\w\s]+', '')


# ### Data Exploration

# In[ ]:


#number of posts by year
reddit['n_posts']=1
reddit['timestamp']= pd.to_datetime(reddit['timestamp'])
reddit['year'] = reddit['timestamp'].dt.strftime('%Y')
#reddit['month_year'] = reddit['timestamp'].dt.strftime('%B-%Y')

year_val= reddit[["year", "n_posts"]]
plot_dat=year_val.groupby('year').sum()
plot_dat


# #### Implications in peak number of posts
# **2014 : Ebola virus enters U.S.: https://en.wikipedia.org/wiki/Ebola_virus_cases_in_the_United_States**  
# **2019 : Onset of COVID-19 Pandemic** 
# 
# #### Follow-up questions? 
# 1. What topics were people talking about around vaccine myths during either ebola endemic or Covid-19 pandemic?  
# 2. Do the topics differ in 2014 versus 2019 and 2020? 

# ### BERTopic Implementation
# Determine topics for 2014 (Ebola) and 2019-2020 (Covid)  
# 
# *Note: Install HBDSCAN before BERTopic  
# `conda install -c conda-forge hdbscan` then  
# `pip install BERTopic`*
# 
# #### Ebola topic modeling 

# In[ ]:


from bertopic import BERTopic

ebola= reddit[(reddit['year'].str.contains("2014"))]
ebola_list=ebola.text.to_list()
#ebola_list


# In[ ]:


#dimension reduction method
from umap import UMAP
umap_model = UMAP(random_state=900)


# In[ ]:


#use HBDScan as clustering method
from hdbscan import HDBSCAN
cluster_method=HDBSCAN(min_cluster_size=10)


# In[ ]:


topic_model = BERTopic(hdbscan_model=cluster_method, umap_model=umap_model)
topics, probs = topic_model.fit_transform(ebola_list)


# In[ ]:


topic_model.get_topic_info()


# In[ ]:


topic_model.get_topic(0)


# In[ ]:


topic_model.get_topic(1)


# In[ ]:


topic_model.get_topic(2)


# In[ ]:


#observe which posts topics were derived from
topic_model.get_representative_docs()


# In[ ]:


#visualize the topics
topic_model.visualize_hierarchy()


# #### Covid 19 topic modeling

# In[ ]:


covid= reddit[(reddit['year'].str.contains("2019")) | (reddit['year'].str.contains("2020"))]
covid=covid.text.to_list()
#covid


# In[ ]:


#dimension reduction method
from umap import UMAP
umap_model = UMAP(random_state=81)


# In[ ]:


from hdbscan import HDBSCAN
cluster_method=HDBSCAN(min_cluster_size=10)


# In[ ]:


topic_model2 = BERTopic(hdbscan_model=cluster_method)
topics2, probs2 = topic_model2.fit_transform(covid)


# In[ ]:


topic_model2.get_topic_info()


# In[ ]:


topic_model2.get_topic(0)


# In[ ]:


topic_model2.get_topic(1)


# In[ ]:


#observe which posts topics were derived from
topic_model2.get_representative_docs()


# In[ ]:


topic_model2.visualize_hierarchy()
#illustrates how topics were clustered from agglomerate clusteing approach (bottom-up)


# ### Follow-up question?
# 
# Does the word vaccine (or the like) correlate with uptick posts around each corresponding event (ebola vs. covid19)?

# In[ ]:


#Determine the frequency of antivax in our dataframe
reddit['vax']=reddit.text.str.contains('vax|vaccination|vaccines|vaccine', na=False, regex=True)#where na's present bool value=False

def mentions(col):   
    if col.vax == True:
        return 1
    else:
        return 0

reddit["vax_mention"] = reddit.apply(mentions, axis=1)


# In[ ]:


#change timestamp Object to datetime64
reddit["year"] = reddit["year"].astype("datetime64")
 
# Setting the Date as index
reddit = reddit.set_index("year")
reddit


# In[ ]:


reddit.groupby('year').sum('vax_mention')


# In[ ]:


vax_mention=reddit.groupby('year').sum('vax_mention')


# In[ ]:


#plot the raw number of mentions over time
import matplotlib.pyplot as plt
 
# Using a inbuilt style to change
# the look and feel of the plot
plt.style.use("fivethirtyeight")
 
# setting figure size to 12, 10
plt.figure(figsize=(12, 10))
 
# Labelling the axes and setting
# a title
plt.xlabel("Date")
plt.ylabel("Mentions of Vaccine")
plt.title("# of mentions over time")
 
# plotting the "A" column alone
plt.plot(vax_mention["vax_mention"])


# ### Conclusions
# 1. Data cleaning can increase the coherence of resulting topics in models
# 2. Illustrating trends in social media data allow us to correlate text data to events in time and further increase our understanding of resulting topics
# 3. including clustering methods such as HDBScan can improve the reproducib

# ## Sources
# 1. Pandas documentation: https://pandas.pydata.org/ (popular python data analysis tool)
# 2. Sentiment analysis of vaccine myths data : https://www.kaggle.com/code/khsamaha/reddit-vaccine-myths-eda-and-text-analysis
# 3. BERTopic tutorial: https://maartengr.github.io/BERTopic/index.html
# 4. Plotting time series: https://www.geeksforgeeks.org/how-to-plot-timeseries-based-charts-using-pandas/
