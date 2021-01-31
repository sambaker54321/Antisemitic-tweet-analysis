import snscrape.modules.twitter as sntwitter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Defining functions
def date_times(tweets,no_of_tweets):
    """
    Takes TwitterSearchScraper data objects and returns the
    datetime.datetime data for each tweet. Output is a list of
    datetimes.datetime data.
    """

    datetimes = []
    for i, tweet in tweets:
        if i<no_of_tweets - 1:
            datetimes.append(tweet.date)
        else:
            break
    return datetimes

def tweedate(tweets,no_of_tweets):
    """
    This function takes TwitterSearchScraper data objects and
    returns the number of tweets on each date. Output is a tuple
    (date_list,date_count).
    """
    datetimes = date_times(tweets,no_of_tweets)
    date_list = pd.date_range(start=min(datetimes).date(),end=max(datetimes).date(),freq="D").tolist()
    #List of dates is in units of days from the earliest tweet in the input to the latest.
    dates = []
    for dt in datetimes:
        dates.append(dt.date()) #Only use date from datetime.datetime data.
    dates = dates[::-1] #Reverse the data.
    date_count = np.zeros(len(date_list))
    for i in range(len(date_list)):
        recc = []
        for j in range(len(dates)):
            if dates[j] == date_list[i]:
                recc.append(dates[j]) #When a date is repeated it is added to recc.
        date_count[i] = len(recc) #Number of tweets in one day is the length of recc.
    return date_list,date_count
    
date_since = "2020-12-01" #Date of the oldest tweet in our search

#Total 'antisemitic hashtag tweets
search_term_all_hashtags = f'(#JewishSupremacy OR #JewishPrivilege OR #ProtocolsofZion OR #jewishquestion OR #loxism OR #jewishhypocrisy OR #JewishTruth OR #JewishSupremacists) since:{date_since}'

#Individual hashtags
search_term_jewishsupremacy = f'#JewishSupremacy since:{date_since}'
search_term_jewishprivilege = f'#JewishPrivilege since:{date_since}'
search_term_protocolsofzion = f'#ProtocolsofZion since:{date_since}'
search_term_jewishquestion = f'#jewishquestion since:{date_since}'
search_term_loxism = f'#loxism since:{date_since}'
search_term_jewishhypocrisy = f'#jewishhypocrisy since:{date_since}'
search_term_jewishtruth = f'#JewishTruth since:{date_since}'
search_term_jewishsupremacists = f'#JewishSupremacists since:{date_since}'

#Extra searches
search_term_soros = f'"george soros" (-@georgesoros) since:{date_since}'
search_term_jews = f'"the jews" since:{date_since}'

#Create lists of interesting search terms
search_terms_hashtags = [search_term_jewishsupremacy,search_term_jewishprivilege,search_term_protocolsofzion,search_term_jewishtruth]
search_terms_soros_jews = [search_term_soros,search_term_jews]

#Perform tweedate on interesting hashtags
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
x_ticks = pd.date_range(start = "2020-11-30",end = "2021-02-01",freq = "SM")

hashtag_date_lists = []
hashtag_date_counts = []

for i in range(len(search_terms_hashtags)):
    """
    This loop will return the date_list and date_count for each hashtag search 
    term along with a plot for each (each drawn on the same axes).
    """
    tweets = enumerate(sntwitter.TwitterSearchScraper(search_terms_hashtags[i]).get_items())
    data = tweedate(tweets,10000)
    date_list,date_count = data[0],data[1]
    ax1.plot(date_list,date_count,label=search_terms_hashtags[i][:-17]) #Remove since:date from the label.
    hashtag_date_lists.append(date_list) #Store the date_list data
    hashtag_date_counts.append(date_count) #Store the date_count data

all_hashtags_tweets = enumerate(sntwitter.TwitterSearchScraper(search_term_all_hashtags).get_items())
all_hashtags_data = tweedate(all_hashtags_tweets,10000)
date_list,date_count = all_hashtags_data[0],all_hashtags_data[1] #Return raw data before plotting.
ax1.plot(date_list,date_count,label='Total') #Plot total hashtag data on same axes.
plt.xlabel('Date')
plt.ylabel('Number of tweets')  
plt.xticks(x_ticks)  
plt.legend()
plt.show()

#Perform tweedate on soros and jews
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

soros_jews_date_lists = []
soros_jews_date_counts = []

for i in range(len(search_terms_soros_jews)):
    """
    This loop will return the date_list and date_count for soros and jews 
    search terms along with a plot for each (each drawn on the same axes).
    """    
    tweets = enumerate(sntwitter.TwitterSearchScraper(search_terms_soros_jews[i]).get_items())
    data = tweedate(tweets,100000)
    date_list,date_count = data[0],data[1]
    ax2.plot(date_list,date_count,label=search_terms_soros_jews[i][:-17])
    soros_jews_date_lists.append(date_list) #Store the date_list data
    soros_jews_date_counts.append(date_count) #Store the date_count data

plt.xlabel('Date')
plt.ylabel('Number of tweets')
plt.xticks(x_ticks) 
plt.legend()
plt.show()

#Peak detector function
def peak_detector(data,number_of_peaks):
    """
    Take tweedate data as the input. The function returns
    the positions and height of the number of peaks specified.
    The peaks returned are in order of height.
    """    
    date_list,date_count = data[0].copy(),data[1].copy() #Deep copy the contents of data to avoid overwriting.
    dates = []
    heights = []
    for i in range(number_of_peaks):
        date,height = date_list[np.argmax(date_count)],date_count[np.argmax(date_count)] #Identify maximum height and the date of the peak.
        dates.append(date)
        heights.append(height)
        date_list[np.argmax(date_count)] = 0 #Set maximum to zero so next iteration finds next highest peak.
        date_count[np.argmax(date_count)] = 0
    peaks = [dates,heights]
    return peaks

#Peak detection on total hashtags data
hashtags_peaks = peak_detector(all_hashtags_data,20) #The 20 highest peaks

#Peak ignoring mean function
def peak_ignoring_mean(data,number_of_peaks):
    """
    Input tweedate data and number of peaks as integer. 
    This function turns number of specified peaks to zeros
    in date_count, and computes the mean, ignoring zeros.
    """
    data_no_peaks = []
    date_count = data[1]
    heights = peak_detector(data,number_of_peaks)[1]
    for i in range(len(date_count)):
        anti_match = []
        for j in range(len(heights)):
            if date_count[i] != heights[j]:
                anti_match.append(date_count[i]) #Add the date to a list if it doesn't match
        if len(anti_match) == len(heights): #Only if the date_count is not found at all in heights.
            data_no_peaks.append(anti_match[0])
    mean = np.mean(data_no_peaks)
    return mean

#Peak ignoring mean on total hashtags data
hashtags_mean_without_peaks = peak_ignoring_mean(all_hashtags_data,20)

#Peak detector and mean of George Soros tweets
soros_tweets = enumerate(sntwitter.TwitterSearchScraper(search_term_soros).get_items())
soros_dates_data = tweedate(soros_tweets,100000)
soros_peaks = peak_detector(soros_dates_data,20)
soros_mean_no_peaks = peak_ignoring_mean(soros_dates_data,20)
print(soros_peaks,soros_mean_no_peaks)

#Peak detector and mean of Jews tweets
jews_tweets = enumerate(sntwitter.TwitterSearchScraper(search_term_jews).get_items())
jews_dates_data = tweedate(jews_tweets,100000)
jews_peaks = peak_detector(jews_dates_data,20)
jews_mean_no_peaks = peak_ignoring_mean(jews_dates_data,20)
print(jews_peaks,jews_mean_no_peaks)

print(jews_peaks[0][0:10])
print(hashtags_peaks,hashtags_mean_without_peaks)

#Correlation between jews and soros tweets
d = {'The Jews':jews_dates_data[1],'George Soros':soros_dates_data[1]} #Create dict data
df = pd.DataFrame(data=d) #Create pandas dataframe

cor = df.corr(method='pearson')
print(cor)

#Percentage of Soros tweets that mention "The jews"
jews_tweets = enumerate(sntwitter.TwitterSearchScraper(search_term_jews).get_items())
soros_tweets = enumerate(sntwitter.TwitterSearchScraper(search_term_soros).get_items())

soros_tweets_text = []
for i,soros_tweet in soros_tweets:
    soros_tweets_text.append(soros_tweet.content)
    if i>100000:
        break

soros_tweets_containing_jews = []
for phrase in soros_tweets_text:
    if phrase.find("the jews") != -1:
        soros_tweets_containing_jews.append("yes")
    elif phrase.find("the Jews") != -1:
        soros_tweets_containing_jews.append("yes")
    elif phrase.find("The jews") != -1:
        soros_tweets_containing_jews.append("yes")
    elif phrase.find("The Jews") != -1:
        soros_tweets_containing_jews.append("yes")

print((len(soros_tweets_containing_jews)/len(soros_tweets_text))*100) #Percentage of soros tweets that mention "the jews."

    