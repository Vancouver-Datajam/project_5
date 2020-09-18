# Project 5: Plastics Crisis Twitter Sentiment Analysis

## Project description

### The problem
Since the middle of the 20th century, the rapidly increasing global production of plastics (322 million metric tons per year in 2016) has been accompanied by an unprecedented accumulation of plastic litter in our oceans (Jambeck et al. 2015). In response, there has been an increase in public awareness and concern regarding the plastics crisis, and the growth of the zero waste movement and policy decisions such as plastic bag and plastic straw bans. These decisions are often met with mixed public opinion. Do people have positive, neutral, or negative opinions towards different sustainability initiatives? How do we find and measure these sentiments?

### Libraries used
- pandas
- numpy
- textblob
- collections
- re
- nltk
- sklearn
- emot
- wordcloud
- contractions
- tensorflow
- os

### Dataset
We used Twitter data to conduct a sentiment analysis towards different sustainability topics. Using Tweepy, the Python library for accessing the Twitter API, we scraped some Tweets that contained the following hashtags:
* #noplastic
* #plasticpollutes
* #plasticpollution
* #sustainability
* #zerowaste
<br />
The folder "hashtags" contains all the datasets for this project. Each CSV file name corresponds with the Tweets' hashtag. The CSV with the filename "updated" contains an aditional location column. Because of the API's restrictions, we were only able to scrape week-old data (from August 30, 2020 to September 10, 2020).

### Files included
#### Data cleaning
* stopwords_punctuation_removal_and_wordcloud.ipynb: Removes stop words and punctation from Tweet text to prepare for sentiment analysis. Creates a wordcloud of most common words for each hashtag group.
* extract_english.ipynb: Extracts English-only Tweets.

#### Data analysis
* sentiment_ratio.ipynb: Uses Textblob to calculate the positive and negative percentage and ratio based on Tweet texts. Finds bigrams and trigrams for each hashtag group.
* workflow.ipynb: Combines all cleaning and analysis all into one workflow. Does not include wordclouds. 

#### Tweet bot
* tweet_generator.ipynb: A neural network (with Gated Recurrent Units - GRU) was trained on the obtained tweets. Given the starting word, it generates ‘new’ tweets! [Click here to test it out!](http://ec2-3-238-29-44.compute-1.amazonaws.com/)


### Results

In our scraped Tweets, there are generally more positive than negative sentiments in their text. The ratio of positive to negative sentiments are very high for each hashtag group. 

<img src="https://github.com/Vancouver-Datajam/project_5/blob/master/images/sentiment_analysis_results.png" width="500"/>

The following wordcloud demonstrates the most common words of all Tweets containing the #noplastic hashtag.
<img src="https://github.com/Vancouver-Datajam/project_5/blob/master/images/noplastic_wordcloud.png" width="500"/>

## Team members:
* Jeanette Andrews (Team lead)
* Madan Krishnan (Mentor)
* Jerry Chen
* Willy Deng
* Sayemin Naheen
* Juanita Palomar
* Rita Zeng

## Datajam Schedule
| Time | Description |
| --- | --- |
| 8:30am | Opening Ceremony |
| 9:30am | Official hack start time! Meet together as team and get to know each other|
| 9:45am | Project brainstorming & defining tasks |
| 10:00am | Optional Git workshop|
| 10:30am | Hack & work on tasks |
| 12:30pm | Check-in #1: meet back up as a team |
| 1:00pm | Hack & work on tasks |
| 3:30pm | Check-in #2: meet back up as a team |
| | Debugging, prioritizing remaining tasks |
| 5:00pm | Final repository merging |
| | Prepare demo or slides |
| 6:30pm | Project deadline & final Presentation! |
| 7:30pm | Career Panel & Q&A |
| 8:30pm | Awards Ceremony & Closing |
