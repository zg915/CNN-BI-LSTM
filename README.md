# DS-GA 1003 Machine Learning Final Project

## Optimizing Stock Market Predictions with Hybrid Deep Learning Architectures

## 0. Updates
Alan: Found new links for SCI data; Write BI-LSTM forward function

## 1. Data Preprocessing

### Data Collecting

The Shanghai Composite Index (000001) 7127 trading days from July 1, 1991, to August 31, 2020

data source: https://cn.investing.com/indices/shanghai-composite-historical-data

Another source, using python library "akshare": https://blog.csdn.net/qq_55285829/article/details/136809576?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-4-136809576-blog-109829156.235%5Ev43%5Econtrol&spm=1001.2101.3001.4242.3&utm_relevant_index=7

One more source: http://360gu.janqi.com/p/上证指数的历史数据,近3年,近5年,近10年,1990至今历史,下载以前历史数据/
### Data Cleaning

…

## 2. Modeling (04.23)

### CNN

…

### LSTM

LSTM model explained: https://www.youtube.com/watch?v=YCzL96nL7j0

LSTM code explained: https://www.youtube.com/watch?v=RHGiXPuo_pI

### BI-LSTM

BI-LSTM code explained: https://www.youtube.com/watch?v=jGst43P-TJA

Whole stock prediction LSTM model code explained: https://www.youtube.com/watch?v=q_HS4s1L8UI

Wrote first draft of BI-LSTM forward function. Questions:
- use two BI-LSTM, dropout layer, and ReLU layer exactly as in the paper?
- will we train using solely BI-LSTM, or plug it directly into CNN-BI-LSTM? If yes, what hyperparameter/ structure to use?


### CNN-BI-LSTM

…

## 3. Evaluation

…(model comparison, hyperparameter tuning, different dataset)

## 4. Report

### Poster (04.30)

…

### Paper (05.06)

…