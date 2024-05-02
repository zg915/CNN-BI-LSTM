# DS-GA 1003 Machine Learning Final Project

## Optimizing Stock Market Predictions with Hybrid Deep Learning Architectures

## 0. Updates
Alan: Write template for model training in BI-LSTM.ipynb

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

Best Shot: (20 epoch, 0.0001 learning rate, 1 layer of BILSTM with hidden size 64, 2 layers of fc 128-16(ReLU)-1)

Train RMSE: 0.040643; Test RMSE 38.340305

Train  MAE: 0.023417; Test  MAE 26.461379

Train  R^2: 0.998348; Test  R^2 0.967867

### CNN-LSTM


### CNN-BI-LSTM
best shot: (100 epoch, 0.0001 learning rate, CNN 8-32, 1 layer of BILSTM with hidden size 64, 2 layers of fc 128-16(ReLU)-1)

Train RMSE: 0.040718; Test RMSE 43.307657

Train  MAE: 0.024318; Test  MAE 30.802969

Train  R^2: 0.998337; Test  R^2 0.956623
…

## 3. Evaluation

### Comparsion between LSTM and BI-LSTM on same parameter number
#### LSTM 64 hidden layer = BI-LSTM 32 hidden layer

LSTM:

Train RMSE: 0.041400; Test RMSE 40.406074

Train  MAE: 0.024093; Test  MAE 28.328293

Train  R^2: 0.998291; Test  R^2 0.964189

BI-LSTM:

Train RMSE: 0.042824; Test RMSE 40.418164

Train  MAE: 0.024424; Test  MAE 27.713295

Train  R^2: 0.998167; Test  R^2 0.964581

#### LSTM 128 hidden layer = BI-LSTM 64 hidden layer
LSTM:

Train RMSE: 0.040112; Test RMSE 39.808764

Train  MAE: 0.023203; Test  MAE 28.302415

Train  R^2: 0.998395; Test  R^2 0.963594

BI-LSTM:

Train RMSE: 0.040568; Test RMSE 39.166940

Train  MAE: 0.023319; Test  MAE 27.128253

Train  R^2: 0.998364; Test  R^2 0.966312

### LSTM, BILSTM, CNN-BI-LSTM, CNN-LSTM with the new split


### LSTM, BILSTM, CNN-BI-LSTM, CNN-LSTM with the new data and new train-test-split
LSTM:  
Train RMSE: 0.020860; Test RMSE 21.856915

Train  MAE: 0.012224; Test  MAE 13.633474

Train  R^2: 0.999565; Test  R^2 0.999652


BI-LSTM:  
Train RMSE: 0.021299; Test RMSE 25.945026  

Train  MAE: 0.012339; Test  MAE 15.015306  

Train  R^2: 0.999541; Test  R^2 0.999499  

CNN-LSTM:  
Train RMSE: 0.020806; Test RMSE 23.642113

Train  MAE: 0.012312; Test  MAE 13.792218

Train  R^2: 0.999570; Test  R^2 0.999594

CNN-BI-LSTM:  
Train RMSE: 0.021786; Test RMSE 24.465844

Train  MAE: 0.012207; Test  MAE 13.554134

Train  R^2: 0.999532; Test  R^2 0.999567

## 4. Report

### Poster (04.30)

…

### Paper (05.06)
https://docs.google.com/document/d/16VK6Izbv6z5EJqd9hq8OhnQ2iOai019pXy7xmgFr7mQ/edit?usp=sharing
…