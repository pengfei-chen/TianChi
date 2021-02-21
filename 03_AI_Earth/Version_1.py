import pandas as pd
import numpy  as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import scipy 
from netCDF4 import Dataset
import netCDF4 as nc
import gc

label_path       = './tcdata/enso_round1_train_20210201/SODA_label.nc'
label_trans_path = './tcdata/enso_round1_train_20210201/'
nc_label         = Dataset(label_path,'r')
years            = np.array(nc_label['year'][:])
months           = np.array(nc_label['month'][:])
year_month_index = []
vs               = []
for i,year in enumerate(years):
    for j,month in enumerate(months):
        year_month_index.append('year_{}_month_{}'.format(year,month))
        vs.append(np.array(nc_label['nino'][i,j]))

df_SODA_label               = pd.DataFrame({'year_month':year_month_index}) 
df_SODA_label['year_month'] = year_month_index
df_SODA_label['label']      = vs
# df_SODA_label.to_csv(label_trans_path + 'df_SODA_label.csv',index = None)


SODA_path        = './tcdata/enso_round1_train_20210201/SODA_train.nc'
nc_SODA          = Dataset(SODA_path,'r')

def trans_df(df, vals, lats, lons, years, months):
    '''
        (100, 36, 24, 72) -- year, month,lat,lon 
    '''
    for j,lat_ in enumerate(lats):
        for i,lon_ in enumerate(lons):
            c = 'lat_lon_{}_{}'.format(int(lat_),int(lon_))  
            v = []
            for y in range(len(years)):
                for m in range(len(months)): 
                    v.append(vals[y,m,j,i])
            df[c] = v
    return df


year_month_index = []
years              = np.array(nc_SODA['year'][:])
months             = np.array(nc_SODA['month'][:])
lats             = np.array(nc_SODA['lat'][:])
lons             = np.array(nc_SODA['lon'][:])
for year in years:
    for month in months:
        year_month_index.append('year_{}_month_{}'.format(year,month))
df_sst  = pd.DataFrame({'year_month':year_month_index}) 
df_t300 = pd.DataFrame({'year_month':year_month_index}) 
df_ua   = pd.DataFrame({'year_month':year_month_index}) 
df_va   = pd.DataFrame({'year_month':year_month_index})

df_sst = trans_df(df = df_sst, vals = np.array(nc_SODA['sst'][:]), lats = lats, lons = lons, years = years, months = months)
df_t300 = trans_df(df = df_t300, vals = np.array(nc_SODA['t300'][:]), lats = lats, lons = lons, years = years, months = months)
df_ua   = trans_df(df = df_ua, vals = np.array(nc_SODA['ua'][:]), lats = lats, lons = lons, years = years, months = months)
df_va   = trans_df(df = df_va, vals = np.array(nc_SODA['va'][:]), lats = lats, lons = lons, years = years, months = months)

label_trans_path = './tcdata/enso_round1_train_20210201/'
# df_sst.to_csv(label_trans_path  + 'df_sst_SODA.csv',index = None)
# df_t300.to_csv(label_trans_path + 'df_t300_SODA.csv',index = None)
# df_ua.to_csv(label_trans_path   + 'df_ua_SODA.csv',index = None)
# df_va.to_csv(label_trans_path   + 'df_va_SODA.csv',index = None)

# 2.2 CMIP_label处理
label_path       = './tcdata/enso_round1_train_20210201/CMIP_label.nc'
label_trans_path = './tcdata/enso_round1_train_20210201/'
nc_label         = Dataset(label_path,'r')
years            = np.array(nc_label['year'][:])
months           = np.array(nc_label['month'][:])
year_month_index = []
vs               = []
for i,year in enumerate(years):
    for j,month in enumerate(months):
        year_month_index.append('year_{}_month_{}'.format(year,month))
        vs.append(np.array(nc_label['nino'][i,j]))
df_CMIP_label               = pd.DataFrame({'year_month':year_month_index}) 
df_CMIP_label['year_month'] = year_month_index
df_CMIP_label['label']      = vs
# df_CMIP_label.to_csv(label_trans_path + 'df_CMIP_label.csv',index = None)

# 2.3 CMIP_train处理
CMIP_path       = './tcdata/enso_round1_train_20210201/CMIP_train.nc'
CMIP_trans_path = './tcdata/enso_round1_train_20210201/'
nc_CMIP  = Dataset(CMIP_path,'r')

nc_CMIP.variables.keys()
nc_CMIP['t300'][:].shape


year_month_index = []
years              = np.array(nc_CMIP['year'][:])
months             = np.array(nc_CMIP['month'][:])
lats               = np.array(nc_CMIP['lat'][:])
lons               = np.array(nc_CMIP['lon'][:])
last_thre_years = 1000
for year in years:
    '''
        数据的原因，我们
    '''
    if year >= 4645 - last_thre_years:
        for month in months:
            year_month_index.append('year_{}_month_{}'.format(year,month))
df_CMIP_sst  = pd.DataFrame({'year_month':year_month_index}) 
df_CMIP_t300 = pd.DataFrame({'year_month':year_month_index}) 
df_CMIP_ua   = pd.DataFrame({'year_month':year_month_index}) 
df_CMIP_va   = pd.DataFrame({'year_month':year_month_index})

# 因为内存限制,我们暂时取最后1000个year的数据
def trans_thre_df(df, vals, lats, lons, years, months, last_thre_years = 1000):
    '''
        (4645, 36, 24, 72) -- year, month,lat,lon 
    '''
    for j,lat_ in (enumerate(lats)):
#         print(j)
        for i,lon_ in enumerate(lons):
            c = 'lat_lon_{}_{}'.format(int(lat_),int(lon_))  
            v = []
            for y_,y in enumerate(years):
                '''
                    数据的原因，我们
                '''
                if y >= 4645 - last_thre_years:
                    for m_,m in  enumerate(months): 
                        v.append(vals[y_,m_,j,i])
            df[c] = v
    return df

df_CMIP_sst  = trans_thre_df(df = df_CMIP_sst,  vals   = np.array(nc_CMIP['sst'][:]),  lats = lats, lons = lons, years = years, months = months)
# df_CMIP_sst.to_csv(CMIP_trans_path + 'df_CMIP_sst.csv',index = None)
del df_CMIP_sst
gc.collect()

df_CMIP_t300 = trans_thre_df(df = df_CMIP_t300, vals   = np.array(nc_CMIP['t300'][:]), lats = lats, lons = lons, years = years, months = months)
# df_CMIP_t300.to_csv(CMIP_trans_path + 'df_CMIP_t300.csv',index = None)
del df_CMIP_t300
gc.collect()

df_CMIP_ua   = trans_thre_df(df = df_CMIP_ua,   vals   = np.array(nc_CMIP['ua'][:]),   lats = lats, lons = lons, years = years, months = months)
# df_CMIP_ua.to_csv(CMIP_trans_path + 'df_CMIP_ua.csv',index = None)
del df_CMIP_ua
gc.collect()

df_CMIP_va   = trans_thre_df(df = df_CMIP_va,   vals   = np.array(nc_CMIP['va'][:]),   lats = lats, lons = lons, years = years, months = months)
# df_CMIP_va.to_csv(CMIP_trans_path + 'df_CMIP_va.csv',index = None)
del df_CMIP_va
gc.collect()

# 数据建模
# 1. 工具包导入
import pandas as pd
import numpy  as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import scipy 
import joblib
from netCDF4 import Dataset
import netCDF4 as nc 
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Input 
import gc

# 2. 数据读取
# 1.SODA_label处理
label_path       = './tcdata/enso_round1_train_20210201/SODA_label.nc'
nc_label         = Dataset(label_path,'r')
tr_nc_labels     = nc_label['nino'][:]

# 2. 原始特征数据读取
SODA_path        = './tcdata/enso_round1_train_20210201/SODA_train.nc'
nc_SODA          = Dataset(SODA_path,'r') 
nc_sst           = np.array(nc_SODA['sst'][:])
nc_t300          = np.array(nc_SODA['t300'][:])
nc_ua            = np.array(nc_SODA['ua'][:])
nc_va            = np.array(nc_SODA['va'][:])

#模型构建
#  神经网络框架
def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def RMSE_fn(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.array(y_true, float).reshape(-1, 1) - np.array(y_pred, float).reshape(-1, 1), 2)))

def build_model():
    inp    = Input(shape=(12,24,72,4))      
    x_4    = Dense(1, activation='relu')(inp)   
    x_3    = Dense(1, activation='relu')(tf.reshape(x_4,[-1,12,24,72]))
    x_2    = Dense(1, activation='relu')(tf.reshape(x_3,[-1,12,24]))
    x_1    = Dense(1, activation='relu')(tf.reshape(x_2,[-1,12]))    
    x = Dense(64, activation='relu')(x_1)  
    x = Dropout(0.25)(x) 
    x = Dense(32, activation='relu')(x)   
    x = Dropout(0.25)(x)  
    output = Dense(24, activation='linear')(x)   
    model  = Model(inputs=inp, outputs=output)
    adam = tf.optimizers.Adam(lr=1e-3,beta_1=0.99,beta_2 = 0.99) 
    model.compile(optimizer=adam, loss=RMSE)
    return model

# 2. 训练集验证集划分
### 训练特征，保证和训练集一致
tr_features = np.concatenate([nc_sst[:,:12,:,:].reshape(-1,12,24,72,1),nc_t300[:,:12,:,:].reshape(-1,12,24,72,1),\
                              nc_ua[:,:12,:,:].reshape(-1,12,24,72,1),nc_va[:,:12,:,:].reshape(-1,12,24,72,1)],axis=-1)

### 训练标签，取后24个
tr_labels = tr_nc_labels[:,12:] 
### 训练集验证集划分
tr_len     = int(tr_features.shape[0] * 0.8)
tr_fea     = tr_features[:tr_len,:].copy()
tr_label   = tr_labels[:tr_len,:].copy()
val_fea     = tr_features[tr_len:,:].copy()
val_label   = tr_labels[tr_len:,:].copy()

# 3. 模型训练
model_mlp     = build_model()
#### 模型存储的位置
model_weights = './model_baseline/model_mlp_baseline.h5'

checkpoint = ModelCheckpoint(model_weights, monitor='val_loss', verbose=0, save_best_only=True, mode='min',
                             save_weights_only=True)

plateau        = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_delta=1e-4, mode='min')
early_stopping = EarlyStopping(monitor="val_loss", patience=20)
history        = model_mlp.fit(tr_fea, tr_label,
                    validation_data=(val_fea, val_label),
                    batch_size=4096, epochs=200,
                    callbacks=[plateau, checkpoint, early_stopping],
                    verbose=2)

# 4. 模型预测
prediction = model_mlp.predict(val_fea)

# 5. Metrics
from   sklearn.metrics import mean_squared_error
def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_pred = y_preds, y_true = y_true))

def score(y_true, y_preds):
    accskill_score = 0
    rmse_scores    = 0
    a = [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6
    y_true_mean = np.mean(y_true,axis=0) 
    y_pred_mean = np.mean(y_preds,axis=0) 
#     print(y_true_mean.shape, y_pred_mean.shape)

    for i in range(24): 
        fenzi = np.sum((y_true[:,i] -  y_true_mean[i]) *(y_preds[:,i] -  y_pred_mean[i]) ) 
        fenmu = np.sqrt(np.sum((y_true[:,i] -  y_true_mean[i])**2) * np.sum((y_preds[:,i] -  y_pred_mean[i])**2) ) 
        cor_i = fenzi / fenmu
    
        accskill_score += a[i] * np.log(i+1) * cor_i
        rmse_score   = rmse(y_true[:,i], y_preds[:,i])
#         print(cor_i,  2 / 3.0 * a[i] * np.log(i+1) * cor_i - rmse_score)
        rmse_scores += rmse_score 
    
    return 2 / 3.0 * accskill_score - rmse_scores

print('score', score(y_true = val_label, y_preds = prediction))


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Input 
import numpy as np
import os
import zipfile

def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def build_model():
    inp    = Input(shape=(12,24,72,4))  
    
    x_4    = Dense(1, activation='relu')(inp)   
    x_3    = Dense(1, activation='relu')(tf.reshape(x_4,[-1,12,24,72]))
    x_2    = Dense(1, activation='relu')(tf.reshape(x_3,[-1,12,24]))
    x_1    = Dense(1, activation='relu')(tf.reshape(x_2,[-1,12]))
     
    x = Dense(64, activation='relu')(x_1)  
    x = Dropout(0.25)(x) 
    x = Dense(32, activation='relu')(x)   
    x = Dropout(0.25)(x)  
    output = Dense(24, activation='linear')(x)   
    model  = Model(inputs=inp, outputs=output)

    adam = tf.optimizers.Adam(lr=1e-3,beta_1=0.99,beta_2 = 0.99) 
    model.compile(optimizer=adam, loss=RMSE)

    return model 

model = build_model()
model.load_weights('./model_baseline/model_mlp_baseline.h5')

test_path = './tcdata/enso_round1_test_20210201/'

### 1. 测试数据读取
files = os.listdir(test_path)
test_feas_dict = {}
for file in files:
    test_feas_dict[file] = np.load(test_path + file)
    
### 2. 结果预测
test_predicts_dict = {}
for file_name,val in test_feas_dict.items():
    test_predicts_dict[file_name] = model.predict(val).reshape(-1,)
#     test_predicts_dict[file_name] = model.predict(val.reshape([-1,12])[0,:])

### 3.存储预测结果
for file_name,val in test_predicts_dict.items(): 
    np.save('./result/' + file_name,val)

# 预测结果打包
#打包目录为zip文件（未压缩）
def make_zip(source_dir='./result/', output_filename = 'result.zip'):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    source_dirs = os.walk(source_dir)
    print(source_dirs)
    for parent, dirnames, filenames in source_dirs:
        print(parent, dirnames)
        for filename in filenames:
            if'.npy'not in filename:
                continue
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)   #相对路径
            zipf.write(pathfile, arcname)
    zipf.close()
make_zip()

