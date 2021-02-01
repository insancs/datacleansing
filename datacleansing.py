import pandas as pd
import numpy as np
import io

retail_raw = pd.read_csv('E:/DQLab/Dataset/retail_raw_reduced_data_quality.csv')
# Menampilkan semua kolom tanpa terpotong
# pd.options.display.max_columns = None
pd.options.display.width = None
print(retail_raw.head())

# Data Profiling
# [1] Inspeksi data
print("[1] Data Type :\n", retail_raw.dtypes)

# [2] Descriptive Statistic
print("\n[2] Descriptive Statistic")
# Length column
length_city = len(retail_raw['city'])
print("Length city :", length_city)

# Count
count_city = retail_raw['city'].count()
print("Count city :", count_city)

# Missing value, selisih antara length dan count
number_of_misval_city = length_city - count_city
float_of_misval_city = float(number_of_misval_city / length_city)
pct_of_misval_city = "{0: 1f}%".format(float_of_misval_city * 100)
print("Persentase missing value city :", pct_of_misval_city)

# Min, Max, Mean, Median, Modus, Standart deviation
print('Kolom quantity')
print('Minimum value: ', retail_raw['quantity'].min())
print('Maximum value: ', retail_raw['quantity'].max())
print('Mean value: ', retail_raw['quantity'].mean())
print('Mode value: ', retail_raw['quantity'].mode())
print('Median value: ', retail_raw['quantity'].median())
print('Standard Deviation value: ', retail_raw['quantity'].std())

# Quantile statistic
print("\nQuantile kolom quantity :\n", retail_raw['quantity'].quantile([0.25, 0.5, 0.75]))

# Correlation
print("\nKorelasi quantity dan item_price\n", retail_raw[['quantity', 'item_price']].corr())

# [3] Data Cleansing
print("\n[3] Data Cleansing")
# Cek kolom yang memiliki missing value
print('Check kolom yang memiliki missing data:')
print(retail_raw.isnull().any())

# Filling missing value
retail_raw['quantity'] = retail_raw['quantity'].fillna(retail_raw['quantity'].mean())
retail_raw['item_price'] = retail_raw['item_price'].fillna(retail_raw['item_price'].mean())
print("\nCek hasil imputasi :\n", retail_raw.isnull().any())

# [4] Deklarasi variabel
Q1 = retail_raw['quantity'].quantile(0.25)
Q3 = retail_raw['quantity'].quantile(0.75)
IQR = Q3 - Q1

Q11 = retail_raw['item_price'].quantile(0.25)
Q33 = retail_raw['item_price'].quantile(0.75)
IQR1 = Q33 - Q11

# Cek dimensi sebelum
print("\n[4] Hapus Outliers")
print("Dimensi data :", retail_raw.shape)

# Removing outliers
retail_raw = retail_raw[~((retail_raw['quantity'] < (Q1 - 1.5 * IQR)) | (retail_raw['quantity'] > (Q3 + 1.5 * IQR)))]
# Cek dimensi
print("Hasil removing outliers quantity:", retail_raw.shape)
retail_raw = retail_raw[
    ~((retail_raw['item_price'] < (Q11 - 1.5 * IQR1)) | (retail_raw['item_price'] > (Q33 + 1.5 * IQR1)))]
# Cek dimensi sesudah
print("Hasil removing outliers item_price:", retail_raw.shape)

# Cek duplikasi data
print("\nCek duplikasi :\n", retail_raw.duplicated(subset=None))
# Hapus duplikasi
retail_raw.drop_duplicates(inplace=True)
# Cek dimensi data
print(retail_raw.shape)
