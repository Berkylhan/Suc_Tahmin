import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


#DR_NO: Olay için atanmış benzersiz rapor numarası. Her olay için tekil bir kimlik numarası olarak kullanılır.
#DATE_RPTD: Olayın raporlandığı tarih. Bu, suçun polise bildirildiği tarih olabilir.
#DATE_OCC: Olayın gerçekleştiği tarih. Bu tarih, suçun işlendiği günü belirtir.
#TIME_OCC: Olayın gerçekleştiği saat. Genellikle 24 saat formatında (HHMM) kaydedilir, örneğin 2130 saati 21:30’a denk gelir.
#AREA: Olayın gerçekleştiği ana bölgeyi veya bölge kodunu ifade eder. Her bölgenin belirli bir sayısal kodu bulunur.
#AREA_NAME: Olayın gerçekleştiği bölgenin adı (örneğin, "Wilshire" veya "Central").
#RPT_DIST_NO: Raporun kaydedildiği bölgesel dağıtım numarası. Bu, olayın hangi alt bölgede veya bölgesel birimde gerçekleştiğini gösterir.
#PART_1-2: Suçun ciddi (Part 1) veya daha az ciddi (Part 2) olup olmadığını gösteren kod. Suç türlerini belirlemek için kullanılır.
#CRM_CD: Suç kodu. Her suç türü için benzersiz bir sayısal kod atanmıştır.
#CRM_CD_DESC: Suç kodunun açıklaması, yani suçun tanımı (örneğin, "Hırsızlık" veya "Saldırı").
#MOCODES: Suçun işlenme şekli veya kullanılan metodun kodu. Modus Operandi’nin (MO) kısaltmasıdır.
#VICT_AGE: Mağdurun yaşı.
#VICT_SEX: Mağdurun cinsiyeti (örneğin, "M" erkek için, "F" kadın için).
#VICT_DESCENT: Mağdurun kökeni veya etnik grubu.
#PREMIS_CD: Suçun işlendiği yerin kodu. Belirli türdeki mekanlara göre sayısal bir kod atanmıştır.
#PREMIS_DESC: Suçun işlendiği yerin açıklaması (örneğin, "Ev", "Dükkan", "Araç").
#WEAPON_USED_CD: Olay sırasında kullanılan silahın kodu. Eğer silah yoksa boş olabilir.
#WEAPON_DESC: Kullanılan silahın açıklaması (örneğin, "Bıçak", "Tabanca").
#STATUS: Olayın durumu hakkında kısa bilgi veren kod (örneğin, “AA” tutuklama anlamında olabilir).
#STATUS_DESC: Olayın durumuna dair açıklama (örneğin, “Adult Arrest” yetişkin tutuklama anlamında).
#CRM_CD_1: Ana suç kodu. Suç çoklu kodla işaretlenmişse ana suç burada belirtilir.
#CRM_CD_2: İkinci suç kodu. Suçla ilişkili başka bir suç türünü ifade edebilir (eğer mevcutsa).
#CRM_CD_3: Üçüncü suç kodu. Ek bir suç türünü gösterir (eğer mevcutsa).
#CRM_CD_4: Dördüncü suç kodu. Ek bir suç türünü gösterir (eğer mevcutsa).
#LOCATION: Suçun işlendiği tam adres veya konum bilgisi.
#CROSS_STREET: Olayın gerçekleştiği yerin yakınındaki sokak veya kesişim noktası.
#LAT: Olayın gerçekleştiği yerin enlem bilgisi.
#LON: Olayın gerçekleştiği yerin boylam bilgisi.

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.2f}'.format)

df_ = pd.read_excel("datasets/Crime_Data_from_2020_to_Present_Proje_örneklem.xlsx")
#df = df_.sample(n=1000)
df = df_.copy()

df.columns
df.head()
df.shape
df.isnull().sum()
df.isnull().mean() * 100
df.info
df.describe().T

def baslik_duzenle(df):
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.upper()
    return df.head()

baslik_duzenle(df)

def bos_deger_sil(df):
    df.drop(["CRM_CD_4", "CRM_CD_3", "CRM_CD_2", "CROSS_STREET", "WEAPON_DESC", "WEAPON_USED_CD"], axis=1, inplace=True)
    return df.head()

bos_deger_sil(df)


df['VICT_SEX'] = df['VICT_SEX'].apply(lambda x: x if x in ['F', 'M'] else 'X')

df['VICT_SEX'].value_counts()
df[df['VICT_SEX'] == 'X'].head()
df['CRM_CD_DESC'].value_counts()
df['CRM_CD_DESC'].nunique()

#veri tiplerini inceleme
df.dtypes

def tarihi_parcala(df):
    df['DATE_OCC'] = pd.to_datetime(df['DATE_OCC'], errors='coerce')
    df['DATE_OCC'] = pd.to_datetime(df['DATE_OCC'], format='%Y-%m-%d', errors='coerce')
    df['YEAR_OCC'] = df['DATE_OCC'].dt.year
    df['MONTH_OCC'] = df['DATE_OCC'].dt.month
    df['DAY_OF_WEEK_OCC'] = df['DATE_OCC'].dt.dayofweek
    df['DAY_NAME_OCC'] = df['DATE_OCC'].dt.day_name()
    df['TIME_OCC_FORMATTED'] = df['TIME_OCC'].astype(str).str.zfill(4)
    df['TIME_OCC_FORMATTED'] = df['TIME_OCC_FORMATTED'].str[:2] + ':' + df['TIME_OCC_FORMATTED'].str[2:]
    df['TIME_OCC_HOUR'] = df['TIME_OCC_FORMATTED'].str[:2]
    return df.head()

tarihi_parcala(df)

df.head()
df.tail()
df.dtypes


################################################### özellik mühendisliği ##########################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
df.head()
#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

# Aykırı Değerleri Yakalama
#############################################

# Grafik Teknikle Aykırı Değerler
###################

sns.boxplot(x=df["VICT_AGE"])
plt.show()


# Aykırı Değerler Nasıl Yakalanır?
###################

q1 = df["VICT_AGE"].quantile(0.25)
q3 = df["VICT_AGE"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["VICT_AGE"] < low) | (df["VICT_AGE"] > up)]

df[(df["VICT_AGE"] < low) | (df["VICT_AGE"] > up)].index


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "VICT_AGE", q1=0.3, q3=0.7)

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "VICT_AGE")

def grab_col_names(dataframe, cat_th=5, car_th=10):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

grab_col_names(df)


# Aykırı Değerlerin Kendilerine Erişmek
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "VICT_AGE")

# Aykırı Değer Problemini Çözme
#############################################

###################
# Silme
###################


df.shape
df_without_outliers = df[~((df["VICT_AGE"] < low) | (df["VICT_AGE"] > up))]
df_without_outliers.head()

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers.head()

remove_outlier(df, "VICT_AGE")

#################################################################################################
#################################################################################################

# Kategoriler için birleştirme fonksiyonu
def categorize(crime_desc):
    if "BURGLARY" in crime_desc or "THEFT" in crime_desc or "ROBBERY" in crime_desc or "STOLEN" in crime_desc:
        return "THEFT"
    elif "ASSAULT" in crime_desc or "BATTERY" in crime_desc or "BRANDISH" in crime_desc:
        return "ASSAULT"
    elif "RAPE" in crime_desc or "SEXUAL" in crime_desc or "LEWD" in crime_desc or "INDECENT" in crime_desc:
        return "SEXUAL_CRIMES"
    elif "VANDALISM" in crime_desc or "ARSON" in crime_desc or "THROWING OBJECT" in crime_desc:
        return "PROPERTY_DAMAGE"
    elif "HOMICIDE" in crime_desc or "FIREARMS" in crime_desc or "THREATS" in crime_desc:
        return "OTHER_VIOLENT_CRIMES"
    elif "COURT" in crime_desc or "RESTRAINING ORDER" in crime_desc or "OFFENDER" in crime_desc:
        return "JUDICIAL_AND_ADMINISTRATIVE_CRIMES"
    else:
        return "OTHERS"

# Yeni kategori sütununu ekle
df["CATEGORY"] = df["CRM_CD_DESC"].apply(categorize)


# Kategoriler için birleştirme fonksiyonu
def categorize(crime_desc):
    if "BURGLARY" in crime_desc or "THEFT" in crime_desc or "ROBBERY" in crime_desc or "STOLEN" in crime_desc:
        return "THEFT"
    elif "ASSAULT" in crime_desc or "BATTERY" in crime_desc or "BRANDISH" in crime_desc:
        return "ASSAULT"
    # Diğer kategoriler buraya eklenir
    else:
        return "DİĞER"

# Kategorileri ekleyin
df["CATEGORY"] = df["CRM_CD_DESC"].apply(categorize)

# Tarih ve kategori bazlı suç sayısı
grouped = df.groupby(["DATE", "CATEGORY"]).size().unstack(fill_value=0)

# HIRSIZLIK kategorisini çekin
theft_data = grouped["THEFT"]

time_grouped = df.groupby(["DATE", "CATEGORY"]).size().unstack(fill_value=0)

time_grouped.plot(figsize=(14, 8), title="Crime Categories Over Time", xlabel="Date", ylabel="Crime Count")
plt.legend(title="Categories")
plt.grid()
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

# Zaman serisi çizimi
theft_data.plot(figsize=(12, 6), title="THEFT Suçlarının Zaman İçindeki Dağılımı", xlabel="Tarih", ylabel="HIRSIZLIK Sayısı")
plt.show()

# Hangi area'nın adını ve oradaki suç miktarını gösterir:
grouped_data = df.groupby(['AREA', 'AREA_NAME']).size().reset_index(name='Count')

##################################   MODEL OLUŞTURMA  ########################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

columns_to_use = ['DAY_OF_WEEK_OCC', 'TIME_OCC_HOUR', 'CRM_CD_DESC', 'LAT', 'LON']  # Varsayım
data = df[columns_to_use].dropna()

data = pd.get_dummies(data, columns=['CRM_CD_DESC'], drop_first=True)

X = data.drop(columns=['LAT', 'LON'])
y = data[['LAT', 'LON']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")

input_data = {'DAY_OF_WEEK_OCC': [2],
              'TIME_OCC_HOUR': [14],
              'CRM_CD_DESC': ['BURGLARY FROM VEHICLE']}

input_df = pd.DataFrame(input_data)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(input_df)
print(f"Tahmini Enlem ve Boylam: {prediction}")


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Test setindeki tahminleri yap
y_pred = model.predict(X_test)

# Performans metriklerini hesapla
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Performansı yazdır
print(f"Model Performansı:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

#################################################################################################
#################################################################################################

################################## eksik değerler ###############################################

# Eksik Değerlerin Yakalanması
#############################################

df.head()
# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()
# degiskenlerdeki eksik deger sayisi
df.isnull().sum()
# degiskenlerdeki tam deger sayisi
df.notnull().sum()
# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()
# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

missing_values_table(df, True)


########################################################################

# Eksik Değer Problemini Çözme
#############################################

missing_values_table(df)

###################
# Çözüm 1: Hızlıca silmek
###################
#df.dropna().shape

###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################

df["VICT_AGE"].fillna(df["VICT_AGE"].mean()).isnull().sum()
df["VICT_AGE"].fillna(df["VICT_AGE"].median()).isnull().sum()
df["VICT_AGE"].fillna(0).isnull().sum()

# df.apply(lambda x: x.fillna(x.mean()), axis=0)

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

df["Embarked"].fillna("missing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()




#tablolar

pivot_table = df.pivot_table(index='CRM_CD_DESC', values='DR_NO', aggfunc='count').rename(columns={'DR_NO': 'OLAY_SAYISI'})
total_count = pivot_table['OLAY_SAYISI'].sum()
pivot_table['ORAN'] = (pivot_table['OLAY_SAYISI'] / total_count) * 100
pivot_table = pivot_table.sort_values(by='OLAY_SAYISI', ascending=False)
print(pivot_table)

#####suç türlerine göre tablo####

# Suç türlerine göre olay sayısını hesaplama
crime_counts = df['CRM_CD_DESC'].value_counts().reset_index()
crime_counts.columns = ['SUÇ_TÜRÜ', 'OLAY_SAYISI']

# Toplam olay sayısını hesaplama
total_crimes = crime_counts['OLAY_SAYISI'].sum()

# Oranları hesaplama ve tabloya ekleme
crime_counts['ORAN'] = (crime_counts['OLAY_SAYISI'] / total_crimes) * 100

# Olay sayısına göre tabloyu sıralama
crime_counts = crime_counts.sort_values(by='OLAY_SAYISI', ascending=False)

# İlk birkaç satırı görüntüleme
print(crime_counts.head())

import matplotlib.pyplot as plt
import matplotlib.cm as cm

############################# ilk 10 suçun grafiği ##########

# En sık görülen ilk 10 suç türünü seçme
top_10_crimes = df['CRM_CD_DESC'].value_counts().head(10)


# Grafik oluşturma

plt.figure(figsize=(10, 6))
top_10_crimes.plot(kind='bar')
plt.title("En Sık Görülen İlk 10 Suç")
plt.xlabel("Suç Türü")
plt.ylabel("Olay Sayısı")
plt.xticks(rotation=45)
plt.show()

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
plt.figure(figsize=(10, 6))
top_10_crimes.plot(kind='barh', color="gray")
plt.title("En Sık Görülen İlk 10 Suç")
plt.xlabel("Olay Sayısı")
plt.ylabel("Suç Türü")
plt.gca().invert_yaxis()  # En yüksekten en aza doğru sıralamak için ekseni ters çeviriyoruz
plt.show()

###########################

###### ilk 10 suçun grafiği yıllara göre #####################################################################

import pandas as pd
import matplotlib.pyplot as plt


# İlk 10 suçu bul
top_10_crimes_index = df['CRM_CD_DESC'].value_counts().head(10).index

# Her suç için ayrı grafik oluştur
for crime in top_10_crimes_index:
    # Suç filtresi ve yıllık olay sayısı
    yearly_data = df[df['CRM_CD_DESC'] == crime].groupby('YEAR_OCC').size()

    # Yeni bir sayfa (figure) oluştur
    plt.figure(figsize=(10, 6))
    yearly_data.plot(kind='bar', color='gray', width=0.3)
    plt.title(f"{crime} Suçunun Yıllara Göre Dağılımı")
    plt.xlabel("Yıl")
    plt.ylabel("Olay Sayısı")
    plt.xticks(rotation=45)
    plt.show()

################################################################################################

###### ilk 10 suçun grafiği aylara göre #####################################################################


import pandas as pd
import matplotlib.pyplot as plt


# İlk 10 suçu bul
top_10_crimes_index = df['CRM_CD_DESC'].value_counts().head(10).index

# Her suç için ayrı grafik oluştur
for crime in top_10_crimes_index:
    # Suç filtresi ve aylık olay sayısı
    monthly_data = df[df['CRM_CD_DESC'] == crime].groupby('MONTH_OCC').size()

    # Yeni bir sayfa (figure) oluştur
    plt.figure(figsize=(10, 6))
    monthly_data.plot(kind='bar', color='gray', width=0.3)  # Bar genişliği ayarlanmış
    plt.title(f"{crime} Suçunun Aylara Göre Dağılımı")
    plt.xlabel("Ay")
    plt.ylabel("Olay Sayısı")
    plt.xticks(rotation=45)
    plt.show()

#################################################################################

###### ilk 10 suçun grafiği günlere göre #####################################################################

import pandas as pd
import matplotlib.pyplot as plt


# İlk 10 suçu bul
top_10_crimes_index = df['CRM_CD_DESC'].value_counts().head(10).index

# Gün sırası tanımla
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# DAY_NAME_OCC kolonunu kategorik olarak ayarla
df['DAY_NAME_OCC'] = pd.Categorical(df['DAY_NAME_OCC'], categories=day_order, ordered=True)

# Her suç için ayrı grafik oluştur
for crime in top_10_crimes_index:
    # Suç filtresi ve günlük olay sayısı
    daily_data = df[df['CRM_CD_DESC'] == crime].groupby('DAY_NAME_OCC').size().reindex(day_order)

    # Yeni bir sayfa (figure) oluştur
    plt.figure(figsize=(10, 6))
    daily_data.plot(kind='bar', color='gray', width=0.6)  # Bar genişliği ayarlanmış
    plt.title(f"{crime} Suçunun Günlere Göre Dağılımı")
    plt.xlabel("Gün")
    plt.ylabel("Olay Sayısı")
    plt.xticks(rotation=45)
    plt.show()

#######################################################################



###### ilk 10 suçun grafiği saatlere göre #####################################################################


# İlk 10 suçu bul
top_10_crimes_index = df['CRM_CD_DESC'].value_counts().head(10).index

# Her suç için ayrı grafik oluştur
for crime in top_10_crimes_index:
    # Suç filtresi ve saatlik olay sayısı
    hourly_data = df[df['CRM_CD_DESC'] == crime].groupby('TIME_OCC_HOUR').size()

    # Yeni bir sayfa (figure) oluştur
    plt.figure(figsize=(10, 6))
    hourly_data.plot(kind='bar', color='gray', width=0.6)  # Bar genişliği ayarlanmış
    plt.title(f"{crime} Suçunun Saatlere Göre Dağılımı")
    plt.xlabel("Saat")
    plt.ylabel("Olay Sayısı")
    plt.xticks(rotation=45)
    plt.show()

######################################################################



#################### yıllara göre çizgi grafik #####################


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# YEAR_OCC sütununu tam sayıya dönüştür
df['YEAR_OCC'] = df['YEAR_OCC'].astype(int)

# İlk 10 suçu belirleme
top_10_crimes = df['CRM_CD_DESC'].value_counts().head(10).index.tolist()

# Grafik oluşturma
plt.figure(figsize=(20, 10))

# Her suç türü için yıllık olay sayısını çiz
for crime in top_10_crimes:
    # Yıllık olay sayılarını hesapla
    yearly_counts = df[df['CRM_CD_DESC'] == crime].groupby('YEAR_OCC').size()

    # Çizgi grafiği ekle
    plt.plot(yearly_counts.index, yearly_counts.values, marker='o', label=crime)

# Eksen etiketlerini ve başlığı ekle, font boyutunu artırdık
plt.xlabel('Yıl', fontsize=14)
plt.ylabel('Olay Sayısı', fontsize=14)
plt.title('İlk 10 Suç Türünün Yıllara Göre Olay Sayısı', fontsize=16)

# Açıklamayı (legend) grafiğin altına almak ve font boyutunu artırmak için ayarlama yapın
plt.legend(title="Suç Türleri", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12, title_fontsize=14)

# X ekseninde tam sayıları kullanmak için ayar yapıyoruz
plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Izgara çizgileri, düzen ve gösterim
plt.grid(True)
plt.tight_layout()
plt.show()

#############################################################

#################### aylara göre çizgi grafik #####################

import matplotlib.pyplot as plt

# İlk 10 suçu belirleme
top_10_crimes = df['CRM_CD_DESC'].value_counts().head(10).index.tolist()

# Grafik oluşturma
plt.figure(figsize=(20, 10))

# Ay isimlerini belirleme
ay_isimleri = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']

# Her suç türü için aylık olay sayısını çiz
for crime in top_10_crimes:
    # Aylık olay sayılarını hesapla
    monthly_counts = df[df['CRM_CD_DESC'] == crime].groupby('MONTH_OCC').size()

    # Çizgi grafiği ekle
    plt.plot(monthly_counts.index, monthly_counts.values, marker='o', label=crime)

# Eksen etiketlerini ve başlığı ekle, font boyutunu artırdık
plt.xlabel('Ay', fontsize=14)
plt.ylabel('Olay Sayısı', fontsize=14)
plt.title('İlk 10 Suç Türünün Aylara Göre Olay Sayısı', fontsize=16)

# X eksenini ay isimleri ile etiketle
plt.xticks(ticks=range(1, 13), labels=ay_isimleri, rotation=45)

# Açıklamayı (legend) grafiğin altına almak ve font boyutunu artırmak için ayarlama yapın
plt.legend(title="Suç Türleri", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12, title_fontsize=14)

# Izgara çizgileri, düzen ve gösterim
plt.grid(True)
plt.tight_layout()
plt.show()



#################### günlere göre çizgi grafik #####################

import matplotlib.pyplot as plt

# İlk 10 suçu belirleme
top_10_crimes = df['CRM_CD_DESC'].value_counts().head(10).index.tolist()

# Grafik oluşturma
plt.figure(figsize=(14, 10))

# Gün isimlerini belirleme ve sıralama
gun_sirasi = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']

# Her suç türü için günlük olay sayısını çiz
for crime in top_10_crimes:
    # Günlük olay sayılarını hesapla ve sıralama düzenine göre yeniden diz
    daily_counts = df[df['CRM_CD_DESC'] == crime].groupby('DAY_NAME_OCC').size().reindex(gun_sirasi)

    # Çizgi grafiği ekle
    plt.plot(gun_sirasi, daily_counts.values, marker='o', label=crime)

# Eksen etiketlerini ve başlığı ekle, font boyutunu artırdık
plt.xlabel('Gün', fontsize=14)
plt.ylabel('Olay Sayısı', fontsize=14)
plt.title('İlk 10 Suç Türünün Günlere Göre Olay Sayısı', fontsize=16)

# X eksenindeki gün isimlerinin okunabilirliğini artırma
plt.xticks(rotation=45)

# Açıklamayı (legend) grafiğin altına almak ve font boyutunu artırmak için ayarlama yapın
plt.legend(title="Suç Türleri", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12, title_fontsize=14)

# Izgara çizgileri, düzen ve gösterim
plt.grid(True)
plt.tight_layout()
plt.show()


##################################

top_10_crimes = df['CRM_CD_DESC'].value_counts().head(10).index

# Her suç için ayrı grafik oluştur
for crime in top_10_crimes:
    # Suç filtresi ve cinsiyete göre olay sayısı
    gender_data = df[df['CRM_CD_DESC'] == crime].groupby('VICT_SEX').size()

    # Yeni bir sayfa (figure) oluştur
    plt.figure(figsize=(10, 6))
    gender_data.plot(kind='bar', color=['#1f77b4', '#ff7f0e'], width=0.6)  # İki renk seçeneği eklendi
    plt.title(f"{crime} Suçunun Cinsiyete Göre Dağılımı")
    plt.xlabel("Cinsiyet")
    plt.ylabel("Olay Sayısı")
    plt.xticks(rotation=0)
    plt.show()




######################################################################

import folium
from folium.plugins import HeatMapWithTime
import pandas as pd

# Veri hazırlığı: Aylık olay sayısı için
df['YEAR_MONTH_OCC'] = df['DATE_OCC'].dt.to_period('M')  # Eğer tarih sütunu henüz yoksa bunu oluşturun
top_10_crimes = df['CRM_CD_DESC'].value_counts().head(10).index.tolist()
top_crimes_data = df[df['CRM_CD_DESC'].isin(top_10_crimes)]

# Harita oluştur ve daha yakın bir merkezi ayarla
map_center = [top_crimes_data['LAT'].mean(), top_crimes_data['LON'].mean()]
m = folium.Map(location=map_center, zoom_start=11)  # Zoom seviyesini 12 yaparak daha yakınlaştırıyoruz

# Aylık veri hazırlığı
months = sorted(top_crimes_data['YEAR_MONTH_OCC'].unique())
heat_data = []

for month in months:
    month_data = top_crimes_data[top_crimes_data['YEAR_MONTH_OCC'] == month]
    heat_data.append([[row['LAT'], row['LON'], 1] for index, row in month_data.iterrows()])

# Son durum için tüm veriyi içeren bir yoğunluk katmanı ekleme
all_data_layer = [[row['LAT'], row['LON'], 1] for index, row in top_crimes_data.iterrows()]
heat_data.append(all_data_layer)

# HeatMapWithTime katmanını kullanarak aylık stop-motion etkisi ve son durumda tüm suçları gösterme
HeatMapWithTime(heat_data, radius=5, auto_play=True, max_opacity=0.8, speed_step=0.1).add_to(m)

# Haritayı HTML dosyası olarak kaydet
m.save("top_10_crimes_monthly_stop_motion_map.html")

# Haritayı görüntüle
m

###############################################################


# Tahmin yap butonu
if st.button("Tahmini Çalıştır"):
    # Kullanıcı girdilerini dönüştürme
    day_encoded = le_day.transform([day])[0]
    crime_encoded = le_crime.transform([crime])[0]
    example = [[day_encoded, time_hour, crime_encoded]]

    # Tahminler
    predicted_lat = model_lat.predict(example)
    predicted_lon = model_lon.predict(example)

    # Sonuçları gösterme
    st.success(f"Tahmin edilen lokasyon: LAT = {predicted_lat[0]:.6f}, LON = {predicted_lon[0]:.6f}")

    # Harita üzerinde gösterim
    st.map(pd.DataFrame({'lat': [predicted_lat[0]], 'lon': [predicted_lon[0]]}))

