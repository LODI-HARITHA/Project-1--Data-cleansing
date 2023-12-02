import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('C:/Users/DELL PC/Downloads/Wind_turbine.csv')
df.describe()
df.info()
df.head()

# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
df.mean()
df.median()
df.mode()

# Measures of Dispersion / Second moment business decision
df.var() # variance
df.std() # standard deviation 

# Third moment business decision
df.skew()

# Fourth moment business decision
df.kurt()



############ plot histograms #############
plt.hist(df.Wind_speed,color='blue') #histogram
plt.hist(df.Power,color='blue')
plt.hist(df.Nacelle_ambient_temperature,color='blue')
plt.hist(df.Generator_bearing_temperature,color='blue')
plt.hist(df.Gear_oil_temperature,color='blue')
plt.hist(df.Ambient_temperature,color='blue')
plt.hist(df.Rotor_Speed,color='blue')
plt.hist(df.Nacelle_temperature,color='blue')
plt.hist(df.Bearing_temperature,color='blue')
plt.hist(df.Generator_speed,color='blue')
plt.hist(df.Yaw_angle,color='blue')
plt.hist(df.Wind_direction,color='blue')
plt.hist(df.Wheel_hub_temperature,color='blue')
plt.hist(df.Gear_box_inlet_temperature,color='blue')



## Data Preprocessing
## Missing values
# check for count of NA'sin each column
df.isna().sum()

# for Mean,Meadian,Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer
# Median Imputer

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df["Wind_speed"] = pd.DataFrame(median_imputer.fit_transform(df[['Wind_speed']]))
df["Wind_speed"].isnull().sum()

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df["Power"] = pd.DataFrame(median_imputer.fit_transform(df[['Power']]))
df["Power"].isnull().sum()

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df["Nacelle_ambient_temperature"] = pd.DataFrame(median_imputer.fit_transform(df[['Nacelle_ambient_temperature']]))
df["Nacelle_ambient_temperature"].isnull().sum()

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df["Nacelle_temperature "] = pd.DataFrame(median_imputer.fit_transform(df[['Nacelle_temperature ']]))
df["Nacelle_temperature "].isnull().sum()

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df["Generator_speed"] = pd.DataFrame(median_imputer.fit_transform(df[['Generator_speed']]))
df["Generator_speed"].isnull().sum()

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df["Yaw_angle"] = pd.DataFrame(median_imputer.fit_transform(df[['Yaw_angle']]))
df["Yaw_angle"].isnull().sum()

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df["Gear_box_inlet_temperature"] = pd.DataFrame(median_imputer.fit_transform(df[['Gear_box_inlet_temperature']]))
df["Gear_box_inlet_temperature"].isnull().sum()




## Handling duplicates

#Identify duplicates records in the data
duplicate = df.duplicated()
sum(duplicate)




## Outlier Analysis/Treatment

sns.boxplot(df.Wind_speed );plt.title('Boxplot');plt.show() # Outliers present
sns.boxplot(df.Power );plt.title('Boxplot');plt.show() # Outliers present
sns.boxplot(df.Nacelle_ambient_temperature);plt.title('Boxplot');plt.show() # Outliers present
sns.boxplot(df.Generator_bearing_temperature);plt.title('Boxplot');plt.show() # Outliers present
sns.boxplot(df.Gear_oil_temperature);plt.title('Boxplot');plt.show() # outliers present
sns.boxplot(df.Ambient_temperature);plt.title('Boxplot');plt.show() # Outliers present
sns.boxplot(df.Rotor_Speed);plt.title('Boxplot');plt.show() # outliers present
sns.boxplot(df.Nacelle_temperature);plt.title('Boxplot');plt.show() # Outliers present
sns.boxplot(df.Bearing_temperature);plt.title('Boxplot');plt.show() # Outliers present
sns.boxplot(df.Generator_speed);plt.title('Boxplot');plt.show()  # Outliers present
sns.boxplot(df.Yaw_angle);plt.title('Boxplot');plt.show() # Outliers present
sns.boxplot(df.Wind_direction);plt.title('Boxplot');plt.show() # Outliers present
sns.boxplot(df.Wheel_hub_temperature);plt.title('Boxplot');plt.show() # No outliers
sns.boxplot(df.Gear_box_inlet_temperature);plt.title('Boxplot');plt.show() # Outliers present




####################### 2.Replace ############################
# Detection of Outliers 
IQR = df['Wind_speed'].quantile(0.75) - df['Wind_speed'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Wind_speed'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Wind_speed'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Wind_speed'] > upper_limit, upper_limit,
                                         np.where(df['Wind_speed'] < lower_limit, lower_limit,
                                                  df['Wind_speed'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

# Detection of Outliers 
IQR = df['Power'].quantile(0.75) - df['Power'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Power'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Power'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Power'] > upper_limit, upper_limit,
                                         np.where(df['Power'] < lower_limit, lower_limit,
                                                  df['Power'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

# Detection of Outliers 
IQR = df['Nacelle_ambient_temperature'].quantile(0.75) - df['Nacelle_ambient_temperature'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Nacelle_ambient_temperature'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Nacelle_ambient_temperature'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Nacelle_ambient_temperature'] > upper_limit, upper_limit,
                                         np.where(df['Nacelle_ambient_temperature'] < lower_limit, lower_limit,
                                                  df['Nacelle_ambient_temperature'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

# Detection of Outliers 
IQR = df['Generator_bearing_temperature'].quantile(0.75) - df['Generator_bearing_temperature'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Generator_bearing_temperature'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Generator_bearing_temperature'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Generator_bearing_temperature'] > upper_limit, upper_limit,
                                         np.where(df['Generator_bearing_temperature'] < lower_limit, lower_limit,
                                                  df['Generator_bearing_temperature'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

# Detection of Outliers 
IQR = df['Gear_oil_temperature'].quantile(0.75) - df['Gear_oil_temperature'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Gear_oil_temperature'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Gear_oil_temperature'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Gear_oil_temperature'] > upper_limit, upper_limit,
                                         np.where(df['Gear_oil_temperature'] < lower_limit, lower_limit,
                                                  df['Gear_oil_temperature'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

# Detection of Outliers 
IQR = df['Ambient_temperature'].quantile(0.75) - df['Ambient_temperature'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Ambient_temperature'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Ambient_temperature'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Ambient_temperature'] > upper_limit, upper_limit,
                                         np.where(df['Ambient_temperature'] < lower_limit, lower_limit,
                                                  df['Ambient_temperature'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

# Detection of Outliers 
IQR = df['Rotor_Speed'].quantile(0.75) - df['Rotor_Speed'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Rotor_Speed'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Rotor_Speed'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Rotor_Speed'] > upper_limit, upper_limit,
                                         np.where(df['Rotor_Speed'] < lower_limit, lower_limit,
                                                  df['Rotor_Speed'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

# Detection of Outliers 
IQR = df['Nacelle_temperature'].quantile(0.75) - df['Nacelle_temperature'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Nacelle_temperature'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Nacelle_temperature'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Nacelle_temperature'] > upper_limit, upper_limit,
                                         np.where(df['Nacelle_temperature'] < lower_limit, lower_limit,
                                                  df['Nacelle_temperature'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

# Detection of Outliers 
IQR = df['Bearing_temperature'].quantile(0.75) - df['Bearing_temperature'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Bearing_temperature'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Bearing_temperature'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Bearing_temperature'] > upper_limit, upper_limit,
                                         np.where(df['Bearing_temperature'] < lower_limit, lower_limit,
                                                  df['Bearing_temperature'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

# Detection of Outliers 
IQR = df['Generator_speed'].quantile(0.75) - df['Generator_speed'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Generator_speed'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Generator_speed'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Generator_speed'] > upper_limit, upper_limit,
                                         np.where(df['Generator_speed'] < lower_limit, lower_limit,
                                                  df['Generator_speed'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

# Detection of Outliers 
IQR = df['Yaw_angle'].quantile(0.75) - df['Yaw_angle'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Yaw_angle'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Yaw_angle'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Yaw_angle'] > upper_limit, upper_limit,
                                         np.where(df['Yaw_angle'] < lower_limit, lower_limit,
                                                  df['Yaw_angle'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

# Detection of Outliers 
IQR = df['Wind_direction'].quantile(0.75) - df['Wind_direction'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Wind_direction'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Wind_direction'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Wind_direction'] > upper_limit, upper_limit,
                                         np.where(df['Wind_direction'] < lower_limit, lower_limit,
                                                  df['Wind_direction'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

# Detection of Outliers 
IQR = df['Wheel_hub_temperature'].quantile(0.75) - df['Wheel_hub_temperature'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Wheel_hub_temperature'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Wheel_hub_temperature'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Wheel_hub_temperature'] > upper_limit, upper_limit,
                                         np.where(df['Wheel_hub_temperature'] < lower_limit, lower_limit,
                                                  df['Wheel_hub_temperature'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()

# Detection of Outliers 
IQR = df['Gear_box_inlet_temperature'].quantile(0.75) - df['Gear_box_inlet_temperature'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['Gear_box_inlet_temperature'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['Gear_box_inlet_temperature'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['Gear_box_inlet_temperature'] > upper_limit, upper_limit,
                                         np.where(df['Gear_box_inlet_temperature'] < lower_limit, lower_limit,
                                                  df['Gear_box_inlet_temperature'])))
                                 





######## One-hot encode the categorical column #############
one_hot = pd.get_dummies(df['Failure_status'])

# Print the one-hot encoded dataframe
print(one_hot)





############### Distribution plot ################

sns.distplot(df['Power'], kde=False,label='Power')
sns.distplot(df['Wind_speed'], kde=False,label='Wind_speed')
sns.distplot(df['Nacelle_ambient_temperature'], kde=False,label='Nacelle_ambient_temperature')
sns.distplot(df['Generator_bearing_temperature'], kde=False,label='Generator_bearing_temperature')
sns.distplot(df['Gear_oil_temperature'], kde=False,label='Gear_oil_temperature')
sns.distplot(df['Ambient_temperature'], kde=False,label='Ambient_temperature')
sns.distplot(df['Rotor_Speed'], kde=False,label='Rotor_Speed')
sns.distplot(df['Nacelle_temperature'], kde=False,label='Nacelle_temperature')
sns.distplot(df['Generator_speed'], kde=False,label='Generator_speed')
sns.distplot(df['Yaw_angle'], kde=False,label='Yaw_angle')
sns.distplot(df['Wind_direction'], kde=False,label='Wind_direction')
sns.distplot(df['Wheel_hub_temperature'], kde=False,label='Wheel_hub_temperature')
sns.distplot(df['Gear_box_inlet_temperature'], kde=False,label='Gear_box_inlet_temperature')

plt.legend()




################ FINDING ZERO VARIANCE ################
print(df['Power'].var())
print(df['Wind_speed'].var())
print(df['Nacelle_ambient_temperature'].var())
print(df['Generator_bearing_temperature'].var())
print(df['Gear_oil_temperature'].var())
print(df['Ambient_temperature'].var())
print(df['Rotor_Speed'].var())
print(df['Nacelle_temperature'].var())
print(df['Generator_speed'].var())
print(df['Yaw_angle'].var())
print(df['Wind_direction'].var())
print(df['Wheel_hub_temperature'].var())
print(df['Gear_box_inlet_temperature'].var())




################# Standardization ################
# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std()) 
    return(x)

df1_norm = norm_func(df)
df1_norm.describe()

#Normalization
# or denominator (i.max()-i.min())
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min()) # or denominator (i.max()-i.min())
    return(x)

df2_norm = norm_func(df)
df2_norm.describe()

#Normal Quantile-Quantile Plot
import scipy.stats as stats
import pylab





################## Checking Whether data is normally distributed #################

stats.probplot(df['Wind_speed'], dist='norm',plot=pylab);plt.show() #pylab is visual representation

stats.probplot(df['Power'], dist='norm',plot=pylab);plt.show() # it is normally distributed

stats.probplot(df['Nacelle_ambient_temperature'], dist='norm',plot=pylab);plt.show() #pylab is visual representation

stats.probplot(df['Generator_bearing_temperature'], dist='norm',plot=pylab);plt.show() # it is normally distributed

stats.probplot(df['Gear_oil_temperature'], dist='norm',plot=pylab);plt.show() #pylab is visual representation

stats.probplot(df['Ambient_temperature'], dist='norm',plot=pylab);plt.show() # it is normally distributed

stats.probplot(df['Rotor_Speed'], dist='norm',plot=pylab);plt.show() #pylab is visual representation

stats.probplot(df['Nacelle_temperature'], dist='norm',plot=pylab);plt.show() # it is normally distributed

stats.probplot(df['Bearing_temperature'], dist='norm',plot=pylab);plt.show() #pylab is visual representation

stats.probplot(df['Generator_speed'], dist='norm',plot=pylab);plt.show() # it is normally distributed

stats.probplot(df['Yaw_angle'], dist='norm',plot=pylab);plt.show() #pylab is visual representation

stats.probplot(df['Wind_direction'], dist='norm',plot=pylab);plt.show() # it is normally distributed


stats.probplot(df['Wheel_hub_temperature'], dist='norm',plot=pylab);plt.show() #pylab is visual representation

stats.probplot(df['Gear_box_inlet_temperature'], dist='norm',plot=pylab);plt.show() # it is normally distributed

stats.probplot(df['Wheel_hub_temperature'], dist='norm',plot=pylab);plt.show() #pylab is visual representation

stats.probplot(df['Gear_box_inlet_temperature'], dist='norm',plot=pylab);plt.show() # it is normally distributed





###################### transformation ################### 
import numpy as np
stats.probplot(np.log(df['Wind_speed']),dist="norm",plot=pylab) #best transformation
df.describe()

#transformation to make Weight .gained..grams variable normal
stats.probplot(np.log(df['Power']),dist="norm",plot=pylab) #best transformation
df.describe()

stats.probplot(np.log(df['Nacelle_ambient_temperature']),dist="norm",plot=pylab) #best transformation
df.describe()

stats.probplot(np.log(df['Generator_bearing_temperature']),dist="norm",plot=pylab) #best transformation
df.describe()

stats.probplot(np.log(df['Gear_oil_temperature']),dist="norm",plot=pylab) #best transformation
df.describe()

stats.probplot(np.log(df['Ambient_temperature']),dist="norm",plot=pylab) #best transformation
df.describe()

stats.probplot(np.log(df['Rotor_Speed']),dist="norm",plot=pylab) #best transformation
df.describe()

stats.probplot(np.log(df['Nacelle_temperature']),dist="norm",plot=pylab) #best transformation
df.describe()

stats.probplot(np.log(df['Bearing_temperature']),dist="norm",plot=pylab) #best transformation
df.describe()

stats.probplot(np.log(df['Generator_speed']),dist="norm",plot=pylab) #best transformation
df.describe()

stats.probplot(np.log(df['Yaw_angle']),dist="norm",plot=pylab) #best transformation
df.describe()

stats.probplot(np.log(df['Wind_direction']),dist="norm",plot=pylab) #best transformation
df.describe()

stats.probplot(np.log(df['Wheel_hub_temperature']),dist="norm",plot=pylab) #best transformation
df.describe()

stats.probplot(np.log(df['Gear_box_inlet_temperature']),dist="norm",plot=pylab) #best transformation
df.describe()

#exploratory data analysis


#first moment business decision
df.mean()   
df.median()
df.mode()


#second moment business decision
#variance
df.var()
df.std()


#third moment business decision
#skewness
df.skew()

#Fourth moment business decision
#kurtosis
df.kurt()

plt.hist(df.Wind_speed,color='blue') #histogram
plt.hist(df.Power,color='blue')
plt.hist(df.Nacelle_ambient_temperature,color='blue')
plt.hist(df.Generator_bearing_temperature,color='blue')
plt.hist(df.Gear_oil_temperature,color='blue')
plt.hist(df.Ambient_temperature,color='blue')
plt.hist(df.Rotor_Speed,color='blue')
plt.hist(df.Nacelle_temperature,color='blue')
plt.hist(df.Bearing_temperature,color='blue')
plt.hist(df.Generator_speed,color='blue')
plt.hist(df.Yaw_angle,color='blue')
plt.hist(df.Wind_direction,color='blue')
plt.hist(df.Wheel_hub_temperature,color='blue')
plt.hist(df.Gear_box_inlet_temperature,color='blue')


from imblearn.over_sampling import SMOTE


x = df
y = x['Failure_status']
X = x.drop(["Failure_status"],axis=1)

smote = SMOTE()# Create the SMOTE object
X_resampled, y_resampled = smote.fit_resample(X, y)# Fit and transform the data

# Print the class distribution
print(y_resampled.value_counts()) 
X_resampled.shape

