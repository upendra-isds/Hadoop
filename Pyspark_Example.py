#export PYSPARK_PYTHON="/home/appsadm/anaconda2/bin/python"

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
#from impala.util import as_pandas
#from impala.dbapi import connect
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import sys
import os
os.environ["SPARK_HOME"] = "/opt/cloudera/parcels/CDH-5.6.0-1.cdh5.6.0.p0.45/lib/spark"
sys.path.append("/opt/cloudera/parcels/CDH-5.6.0-1.cdh5.6.0.p0.45/lib/spark/python/pyspark")
from pyspark import SparkConf, SparkContext
sconf = SparkConf().setAppName("Database Security Product Family").setMaster("yarn-client")
sc = SparkContext(conf = sconf)
from pyspark.sql import HiveContext
from pyspark.sql import SQLContext


hiveContext = HiveContext(sc)
sqlContext = SQLContext(sc)


train_query ='select country, purchase_last3years as Purchase, bookings, sr as Sr, sam_rsam, products as Products, discount as Discount, renewal_pc, lower(segment) as segment_lower, tenure as Tenure, early_adopter as Early_Adopter, max_level_support as highlevel_support, employees_total as Employees, revenue_score, subsidiary_ind, manufacturing_ind, public_private_ind, major_industry_category, percent_growth_sales_3yr, number_sales_escalations, contract_length, renewal_flag, reseller_indicator, distributer_indicator, partner_indicator, customer_segments, system_security, app_n_change, total_protection_service, web_gateway, web_saas, atd, tie, host_dlp, network_dlp, endpoint_encryption, siem, scanalert, policy_compliance, server_security, security_for_storage, security_for_mssharepoint, ips, database_security as label from mcafee_masterdatabase.internall_final_pf_id_attributes_2014 where lower(segment) <> "smb"'

hqlChurnRDD = hiveContext.sql(train_query)

df = hqlChurnRDD.toPandas()

cols = df.columns
X_df = df[cols[:-1]]
Y_df = df[cols[-1]]

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
		
X_df_encoded = MultiColumnLabelEncoder(columns = ['country','sam_rsam','segment_lower','highlevel_support','subsidiary_ind','manufacturing_ind','public_private_ind','major_industry_category','renewal_flag','reseller_indicator','distributer_indicator','partner_indicator','customer_segments']).fit_transform(X_df)

encoder = OneHotEncoder(categorical_features=[0,4,8,11,14,15,16,17,21,22,23,24,25], sparse=False).fit(X_df_encoded)
X_one_hot = encoder.transform(X_df_encoded)
#X_one_hot


clf = GradientBoostingClassifier(max_depth=8,n_estimators=100).fit(X_one_hot, Y_df)

test_query = 'select in16.country, in16.purchase_last3years as Purchase, in16.bookings, in16.sr as Sr, in16.sam_rsam, in16.products as Products, in16.discount as Discount, in16.renewal_pc, lower(in16.segment) as segment_lower, in16.tenure as Tenure, in16.early_adopter as Early_Adopter, in16.max_level_support as highlevel_support, in16.employees_total as Employees, in16.revenue_score, in16.subsidiary_ind, in16.manufacturing_ind, in16.public_private_ind, in16.major_industry_category, in16.percent_growth_sales_3yr, in16.number_sales_escalations, in16.contract_length, in16.renewal_flag, in16.reseller_indicator, in16.distributer_indicator, in16.partner_indicator, in16.customer_segments, in16.system_security, in16.app_n_change, in16.total_protection_service, in16.web_gateway, in16.web_saas, in16.atd, in16.tie, in16.host_dlp, in16.network_dlp, in16.endpoint_encryption, in16.siem, in16.scanalert, in16.policy_compliance, in16.server_security, in16.security_for_storage, in16.security_for_mssharepoint, in16.ips, in16.mdm_parent_id, in16.database_security as label from  mcafee_masterdatabase.internall_final_pf_id_attributes_2016 IN16 WHERE lower(in16.segment) <> "smb" and in16.database_security = 0'

testRDD = hiveContext.sql(test_query)


df_test = testRDD.toPandas()

cols_test = df_test.columns
#df.drop([col for col in ['country','Purchase'] if col in df_test], axis=1, inplace=True, errors = 'ignore')
#X_df = df.drop('column_name', 1)
X_df1 = df_test[cols_test[:-2]]
Y_df1 = df_test[cols_test[:-2:-1]]

X_df_encoded1 = MultiColumnLabelEncoder(columns = ['country','sam_rsam','segment_lower','highlevel_support','subsidiary_ind','manufacturing_ind','public_private_ind','major_industry_category','renewal_flag','reseller_indicator','distributer_indicator','partner_indicator','customer_segments']).fit_transform(X_df1)

encoder1 = OneHotEncoder(categorical_features=[0,4,8,11,14,15,16,17,21,22,23,24,25], sparse=False).fit(X_df_encoded1)
X_one_hot1 = encoder1.transform(X_df_encoded1)
#X_one_hot1

predicted=clf.predict(X_one_hot1)
predicted_proba=clf.predict_proba(X_one_hot1)
#mergedlist = list(set(predicted + predicted_proba))
predicted_df = pd.DataFrame(predicted)
predicted_proba_df=pd.DataFrame(predicted_proba)
predicted_final_df=pd.concat([predicted_df,predicted_proba_df],axis=1)
predicted_final_df.columns=['Prediction','FalseProbability','TrueProbability']

predicted_final_df1=pd.concat([df_test['mdm_parent_id'],predicted_final_df],axis=1)

outPutDir = "/home/appsadm/mcafee_masterdatabase/etl/Prediction/Pred_code/Product_family_prod/Results/"
finaloutPutFilePath	= outPutDir + "database_security_prediction_GBT.csv"

predicted_final_df1.to_csv(finaloutPutFilePath, sep=',')