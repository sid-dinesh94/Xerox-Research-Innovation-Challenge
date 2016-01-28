import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
import csv
from sys import argv

#kfeatures = ['MEDIAN_L6', 'MEDIAN_L7', 'MEDIAN_L12', 'MEDIAN_L15', 'MEDIAN_L20', 'MEDIAN_L21', 'MEDIAN_L22', 'MEDIAN_V3', 'MEDIAN_V4','STD_L2', 'STD_L3', 'STD_L4', 'STD_L6', 'STD_L12', 'STD_L15', 'STD_L20', 'STD_L22', 'STD_L23', 'STD_V3', 'STD_V5', 'STD_V6', 'MIN_L2', 'MIN_L3', 'MIN_L6', 'MIN_L11','MIN_L21', 'MIN_V1', 'MIN_V2', 'MIN_V5', 'MAX_L12', 'MAX_L15', 'MAX_L20', 'MAX_L21', 'MAX_L22', 'MAX_L23', 'MAX_V3', 'MAX_V4', 'MAX_V6', 'COUNT_L15', 'COUNT_L20']
kfeaturestrain = ['ID', 'TIME', 'AGE', 'MEDIAN_L2', 'MEDIAN_L6', 'MEDIAN_L7', 'MEDIAN_L11', 'MEDIAN_L12', 'MEDIAN_L15', 'MEDIAN_L20', 'MEDIAN_L21', 'MEDIAN_L22', 'MEDIAN_V1', 'MEDIAN_V2', 'MEDIAN_V3', 'MEDIAN_V4', 'MEDIAN_V5', 'STD_L2', 'STD_L3', 'STD_L4', 'STD_L6', 'STD_L7', 'STD_L9', 'STD_L12', 'STD_L15', 'STD_L20', 'STD_L21', 'STD_L22', 'STD_L23', 'STD_L25', 'STD_V1', 'STD_V2', 'STD_V3', 'STD_V4', 'STD_V5', 'STD_V6', 'MIN_L2', 'MIN_L3', 'MIN_L4', 'MIN_L6', 'MIN_L11', 'MIN_L16', 'MIN_L21', 'MIN_L25', 'MIN_V1', 'MIN_V2', 'MIN_V5', 'MIN_V6', 'MAX_L2', 'MAX_L3', 'MAX_L12', 'MAX_L15', 'MAX_L20', 'MAX_L21', 'MAX_L22', 'MAX_L23', 'MAX_V3', 'MAX_V4', 'MAX_V6', 'COUNT_L1', 'COUNT_L2', 'COUNT_L3', 'COUNT_L6', 'COUNT_L15', 'COUNT_L16', 'COUNT_L20', 'COUNT_L23', 'ICU', 'LABEL']
kfeaturestrain = ['ID', 'TIME', 'AGE', 'MEDIAN_L2', 'MEDIAN_L6', 'MEDIAN_L7', 'MEDIAN_L11', 'MEDIAN_L12', 'MEDIAN_L15', 'MEDIAN_L20', 'MEDIAN_L21', 'MEDIAN_L22', 'MEDIAN_V1', 'MEDIAN_V2', 'MEDIAN_V3', 'MEDIAN_V4', 'MEDIAN_V5', 'STD_L2', 'STD_L3', 'STD_L4', 'STD_L6', 'STD_L7', 'STD_L9', 'STD_L12', 'STD_L15', 'STD_L20', 'STD_L21', 'STD_L22', 'STD_L23', 'STD_L25', 'STD_V1', 'STD_V2', 'STD_V3', 'STD_V4', 'STD_V5', 'STD_V6', 'MIN_L2', 'MIN_L3', 'MIN_L4', 'MIN_L6', 'MIN_L11', 'MIN_L16', 'MIN_L21', 'MIN_L25', 'MIN_V1', 'MIN_V2', 'MIN_V5', 'MIN_V6', 'MAX_L2', 'MAX_L3', 'MAX_L12', 'MAX_L15', 'MAX_L20', 'MAX_L21', 'MAX_L22', 'MAX_L23', 'MAX_V3', 'MAX_V4', 'MAX_V6', 'COUNT_L1', 'COUNT_L2', 'COUNT_L3', 'COUNT_L6', 'COUNT_L15', 'COUNT_L16', 'COUNT_L20', 'COUNT_L23', 'ICU']

def read_train():
	data_age = pd.read_csv("id_age_train.csv", sep=",")
	data_vitals = pd.read_csv("id_time_vitals_train.csv", sep=",")
	data_labels = pd.read_csv("id_label_train.csv", sep=",").set_index('ID')
	data_labs = pd.read_csv("id_time_labs_train.csv", sep = ",")
	data_timeseries = pd.merge(data_labs, data_vitals)
	patients = data_timeseries['ID']
	patient_ids = list(data_timeseries['ID'].unique())
	labels = np.zeros(len(patients))
	for p in patient_ids:
		if int(data_labels.ix[p]['LABEL'])==1:
			data_id = data_timeseries[data_timeseries['ID']==p]
			indices = data_id[data_id['ICU']==1].index
			num = len(indices)
			for i in indices:
				labels[i]=1
	data_timeseries['LABEL'] = labels
	data = pd.merge( data_age, data_timeseries, on='ID')
	data.loc[445378,'L20'] = 20.5267
	return data

def read_test(testvitals, testlabs, testage, val = False):
	data_age_test = pd.read_csv(testage, sep=",")
	data_vitals_test = pd.read_csv(testvitals, sep=",")
	data_labs_test = pd.read_csv(testlabs, sep = ",")
	data_timeseries_test = pd.merge(data_labs_test, data_vitals_test)
	#patient_ids_test = list(data_timeseries_test['ID'].unique())
	data_test = pd.merge( data_age_test, data_timeseries_test, on='ID')
	return data_test

def preprocess(data, normal):
	data['V6'] = data['V6'].apply(lambda x: 80 if x<80 else 112 if x>112 else x)
	data['V5'] = data['V5'].apply(lambda x: 100 if x>100 else x if x>0 else np.nan )
	data['V4'] = data['V4'].apply(lambda x: x if x>0 else np.nan )
	data['V3'] = data['V3'].apply(lambda x: x if (x>30 and x<220) else np.nan )
	data['V2'] = data['V2'].apply(lambda x: x if (x>15 and x<200) else np.nan )
	data['V1'] = data['V1'].apply(lambda x: x if (x>30 and x<300) else np.nan )
	data['L1'] = data['L1'].apply(lambda x: x if (x>0 and x<14) else np.nan)
	data['L2'] = data['L2'].apply(lambda x: 132 if x>132 else x if x>0 else np.nan )
	data[['L3', 'L4', 'L5', 'L6', 'L16','L17', 'L13', 'L14', 'L21', 'L22', 'L24', 'L25']] = data[['L3', 'L4', 'L5', 'L6', 'L16','L17', 'L13', 'L14', 'L21', 'L22', 'L24', 'L25']].applymap(lambda x: x if x>0 else np.nan )
	data['L7'] = data['L7'].apply(lambda x: x if x<700 else 700 if x>700 else np.nan)
	data['L8'] = data['L8'].apply(lambda x: x if x<200 else 200 if x>200 else np.nan)
	data['L9'] = data['L9'].apply(lambda x: x if x<100 else x/1000)
	data['L10'] = data['L10'].apply(lambda x: x if (x>0 and x<100) else np.nan)
	data['L11'] = data['L11'].apply(lambda x: x if (x>0 and x<2000) else 2000 if x>2000 else np.nan)
	data['L12'] = data['L12'].apply(lambda x: x if x<5 else 5+(x-5)/10)
	data['L15'] =data['L15'].apply(lambda x: x if (x>0 and x<20) else 20 if x>20 else np.nan)
	data['L18'] =data['L18'].apply(lambda x: x if x<1000 else 1000 if x>1000 else np.nan)
	data['L19'] =data['L19'].apply(lambda x: x if x<800 else 800 if x>800 else np.nan)
	data['L20'] =data['L20'].apply(lambda x: x/100 if x>1000 else x if x>0 else np.nan)
	data['L23'] =data['L23'].apply(lambda x: 3000 if x>3000 else x if x>0 else np.nan)
	data_grouped = data.groupby(['ID','TIME']).mean()
	patient_ids = data['ID'].unique()
	for patient in patient_ids:
	    data_grouped.loc[patient, 0] = data_grouped.loc[patient, 0].fillna(normal)
	data_p = data_grouped.reset_index()
	return data_p

def make_features_train(data_p):
	data_without_labels = data_p[data_p.columns[:-2]]
	data_by_id = data_without_labels.groupby('ID')
	timecolumns = data_p.columns[3:-2]
	data_medians = data_by_id.apply(pd.expanding_median)[timecolumns].rename(columns=lambda x: 'MEDIAN_'+x)
	data_stds = data_by_id.apply(pd.expanding_std).fillna(0)[timecolumns].rename(columns=lambda x: 'STD_'+x)
	data_mins = data_by_id.apply(pd.expanding_min).fillna(0)[timecolumns].rename(columns=lambda x: 'MIN_'+x)
	data_maxs = data_by_id.apply(pd.expanding_max)[timecolumns].rename(columns=lambda x: 'MAX_'+x)
	data_counts = data_by_id.apply(pd.expanding_count)[timecolumns].rename(columns=lambda x: 'COUNT_'+x)
	data_with_stats = pd.concat([data_medians, data_stds, data_mins, data_maxs, data_counts], axis = 1)
	#data_with_stats = data_with_stats[kfeatures]
	data_stats_full = pd.concat([data_without_labels[['ID','TIME', 'AGE']],data_with_stats, data_p[['ICU','LABEL']]], axis=1)[kfeaturestrain]
	data_stats_np = data_stats_full.as_matrix()
	#data_stats_np = data_with_stats.as_matrix()
	return data_stats_np

def make_features_test(data_p):
	data_without_labels = data_p[data_p.columns[:-1]]
	data_by_id =  data_without_labels.groupby('ID')
	timecolumns = data_p.columns[3:-1]
	data_medians = data_by_id.apply(pd.expanding_median)[timecolumns].rename(columns=lambda x: 'MEDIAN_'+x)
	data_stds = data_by_id.apply(pd.expanding_std).fillna(0)[timecolumns].rename(columns=lambda x: 'STD_'+x)
	data_mins = data_by_id.apply(pd.expanding_min)[timecolumns].rename(columns=lambda x: 'MIN_'+x)
	data_maxs = data_by_id.apply(pd.expanding_max)[timecolumns].rename(columns=lambda x: 'MAX_'+x)
	data_counts = data_by_id.apply(pd.expanding_count)[timecolumns].rename(columns=lambda x: 'COUNT_'+x)
	data_with_stats = pd.concat([data_medians, data_stds, data_mins, data_maxs, data_counts], axis = 1)
	#data_subset = data_with_stats[kfeatures]
	data_subset_full = pd.concat([data_p[['ID','TIME', 'AGE']], data_with_stats,data_p[['ICU']]], axis=1)[kfeaturestest]
	return data_subset_full


def predict(data_train,data_test):
	clf1 = SGDClassifier(loss = 'squared_hinge',warm_start=True, random_state = 41,penalty='elasticnet',n_iter=30, l1_ratio=0.20, class_weight= {0:0.2, 1:0.8})
	clf2 = SGDClassifier(loss = 'squared_hinge',warm_start=True, random_state = 41,penalty='elasticnet',n_iter=30, l1_ratio=0.20, class_weight= {0: 0.3, 1:0.7})
	clf3 = SGDClassifier(loss = 'log',warm_start=True, random_state = 41,penalty='elasticnet',n_iter=30, l1_ratio=0.20, class_weight={0:0.35, 1:0.65})
	clf4 = MultinomialNB(fit_prior = True, class_prior= [0.3,0.7])
	clfs = [clf1,clf2,clf3,clf4]
	clf4.partial_fit(data_train[:,2:-2],data_train[:,-1],[0,1])
	for clf in clfs[:-1]:
		clf.fit(data_train[:,2:-2],data_train[:,-1])

	test_feats_grouped = data_test.set_index('ID')
	test_ids = test_feats_grouped.index.unique()
	final_predictions = []
	for id in test_ids:
		test_id = test_feats_grouped.ix[id]
		test_id_np = np.atleast_2d(test_id.as_matrix())
		icu_indices = np.nonzero(test_id_np[:,-1])[0]
		prev_prediction = [0,0,0,0]
		icui=0
		for ind in icu_indices:
			partial_data_feats = test_id_np[:ind+1,1:-1]
			if icui==0 and ind>0:
				for clf in clfs:
					clf.partial_fit(partial_data_feats[:-1,:], np.zeros((partial_data_feats[:-1,:].shape[0],)))
			if icui>0:
				for i in range(0,len(clfs)):
					if ind - icu_indices[icui-1]>1:
						results = np.empty((partial_data_feats[icu_indices[icui-1]:ind,:].shape[0],))
						results.fill(int(prev_prediction[i]))
						clfs[i].partial_fit(partial_data_feats[icu_indices[icui-1]:ind,:], results)
					else:
						prev_icu = np.atleast_2d(partial_data_feats[icu_indices[icui-1],:])
						clfs[i].partial_fit(prev_icu, np.atleast_1d(prev_prediction[i]))
				bagged_prediction = 0
				for i in range(0,len(clfs)-1):
					prediction = int(clfs[i].predict(partial_data_feats[-1,:])[0])
					prev_prediction[i] = prediction
					bagged_prediction += prediction
				prediction = int(clf4.predict(partial_data_feats[-1,:])[0])
				prev_prediction[3]=prediction
				bagged_prediction += 2*prediction
				if(bagged_prediction>2):
					bagged_prediction_final = 1
				else:
					bagged_prediction_final = 0
				final_predictions.append((int(id), int(test_id_np[ind, 0]), int(prediction[0])))
				icui+=1
	return final_predictions

def train(data_stats_np):
	train_subset = data_stats_np[:,2:-2]
	clf = SGDClassifier(loss='modified_huber', penalty = 'elasticnet', l1_ratio=0.2, n_iter= 30, warm_start=True, random_state = 3)
	#train_feats = data_np_medians[:,2:-2]
	clf.fit(train_subset, data_stats_np[:,-1])
	return clf

def main():
	script, testvitals, testlabs, testage = argv
	print "Reading train files"
	data_train = read_train()
	print "...done"	
	normal = {attribute: data_train[attribute].mean() for attribute in data_train.columns}
	print "Preprocess training data"
	data_train_p = preprocess(data_train, normal)
	print "...done"
	print "Reading test files"
	data_test = read_test(testvitals, testlabs, testage)
	print "...done"
	print "Preprocess test data"
	data_test_p = preprocess(data_test, normal)
	print "...done"
	print "Making train features"
	data_train_feats = make_features_train(data_train_p)
	#data_train_feats = pd.read_csv('Train_processed.csv',index).as_matrix()
	print "...done"
	print "Training"
	
	print "...done"
	print "Making test features"
	data_test_feats = make_features_test(data_test_p)
	print "...done"
	print "Predciting"
	final_predictions = predict(data_train_feats, data_test_feats)
	print "...done"
	output = pd.DataFrame(final_predictions, columns = ['ID','TIME','LABEL'])
	dead = output.groupby('ID').sum()
	dead_ind = dead[dead['LABEL']>0].index
	for i in dead_ind:
		deadid = output[output.ID==i]
		indices = deadid.index
		for id in indices:
			output.ix[id]['LABEL']=1
	print "Writing csv"
	output.to_csv('output.csv', header = False, index = False)
	print "...done"

if __name__ == '__main__':
	main()








