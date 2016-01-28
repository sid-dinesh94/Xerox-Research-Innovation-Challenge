import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sys import argv
import csv
from sklearn.preprocessing import scale

def read_train():
	data_age = pd.read_csv("id_age_train.csv", sep=",")
	data_vitals = pd.read_csv("id_time_vitals_train.csv", sep=",")
	data_labels = pd.read_csv("id_label_train.csv", sep=",")
	data_labs = pd.read_csv("id_time_labs_train.csv", sep = ",")
	data_timeseries = pd.merge(data_labs, data_vitals)
	patients = list(data_timeseries['ID'])
	patient_ids = list(data_timeseries['ID'].unique())
	labels_zeros = np.zeros(len(patients))
	for p in patient_ids:
		if int(data_labels[data_labels['ID']==p]['LABEL'])==1:
			#labels[data_timeseries[data_timeseries['ID']==p].index[-1]] = 1
			data_id = data_timeseries[data_timeseries['ID']==p]
			indices = data_id[data_id['ICU']==1].index
			for i in indices:
				labels_zeros[i]=1
	data_timeseries['LABEL'] = labels_zeros
	data = pd.merge( data_age, data_timeseries, on='ID')
	data.loc[445378,'L20'] = 20.5267
	data_grouped = data.groupby(['ID','TIME']).sum()
	normal = {attribute: data[attribute].mean() for attribute in data.columns}
	for patient in patient_ids:
		data_grouped.loc[patient, 0] = data_grouped.loc[patient, 0].fillna(normal)
	data_grouped.reset_index(level = 1, inplace=True)
	variables =data.columns.difference(['AGE','TIME','ID','LABEL','ICU'])
	filled_featured_data = get_features_in_pandas(patient_ids,data_grouped,variables,normal)
	data_np = filled_featured_data.as_matrix()
	return data_np, filled_featured_data,normal



def get_features(patient,data_grouped,variables,normal,test=False):
    patient_table = data_grouped.ix[patient]
    if((type(patient_table)) == type(pd.Series())):
    	if not test:
    		patient_features = np.atleast_2d(np.zeros(((1,6+(5*31)))))
    	else:
    		patient_features = np.atleast_2d(np.zeros(((1,5+(5*31)))))
    	patient_features[0,:4] = patient,patient_table.ix['TIME'],patient_table.ix['AGE'],1
        parent = dict({variable: dict({'count':0, 'first':0,'last':0,'min':0,'max':0,'median':[]}) for variable in variables})
        i=0
        for variable in sorted(variables):
            parent[variable]['max'] = patient_table.ix[variable]
            parent[variable]['min'] = patient_table.ix[variable]
            parent[variable]['count'] += 1 
            parent[variable]['last'] = patient_table.ix[variable]
            parent[variable]['first'] = patient_table.ix[variable]
            patient_features[0,i*5+4]=    parent[variable]['max']
            patient_features[0,i*5+5]=    parent[variable]['min']
            patient_features[0,i*5+6]=    parent[variable]['first']
            patient_features[0,i*5+7]=    parent[variable]['last']
            patient_features[0,i*5+8]=    parent[variable]['count']
            i+=1
        if(not test):
        	patient_features[0,-2:] = patient_table.ix['ICU'],patient_table.ix['LABEL']
        else:
        	patient_features[0,-1] = patient_table.ix['ICU']

        return patient_features
    else:
        patient_table.reset_index(inplace = True)
        parent = dict({variable: dict({'count':0, 'first':0,'last':0,'min':0,'max':0,'median':[]}) for variable in variables})
        if(not test):
        	patient_features = np.zeros((len(patient_table),6+(5*31)))
        else:
         	patient_features = np.zeros((len(patient_table),5+(5*31)))
        for num in range(len(patient_table)):
            row = patient_table.ix[num]
            patient_features[num,:4] = patient,row['TIME'],row['AGE'],num
            #print patient_features[num,:2]
            #print row['TIME'],row['AGE']
            i = 0
            for variable in sorted(variables):
                if(num==0):
                    parent[variable]['min'] = normal[variable]
                if not np.isnan(row[variable]):
                    parent[variable]['max'] = max(row[variable],parent[variable]['max'])
                    parent[variable]['min'] = min(row[variable],parent[variable]['min'])
                    parent[variable]['count'] += 1 
                    parent[variable]['last'] = row[variable]
                if row['TIME']==0:
                    parent[variable]['first'] = row[variable]
                patient_features[num,i*5+4]=    parent[variable]['max']
                patient_features[num,i*5+5]=    parent[variable]['min']
                patient_features[num,i*5+6]=    parent[variable]['first']
                patient_features[num,i*5+7]=    parent[variable]['last']
                patient_features[num,i*5+8]=    parent[variable]['count']
                i+=1
            if(not test):
            	patient_features[num,-2:] = row['ICU'],row['LABEL']
            else:
            	patient_features[num,-1] = row['ICU']
        return patient_features

def get_features_in_pandas(patients,data_grouped,variables,normal, test = False):
    patients_data_array = []
    for patient in patients:
        patients_data_array.append(get_features(patient,data_grouped,variables,normal, 	test))
        if(patient%100 ==0):
        	print patient 
    conc_array = np.concatenate(patients_data_array)
    columns = ['ID','TIME','AGE','LENGTH']
    for variable in sorted(variables):
        for item in ['max','min','first','last','count']:
            columns.append(variable+'_'+item)
    columns.append('ICU')
    if(not test):
    	columns.append('LABEL')
    #print columns
    return pd.DataFrame(conc_array,columns = columns)


def fit(train):
	#train[:,4:-2] = scale(train_feats)
	train_feats = scale(train[:,4:-2])
	clf = SGDClassifier(loss='perceptron', warm_start=True)
	clf.fit(train_feats, train[:,-1])
	return clf

def predict(clf, test):
	test = test.set_index('ID')
	test_ids = set(test.index)
	final_predictions = []
	for id in test_ids:
		if id%100==0:
			print id
		test_df = test.ix[id]
		test_np = np.atleast_2d(test_df.as_matrix())
		icu_indices = np.nonzero(test_np[:,-1])[0]
		previous_prediction = 0
		flag = 0
		final = 0
		icuindex = 0
		for ind in icu_indices:
			partial_data = test_np[:ind+1,:-1]
			partial_data[:,3:] = scale(partial_data[:,3:])
			partial_data_feats = partial_data[:,3:]
			if icuindex==0 and ind>0:
				clf.partial_fit(partial_data_feats[:-1,:], np.zeros((partial_data_feats[:-1].shape[0],)))
			if icuindex>0:
				if ind - icu_indices[icuindex-1]>1:	
					clf.partial_fit(partial_data_feats[icu_indices[icuindex-1]+1:-1,:], np.zeros((partial_data_feats[icu_indices[icuindex-1]+1:ind,:].shape[0],)))
				previous_icu = np.atleast_2d(partial_data_feats[icu_indices[icui-1],:])
				clf.partial_fit(previous_icu, previous_prediction)
			prediction = clf.predict(partial_data_feats[-1,:])
			previous_prediction = prediction
			final_predictions.append((id, int(partial_data[ind,0]),int(prediction[0])))
	return final_predictions

def read_test(testvitals, testlabs, testage, normal):
	data_age = pd.read_csv(testage, sep=",")
	data_vitals = pd.read_csv(testvitals, sep=",")
	data_labs = pd.read_csv(testlabs, sep = ",")
	data_timeseries = pd.merge(data_labs, data_vitals)
	patients = list(data_timeseries['ID'])
	patient_ids = list(data_timeseries['ID'].unique())
	data = pd.merge( data_age, data_timeseries, on='ID')
	data_grouped = data.groupby(['ID','TIME']).mean()
	for patient in patient_ids:
		data_grouped.loc[patient, 0] = data_grouped.loc[patient, 0].fillna(normal)
	data_grouped.reset_index(level = 1, inplace=True)
	variables =data.columns.difference(['AGE','TIME','ID','ICU'])
	filled_featured_data = get_features_in_pandas(patient_ids,data_grouped,variables,normal,True)
	data_np = filled_featured_data.as_matrix()
	return data_np, filled_featured_data

def main():
	script, testvitals, testlabs, testage = argv
	print "Making train data"
	train, train_df, normal = read_train()
	print "...done"
	print "Training"
	clf = fit(train)
	print "...done"
	print "Making test set"
	test, test_df = read_test(testvitals, testlabs, testage, normal)
	print "...done"
	print "Writing predictions"
	with open('output.csv', 'w') as f:
		writer = csv.writer(f, delimiter=',')
		predictions = predict(clf, test_df)
		for pred in predictions:
			writer.writerow([int(pred[0]), int(pred[1]), int(pred[2])])
	print "...done"

if __name__ == '__main__':
	main()





