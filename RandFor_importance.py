
def Forest_importance(start, df_train_feats, df_train_labels ):
	train_cols = ['num_' + str(k) for k in range(start, start+100)]
	
	X = df_train_feats[train_cols]
	y = df_train_labels
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	pipeline = make_pipeline(SimpleImputer(strategy='mean'), RandomForestClassifier(n_estimators=60))
	hparams = {'randomforestclassifier__max_features' : [0.4,0.6],'randomforestclassifier__max_depth': [4, 3, 2], 'randomforestclassifier__min_samples_leaf':[4,5]}

	clf = GridSearchCV(pipeline, hparams, cv=5)
	clf.fit(X_train, y_train)
	print (clf.best_params_)
	y_pred = clf.predict(X_test)
	print (y_pred)
	
	feat_imp = clf.best_estimator_.steps[1][1].feature_importances_
	feat_imp_rank = pd.DataFrame({'var_name': train_cols, 'importance': feat_imp}).sort_values(by='importance', ascending=False)
	print (feat_imp_rank)
	
	conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
	print (conf_matrix)
	print('Precision: %.3f' % precision_score(y_test, y_pred, average='micro'))
	print('Recall: %.3f' % recall_score(y_test, y_pred, average='micro'))
	print('F1 Score: %.3f' % f1_score(y_test, y_pred, average='micro'))
	
	return
