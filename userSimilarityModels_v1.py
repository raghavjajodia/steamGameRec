#Importing Libraries:
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
import time
from sklearn.metrics.pairwise import cosine_similarity

class prepareData(object):

	def makeData(self, fileName):
		 
		'''''
		Method   - Loads the Data from the given fileName, cleans it.
				   Also, appends new columns ['playtime_cdf', 'playtime_scaled'].
		Input    - File from which the data has to be loaded
		Returns  - Returns loaded Data from the given fileName.
		'''''   

		# Reading the Data:
		df = pd.read_csv(fileName)
		df = df.drop('Unnamed: 0', axis = 1)

		# Dropping rows with zero as they're equivalent to not playing the game:
		df = df[df['playtime_forever'] != 0].reset_index(drop = True)

		# Transforming the Game Playtime of each user based on the CDF of Playtime of the Game:
		# This can help us normalize the data and standardize them so as to use for explicit rating based models:
		df = self.fitTransform(df)

		# Scaling down the playtime:
		df['playtime_forever_scaled'] = df['playtime_forever'] **(1/3)

		df = df[['steamid', 'appid', 'playtime_forever', 'playtime_cdf', 'playtime_forever_scaled']]
		
		return df

	def sparsity(self, dataset):
		  
		'''''
		Input    - Dataframe whose sparsity has to be calculated
		Returns  - Returns sparsity of the dataframe.
		'''''      
		
		return round(100*(1 - len(dataset)/(dataset['steamid'].nunique() * dataset['appid'].nunique())), 2)

	
	def train_test_split(self, input_data, train_ratio = 0.8):
		
		'''''
		Inputs: input_data  - DataFrame that has to be split into Train & Test by given split-ratio.
				train_ratio - Ratio in which the given DataFrame has to be split
				
		Returns: Returns two DataFrames Train and Test that are randomly splitted from the given DataFrame as per given ratio.
		
		Method: Randomly train_ratio% of samples of every user into train and remaining into the test DataFrame.
		'''''
		
		train = input_data.groupby('steamid').apply(lambda x:x.sample(frac = train_ratio))
		train_index = list(zip(*train.index))[1]
		test = input_data[(input_data.index.isin(train_index) == False)]
		
		train.index.rename(['id', 'appid_level'], inplace=True)
		train.reset_index(drop = True, inplace = True)
		test.reset_index(drop = True, inplace = True)
		
		return train, test

	def getMatrix(self, dataset, col_name = 'playtime_cdf'):
		
		'''''
		Inputs: input_data  - DataFrame that has to be transformed into Matrix form.
				col_name	- Values of column that are to be filled in the matrix.
				
		Returns: Returns matrix of shape (Users, Games) with (i, j)th cell having the value of the column given.
		
		Method: Randomly train_ratio% of samples of every user into train and remaining into the test DataFrame.
		'''''

		return dataset.pivot(index = 'steamid', columns = 'appid', values = col_name)

	def getBinaryMatrix(self, dataset, col_name = 'playtime_cdf'):
		
		'''''
		Inputs: input_data  - DataFrame that has to be transformed into Matrix form.
				
		Returns: Returns matrix of shape (Users, Games) with (i, j)th cell having 1 if the ith user had played jth game.
		
		Method: Randomly train_ratio% of samples of every user into train and remaining into the test DataFrame.
		'''''		
		matrix = dataset.pivot(index = 'steamid', columns = 'appid', values = col_name)
		matrix = (pd.notnull(matrix))*1
		return matrix

	transformerSeries = {}
			
	def getLowerOrEqualIndex(self,playtimeList, playtime):
		if playtime < playtimeList[0]:
			return 0
		ans = 0
		low = 0
		high = len(playtimeList) - 1
		while(low<=high):
			mid = low + (high-low)//2
			if playtime > playtimeList[mid]:
				ans = mid
				low = mid + 1
			elif playtime == playtimeList[mid]:
				return mid
			else:
				high = mid - 1
		return ans      
	
	def getNearestCdf(self,appid, playtime):
		playtimeList = self.transformerSeries[appid].index
		bestpos = self.getLowerOrEqualIndex(playtimeList, playtime)
		return self.transformerSeries[appid].iloc[bestpos]
	
	def fitTransform(self,tupledata):
		grouped1 = tupledata.groupby(["appid","playtime_forever"]).count()
		grouped2 = grouped1.groupby(level=[0]).cumsum()
		grouped3 = grouped2.groupby(level = [0]).max()
		withcdf = grouped2/grouped3
		self.transformerSeries = pd.Series(withcdf['steamid'],index=withcdf.index)
		withcdf_df = withcdf.reset_index(level=[0,1])
		withcdf_df.rename(columns={"steamid":"playtime_cdf"}, inplace=True)
		finaltuple = pd.merge(withcdf_df,tupledata, on=['appid','playtime_forever'],how='inner',suffixes=('_newdf',''))
		return finaltuple

	def Transform(self,tupledata):
		ansdata = tupledata.groupby(["appid","playtime_forever"]).count().reset_index()
		ansdata.drop('steamid', inplace = True, axis = 1)
		ansdata['playtime_cdf'] =  ansdata.apply(lambda x: self.getNearestCdf(x['appid'],x['playtime_forever']), axis = 1)
		return ansdata


class correlationMethods(object):

	# Initializing small epsilon value to avoid division by zero
	eps = pow(10, -8)

	## Computes Pearson Correlation: 
	def pearson_correlation(self, dataset):
		
		'''''
		Method   - Computes Pearson Correlation between every possible pair of users in the input matrix.
		Input    - Input Train Data of shape (Users, Games) 
				   with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
		Returns  - Matrix of shape (Users, Users) where (i, j)th cell has Pearson Correlation between ith and jth user.
		'''''
		
		# Subtracting the mean of each User from the input matrix
		dataset_submean = (dataset.T - np.nanmean(dataset.T, axis = 0)).T
		n = len(dataset_submean)
		
		# Computing the Pearson Correlation:
		dataset_submean_nonNaN = np.nan_to_num(dataset_submean)
		corr_num = np.matmul(dataset_submean_nonNaN, dataset_submean_nonNaN.T)
		corr_den = np.sqrt(np.matmul(np.sum(pow(dataset_submean, 2), axis = 1).reshape((n,1)), np.sum(pow(dataset_submean, 2), axis = 1).reshape((1,n)))) + self.eps
		corr_matrix = corr_num/corr_den
		np.fill_diagonal(corr_matrix, 0)
		corr_matrix_df = pd.DataFrame(corr_matrix, index = dataset.index, columns= dataset.index)
		
		return corr_matrix_df


	## Computes Constrained Pearson Correlation Matrix:
	def constrained_pearson_correlation(self, dataset):
		
		'''''
		Method   - Computes Constrained Pearson Correlation between every possible pair of users in the input matrix.
		Input    - Input Train Data of shape (Users, Games) 
				   with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
		Returns  - Matrix of shape (Users, Users) where (i, j)th cell has Constrained Pearson Correlation between ith and jth user.
		'''''
		
		# Subtracting the mean of each User from the input matrix
		dataset_submedian = (dataset.T - np.nanmedian(dataset.T, axis = 0)).T
		n = len(dataset_submedian)
		
		# Computing the Constrained Pearson Correlation:
		dataset_submedian_nonNaN = np.nan_to_num(dataset_submedian)
		corr_num = np.matmul(dataset_submedian_nonNaN, dataset_submedian_nonNaN.T)
		corr_den = np.sqrt(np.matmul(np.sum(pow(dataset_submedian, 2), axis = 1).reshape((n,1)), np.sum(pow(dataset_submedian, 2), axis = 1).reshape((1,n)))) + self.eps
		corr_matrix = corr_num/corr_den
		np.fill_diagonal(corr_matrix, 0)
		corr_matrix_df = pd.DataFrame(corr_matrix, index = dataset.index, columns= dataset.index)
		corr_matrix_df.head()
		
		return corr_matrix_df


	## Computes Cosine Similarity:
	def cosine_similarity(self, dataset):
		
		'''''
		Method   - Computes Cosine Similarity between every possible pair of users in the input matrix.
		Input    - Input Train Data of shape (Users, Games) 
				   with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
		Returns  - Matrix of shape (Users, Users) where (i, j)th cell has Cosine Similarity between ith and jth user.
		'''''
		
		return cosine_similarity(np.nan_to_num(dataset))


	## Computes Adjusted Cosine Similarity:
	def adjusted_cosine_similarity(self, dataset):
			
		'''''
		Method   - Computes Cosine Similarity between every possible pair of users in the input matrix.
		Input    - Input Train Data of shape (Users, Games) 
				   with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
		Returns  - Matrix of shape (Users, Users) where (i, j)th cell has Cosine Similarity between ith and jth user.
		'''''
		
		# Subtracting the mean of each User from the input matrix
		dataset_submean = (dataset.T - np.nanmean(dataset.T, axis = 0)).T

		#Computes Cosine Similarity on the mean reduced entries and return it
		return self.cosine_similarity(np.nan_to_num(dataset_submean))


class binaryRecommenders(object):

	def get_recommendations(self, person, matrix, corr_matrix, k = 10):
		
		'''''
		Inputs: person      - Steam ID for whom recommendations will be returned
				matrix      - Input Train Data of shape (Users, Games) 
							  with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
				corr_matrix - Matrix of shape (Users, Users) having the user to user similarity
				k           - Number of recommendation required
				
		Returns: Top k recommendations for the given user
		
		Method: Calculates the top k recommendations based on the user to user similarity excluding already played games
		'''''
		
		# Taking similarity of given user with other users
		correlations = np.array(corr_matrix[person]).reshape((1, corr_matrix.shape[0]))
		
		# Removing the games already played the user
		target_games = matrix[matrix.T[person].loc[matrix.T[person] == 0].index]
		
		# Computing scores for the games that the user hasn't played yet for recommendation
		total_scores = np.matmul(correlations, target_games)
		
		# Sorting the scores based on the scores computed above
		game_wise_scores = pd.DataFrame([list(target_games.columns), list(total_scores[0])]).T.rename(str, columns = {0:'appid', 1:'score'}).sort_values(by = 'score', ascending = False)     
		
		return list(game_wise_scores.iloc[0:k]['appid'])



	def predict(self, person, matrix, corr_matrix, appid, k = 10):
		
		'''''
		Inputs: person      - Steam ID for whom recommendations will be returned
				matrix      - Input Train Data of shape (Users, Games) 
							  with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
				corr_matrix - Matrix of shape (Users, Users) having the user to user similarity
				appid       - List of Game IDs for which we want to check if it will be recommended or not in top k items.
				k           - Number of recommendation required
				
		Returns: Returns binary list with 1(s) if the game will be recommended to user as per selected threshold and 0(s) otherwise.
		
		Method: Calculates the top k recommendations based on the user to user similarity excluding already played games
				and checks if the games in the given input list are there.
		'''''
		
		# Getting the top k recommended items
		predicted_games = self.get_recommendations(person, matrix, corr_matrix, k)
			
		# Checking if the item from the given list is in the top k recommendations
		return [1 if i in predicted_games else 0 for i in appid]



	def get_precision_recall_overall(self, test, matrix, corr_matrix, k = 10):

		'''''
		Inputs: test        - Test Data: 20% of the data that we've masked to be given here
				matrix      - Input Train Data of shape (Users, Games) 
							  with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
				corr_matrix - Matrix of shape (Users, Users) having the user to user similarity
				k           - Number of recommendation required
				
		Returns: Returns two lists having pairs of (User ID, Recall Value) & (User ID, Precision Value).
		
		Method: Calculates the top k recommendations based on the user to user similarity excluding already played games
				and computes recall & precision for each user.
		'''''
		
		# Computing percentile cut-off equivalent to picking up top k recommendations
		#threshold = 100 - 100 * k / len(matrix)
		
		# Computing scores for the games that the user hasn't played yet for recommendation
		preds_calc = np.matmul(corr_matrix, matrix)
		
		# Re-ordering the games already played the user with very low score (Practically impossible for our setting)
		preds_calc[matrix == 1] = -1500000
		
		# Picking up the top k items to be recommended items for each user
		#preds = pd.DataFrame((preds_calc.T >= np.percentile(preds_calc, threshold, axis=1))*1, index = matrix.columns, columns = matrix.index)
		preds = pd.DataFrame(preds_calc.T, index = matrix.columns, columns = matrix.index)
		
		# Initializing Recall & Precision Lists
		recall_value_overall = []
		precision_value_overall = []
		
		# Computing Recall & Precision for all the users:
		for i in preds.columns:
			recall_value_overall.append([i, len(set(preds[i].sort_values(ascending = False)[0:k].index) & set(test[test['steamid'] == i]['appid'])) / len(set(test[test['steamid'] == i]['appid']))])    
			precision_value_overall.append([i, len(set(preds[i].sort_values(ascending = False)[0:k].index) & set(test[test['steamid'] == i]['appid'])) / k])

		return recall_value_overall, precision_value_overall



	def get_precision_recall(self, person, test, matrix, corr_matrix, k = 10):
		
		'''''
		Inputs: person      - User ID for which Recall & Precision are to be calculated.
				test        - Test Data: 20% of the data that we've masked to be given here.
				matrix      - Input Train Data of shape (Users, Games)
							  with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'.
				corr_matrix - Matrix of shape (Users, Users) having the user to user similarity.
				k           - Number of recommendation required.
				
		Returns: Returns Recall Value & Precision Value for the given User ID.
		
		Method: Calculates the top k recommendations based on the user to user similarity excluding already played games
				and computes recall & precision for each user.
		'''''
		
		# Computing percentile cut-off equivalent to picking up top k recommendations
		#threshold = 100 - 100 * k / len(matrix)
		
		# Removing the games already played the user
		target_games = matrix[matrix.T[person].loc[matrix.T[person] == 0].index]
		
		# Computing scores for the games that the user hasn't played yet for recommendation
		preds_calc = np.matmul(corr_matrix.loc[person], target_games)
		
		# Picking up the top k items to be recommended items for each user
		#preds = pd.DataFrame((preds_calc.T >= np.percentile(preds_calc, threshold))*1, index = target_games.columns)
		preds = pd.DataFrame(preds_calc.T, index = target_games.columns)
		
		# Computing Recall & Precision for the given user:
		recall_value = len(set(preds.sort_values(ascending = False)[0:k].index) & set(test['appid'])) / len(test['appid'])
		precision_value = len(set(preds.sort_values(ascending = False)[0:k].index) & set(test['appid'])) / k
		
		return recall_value, precision_value


class binaryRecommenders_high_correlation(object):

	def get_recommendations(self, person, matrix, corr_matrix, corr_threshold = 0.25, k = 10):
		
		'''''
		Inputs: person         - Steam ID for whom recommendations will be returned
				matrix         - Input Train Data of shape (Users, Games) 
								 with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
				corr_matrix    - Matrix of shape (Users, Users) having the user to user similarity
				corr_threshold - The minimum correlation between two users to be considered for recommendation
				k              - Number of recommendation required
				
		Returns: Top k recommendations for the given user
		
		Method: Calculates the top k recommendations based on the user to user similarity excluding already played games
		'''''
		
		# Taking similarity of given user with other users
		correlations = np.array(corr_matrix[person]).reshape((1, corr_matrix.shape[0]))

		# Zeroing all the correlation values less than the given threshold
		correlations[correlations<=corr_threshold] = 0
		
		# Removing the games already played the user
		target_games = matrix[matrix.T[person].loc[matrix.T[person] == 0].index]
		
		# Computing scores for the games that the user hasn't played yet for recommendation
		total_scores = np.matmul(correlations, target_games)
		
		# Sorting the scores based on the scores computed above
		game_wise_scores = pd.DataFrame([list(target_games.columns), list(total_scores[0])]).T.rename(str, columns = {0:'appid', 1:'score'}).sort_values(by = 'score', ascending = False)     
		
		return list(game_wise_scores.iloc[0:k]['appid'])

	def predict(self, person, matrix, corr_matrix, appid, corr_threshold = 0.25, k = 10):
		
		'''''
		Inputs: person      	- Steam ID for whom recommendations will be returned
				matrix      	- Input Train Data of shape (Users, Games) 
								  with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
				corr_matrix 	- Matrix of shape (Users, Users) having the user to user similarity
				appid       	- List of Game IDs for which we want to check if it will be recommended or not in top k items.
				corr_threshold 	- The minimum correlation between two users to be considered for recommendation
				k           	- Number of recommendation required
				
		Returns: Returns binary list with 1(s) if the game will be recommended to user as per selected threshold and 0(s) otherwise.
		
		Method: Calculates the top k recommendations based on the user to user similarity excluding already played games
				and checks if the games in the given input list are there.
		'''''
		
		# Getting the top k recommended items
		predicted_games = self.get_recommendations(person, matrix, corr_matrix, corr_threshold, k)
			
		# Checking if the item from the given list is in the top k recommendations
		predicted = [1 if i in predicted_games else 0 for i in appid]
		return precicted


	def get_precision_recall_overall(self, test, matrix, corr_matrix, corr_threshold = 0.25, k = 10):

		'''''
		Inputs: test        	- Test Data: 20% of the data that we've masked to be given here
				matrix      	- Input Train Data of shape (Users, Games) 
								  with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
				corr_matrix 	- Matrix of shape (Users, Users) having the user to user similarity
				corr_threshold 	- The minimum correlation between two users to be considered for recommendation
				k           	- Number of recommendation required
				
		Returns: Returns two lists having pairs of (User ID, Recall Value) & (User ID, Precision Value).
		
		Method: Calculates the top k recommendations based on the user to user similarity excluding already played games
				and computes recall & precision for each user.
		'''''
		
		# Computing percentile cut-off equivalent to picking up top k recommendations
		#threshold = 100 - 100 * k / len(matrix)
		
		# Zeroing all the correlation values less than the given threshold
		corr_matrix[corr_matrix <= corr_threshold] = 0
		
		# Computing scores for the games that the user hasn't played yet for recommendation
		preds_calc = np.matmul(corr_matrix, matrix)
		
		# Re-ordering the games already played the user with very low score (Practically impossible for our setting)
		preds_calc[matrix == 1] = -1500000
		
		# Picking up the top k items to be recommended items for each user
		#preds = pd.DataFrame((preds_calc.T >= np.percentile(preds_calc, threshold, axis=1))*1, index = matrix.columns, columns = matrix.index)
		preds = pd.DataFrame(preds_calc.T, index = matrix.columns, columns = matrix.index)
		
		# Initializing Recall & Precision Lists
		recall_value_overall = []
		precision_value_overall = []
		
		# Computing Recall & Precision for all the users:
		for i in preds.columns:
			recall_value_overall.append([i, len(set(preds[i].sort_values(ascending = False)[0:k].index) & set(test[test['steamid'] == i]['appid'])) / len(set(test[test['steamid'] == i]['appid']))])    
			precision_value_overall.append([i, len(set(preds[i].sort_values(ascending = False)[0:k].index) & set(test[test['steamid'] == i]['appid'])) / k])

		return recall_value_overall, precision_value_overall


	def get_precision_recall(self, person, test, matrix, corr_matrix, corr_threshold = 0.25, k = 10):
		
		'''''
		Inputs: person      	- User ID for which Recall & Precision are to be calculated.
				test        	- Test Data: 20% of the data that we've masked to be given here.
				matrix      	- Input Train Data of shape (Users, Games)
								  with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'.
				corr_matrix 	- Matrix of shape (Users, Users) having the user to user similarity.
				corr_threshold 	- The minimum correlation between two users to be considered for recommendation
				k           	- Number of recommendation required.
				
		Returns: Returns Recall Value & Precision Value for the given User ID.
		
		Method: Calculates the top k recommendations based on the user to user similarity excluding already played games
				and computes recall & precision for each user.
		'''''
		
		# Computing percentile cut-off equivalent to picking up top k recommendations
		#threshold = 100 - 100 * k / len(matrix)
		
		# Removing the games already played the user
		target_games = matrix[matrix.T[person].loc[matrix.T[person] == 0].index]

		# Taking similarity of given user with other users
		correlations = np.array(corr_matrix[person]).reshape((1, corr_matrix.shape[0]))

		# Zeroing all the correlation values less than the given threshold
		correlations[correlations<=corr_threshold] = 0
		
		# Computing scores for the games that the user hasn't played yet for recommendation
		preds_calc = np.matmul(correlations, target_games)
		
		# Picking up the top k items to be recommended items for each user
		#preds = pd.DataFrame((preds_calc.T >= np.percentile(preds_calc, threshold))*1, index = target_games.columns)
		preds = pd.DataFrame(preds_calc.T, index = target_games.columns)

		# Computing Recall & Precision for the given user:
		recall_value = len(set(preds.sort_values(ascending = False)[0:k].index) & set(test['appid'])) / len(test['appid'])
		precision_value = len(set(preds.sort_values(ascending = False)[0:k].index) & set(test['appid'])) / k

		return recall_value, precision_value
	


class playtimePredictionMethods(prepareData):

	def get_recommendations(self, person, dataset, corr_matrix, k = 10):
		
		'''''
		Inputs: person      - Steam ID for whom recommendations will be returned
				dataset     - Input Train Data of shape (Users, Games) 
							  with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
				corr_matrix - Matrix of shape (Users, Users) having the user to user similarity
				k           - Number of recommendation required
				
		Returns: Top k recommendations for the given user
		
		Method: Calculates the top k recommendations based on the user to user similarity excluding already played games
		'''''
		
		# Taking similarity of given user with other users
		correlations = np.array(corr_matrix[person]).reshape((1, corr_matrix.shape[0]))
		
		# Removing the games already played the user
		target_games = dataset[dataset.columns[pd.isnull(dataset.T[person])]]
		
		# Computing scores for the games that the user hasn't played yet for recommendation
		total_scores = np.matmul(correlations, np.nan_to_num(target_games))
		
		# Computing sum of similarities of a given user for normalizing the obtained scores
		total_sums = np.matmul(correlations, pd.isnull(target_games)*1)
		
		# Normalizing the scores to obtain the predicted game wise play-time
		predicted_scores = total_scores/total_sums
		
		# Sorting the scores based on the scores computed above
		game_wise_scores = pd.DataFrame([list(target_games.columns), list(predicted_scores[0])]).T.rename(str, columns = {0:'appid', 1:'score'}).sort_values(by = 'score', ascending = False)     
		
		return list(game_wise_scores.iloc[0:k]['appid'])



	def predict(self, person, dataset, corr_matrix, appid):
		
		'''''
		Inputs: person      - Steam ID for whom recommendations will be returned
				dataset     - Input Train Data of shape (Users, Games) 
							  with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
				corr_matrix - Matrix of shape (Users, Users) having the user to user similarity
				appid       - List of Game IDs for which we want to check if it will be recommended or not in top k items.
				k           - Number of recommendation required
				
		Returns: Returns binary list with 1(s) if the game will be recommended to user as per selected threshold and 0(s) otherwise.
		
		Method: Calculates the top k recommendations based on the user to user similarity excluding already played games
				and checks if the games in the given input list are there.
		'''''
		
		# Taking similarity of given user with other users
		correlations = np.array(corr_matrix[person]).reshape((1, corr_matrix.shape[0]))
		
		# Removing the games already played the user
		target_games = dataset[dataset.columns[pd.isnull(dataset.T[person])]]
		
		# Computing scores for the games that the user hasn't played yet for recommendation
		total_scores = np.matmul(correlations, np.nan_to_num(target_games[appid]).reshape((len(np.nan_to_num(target_games[appid])), len(appid))))
		
		# Computing sum of similarities of a given user for normalizing the obtained scores
		total_sums = np.matmul(correlations, pd.isnull(target_games[appid]))
		
		# Normalizing the scores to obtain the predicted game wise play-time
		predicted_scores = total_scores/total_sums
		predicted_scores[predicted_scores <= 0] = 0
		
		return predicted_scores


	'''''
	def predict_game_playtime_overall(self, dataset, corr_matrix, k = 10):
		

		Inputs: dataset     - Input Train Data of shape (Users, Games) 
							  with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
				corr_matrix - Matrix of shape (Users, Users) having the user to user similarity
				k           - Number of recommendation required
				
		Returns: Returns matrix of shape (Users, Games) with the predicted playtime of the top k games for each user.
		
		Method: Predicts the playtime for all the games, excluding already played games, 
				and picks up top k recommendations for each user based on user to user similarity.

		
		# Computing percentile cut-off equivalent to picking up top k recommendations
		#threshold = 100 - 100 * k / len(dataset)
		
		# Computing scores for the games that the user hasn't played yet for recommendation
		total_scores = np.matmul(corr_matrix, np.nan_to_num(dataset))
		
		# Computing sum of similarities of a given user for normalizing the obtained scores
		total_sums = np.matmul(corr_matrix, pd.notnull(dataset))
		
		# Normalizing the scores to obtain the predicted game wise play-time
		predicted_scores = total_scores/total_sums
		
		# Re-ordering the games already played the user with very low score (Practically impossible for our setting)
		predicted_scores[dataset >= 1] = 0
		
		# Making the DataFrame of the obtained data:
		predicted_scores = pd.DataFrame(predicted_scores, index = corr_matrix.index, columns = dataset.columns)

		# Picking up the top k items to be recommended items for each user
		predicted_scores = predicted_scores[(predicted_scores.T >= np.percentile(predicted_scores, threshold, axis=1)).T]


		return predicted_scores
	'''''

	def get_rmse_overall(self, test, dataset, corr_matrix, col_name = 'playtime_cdf'):

		'''''
		Inputs: test        - Test Data: 20% of the data that we've masked to be given here
				dataset     - Input Train Data of shape (Users, Games) 
							  with each cell indicating if user has played the game or not in case of binary, otherwise 'playtime_cdf'
				corr_matrix - Matrix of shape (Users, Users) having the user to user similarity
				col_name    - Playtime format to be taken for computations.
				
		Returns: Returns two lists having pairs of (User ID, Recall Value) & (User ID, Precision Value).
		
		Method: Calculates the top k recommendations based on the user to user similarity excluding already played games
				and computes recall & precision for each user.
		'''''    
		
		# Creating matrix form for test data:
		test_matrix = prepareData.getMatrix(self, test, col_name = col_name)
		
		# Computing scores for the games that the user hasn't played yet for recommendation
		total_scores = np.matmul(corr_matrix, np.nan_to_num(dataset))
		
		# Computing sum of similarities of a given user for normalizing the obtained scores
		total_sums = np.matmul(corr_matrix, pd.notnull(dataset))
		
		# Normalizing the scores to obtain the predicted game wise play-time
		predicted_scores = total_scores/total_sums
		
		# Making the DataFrame of the obtained data:
		predicted_scores = pd.DataFrame(predicted_scores, index = dataset.index, columns = dataset.columns)

		# Computing RMSE
		rmse = np.sqrt(np.nansum((np.nan_to_num(predicted_scores[test_matrix.columns]) - test_matrix)**2)/np.count_nonzero(np.nan_to_num(test_matrix)))

		return rmse



class Model(correlationMethods, binaryRecommenders, binaryRecommenders_high_correlation, playtimePredictionMethods):

	'''''
	def __init__(self):
		self.prepData = prepareData(self)
		self.cm = correlationMethods(self)
		self.binaryRec = binaryRecommenders(self)
		self.binaryRecHighCorr = binaryRecommenders_high_correlation(self)
		self.playPred = playtimePredictionMethods(self)
	'''''

	def runModel(self, dataset, method = 'binary', col_name = 'playtime_cdf', corr_method = 'pearson', nruns = 1, train_ratio = 0.8, corr_threshold = 0.25, k = 10, p_out = False):

		start_time = time.time()
		
		if (method == 'binary') | (method == 'binary_highcorrelation'):
			mean_recall = []
			mean_precision = []

		elif method == 'predict_playtime':
			mean_rmse = []
		
		for i in range(nruns):
			
			train, test = prepareData.train_test_split(self, dataset, train_ratio)

			if method == 'binary':
				matrix = prepareData.getBinaryMatrix(self, train)
			
			elif method == 'predict_playtime':
				matrix = prepareData.getMatrix(self, train, col_name)
			
			elif method == 'binary_highcorrelation':
				matrix = prepareData.getBinaryMatrix(self, train)

			
			if corr_method == 'pearson':
				corr_matrix = correlationMethods.pearson_correlation(self, matrix)
			
			elif corr_method == 'constrained_pearson':
				corr_matrix = correlationMethods.constrained_pearson_correlation(self, matrix)
			
			elif corr_method == 'cosine':
				corr_matrix = correlationMethods.cosine_similarity(self, matrix)
			
			elif corr_method == 'adjusted_cosine':
				corr_matrix = correlationMethods.adjusted_cosine_similarity(self, matrix)
			

			if method == 'binary':
				recall, precision = binaryRecommenders.get_precision_recall_overall(self, test, matrix, corr_matrix, k)
				mean_recall.append(np.mean(np.array(recall).T[1]))
				mean_precision.append(np.mean(np.array(precision).T[1]))

			elif method == 'predict_playtime':
				rmse = playtimePredictionMethods.get_rmse_overall(self, test, matrix, corr_matrix, col_name = col_name)
				mean_rmse.append(np.mean(rmse))

			elif method == 'binary_highcorrelation':
				recall, precision = binaryRecommenders_high_correlation.get_precision_recall_overall(self, test, matrix, corr_matrix, corr_threshold, k)
				mean_recall.append(np.mean(np.array(recall).T[1]))
				mean_precision.append(np.mean(np.array(precision).T[1]))
			
			if (i % 5 == 0) & p_out == True:
				print('After {} iteration: {} seconds elapsed'.format(i+1, (time.time() - start_time)))

		print("--- Total %s seconds elapsed ---" % (time.time() - start_time))

		if (method == 'binary') | (method == 'binary_highcorrelation'):
			return mean_recall, mean_precision
		
		else:
			return mean_rmse


class evaluatePerformace(Model):
	'''''
	def __init__(self):
		self.model = Model(self)
	'''''
	def plotRec_k_GamesPerformance(self, dataset, k_valuesList = [5, 10, 20, 50], nruns = 3, method = 'binary', col_name = 'playtime_cdf', corr_method = 'pearson', train_ratio = 0.8, p_out = False):

		if method == 'binary':
			recall = []
			precision = []
			mean_recall = []
			mean_precision = []

			for i in k_valuesList:
				recall, precision = Model.runModel(self, dataset = dataset, method = 'binary', col_name = col_name, corr_method = corr_method, nruns = nruns, train_ratio = train_ratio, k = i, p_out = p_out)
				mean_recall.append(recall)
				mean_precision.append(precision)

			plt.figure(figsize = (12, 8))
			plt.plot(k_valuesList, mean_recall, lw=2, label='Mean Recall Value on {} runs'.format(nruns))
			plt.plot(k_valuesList, mean_precision, lw=2, label='Precision Value on {} runs'.format(nruns))
			plt.legend(loc = 'lower right')
			plt.xlabel('Recommending k items')
			plt.ylabel('Recall & precision')
			plt.title('Model Performance (Recall & Precision) vs Recommending k items')
			plt.show()

		elif method == 'predict_playtime':
			rmse = []
			mean_rmse = []

			for i in k_valuesList:
				rmse = Model.runModel(self, dataset, method = 'predict_playtime', col_name = col_name, corr_method = corr_method, nruns = nruns, train_ratio = train_ratio, k = i, p_out = p_out)
				mean_rmse.append(rmse)

			plt.figure(figsize = (12, 8))
			plt.plot(mean_rmse, mean_recall, lw=2, label='Mean RMSE Value on {} runs'.format(nruns))
			plt.legend(loc = 'lower right')
			plt.xlabel('Recommending k items')
			plt.ylabel('RMSE')
			plt.title('Model Performance (RMSE) vs Recommending k items')
			plt.show()

		elif method == 'binary_highcorrelation':
			recall = []
			precision = []
			mean_recall = []
			mean_precision = []

			for i in k_valuesList:
				recall, precision = Model.runModel(self, dataset = dataset, method = 'binary_highcorrelation', col_name = col_name, corr_method = corr_method, nruns = nruns, train_ratio = train_ratio, corr_threshold = corr_threshold, k = i, p_out = p_out)
				mean_recall.append(recall)
				mean_precision.append(precision)

			plt.figure(figsize = (12, 8))
			plt.plot(k_valuesList, mean_recall, lw=2, label='Mean Recall Value on {} runs'.format(nruns))
			plt.plot(k_valuesList, mean_precision, lw=2, label='Precision Value on {} runs'.format(nruns))
			plt.legend(loc = 'lower right')
			plt.xlabel('Recommending k items')
			plt.ylabel('Recall & precision')
			plt.title('Model Performance (Recall & Precision) vs Recommending k items')
			plt.show()

		else:

			print("Choose valid method from: \n 1) binary \n 2) predict_playtime \n 3) binary_highcorrelation")




