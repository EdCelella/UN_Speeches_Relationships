import re
import csv
import sys
import os
import pickle
import code

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import metrics

# INCLUDE THESE SOMEWHERE
# nltk.download('stopwords')
# nltk.download('punkt')

# scree


def main():

	# Data set files.
	debate_file = "data_sets/un-general-debates.csv"
	hdi_file = "data_sets/hdi.csv"
	country_name_file = "data_sets/country_names.csv"
	country_continent_file = "data_sets/country_continent.csv"
	stop_words_file = "data_sets/stop_words.txt"

	data, data_2015 = preprocess(debate_file, hdi_file, country_name_file, country_continent_file, stop_words_file)

	print("Running PCA on all data.")
	pdf = principle_component_analysis(data)

	print("Running PCA on 2015 data")
	pdf_2015 = principle_component_analysis(data_2015)

	# Tagging information for plots
	hdi_tag = (data["hdi"], dict(zip(["VH", "HI", "ME", "LO", "NaN"], ['g', 'b', 'r', 'k', 'w'])) )
	continent_tag = (data["continent"], dict(zip(["AF", "NA", "OC", "AN", "AS", "EU", "SA"],['g', 'b', 'k', 'r', 'c', 'm', 'y'])) )
	decade_tag = (data["decade"], dict(zip(["2010s", "2000s", "1990s", "1980s", "1970s"], ['g', 'b', 'r', 'k', 'y'])) )

	hdi_tag_2015 = (data_2015["hdi"], dict(zip(["VH", "HI", "ME", "LO", "NaN"], ['g', 'b', 'r', 'k', 'w'])) )
	continent_tag_2015 = (data_2015["continent"], dict(zip(["AF", "NA", "OC", "AN", "AS", "EU", "SA"],['g', 'b', 'k', 'r', 'c', 'm', 'y'])) )

	code.interact(local=dict(globals(), **locals()))

'''
----------------------------------------------------------------------------------------------------------------------

ANALYSIS

----------------------------------------------------------------------------------------------------------------------
'''

def latent_semantic_analysis(data, tfidf, tfidf_vec):

	# Specifies number of topics for LSA
	num_components = 10
	components = ["topic_" + str(i+1) for i in range(0, num_components)]

	# Initialises SVD object
	lsa = TruncatedSVD(num_components, algorithm = 'randomized')

	topics = lsa.fit_transform(tfidf_vec)
	topics = Normalizer(copy=False).fit_transform(topics)

	topic_terms = pd.DataFrame(lsa.components_,index = components,columns = tfidf.get_feature_names())

	for i in range(0, topic_terms.shape[0]):
		print("\n------------------------------------------\n")
		print(topic_terms.iloc[i].nlargest(10))
		print("\n------------------------------------------")

	# Replaces text with LSD topic distances.
	data = data.reset_index()
	data = pd.concat([data,pd.DataFrame(topics, columns=components)], axis=1).drop(["index"],axis=1)

	return data, lsa

def principle_component_analysis(data):
	
	pca = PCA(0.99)

	topics = data[data.columns[5:]]

	pca_comp = pca.fit(topics)
	print(pca.explained_variance_ratio_)

	components = pca.n_components_
	component_labels = ["PC" + str(i+1) for i in range(0, components)]

	pca_matrix = pca.fit_transform(topics)

	pdf = pd.DataFrame(data = pca_matrix, columns = component_labels)

	data = (data[["hdi", "continent","decade"]].reset_index()).drop(["index"],axis=1)
	pdf = pdf.reset_index().drop(["index"],axis=1)

	pdf = pd.concat([data, pdf], axis=1)

	return pdf

	


'''
----------------------------------------------------------------------------------------------------------------------

PREPROCESSING

----------------------------------------------------------------------------------------------------------------------
'''


# -------------------
# NAME   : preprocess
# RETURN : dataframe
# DESC   : Preprocesses data set by adding meta data, removing stop words, stemming, calculating tfidf matrix and applying LSA.
# -------------------
def preprocess(debate_file, hdi_file, country_name_file, country_continent_file, stop_words_file):

	# Program generated filenames
	pt_file = "generated/parsed_text.csv"

	tf_file = "generated/tfidf_all.csv"
	tf_vec_obj = "generated/tfidf_vec_all.p"
	tf_object = "generated/tfidf_obj_all.p"

	tf_2015_file = "generated/tfidf_2015.csv"
	tf_2015_vec_obj = "generated/tfidf_vec_all.p"
	tf_2015_object = "generated/tfidf_obj_2015.p"

	lsa_file = "generated/lsa_all.csv"
	lsa_obj = "generated/lsa_obj.p"

	lsa_2015_file = "generated/lsa_2015.csv"
	lsa_2015_obj = "generated/lsa_2015_obj.p"

	# Checks if generated file directory exists, if not it creates the directory.
	if not os.path.isdir("generated"): os.mkdir("generated")

	# Checks if parsed text data set exists.
	if not os.path.isfile(pt_file):

		print("Reading debate file.")
		data = read_csv(debate_file)

		# Adds continent and human development index data to each speech.
		data = add_meta(data, hdi_file, country_continent_file, country_name_file)
	
		# Cleans, removes stops words, and stems speeches.
		data = clean_text(data.copy(), stop_words_file)

		data.to_csv(pt_file)

	# If text is already parsed read file.
	else:
		print("Reading parsed text file.")
		data = read_csv(pt_file)
		data = data.drop([""],axis=1)

	# Creates a subset of the dataset containing only the speeches from 2015.
	data_2015 = data.loc[data['year'] == "2015"]

	# Gets tfidf vectors for each speech.
	data, tfidf, tfidf_vec = produce_tfidf(data, tf_file, tf_vec_obj, tf_object)
	data_2015, tfidf_2015, tfidf_2015_vec = produce_tfidf(data_2015, tf_2015_file, tf_2015_vec_obj, tf_2015_object)


	# Checks if LSA data exists.
	if not os.path.isfile(lsa_file) or not os.path.isfile(lsa_obj):
		print("Running LSA on all data.")

		# Runs LSA on TFIDF vectors.
		data, lsa = latent_semantic_analysis(data[["year","country","continent","hdi","decade"]], tfidf, tfidf_vec)

		# Saves LSA data.
		data.to_csv(lsa_file)
		pickle.dump(lsa, open(lsa_obj, "wb"))


	else:
		print("Reading all data LSA files.")
		data = read_csv(lsa_file).drop([""],axis=1)
		lsa = pickle.load(open(lsa_obj, "rb"))

	print(lsa.explained_variance_ratio_)

	# Checks if 2015 LSA data exists.
	if not os.path.isfile(lsa_2015_file) or not os.path.isfile(lsa_2015_obj):
		print("Running LSA on 2015 data.")

		# Runs LSA on TFIDF vectors.
		data_2015, lsa_2015 = latent_semantic_analysis(data_2015[["year","country","continent","hdi","decade"]], tfidf_2015, tfidf_2015_vec)

		# Saves LSA data.
		data_2015.to_csv(lsa_2015_file)
		pickle.dump(lsa_2015, open(lsa_2015_obj, "wb"))

	else:
		print("Reading 2015 LSA files.")
		data_2015 = read_csv(lsa_2015_file).drop([""],axis=1)
		lsa_2015 = pickle.load(open(lsa_2015_obj, "rb"))

	print(lsa_2015.explained_variance_ratio_)

	return data, data_2015


# -------------------
# NAME   : produce_tfidf
# RETURN : dataframe, tfidf vectorizer object, tfidf vector
# DESC   : Produces a tfidf vector from the given corpus.
# -------------------
def produce_tfidf(data, data_filename, vec_filename, obj_filename):

	# Checks if TFIDF data exists.
	if not os.path.isfile(data_filename) or not os.path.isfile(vec_filename) or not os.path.isfile(obj_filename):

		print("Producing TFIDF vector.")

		# Creates TFIDF vectorizer object. Parameters set to keep the top 5000 words in the vocabulary that appear in less
		# than 80% of the documents, and more than 20% of the documents.
		tfidf = TfidfVectorizer(analyzer="word", use_idf=True, smooth_idf=True, max_df=0.8, min_df=0.2, max_features=5000)

		# Produces tfidf vector using the speech corpus, adhering to the above set parameters.
		tfidf_vec = tfidf.fit_transform(data['text'])

		# Converts vector to dataframe.
		tfidf_df = pd.DataFrame(tfidf_vec.toarray(), columns= tfidf.get_feature_names())

		# Redundant information in data set removed (session and text), and replaces it with TFIDF vector for each speech.
		data = data.drop(['session', 'text'], axis=1).reset_index()
		data = pd.concat([data, tfidf_df], axis=1)

		# Remove redundant column.
		data = data.drop(["index"],axis=1)

		# Save TFIDF data.
		data.to_csv(data_filename)
		pickle.dump(tfidf, open(obj_filename, "wb"))
		pickle.dump(tfidf_vec, open(vec_filename, "wb"))

	# Read TFIDF data from files.
	else:

		print("Reading TFIDF vector data.")

		data = read_csv(data_filename).drop([""],axis=1)
		tfidf = pickle.load(open(obj_filename, "rb"))
		tfidf_vec = pickle.load(open(vec_filename, "rb"))
		

	return data, tfidf, tfidf_vec


# -------------------
# NAME   : read_csv
# RETURN : dataframe
# DESC   : Reads a csv file into a dataframe.
# -------------------
def read_csv(filename):

	# Opens csv file.
	with open(filename) as f:
		reader = csv.reader(f)
		csv_rows = list(reader)

	# Converts to dataframe.
	data = pd.DataFrame(csv_rows) 

	# Sets column names of dataframe.
	header = data.iloc[0]
	data = data[1:]
	data.columns = header

	return data


# -------------------
# NAME   : add_meta
# RETURN : dataframe
# DESC   : Adds continent and human development index categories to data.
# -------------------
def add_meta(data, hdi_file, country_continent_file, country_name_file):

	print("Adding meta data.")

	# Reads CSV files to get HDI, continent and country name data.
	continent = read_csv(country_continent_file).set_index('CoC').to_dict()
	hdi = read_csv(hdi_file)
	country_names_to_codes = read_csv(country_name_file).set_index("name").to_dict()
	
	# Creates new column in dataframe to store the continent of each country, and inserts correct continent.
	data['continent'] = data['country'].apply(lambda x: continent['CC'][x])

	# Creates new column in dataframe to store the decade of each speech was made.
	data['decade'] = data['year'].apply(lambda x: "2010s" if int(x) >= 2010 else ("2000s" if int(x) >= 2000 else ("1990s" if int(x) >= 1990 else ("1980s" if int(x) >= 1980 else "1970s"))))

	# Replaces country name in HCI data set with the country code.
	hdi["country"] = hdi["country"].apply(lambda x: country_names_to_codes["alpha-3"][x])

	# Replaces HDI value with a code for the band each score is in.
	hdi["hdi"] = hdi["hdi"].apply(lambda x: "VH" if float(x) >= 0.8 else ("HI" if float(x) >= 0.7 else ("ME" if float(x) >= 0.55 else "LO") ))

	# Creates new column in dataframe to store each countries HDI band and inserts. If country does not exist anymore inserts the value NaN.
	code_to_hdi = hdi.set_index("country").to_dict()
	data["hdi"] = data["country"].apply(lambda x: code_to_hdi['hdi'].get(x, "NaN"))

	return data


# -------------------
# NAME   : clean_text
# RETURN : dataframe
# DESC   : removes special characters, numbers and punctuation. Removes stop words, and then stems text.
# -------------------
def clean_text(data, stop_words_file):

	# Get stop word set, and initialise stemmer.
	stop_words = set(stopwords.words('english'))
	stemmer = PorterStemmer()

	# Produces list of country names to use as stop words.
	f = open(stop_words_file, 'r')
	countries = f.readlines()
	f.close()
	countries = [(x.lower()).strip() for x in countries]

	# Adds country names and other common words to stop word set.
	stop_words = stop_words.union(set(countries))
	stop_words = stop_words.union(set(['countries', 'country' 'per', 'cent', 'reform', 'crisis', 'problem', 'global', 'republic', 'welcome', 'east', 'west', 'cooperation', 'democratic', 'operation', 'dialogue', 'small', 'goal', 'address', 'let']))

	print("Removing stop words and stemming text: ")
	rows = data.shape[0]
	for i in range(0, rows):

		# Progress bar.
		p = (i + 1) / rows
		sys.stdout.write('\r')
		sys.stdout.write("[%-20s] %d%%" % ('='*int(20*p), 100*p))
		sys.stdout.flush()

		# Gets speech text
		txt = data.iloc[i]['text']

		# Cleans speech text
		txt = txt.lower()									# Lowercase all text
		txt = re.sub('\s+',' ', txt)						# Remove punctuation
		txt = re.sub(r'\d+', '', txt)						# Remove numbers
		txt = re.sub(r"(?s)<.?>", " ", txt) 				# Remove tags
		txt = re.sub(r"[^A-Za-z0-9(),*!?\'\`]", " ", txt)	# Remove irregular characters
		txt = re.sub("\\\\u(.){4}", " ", txt)				# Remove unicode characters 
		txt = txt.strip()									# Remove leading and trailing spaces.

	    # Tokenise text
		tokenizer = RegexpTokenizer(r'\w+')
		txt_tokens = tokenizer.tokenize(txt)

		# Remoce stop words
		txt = [w for w in txt_tokens if w not in stop_words]

		# Stem words
		for j in range(0, len(txt)):
			txt[j] = stemmer.stem(txt[j])

		# Replaces speech with cleaned and stemmed speech text
		data.iloc[i]['text'] = ' '.join(txt)

	print()

	return data



'''
----------------------------------------------------------------------------------------------------------------------

VISUALISATION

----------------------------------------------------------------------------------------------------------------------
'''


# -------------------
# NAME   : plot_2d
# RETURN : None
# DESC   : Plots a 2D scatter graph and colours points based on given targets.
# -------------------
def plot_2d(x, y, tag_zip, x_label='Component 1', y_label='Component 2'):

	# Retrieves tagging information
	tag = tag_zip[0]
	targets = list(tag_zip[1].keys())
	colours = list(tag_zip[1].values())

	# Creates dataframe of passed plotting axis
	data = pd.concat([x,y,tag],axis=1)
	data.columns = ["x","y","tag"]

	# Converts x and y values to float.
	data["x"] = data["x"].apply(lambda x: float(x))
	data["y"] = data["y"].apply(lambda x: float(x))

	# Defines plot figure.
	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(111)

	# Sets plot axis labels.
	ax.set_xlabel(x_label, fontsize = 15)
	ax.set_ylabel(y_label, fontsize = 15)

	# Plots points and colours based on tagging information.
	for target, colour in zip(targets,colours):
		indicesToKeep = data["tag"] == target
		ax.scatter(data.loc[indicesToKeep, 'x'], data.loc[indicesToKeep, 'y'], c = colour, s = 5)

	# Adds legend and grid to plot.
	ax.legend(targets)
	ax.grid()

	# Shows plot.
	plt.show()


# -------------------
# NAME   : plot_3d
# RETURN : None
# DESC   : Plots a 3D scatter graph and colours points based on given targets.
# -------------------
def plot_3d(x, y, z, tag_zip, x_label='Component 1', y_label='Component 2', z_label='Component 3'):

	# Retrieves tagging information
	tag = tag_zip[0]
	targets = list(tag_zip[1].keys())
	colours = list(tag_zip[1].values())

	# Creates dataframe of passed plotting axis
	data = pd.concat([x,y,z,tag],axis=1)
	data.columns = ["x","y","z","tag"]

	# Converts x, y and z values to float.
	data["x"] = data["x"].apply(lambda x: float(x))
	data["y"] = data["y"].apply(lambda x: float(x))
	data["z"] = data["z"].apply(lambda x: float(x))

	# Defines plot figure.
	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(111, projection='3d') 

	# Sets plot axis labels.
	ax.set_xlabel(x_label, fontsize = 15)
	ax.set_ylabel(y_label, fontsize = 15)
	ax.set_zlabel(z_label, fontsize = 15)

	# Plots points and colours based on tagging information.
	for target, colour in zip(targets,colours):
		indicesToKeep = data['tag'] == target
		ax.scatter(data.loc[indicesToKeep, 'x'], data.loc[indicesToKeep, 'y'], data.loc[indicesToKeep, 'z'], c = colour, s = 5)

	# Adds legend and grid to plot.
	ax.legend(targets)
	ax.grid()

	# Shows plot.
	plt.show()



if __name__ == "__main__":
	main()