import tensorflow as tf
import tflearn 
import pandas as pd
import tempfile 
import urllib


import tempfile
import urllib
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "maritial_status", "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
df_train = pd.read_csv(train-file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["incomebracket"].apply(lambda x: ">50K" in x)).astype(int)

CATEGORICAL_COLUMNS = ["workclass", "education", "maritial_status", "occupation", "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

def input_fn(df):
	continuous_cols = {k: tf.constant(df[k].values)for k in CONTINUOUS_COLUMNS}

	categorical_cols = {k: tf.SparseTensor(indices = [[i, 0]for i in range(df[k].size)],values=df[k].values,dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}

	feature_cols = dict(continuous_cols.items() + categorical_cols.items())
	label = tf.constant(df[LABEL_COLUMN].values)
	return feature_cols, label

def train_input_fn():
	return input_fn(df_train)

def eval_input_fn():
	return input_fn(df_test)

gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])
education = tf.contrib.layers.sparse-column_with_hash_bucket("education", hash_bucket_size = 1000)
race = tf.contrib.layers.sparse-column_with_hash_bucket("race", hash_bucket_size = 100)
maritial_status = tf.contrib.layers.sparse_column_with_hash_bucket("maritial_status", hash_bucket_size = 100)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size = 100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size = 100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size = 1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size = 1000)

age = tf.contrib.layers.real_valued_column("age")
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hourse_per_week")

age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18,25,30,35,40,45,50,55,60,65])

education_x_occupation = tf.contrib.layers.crossed_column([education, occupation]), hash_bucket_size=int(1e4)

age_buckets_x_education_x_occupation = tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6))

model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassification(feature_columns=[gender, native_country, education, occupation ,workclass, maritial_status, race,age_buckets, education_x_occupation, age_buckets_x_education_x_occupation], model_dir= model_dir)

m.fit(input_fn = train_input_fn, steps=200)

results = m.evaluate(input_fn=eval.eval_input_fn, steps=1)
for key in sorted(results):
	print("%s: %s" % (key, results[key]))
