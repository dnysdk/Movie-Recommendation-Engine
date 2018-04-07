import re
import sys
import pandas
import os
import zipfile
import requests
import tqdm
from sklearn import dummy, metrics, cross_validation, ensemble

import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0,3"
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

base_dir = './.ml-1m'

# Download the dataset. It's small, only about 6 MB.
if not os.path.exists(base_dir):
    url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    response = requests.get(url, stream=True)
    total_length = response.headers.get('content-length')
    bar = tqdm.tqdm_notebook(total=int(total_length))
    with open('./ml-1m.zip', 'wb') as f:
        for data in response.iter_content(chunk_size=4096):
            f.write(data)
            bar.update(4096)
    zip_ref = zipfile.ZipFile('./ml-1m.zip', 'r')
    zip_ref.extractall('.')
    zip_ref.close()
    
# Read in the dataset, and do a little preprocessing,
# mostly to set the column datatypes.
users = pandas.read_csv(base_dir+'/users.dat', sep='::', 
                        engine='python', 
                        names=['userid', 'gender', 'age', 'occupation', 'zip']).set_index('userid')
ratings = pandas.read_csv(base_dir+'/ratings.dat', engine='python', 
                          sep='::', names=['userid', 'movieid', 'rating', 'timestamp'])
movies = pandas.read_csv(base_dir+'/movies.dat', engine='python',
                         sep='::', names=['movieid', 'title', 'genre']).set_index('movieid')

#print(movies)
#sys.exit(1)
#print movies.title.str.contains('\(\d\d\d\d\)',regex=True)
movies['year'] = movies.title.str.extract('(\d\d\d\d)',expand=True)
#print movies['year']
#sys.exit(0)

genre_values = ["Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]

genre_hash = {}
for i,val in enumerate(genre_values):
    genre_hash[val] = i

movies['genre'] = movies.genre.str.split('|')

movies.year = movies.year.astype('category')
users.age = users.age.astype('category')
users.gender = users.gender.astype('category')
users.occupation = users.occupation.astype('category')
ratings.movieid = ratings.movieid.astype('category')
ratings.userid = ratings.userid.astype('category')

# Also, make vectors of all the movie ids and user ids. These are
# pandas categorical data, so they range from 1 to n_movies and 1 to n_users, respectively.
movieid = ratings.movieid.cat.codes.values
userid = ratings.userid.cat.codes.values

movie_year = np.zeros((ratings.shape[0], 1))
movie_genre = np.zeros((ratings.shape[0], len(genre_values)))
for i,m_id in enumerate(ratings.movieid): 
    movie_year[i,0] = movies.loc[m_id]['year']
    for val in movies.loc[m_id]['genre']:
        movie_genre[i,genre_hash[val]] = 1
        
# And finally, set up a y variable with the rating,
# as a one-hot encoded matrix.
#
# note the '- 1' for the rating. That's because ratings
# go from 1 to 5, while the matrix columns go from 0 to 4

y = np.zeros((ratings.shape[0], 5))
y[np.arange(ratings.shape[0]), ratings.rating - 1] = 1

import pickle
pickle.dump(movie_genre,open(base_dir+'/moviegenre.pkl','wb'))
pickle.dump(movie_year,open(base_dir+'/movieyear.pkl','wb'))
pickle.dump(y,open(base_dir+'/y.pkl','wb'))

print(len(movieid))
print(len(userid))
print(movie_genre[0:3])
print("*"*100)
print(movie_year[0:3])

abc = pandas.DataFrame({})
abc['movieid'] = ratings.movieid
abc['userid'] = ratings.userid
abc['year'] = movie_year


# Dummy classifier! Just see how well stupid can do.
pred = dummy.DummyClassifier(strategy='prior')
pred.fit(abc, ratings.rating)
print(metrics.mean_absolute_error(ratings.rating, pred.predict(abc)))


#2d to 1d array
movie_year = movie_year.ravel()

# Count the movies and users
n_movies = movies.shape[0]
n_users = users.shape[0]

# Now, the deep learning classifier

# First, we take the movie and vectorize it.
# The embedding layer is normally used for sequences (think, sequences of words)
# so we need to flatten it out.
# The dropout layer is also important in preventing overfitting
movie_input = keras.layers.Input(shape=[1])
movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 32)(movie_input))
movie_vec = keras.layers.Dropout(0.5)(movie_vec)

movie_year_input = keras.layers.Input(shape=[1])
movie_year_vec = keras.layers.Flatten()(keras.layers.Embedding(2018 + 1, 32)(movie_year_input))
movie_year_vec = keras.layers.Dropout(0.5)(movie_year_vec)

movie_genre_input = keras.layers.Input(shape=[18])
#movie_genre_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 32)(movie_genre_input))
movie_genre_vec = keras.layers.Dropout(0.5)(movie_genre_input)

# Same thing for the users
user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 32)(user_input))
user_vec = keras.layers.Dropout(0.5)(user_vec)

# Next, we join them all together and put them
# through a pretty standard deep learning architecture
input_vecs = keras.layers.merge([movie_vec, movie_year_vec, movie_genre_vec, user_vec], mode='concat')
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(256, activation='relu')(input_vecs))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(256, activation='relu')(nn))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dense(256, activation='relu')(nn)

# Finally, we pull out the result!
result = keras.layers.Dense(5, activation='softmax')(nn)

# And make a model from it that we can actually run.
model = kmodels.Model([movie_input, movie_year_input, movie_genre_input, user_input], result)

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint
# checkpoint
filepath=base_dir+"/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]


# If we wanted to inspect part of the model, for example, to look
# at the movie vectors, here's how to do it. You don't need to 
# compile these models unless you're going to train them.
#final_layer = kmodels.Model([movie_input, movie_year_input, movie_genre_input, user_input], nn)
#movie_vec = kmodels.Model(movie_input, movie_vec)

# Split the data into train and test sets...
a_movieid, b_movieid, a_movie_year, b_movie_year, a_movie_genre, b_movie_genre, a_userid, b_userid, a_y, b_y = cross_validation.train_test_split(movieid, movie_year, movie_genre, userid, y)


# And of _course_ we need to make sure we're improving, so we find the MAE before
# training at all.
metrics.mean_absolute_error(np.argmax(b_y, 1)+1, np.argmax(model.predict([b_movieid, b_movie_year, b_movie_genre, b_userid]), 1)+1)

print(a_movie_genre[:3])
print(a_movieid)
print(a_movie_genre.shape)
print(a_movieid.shape)
model.summary()


try:
    history = model.fit([a_movieid, a_movie_year, a_movie_genre, a_userid], a_y, nb_epoch=20, callbacks=callbacks_list, verbose=1, validation_data=([b_movieid, b_movie_year, b_movie_genre, b_userid], b_y))
    plot(history.history['loss'])
    plot(history.history['val_loss'])
except KeyboardInterrupt:
    pass
model.save(base_dir+'/second.h5')

import matplotlib.pyplot as plt
print(history.history['acc'])
print(history.history['val_acc'])

vals = {'acc':history.history['acc'],
        'val_acc':history.history['val_acc'],
        'loss':history.history['loss'],
        'val_loss':history.history['val_loss']
       }
pickle.dump(vals,open(base_dir+"/data_history.pkl","wb"))

plot(vals['acc'])
plot(vals['val_acc'])

try:
    history = model.fit([a_movieid, a_movie_year, a_movie_genre, a_userid], a_y, nb_epoch=20, callbacks=callbacks_list, verbose=1, validation_data=([b_movieid, b_movie_year, b_movie_genre, b_userid], b_y))
    plot(history.history['loss'])
    plot(history.history['val_loss'])
except KeyboardInterrupt:
    pass
model.save(base_dir+'/third.h5')

vals['loss']+=history.history['loss']
vals['val_loss']+=history.history['val_loss']
vals['acc']+=history.history['acc']
vals['val_acc']+=history.history['val_acc']

pickle.dump(vals,open(base_dir+"/data_history.pkl","wb"))

plt.figure(2)
plt.plot(vals['loss'])
plt.plot(vals['val_loss'])
plt.figure(3)
plt.plot(vals['acc'])
plt.plot(vals['val_acc'])

# This is the number that matters. It's the held out 
# test set score. Note the + 1, because np.argmax will
# go from 0 to 4, while our ratings go 1 to 5.
metrics.mean_absolute_error(
    np.argmax(b_y, 1)+1, 
    np.argmax(model.predict([b_movieid, b_movie_year, b_movie_genre, b_userid]), 1)+1)

# For comparison's sake, here's the score on the training set.
metrics.mean_absolute_error(
    np.argmax(a_y, 1)+1, 
    np.argmax(model.predict([a_movieid, a_movie_year, a_movie_genre, a_userid]), 1)+1)

import os
from keras.models import load_model
for ff in os.listdir(base_dir):
    if ff.endswith('.hdf5'):
        model1 = load_model(base_dir+"/"+ff)
        print('Model:',ff)
        print('TEST:')
        print metrics.mean_absolute_error(
            np.argmax(b_y, 1)+1, 
            np.argmax(model1.predict([b_movieid, b_movie_year, b_movie_genre, b_userid]), 1)+1)
        print('')
        print('TRAIN:')
        print metrics.mean_absolute_error(
            np.argmax(a_y, 1)+1, 
            np.argmax(model1.predict([a_movieid, a_movie_year, a_movie_genre, a_userid]), 1)+1)
        print('')

