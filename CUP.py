# %% [markdown]
# # **<font color="#34ebdb">0.0 IMPORTS</font>**

# %%
!pip install scikeras

# %%
# General
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import warnings

# Model
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, OneHotEncoder

# Neural Networks
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from sklearn.neural_network import MLPClassifier, MLPRegressor
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.regularizers import L1, L2
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier, KerasRegressor
# from keras_tuner import RandomSearch, HyperParameters
from sklearn.exceptions import ConvergenceWarning

# Regression
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
# Support Vector Machines
from sklearn.svm import LinearSVC, SVC, SVR
# Neighbors
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# Tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# Bayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
# Ensemble
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor

# Evaluation
from sklearn.metrics import accuracy_score, f1_score, classification_report, r2_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
from sklearn.metrics.pairwise import paired_distances, euclidean_distances
from sklearn.metrics import make_scorer

# %% [markdown]
# # **<font color="#34ebdb">0.1 FUNCTIONS</font>**

# %%
sns.set_context(font_scale=2, rc={"font.size":10,"axes.titlesize":16,"axes.labelsize":14})
sns.set_style("whitegrid", {'grid.linestyle': '--'})
sns.set_style({'font.family':'serif', 'font.serif':'Computer Modern'})

# %%
# Function to build keras model for CUP regression
def build_fn_cup(n_hidden_units1, n_hidden_units2, n_hidden_units3, learning_rate, lambd, ):
    keras.backend.clear_session()
    tf.random.set_seed(42)

    model = Sequential()
    model.add(Dense(n_hidden_units1, input_dim = X_train.shape[1], kernel_regularizer = L2(lambd), kernel_initializer = 'glorot_normal', activation = 'relu'))
    model.add(Dense(n_hidden_units2, activation = 'relu'))
    model.add(Dense(n_hidden_units3, activation = 'relu'))
    model.add(Dense(y_train.shape[1], activation = 'linear'))
    model.compile(loss = mean_euclidean_error_keras, optimizer = Adam(learning_rate = learning_rate))

    return model

# %%
# Making MEE scorer for gridsearch
# def mean_euclidean_error(y_true, y_pred):
#     return np.mean(euclidean_distances(y_true, y_pred))

def mean_euclidean_error(y_true, y_pred):
    return np.mean(np.sqrt(np.sum(np.square(y_pred-y_true), axis=-1)))

mee_scoring = make_scorer(mean_euclidean_error, greater_is_better = False)

# Keras custom loss function
def mean_euclidean_error_keras(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))

# %%
# Function to get results from regression
def regression_results(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return r2, mse, mae

# %%
# Function to fit mlp classifier with partial fit to get scores and loss
def mlp_fit(mlp, max_iter):
    train_loss = []
    test_loss = []

    for i in range(max_iter):
        mlp.partial_fit(X_train, y_train)
        y_pred = mlp.predict(X_train)
        y_pred_test = mlp.predict(X_test)

        train_loss.append(mean_euclidean_error(y_train, y_pred))
        test_loss.append(mean_euclidean_error(y_test, y_pred_test))

    print('Training loss:   ', round(train_loss[-1],4))
    print('Testing loss:    ', round(test_loss[-1],4), '\n')

    return train_loss, test_loss

# %%
# Function to create score and loss plots
def plot_score_loss(train_loss, test_loss, save, name):
    plt.figure(figsize = (6, 4))
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='test', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    if save == 1:
        plt.savefig(name+'_loss', dpi=400)
    plt.show()

# %%
# Function to apply nested cross validation on a series of models and parameters
def nested_cross_validation(folds, model_to_use, model_params, param_grid):
    outer_kfold = KFold(folds, shuffle=True, random_state=42)
    inner_kfold = KFold(folds, shuffle=True, random_state=77)

    test_scores = []

    X = X_train
    y = y_train

    # Outer K-Fold (Evaluation)
    for train_indices, test_indices in outer_kfold.split(X, y):
        # Split data into train and test
        X_train_nested, y_train_nested = X[train_indices], y[train_indices]
        X_test_nested, y_test_nested = X[test_indices], y[test_indices]

        models = {}

        # Initializing param_grid
        list_params = ['param'+str(i) for i in range(len(model_params))]
        for list_params[0] in param_grid[0]:
            for list_params[1] in param_grid[1]:
                for list_params[2] in param_grid[2]:

                    val_scores = []

                    # Inner K-Fold for each hyper-parameter configuration
                    for selection_indices, val_indices in inner_kfold.split(X_train_nested, y_train_nested):

                        # Split data into selection and validation
                        X_selection, y_selection = X_train_nested[selection_indices], y_train_nested[selection_indices]
                        X_val, y_val = X_train_nested[val_indices], y_train_nested[val_indices]

                        # Fit the model
                        params = {model_params[i]: list_params[i] for i in range(len(model_params))}

                        if name == "Support Vector Machine":
                            model = model_to_use(SVR(**params))
                        else:
                            model = model_to_use(**params)

                        model.fit(X_selection, y_selection)
                        y_pred_val = model.predict(X_val)
                        val_scores.append(mean_euclidean_error(y_val, y_pred_val))

                    # Validation score of a model is the mean over the inner k-folds
                    models[(list_params[0], list_params[1], list_params[2])] = np.mean(val_scores)

        best_params = max(models, key=models.get)

        best_params_dict = {model_params[i]: best_params[i] for i in range(len(model_params))}

        if name == "Support Vector Machine":
            model = model_to_use(SVR(**best_params_dict))
        else:
            model = model_to_use(**best_params_dict)

        model.fit(X_train_nested, y_train_nested)
        y_pred_test = model.predict(X_test_nested)
        test_scores.append(mean_euclidean_error(y_test_nested, y_pred_test))

    avg_mee = round(np.mean(test_scores), 4)
    std_mee = round(np.std(test_scores), 4)

    return avg_mee, std_mee

# %% [markdown]
# # **<font color="#34ebdb">1.0 DATA UNDERSTANDING & PREPARATION</font>**

# %%
# Monto il Drive per accedere ai file, basta avere una scorciatoia alla cartella "CUP" nella cartella principale del vostro drive
from google.colab import drive
drive.mount('/content/drive')

# Creating local files to access more easily
!mkdir dataset

!cp -r /content/drive/MyDrive/CUP /content/dataset

# %%
# Reading csv file to create pandas dataframe, assigning target_x and target_y column names
TR_CUP = pd.read_csv('/content/dataset/CUP/ML-CUP22-TR.csv', header=None, comment='#')
TR_CUP.drop(labels=0, axis=1, inplace=True)
TR_CUP.rename({10: 'target_x', 11: 'target_y'}, axis=1, inplace=True)

# Overview of the structure and info of dataframe
print(TR_CUP.info())
TR_CUP.head()

# %%
# Repeating the previous steps, this time for test set
TS_CUP = pd.read_csv('/content/dataset/CUP/ML-CUP22-TS.csv', header=None, comment='#')
TS_CUP.drop(labels=0, axis=1, inplace=True)
print(TS_CUP.info())
TS_CUP.head()

# %%
# X takes values of attributes 1 to 9, while y takes the targets x and y
X_CUP_train = TR_CUP.values[:,0:9].astype(np.float32)
y_CUP_train = TR_CUP.values[:,9:11].astype(np.float32)

X_CUP_test = TS_CUP.values.astype(np.float32)

# Looking at the shape of the values
print(X_CUP_train.shape, y_CUP_train.shape)
print(X_CUP_test.shape)

# %%
# Validation schema
X_train, X_test, y_train, y_test = train_test_split(X_CUP_train, y_CUP_train, test_size = .1, random_state = 42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size = .2, random_state = 0)
X_test_blind = X_CUP_test

min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
# X_val = min_max_scaler.fit_transform(X_val)
X_test = min_max_scaler.fit_transform(X_test)

print('Dataset:', X_CUP_train.shape[0])
print('TR:     ', X_train.shape[0])
# print('Validation set:   ', X_val.shape[0])
print('TS:     ', X_test.shape[0])
print('Blind:  ', X_test_blind.shape[0])

# %%
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
regression_results(y_test, y_pred)

# %%
# blind_test_results = pd.DataFrame({'Pred x_target': y_pred[:,0], 'Pred y_target': y_pred[:,1]})
test_results = pd.DataFrame({'True x_target': y_test[:,0], 'Pred x_target': y_pred[:,0], 'True y_target': y_test[:,1], 'Pred y_target': y_pred[:,1]})
test_results

# %% [markdown]
# # **<font color="#34ebdb">2.0 NEURAL NETWORKS</font>**

# %% [markdown]
# ### **<font color="#CEFF5E">KERAS</font>**

# %%
model = KerasRegressor(build_fn_cup, n_hidden_units1 = 0, n_hidden_units2 = 0, n_hidden_units3 = 0, learning_rate = 0, lambd = 0, epochs = 150, verbose = 0)

param_grid = {'n_hidden_units1': range(10, 100, 20),
              'n_hidden_units2': range(0, 100, 20),
              'n_hidden_units3': range(0, 100, 20),
              'learning_rate': [.01],
              'lambd': [0],
              'batch_size': [512, 2048],
              }

search = GridSearchCV(estimator = model,
                      param_grid = param_grid,
                      cv = KFold(5, shuffle = True, random_state = 42),
                      scoring = mee_scoring,
                      verbose = 1).fit(X_train, y_train, verbose = 0)

print('Best score:', search.best_score_, '\nBest params', search.best_params_)

# %%
model = KerasRegressor(build_fn_cup, n_hidden_units1 = 30, n_hidden_units2 = 80, n_hidden_units3 = 80, learning_rate = 0, lambd = 0, epochs = 150, verbose = 0)

param_grid = {'learning_rate': [.001, .01, .1, 1],
              'lambd': [.001, .01, .1],
              'batch_size': [512],
              }

search = GridSearchCV(estimator = model,
                      param_grid = param_grid,
                      cv = KFold(5, shuffle = True, random_state = 42),
                      scoring = mee_scoring,
                      verbose = 1).fit(X_train, y_train, verbose = 0)

print('Best score:', search.best_score_, '\nBest params', search.best_params_)

# %%
keras.backend.clear_session()
tf.random.set_seed(42)

es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = 10,
                   restore_best_weights = True)

model = Sequential()
model.add(Dense(30, input_dim = X_train.shape[1], kernel_initializer = 'glorot_normal', activation = 'relu', kernel_regularizer=L2(.01)))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(y_train.shape[1], activation = 'linear'))
model.compile(loss = mean_euclidean_error_keras, optimizer = Adam(learning_rate=.01))
model.summary()

history = model.fit(X_train, y_train,
                    validation_split = .2,
                    callbacks = [es],
                    epochs = 200,
                    batch_size = 512,
                    verbose = 0)

loss = model.evaluate(X_train, y_train, verbose = 0)
print('Training MEE: ', history.history['loss'][-1])
print('Validation MEE', history.history['val_loss'][-1])

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss', linestyle="--")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('CUP_Keras.png', dpi=400)
plt.show()

# %%
iterations = 50
mee_train = []
mee_val = []
mee_test = []

list_act = ['relu', 'sigmoid', 'tanh']
list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

for activation in list_act:
    print('------', activation,'------\n')
    for i in list_seeds:
        keras.backend.clear_session()
        tf.random.set_seed(i)

        es = EarlyStopping(monitor = 'val_loss',
                          mode = 'min',
                          patience = 10,
                          restore_best_weights = True)

        model = Sequential()
        model.add(Dense(30, input_dim = X_train.shape[1], kernel_initializer = 'glorot_normal', activation = activation, kernel_regularizer=L2(.01)))
        model.add(Dense(80, activation = activation))
        model.add(Dense(80, activation = activation))
        model.add(Dense(y_train.shape[1], activation = 'linear'))
        model.compile(loss = mean_euclidean_error_keras, optimizer = Adam(learning_rate=.01))

        history = model.fit(X_train, y_train,
                            validation_split = .2,
                            callbacks = [es],
                            epochs = 200,
                            batch_size = 512,
                            verbose = 0)

        mee_train.append(history.history['loss'][-1])
        mee_val.append(history.history['val_loss'][-1])

    print('Avg mee train:', round(np.mean(mee_train),4))
    print('Std dev:      ', round(np.std(mee_train),4))
    print('Avg mee val:  ', round(np.mean(mee_val),4))
    print('Std dev:      ', round(np.std(mee_val),4),'\n')

# %%
iterations = 50
mee_train = []
mee_val = []
mee_test = []

list_init = ['glorot_normal', 'random_normal', 'he_normal']
list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

for initializer in list_init:
    print('------', initializer,'------\n')
    for i in list_seeds:
        keras.backend.clear_session()
        tf.random.set_seed(i)

        es = EarlyStopping(monitor = 'val_loss',
                          mode = 'min',
                          patience = 10,
                          restore_best_weights = True)

        model = Sequential()
        model.add(Dense(30, input_dim = X_train.shape[1], kernel_initializer = initializer, activation = 'relu', kernel_regularizer=L2(.01)))
        model.add(Dense(80, activation = 'relu'))
        model.add(Dense(80, activation = 'relu'))
        model.add(Dense(y_train.shape[1], activation = 'linear'))
        model.compile(loss = mean_euclidean_error_keras, optimizer = Adam(learning_rate=.01))

        history = model.fit(X_train, y_train,
                            validation_split = .2,
                            callbacks = [es],
                            epochs = 200,
                            batch_size = 512,
                            verbose = 0)

        mee_train.append(history.history['loss'][-1])
        mee_val.append(history.history['val_loss'][-1])

    print('Avg mee train:', round(np.mean(mee_train),4))
    print('Std dev:      ', round(np.std(mee_train),4))
    print('Avg mee val:  ', round(np.mean(mee_val),4))
    print('Std dev:      ', round(np.std(mee_val),4),'\n')

# %%
iterations = 50
mee_train = []
mee_val = []
mee_test = []

list_eta = [.001, .01, .1, 1]
list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

for eta in list_eta:
    print('------', eta,'------\n')
    for i in list_seeds:
        keras.backend.clear_session()
        tf.random.set_seed(i)

        es = EarlyStopping(monitor = 'val_loss',
                          mode = 'min',
                          patience = 10,
                          restore_best_weights = True)

        model = Sequential()
        model.add(Dense(30, input_dim = X_train.shape[1], kernel_initializer = 'glorot_normal', activation = 'relu', kernel_regularizer=L2(.01)))
        model.add(Dense(80, activation = 'relu'))
        model.add(Dense(80, activation = 'relu'))
        model.add(Dense(y_train.shape[1], activation = 'linear'))
        model.compile(loss = mean_euclidean_error_keras, optimizer = Adam(learning_rate=eta))

        history = model.fit(X_train, y_train,
                            validation_split = .2,
                            callbacks = [es],
                            epochs = 200,
                            batch_size = 512,
                            verbose = 0)

        mee_train.append(history.history['loss'][-1])
        mee_val.append(history.history['val_loss'][-1])

    print('Avg mee train:', round(np.mean(mee_train),4))
    print('Std dev:      ', round(np.std(mee_train),4))
    print('Avg mee val:  ', round(np.mean(mee_val),4))
    print('Std dev:      ', round(np.std(mee_val),4),'\n')

# %%
iterations = 50

train_loss = []
train_r2 = []
train_mse = []
train_mae = []

test_loss = []
test_r2 = []
test_mse = []
test_mae = []

list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

for i in list_seeds:
  keras.backend.clear_session()
  tf.random.set_seed(i)

  es = EarlyStopping(monitor = 'val_loss',
                    mode = 'min',
                    patience = 10,
                    restore_best_weights = True)

  model = Sequential()
  model.add(Dense(30, input_dim = X_train.shape[1], kernel_initializer = 'glorot_normal', activation = 'relu', kernel_regularizer=L2(.01)))
  model.add(Dense(80, activation = 'relu'))
  model.add(Dense(80, activation = 'relu'))
  model.add(Dense(y_train.shape[1], activation = 'linear'))
  model.compile(loss = mean_euclidean_error_keras, optimizer = Adam(learning_rate=.001))

  history = model.fit(X_train, y_train,
                         validation_split = .2,
                         callbacks = [es],
                         epochs = 200,
                         batch_size = 512,
                         verbose = 0)

  # train_loss.append(history.history['loss'][-1])
  y_pred = model.predict(X_train, verbose = 0)
  y_pred_test = model.predict(X_test, verbose = 0)

  tr_r2, tr_mse, tr_mae = regression_results(y_train, y_pred)
  ts_r2, ts_mse, ts_mae = regression_results(y_test, y_pred_test)

  train_loss.append(mean_euclidean_error(y_train, y_pred))
  train_r2.append(tr_r2)
  train_mse.append(tr_mse)
  train_mae.append(tr_mae)

  test_loss.append(mean_euclidean_error(y_test, y_pred_test))
  test_r2.append(ts_r2)
  test_mse.append(ts_mse)
  test_mae.append(ts_mae)

print('Train loss:  ', round(np.mean(train_loss),4))
print('Std dev:     ', round(np.std(train_loss),4))
print('Train r2:    ', round(np.mean(train_r2),4))
print('Std dev:     ', round(np.std(train_r2),4))
print('Train mse:   ', round(np.mean(train_mse),4))
print('Std dev:     ', round(np.std(train_mse),4))
print('Train mae:   ', round(np.mean(train_mae),4))
print('Std dev:     ', round(np.std(train_mae),4))

print('Testing loss:', round(np.mean(test_loss),4))
print('Std dev:     ', round(np.std(test_loss),4))
print('Testing r2:  ', round(np.mean(test_r2),4))
print('Std dev:     ', round(np.std(test_r2),4))
print('Testing mse: ', round(np.mean(test_mse),4))
print('Std dev:     ', round(np.std(test_mse),4))
print('Testing mae: ', round(np.mean(test_mae),4))
print('Std dev:     ', round(np.std(test_mae),4))

# %%
list_seeds

# %%
keras.backend.clear_session()
tf.random.set_seed(45)

es = EarlyStopping(monitor = 'val_loss',
                  mode = 'min',
                  patience = 5,
                  restore_best_weights = True)

model = Sequential()
model.add(Dense(30, input_dim = X_train.shape[1], kernel_initializer = 'glorot_normal', activation = 'relu', kernel_regularizer=L2(.01)))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(y_train.shape[1], activation = 'linear'))
model.compile(loss = mean_euclidean_error_keras, optimizer = Adam(learning_rate=.001))

history = model.fit(X_train, y_train,
                    validation_split = .2,
                    callbacks = [es],
                    epochs = 150,
                    batch_size = 512,
                    verbose = 0)

y_pred = model.predict(X_test, verbose = 0)
print('TR:', round(history.history['loss'][-1],4))
print('VL:', round(history.history['val_loss'][-1],4))
print('TS:', round(mean_euclidean_error(y_test, y_pred),4))

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss', linestyle="--")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('CUP_Keras.png', dpi=400)
plt.show()

# %% [markdown]
# ### **<font color="#CEFF5E">SCIKITLEARN'S MLP</font>**

# %%
# different learning rate schedules and momentum parameters
params = [
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "momentum": 0,
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "nesterovs_momentum": True,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "adaptive",
        "momentum": 0,
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "adaptive",
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "adaptive",
        "nesterovs_momentum": True,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "adam",
        "learning_rate_init": 0.02
    },
]

labels = [
    "constant learning w/o momentum",
    "constant with momentum",
    "constant with Nesterov's momentum",
    "adaptive learning w/o momentum",
    "adaptive with momentum",
    "adaptive with Nesterov's momentum",
    "adam",
]

# %%
models = []
for label, param in zip(labels, params):
    print("training", label)
    mlp = MLPRegressor(random_state=42, max_iter=100, **param)

    # some parameter combinations will not converge so they are ignored here
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
      mlp.fit(X_train, y_train)

    models.append(mlp)
    # print("Score: %f" % mlp.score(X3_val, y3_val))
    print("Loss: %f" % mlp.loss_,'\n')

# %%
plt.figure(figsize=(12,4))
for i, label in zip(range(len(models)), labels):
  plt.plot(models[i].loss_curve_, label = label)
plt.title("Comparing different learning methods for MLP")
plt.ylim(0,50)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
plt.show()

# %%
max_iter = 150
model = MLPRegressor(solver='adam', max_iter=max_iter, random_state=42)

param_grid = {'hidden_layer_sizes': [(10, 10, 10), (30, 30, 30), (50, 50, 50),(70, 70, 70), (100, 100, 100)],
              'momentum': [.01, .1, 1],
              'learning_rate_init': [.01, .1, 1],
              'batch_size': [512, 'auto']}

with warnings.catch_warnings():
  warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
  search = GridSearchCV(estimator = model,
                        param_grid = param_grid,
                        cv = KFold(5, shuffle = True, random_state = 42),
                        scoring = mee_scoring,
                        verbose = 1).fit(X_train, y_train)

print('Best score:', search.best_score_, '\nBest params', search.best_estimator_)

# %%
max_iter = 150
model = MLPRegressor(solver='adam', max_iter=max_iter, random_state=42)

param_grid = {'hidden_layer_sizes': [(100, 100, 100), (125, 125, 125), (150, 150, 150)],
              'momentum': [.08, .01, .03],
              'learning_rate_init': [.08, .01, .03],
              'batch_size': ['auto']}

with warnings.catch_warnings():
  warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
  search = GridSearchCV(estimator = model,
                        param_grid = param_grid,
                        cv = KFold(5, shuffle = True, random_state = 42),
                        scoring = mee_scoring,
                        verbose = 1).fit(X_train, y_train)

print('Best score:', search.best_score_, '\nBest params', search.best_estimator_)

# %%
iterations = 50

train_loss = []
train_r2 = []
train_mse = []
train_mae = []

test_loss = []
test_r2 = []
test_mse = []
test_mae = []

list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

for i in list_seeds:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        mlp = MLPRegressor(batch_size = 'auto',
                           hidden_layer_sizes = (125, 125, 125),
                           learning_rate_init = .01,
                           max_iter = 150,
                           momentum = .08,
                           random_state = i)

        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_train)
        y_pred_test = mlp.predict(X_test)

        tr_r2, tr_mse, tr_mae = regression_results(y_train, y_pred)
        ts_r2, ts_mse, ts_mae = regression_results(y_test, y_pred_test)

        train_loss.append(mean_euclidean_error(y_train, y_pred))
        train_r2.append(tr_r2)
        train_mse.append(tr_mse)
        train_mae.append(tr_mae)

        test_loss.append(mean_euclidean_error(y_test, y_pred_test))
        test_r2.append(ts_r2)
        test_mse.append(ts_mse)
        test_mae.append(ts_mae)


print('Train loss:', round(np.mean(train_loss),4))
print('Std dev:   ', round(np.std(train_loss),4))
print('Train r2:  ', round(np.mean(train_r2),4))
print('Std dev:   ', round(np.std(train_r2),4))
print('Train mse: ', round(np.mean(train_mse),4))
print('Std dev:   ', round(np.std(train_mse),4))
print('Train mae: ', round(np.mean(train_mae),4))
print('Std dev:   ', round(np.std(train_mae),4))

print('Testing loss:', round(np.mean(test_loss),4))
print('Std dev:     ', round(np.std(test_loss),4))
print('Testing r2:  ', round(np.mean(test_r2),4))
print('Std dev:     ', round(np.std(test_r2),4))
print('Testing mse: ', round(np.mean(test_mse),4))
print('Std dev:     ', round(np.std(test_mse),4))
print('Testing mae: ', round(np.mean(test_mae),4))
print('Std dev:     ', round(np.std(test_mae),4))

# %%
list_seeds

# %%
mlp = MLPRegressor(batch_size = 'auto',
                   hidden_layer_sizes = (125, 125, 125),
                   learning_rate_init = .01,
                   max_iter = 100,
                   momentum = .08,
                   random_state = 32)

train_loss, test_loss = mlp_fit(mlp, max_iter = 100)

plot_score_loss(train_loss = train_loss,
                test_loss = test_loss,
                save = 1,
                name = 'CUP_sklearn')

# %% [markdown]
# # **<font color="#34ebdb">3.0 NESTED CROSS VALIDATION</font>**

# %%
# Models, parameters and hyperparameters to test with nested cross validation for MONK 1
knn_params = ['n_neighbors', 'weights', 'metric']
knn_param_grid = [range(2,78,2), ["uniform", "distance"], ["euclidean", "cityblock", "chebyshev"]]

lr_params = ['fit_intercept', 'positive', 'n_jobs']
lr_param_grid = [[True, False], [False, True], [-1]]

rf_params = ['criterion', 'max_depth', 'n_estimators']
rf_param_grid = [['squared_error', 'absolute_error'], [1, 3, 5, 7, 10, None], [50, 100, 150]]

ridge_params = ['alpha', 'max_iter', 'solver']
ridge_param_grid = [[.1, 1., 10, 100], [None, 500, 1000], ['auto', 'svd', 'lsqr', 'saga']]

las_params = ['alpha', 'max_iter', 'tol']
las_param_grid = [[.001, .01, .1, 1.], [500, 1000, 1500], [5e-5, 1e-4, 5e-4]]

nested_dict = {
    'K-Nearest Neighbors': [KNeighborsRegressor, knn_params, knn_param_grid],
    'Linear Regression': [LinearRegression, lr_params, lr_param_grid],
    'Ridge': [Ridge, ridge_params, ridge_param_grid],
    'Lasso': [Lasso, las_params, las_param_grid]
    }

# %%
# Ignoring convergence warning for logistic regression when max_iter is not enough to converge
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    for name, [model_to_use, model_params, param_grid] in nested_dict.items():
        avg_mee, std_mee = nested_cross_validation(folds = 5,
                                                   model_to_use = model_to_use,
                                                   model_params = model_params,
                                                   param_grid = param_grid)
        print('------', name, '------')
        print('Average MEE:', avg_mee)
        print('Std dev:    ', std_mee, '\n')

# %%
# Models, parameters and hyperparameters to test with nested cross validation for MONK 1
rf_params = ['criterion', 'max_depth', 'n_estimators']
rf_param_grid = [['squared_error', 'absolute_error'], [1, 3, 5], [50, 100, 150]]

nested_dict = {'Random Forests': [RandomForestRegressor, rf_params, rf_param_grid]}

# Ignoring convergence warning for logistic regression when max_iter is not enough to converge
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    for name, [model_to_use, model_params, param_grid] in nested_dict.items():
        avg_mee, std_mee = nested_cross_validation(folds = 5,
                                                   model_to_use = model_to_use,
                                                   model_params = model_params,
                                                   param_grid = param_grid)
        print('------', name, '------')
        print('Average MEE:', avg_mee)
        print('Std dev:    ', std_mee, '\n')

# %% [markdown]
# ## **<font color="#CEFF5E">K NEIGHBORS</font>**

# %%
model = KNeighborsRegressor()

parameters = {'n_neighbors': range(2,78,2),
              'weights': ["uniform", "distance"],
              'metric' : ["euclidean", "cityblock", "chebyshev"]
             }

search = GridSearchCV(estimator = model,
                      param_grid = parameters,
                      cv = KFold(5, shuffle=True, random_state=42),
                      n_jobs = -1,
                      scoring=mee_scoring,
                      verbose = 1).fit(X_train, y_train)

print('Best score:', search.best_score_, '\nBest params', search.best_estimator_)

# %%
model = KNeighborsRegressor()

parameters = {'n_neighbors': [18, 19, 20, 21, 22, 23, 24],
              'weights': ["uniform", "distance"],
              'metric' : ["euclidean", "cityblock", "chebyshev"]
             }

search = GridSearchCV(estimator = model,
                      param_grid = parameters,
                      cv = KFold(5, shuffle=True, random_state=42),
                      n_jobs = -1,
                      scoring=mee_scoring,
                      verbose = 1).fit(X_train, y_train)

print('Best score:', search.best_score_, '\nBest params', search.best_estimator_)

# %%
model = KNeighborsRegressor(metric='euclidean', n_neighbors=22, weights='distance')
model.fit(X_train, y_train)

train_loss = []
train_r2 = []
train_mse = []
train_mae = []

test_loss = []
test_r2 = []
test_mse = []
test_mae = []

y_pred = model.predict(X_train)
y_pred_test = model.predict(X_test)

tr_r2, tr_mse, tr_mae = regression_results(y_train, y_pred)
ts_r2, ts_mse, ts_mae = regression_results(y_test, y_pred_test)

train_loss.append(mean_euclidean_error(y_train, y_pred))
train_r2.append(tr_r2)
train_mse.append(tr_mse)
train_mae.append(tr_mae)

test_loss.append(mean_euclidean_error(y_test, y_pred_test))
test_r2.append(ts_r2)
test_mse.append(ts_mse)
test_mae.append(ts_mae)

print('Train loss:  ', round(np.mean(train_loss),4))
print('Train r2:    ', round(np.mean(train_r2),4))
print('Train mse:   ', round(np.mean(train_mse),4))
print('Train mae:   ', round(np.mean(train_mae),4))

print('Testing loss:', round(np.mean(test_loss),4))
print('Testing r2:  ', round(np.mean(test_r2),4))
print('Testing mse: ', round(np.mean(test_mse),4))
print('Testing mae: ', round(np.mean(test_mae),4))

# %% [markdown]
# ## **<font color="#CEFF5E">RAMDOM FOREST</font>**

# %%
model = RandomForestRegressor()

parameters = {'n_estimators': [50, 100, 150],
              'criterion': ['squared_error', 'absolute_error'],
              'max_depth' : [1, 3, 5]
             }

search = GridSearchCV(estimator = model,
                      param_grid = parameters,
                      cv = KFold(5, shuffle=True, random_state=42),
                      n_jobs = -1,
                      scoring= mee_scoring,
                      verbose = 1).fit(X_train, y_train)

print('Best score:', search.best_score_, '\nBest params', search.best_estimator_)

# %%
model = RandomForestRegressor()

parameters = {'n_estimators': [100],
              'criterion': ['absolute_error'],
              'max_depth' : [5, 10, 15]
             }

search = GridSearchCV(estimator = model,
                      param_grid = parameters,
                      cv = KFold(5, shuffle=True, random_state=42),
                      n_jobs = -1,
                      scoring= mee_scoring,
                      verbose = 1).fit(X_train, y_train)

print('Best score:', search.best_score_, '\nBest params', search.best_estimator_)

# %%
iterations = 20

train_loss = []
train_r2 = []
train_mse = []
train_mae = []

test_loss = []
test_r2 = []
test_mse = []
test_mae = []

list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

for i in list_seeds:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        model = RandomForestRegressor(criterion='absolute_error', max_depth=10, random_state=i)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        tr_r2, tr_mse, tr_mae = regression_results(y_train, y_pred)
        ts_r2, ts_mse, ts_mae = regression_results(y_test, y_pred_test)

        train_loss.append(mean_euclidean_error(y_train, y_pred))
        train_r2.append(tr_r2)
        train_mse.append(tr_mse)
        train_mae.append(tr_mae)

        test_loss.append(mean_euclidean_error(y_test, y_pred_test))
        test_r2.append(ts_r2)
        test_mse.append(ts_mse)
        test_mae.append(ts_mae)


print('Train loss:', round(np.mean(train_loss),4))
print('Std dev:   ', round(np.std(train_loss),4))
print('Train r2:  ', round(np.mean(train_r2),4))
print('Std dev:   ', round(np.std(train_r2),4))
print('Train mse: ', round(np.mean(train_mse),4))
print('Std dev:   ', round(np.std(train_mse),4))
print('Train mae: ', round(np.mean(train_mae),4))
print('Std dev:   ', round(np.std(train_mae),4))
print('-------------')
print('Testing loss:', round(np.mean(test_loss),4))
print('Std dev:     ', round(np.std(test_loss),4))
print('Testing r2:  ', round(np.mean(test_r2),4))
print('Std dev:     ', round(np.std(test_r2),4))
print('Testing mse: ', round(np.mean(test_mse),4))
print('Std dev:     ', round(np.std(test_mse),4))
print('Testing mae: ', round(np.mean(test_mae),4))
print('Std dev:     ', round(np.std(test_mae),4))

# %% [markdown]
# ## **<font color="#CEFF5E">RIDGE</font>**

# %%
model = Ridge()

parameters = {'alpha': [.0001, .0002, .0003, .01, 1, 2, 5],
              'tol': [.00001, .0001, .01, .1, 1],
              'solver': ['auto', 'svd', 'saga', 'lsqr', 'cholesky'],
             }

search = GridSearchCV(estimator = model,
                      param_grid = parameters,
                      cv = KFold(5, shuffle=True, random_state=42),
                      n_jobs = -1,
                      scoring=mee_scoring,
                      verbose = 1).fit(X_train, y_train)

print('Best score:', search.best_score_, '\nBest params', search.best_estimator_)

# %%
model = Ridge()

parameters = {'alpha': [.3, .5, .7, 1, 1.5, 2, 3, 5],
              'tol': [.00001, .0001, .09, .01, .05, .1, 1],
              'solver': ['auto', 'svd', 'saga', 'lsqr', 'cholesky'],
             }

search = GridSearchCV(estimator = model,
                      param_grid = parameters,
                      cv = KFold(5, shuffle=True, random_state=42),
                      n_jobs = -1,
                      scoring=mee_scoring,
                      verbose = 1).fit(X_train, y_train)

print('Best score:', search.best_score_, '\nBest params', search.best_estimator_)

# %%
model = Ridge(alpha=0.5, solver='saga', tol=0.09)
model.fit(X_train, y_train)

train_loss = []
train_r2 = []
train_mse = []
train_mae = []

test_loss = []
test_r2 = []
test_mse = []
test_mae = []

y_pred = model.predict(X_train)
y_pred_test = model.predict(X_test)

tr_r2, tr_mse, tr_mae = regression_results(y_train, y_pred)
ts_r2, ts_mse, ts_mae = regression_results(y_test, y_pred_test)

train_loss.append(mean_euclidean_error(y_train, y_pred))
train_r2.append(tr_r2)
train_mse.append(tr_mse)
train_mae.append(tr_mae)

test_loss.append(mean_euclidean_error(y_test, y_pred_test))
test_r2.append(ts_r2)
test_mse.append(ts_mse)
test_mae.append(ts_mae)

print('Train loss:  ', round(np.mean(train_loss),4))
print('Train r2:    ', round(np.mean(train_r2),4))
print('Train mse:   ', round(np.mean(train_mse),4))
print('Train mae:   ', round(np.mean(train_mae),4))
print('--------')
print('Testing loss:', round(np.mean(test_loss),4))
print('Testing r2:  ', round(np.mean(test_r2),4))
print('Testing mse: ', round(np.mean(test_mse),4))
print('Testing mae: ', round(np.mean(test_mae),4))

# %% [markdown]
# # **<font color="#34ebdb">4.0 FINAL MODEL</font>**

# %%
final_model = MLPRegressor(batch_size = 'auto',
                     hidden_layer_sizes = (125, 125, 125),
                     learning_rate_init = .01,
                     max_iter = 100,
                     momentum = .08,
                     random_state = 32).fit(X_CUP_train, y_CUP_train)

y_pred = final_model.predict(X_CUP_test)

# %%
train_loss, test_loss = mlp_fit(final_model, max_iter = 100)

plot_score_loss(train_loss = train_loss,
                test_loss = test_loss,
                save = 0,
                name = 'CUP_sklearn')

# %%
mee_final_model = cross_val_score(model, X_CUP_train, y_CUP_train, cv=5, scoring=mee_scoring)
print('VL loss:', np.mean(mee_final_model))

# %%
blind_test_results = pd.DataFrame({'Pred x_target': y_pred[:,0], 'Pred y_target': y_pred[:,1]})
blind_test_results.to_csv('Pending-Name_ML-CUP22-TS.csv')
blind_test_results

# %%



