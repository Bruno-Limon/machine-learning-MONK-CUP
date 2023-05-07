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
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics import make_scorer

# %% [markdown]
# # **<font color="#34ebdb">1.0 DATA UNDERSTANDING & PREPARATION</font>**

# %%
# Monto il Drive per accedere ai file, basta avere una scorciatoia alle cartelle "MONK" nella cartella principale del vostro drive
from google.colab import drive
drive.mount('/content/drive')

# Creating local files to access more easily
!mkdir dataset

!cp -r /content/drive/MyDrive/MONK /content/dataset

# %%
sns.set_context(font_scale=2, rc={"font.size":10,"axes.titlesize":16,"axes.labelsize":14})
sns.set_style("whitegrid", {'grid.linestyle': '--'})
sns.set_style({'font.family':'serif', 'font.serif':'Computer Modern'})

# %%
# Function that drops the 1st column "NaN", then naming the columns and putting Class after the attributes
def prepare_monk(df):
  df.drop(labels=0, axis=1, inplace=True)
  df.columns =['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']
  df = df[['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'class', 'id']]

  return df

# %%
# List with variable names to iterate over
TR_MONK = ['TR_MONK1', 'TR_MONK2', 'TR_MONK3']
TS_MONK = ['TS_MONK1', 'TS_MONK2', 'TS_MONK3']

# for loop that takes each var from the previous lists and assigns it a pandas dataframe
i = 1
for var in TR_MONK:
  globals()[var] = pd.read_csv('/content/dataset/MONK/monks-' + str(i) + '.train', header=None, delimiter=' ')
  globals()[var] = prepare_monk(globals()[var])
  i += 1

i = 1
for var in TS_MONK:
  globals()[var] = pd.read_csv('/content/dataset/MONK/monks-' + str(i) + '.test', header=None, delimiter=' ')
  globals()[var] = prepare_monk(globals()[var])
  i += 1

# Overview of the structure and info of dataframe before one-hot encoding
print(TR_MONK1.info())
print(TR_MONK1.head())

# One-hot encoding categorical columns
categorical_cols = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']

i = 1
for var in TR_MONK:
  globals()[var] = pd.get_dummies(globals()[var], columns=categorical_cols)
  i += 1

i = 1
for var in TS_MONK:
  globals()[var] = pd.get_dummies(globals()[var], columns=categorical_cols)
  i += 1

# %%
# Looking at the dataset after OHE
print(TR_MONK1.info())
print(TR_MONK1.head())

# %%
def compute_contingency(train_data, test_data):
    """Returns a contingency df with percentage frequencies of classes (0, 1) w.r.t. train and test data"""
    df = pd.DataFrame(columns=["0", "1"], index=["train", "test"])
    train = list(train_data["class"].value_counts() / len(train_data))
    test = list(test_data["class"].value_counts() / len(test_data))
    df.loc["train"], df.loc["test"] = train, test
    return df

# %%
# checking balancing of data

fig, axs = plt.subplots(1, 3)
ax1 = compute_contingency(TR_MONK1, TS_MONK1).plot(kind="bar", stacked="True",ax=axs[0])
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.set_title("MONK1")
ax2 = compute_contingency(TR_MONK2, TS_MONK2).plot(kind="bar", stacked="True",ax=axs[1])
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_title("MONK2")
ax3 = compute_contingency(TR_MONK3, TS_MONK3).plot(kind="bar", stacked="True",ax=axs[2])
ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
ax3.set_title("MONK3")
fig.set_figheight(5)
fig.set_figwidth(15)
fig.tight_layout()
plt.show()

# %%
# X are the values of the attributes for each MONK dataframe, while y corresponds to the class
X_MONK1_train = TR_MONK1.values[:,2:19].astype(np.float32)
y_MONK1_train = TR_MONK1.values[:,0].astype('int')
X_MONK2_train = TR_MONK2.values[:,2:19].astype(np.float32)
y_MONK2_train = TR_MONK2.values[:,0].astype('int')
X_MONK3_train = TR_MONK3.values[:,2:19].astype(np.float32)
y_MONK3_train = TR_MONK3.values[:,0].astype('int')

X_MONK1_test = TS_MONK1.values[:,2:19].astype(np.float32)
y_MONK1_test = TS_MONK1.values[:,0].astype('int')
X_MONK2_test = TS_MONK2.values[:,2:19].astype(np.float32)
y_MONK2_test = TS_MONK2.values[:,0].astype('int')
X_MONK3_test = TS_MONK3.values[:,2:19].astype(np.float32)
y_MONK3_test = TS_MONK3.values[:,0].astype('int')

# Looking at the shape of each set
print(X_MONK1_train.shape, y_MONK1_train.shape)
print(X_MONK2_train.shape, y_MONK2_train.shape)
print(X_MONK3_train.shape, y_MONK3_train.shape)
print(X_MONK1_test.shape, y_MONK1_test.shape)
print(X_MONK2_test.shape, y_MONK2_test.shape)
print(X_MONK3_test.shape, y_MONK3_test.shape)

# %%
# Initializing values
X1_train, X1_val, y1_train, y1_val = train_test_split(X_MONK1_train, y_MONK1_train, test_size=0.3, shuffle=True, random_state=0)
X1_test = X_MONK1_test.astype(np.float32)
y1_test = y_MONK1_test.astype(np.float32)

X2_train, X2_val, y2_train, y2_val = train_test_split(X_MONK2_train, y_MONK2_train, test_size=0.3, shuffle=True, random_state=0)
X2_test = X_MONK2_test.astype(np.float32)
y2_test = y_MONK2_test.astype(np.float32)

X3_train, X3_val, y3_train, y3_val = train_test_split(X_MONK3_train, y_MONK3_train, test_size=0.3, shuffle=True, random_state=0)
X3_test = X_MONK3_test.astype(np.float32)
y3_test = y_MONK3_test.astype(np.float32)

# %% [markdown]
# # **<font color="#34ebdb">2.0 FUNCTIONS</font>**

# %%
# Function to be passed to sklearn's wrapper to build keras model to perform gridsearch
def build_fn(n_hidden_units, learning_rate, momentum, regularizer, lambd):
    model = Sequential()
    model.add(Dense(n_hidden_units,
                    activation = "relu",
                    input_dim = X_MONK1_train.shape[1],
                    kernel_regularizer = regularizer(lambd),
                    kernel_initializer = "random_normal"))

    model.add(Dense(1, activation = "sigmoid", kernel_initializer = "random_normal"))

    model.compile(optimizer = SGD(learning_rate = learning_rate, momentum = momentum),
                  loss = "mse",
                  metrics = ["accuracy"])
    return model

# %%
# Function to get predicted labels from probabilistic predictions on test set done by keras model
def get_pred(predictions):
  y_pred = np.zeros(len(predictions))

  for i in range(len(predictions)):
    if predictions[i] > .5:
      y_pred[i] = 1
    else:
      y_pred[i] = 0

  return y_pred

# %%
# Function to fit mlp classifier with partial fit to get scores and loss
def mlp_fit(mlp, max_iter, monk):
  if monk == 1:
    X_train = X1_train
    y_train = y1_train
    X_val = X1_val
    y_val = y1_val
  elif monk == 2:
    X_train = X2_train
    y_train = y2_train
    X_val = X2_val
    y_val = y2_val
  elif monk == 3:
    X_train = X3_train
    y_train = y3_train
    X_val = X3_val
    y_val = y3_val
  else:
    print('Enter a valid number for MONK')

  train_scores = []
  val_scores = []
  train_loss = []
  val_loss = []

  for i in range(max_iter):
      mlp.partial_fit(X_train, y_train, classes = [0, 1])
      y_pred = mlp.predict(X_train)
      y_pred_val = mlp.predict(X_val)

      train_scores.append(mlp.score(X_train, y_train))
      train_loss.append(log_loss(y_train, y_pred))

      val_scores.append(mlp.score(X_val, y_val))
      val_loss.append(log_loss(y_val, y_pred_val))

  print('Training score:  ', round(mlp.score(X_train, y_train),4))
  print('Validation score:', round(mlp.score(X_val, y_val),4))
  print('Training loss:   ', round(train_loss[-1],4))
  print('Validation loss: ', round(val_loss[-1],4), '\n')

  return train_scores, train_loss, val_scores, val_loss

# %%
# Function to create score and loss plots
def plot_score_loss(train_scores, val_scores, train_loss, val_loss, save, name):
    plt.figure(figsize = (6, 4))
    plt.plot(train_scores, label='Train')
    plt.plot(val_scores, label='val', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if save == 1:
        plt.savefig(name+'_acc', dpi=400)
    plt.show()

    plt.figure(figsize = (6, 4))
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='val', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    if save == 1:
        plt.savefig(name+'_loss', dpi=400)
    plt.show()

# %%
# Lists of parameters to compare within the neural network
list_initializers = ['random_normal', 'random_uniform', 'he_normal', 'he_uniform', 'glorot_normal']
list_activation_hidden = ['relu', 'selu', 'sigmoid', 'tanh']
list_optimizers = [Adam, SGD, RMSprop]
list_batch_size = [1, 2, 8, 32, 64, 128]

# Function to build a neural network based on its necessary characteristics
def build_compare(comparison,
                  part_to_compare,
                  set_to_use,
                  monk,
                  seed,
                  input_unit,
                  epochs,
                  default_batch_size,
                  default_initializer,
                  default_kernel_regularizer,
                  default_act_input,
                  default_act_output,
                  default_optimizer,
                  learning_rate):

    if set_to_use == 'test':
        if monk == 1:
            X_train = X_MONK1_train
            y_train = y_MONK1_train
            X_test = X1_test
            y_test = y1_test
        elif monk == 2:
            X_train = X_MONK2_train
            y_train = y_MONK2_train
            X_test = X2_test
            y_test = y2_test
        elif monk == 3:
            X_train = X_MONK3_train
            y_train = y_MONK3_train
            X_test = X3_test
            y_test = y3_test
        else:
            print('Enter a valid number for MONK (1,2,3)')
    elif set_to_use == 'val':
        if monk == 1:
            X_train = X1_train
            y_train = y1_train
            X_test = X1_val
            y_test = y1_val
        elif monk == 2:
            X_train = X2_train
            y_train = y2_train
            X_test = X2_val
            y_test = y2_val
        elif monk == 3:
            X_train = X3_train
            y_train = y3_train
            X_test = X3_val
            y_test = y3_val
        else:
            print('Enter a valid number for MONK (1,2,3)')
    else:
        print('Enter a valid set (test, val)')

    # Clearing session and assigning new seed before each time the NN is built
    keras.backend.clear_session()
    tf.random.set_seed(seed)

    model = Sequential()

    # In case no comparison is needed and only several iterations are desired, the model is built and the fuction ends here
    if part_to_compare == "testing":
        model.add(Dense(input_unit, input_dim = X_train.shape[1], kernel_initializer = default_initializer, kernel_regularizer = default_kernel_regularizer, activation = default_act_input))
        model.add(Dense(1, activation = default_act_output))
        model.compile(loss = 'mse', metrics = ['accuracy'], optimizer = default_optimizer)

        model.fit(X_train, y_train,
                  epochs = epochs,
                  batch_size = default_batch_size,
                  verbose = 0)

        loss, acc = model.evaluate(X_test, y_test, verbose = 0)

        return loss, acc

    # If on the other hand, several parameters are to be compared:
    # Possibilities for the input layer
    if part_to_compare == "initializer":
        model.add(Dense(input_unit, input_dim = X_train.shape[1], kernel_initializer = comparison, kernel_regularizer = default_kernel_regularizer, activation = default_act_input))
    elif part_to_compare == "activation_input":
        model.add(Dense(input_unit, input_dim = X_train.shape[1], kernel_initializer = default_initializer, kernel_regularizer = default_kernel_regularizer, activation = comparison))
    else:
        model.add(Dense(input_unit, input_dim = X_train.shape[1], kernel_initializer = default_initializer, kernel_regularizer = default_kernel_regularizer, activation = default_act_input))

    # Output layer
    model.add(Dense(1, activation = default_act_output))

    # Possibilities for compiling
    if part_to_compare == "optimizer":
        model.compile(loss = 'mse', metrics = ['accuracy'], optimizer = comparison(learning_rate = learning_rate))
    else:
        model.compile(loss = 'mse', metrics = ['accuracy'], optimizer = default_optimizer)

    if part_to_compare == "batch_size":
        model.fit(X_train, y_train,
                  epochs = epochs,
                  batch_size = comparison,
                  verbose = 0)
    else:
        model.fit(X_train, y_train,
                  epochs = epochs,
                  batch_size = default_batch_size,
                  verbose = 0)

    loss, acc = model.evaluate(X_test, y_test, verbose = 0)

    return loss, acc

# Function to compare the previously built fuction across different iterations and parameters
def begin_comparison(part_to_compare,
                     set_to_use,
                     monk,
                     iter,
                     input_unit,
                     epochs,
                     default_batch_size,
                     default_initializer,
                     default_kernel_regularizer,
                     default_act_input,
                     default_act_output,
                     default_optimizer,
                     learning_rate,
                     plot):
    # Empty lists/dicts to store scores after evaluation
    losses = []
    accs = []
    accs_dict = {}
    losses_dict = {}

    print('comparing', part_to_compare, "on MONK", monk, '\n')

    # Assigning the list of parameters to compare based on what has been passed to the fuction
    if part_to_compare == "testing":
        comparison = ["without comparison"]
    elif part_to_compare == "activation_input":
        comparison = list_activation_hidden
    elif part_to_compare == "batch_size":
        comparison = list_batch_size
    elif part_to_compare == "initializer":
        comparison = list_initializers
    elif part_to_compare == "optimizer":
        comparison = list_optimizers
    else:
        comparison = []
        print("Please enter a valid input (activation_input, activation_output, initializer, optimizer, testing)")

    # For loop to iterate over the different parameters to compare
    for index, item in enumerate(comparison):
        print('-----------', part_to_compare, item, '-----------')
        accs.append([])
        losses.append([])

        # Generating a list of random numbers to be later passed as seeds for tensorflow randomization
        list_seeds = np.random.default_rng().choice(iter*len(comparison)+1, size = iter, replace = False)

        # For loop to build, fit and evaluate a NN for n iterations for each of the items in the comparison list
        for i in range(iter):
            seed = list_seeds[i]

            loss, acc = build_compare(item,
                                      part_to_compare,
                                      set_to_use,
                                      monk,
                                      seed,
                                      input_unit,
                                      epochs,
                                      default_batch_size,
                                      default_initializer,
                                      default_kernel_regularizer,
                                      default_act_input,
                                      default_act_output,
                                      default_optimizer,
                                      learning_rate)

            # Storing scores in their respective list
            losses[index].append(round(loss,6))
            accs[index].append(round(acc,6))

            # Storing scores together with the model and seed they belong
            accs_dict[item, str(i), str(seed)] = round(acc,6)
            losses_dict[item, str(i), str(seed)] = round(loss,6)

        print('Percentual error:', round((np.std(accs[index])/np.mean(accs[index]))*100,6))
        print('Avg accuracy:    ', round(np.mean(accs[index]),6))
        print('Acc std dev:     ', round(np.std(accs[index]),6))
        print('Avg loss:        ', round(np.mean(losses[index]),6))
        print('Loss std dev:    ', round(np.std(losses[index]),6), '\n')

    # Plotting scores obtained before
    if plot == 1:
        plot_comparison(part_to_compare, comparison, iter, accs, losses)

    return accs_dict, losses_dict

def plot_comparison(part_to_compare, comparison, iter, accs, losses):
    for index, item in enumerate(comparison):
        print('\n*******************', part_to_compare, item, 'ACCURACY', '*******************\n')
        plt.plot(range(iter), accs[index])
        plt.title('Accuracy')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.show()
        print('\n*******************', part_to_compare, item, 'LOSS', '*******************\n')
        plt.plot(range(iter), losses[index])
        plt.title('Loss')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
        print('\n')

# %%
# Function to apply nested cross validation on a series of models and parameters
def nested_cross_validation(monk, folds, model_to_use, model_params, param_grid):
    outer_kfold = KFold(folds, shuffle=True, random_state=42)
    inner_kfold = KFold(folds, shuffle=True, random_state=77)

    test_scores = []

    if monk == 1:
        X = X_MONK1_train
        y = y_MONK1_train
    elif monk == 2:
        X = X_MONK2_train
        y = y_MONK2_train
    elif monk == 3:
        X = X_MONK3_train
        y = y_MONK3_train
    else:
        print("Please type a valid input (1,2,3)")

    # Outer K-Fold (Evaluation)
    for train_indices, test_indices in outer_kfold.split(X, y):
        # Split data into train and test
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        models = {}

        # Initializing param_grid
        list_params = ['param'+str(i) for i in range(len(model_params))]
        for list_params[0] in param_grid[0]:
            for list_params[1] in param_grid[1]:
                for list_params[2] in param_grid[2]:

                    val_scores = []

                    # Inner K-Fold for each hyper-parameter configuration
                    for selection_indices, val_indices in inner_kfold.split(X_train, y_train):

                        # Split data into selection and validation
                        X_selection, y_selection = X_train[selection_indices], y_train[selection_indices]
                        X_val, y_val = X_train[val_indices], y_train[val_indices]

                        # Fit the model
                        # for i in range(len(model_params)):
                        params = {model_params[i]: list_params[i] for i in range(len(model_params))}
                        model = model_to_use(**params)
                        model.fit(X_selection, y_selection)
                        val_scores.append(model.score(X_val, y_val))

                    # Validation score of a model is the mean over the inner k-folds
                    models[(list_params[0], list_params[1], list_params[2])] = np.mean(val_scores)
                    # models[(list_params)] = np.mean(val_scores)
                    # models = {key: None for key in keyList}
                    # models = {(list_params[i]): np.mean(val_scores) for i in range(len(model_params))}
                    # models[(list_params[i] for i in range(len(model_params)))] = np.mean(val_scores)

        best_params = max(models, key=models.get)

        best_params_dict = {model_params[i]: best_params[i] for i in range(len(model_params))}
        model = model_to_use(**best_params_dict)
        model.fit(X_train, y_train)
        test_scores.append(model.score(X_test, y_test))

    avg_accuracy = round(np.mean(test_scores), 4)
    std_dev = round(np.std(test_scores), 4)

    return avg_accuracy, std_dev

# %%
# Function to fit mlp classifier with partial fit to get scores and loss
def mlp_final_test(mlp, epochs, monk):
  if monk == 1:
    X_train = X1_train
    y_train = y1_train
    X_val = X1_val
    y_val = y1_val
    X_test = X1_test
    y_test = y1_test
  elif monk == 2:
    X_train = X2_train
    y_train = y2_train
    X_val = X2_val
    y_val = y2_val
    X_test = X2_test
    y_test = y2_test
  elif monk == 3:
    X_train = X3_train
    y_train = y3_train
    X_val = X3_val
    y_val = y3_val
    X_test = X3_test
    y_test = y3_test
  else:
    print('Enter a valid number for MONK')

  train_loss = []
  val_loss = []
  test_loss = []

  for i in range(epochs):
      mlp.partial_fit(X_train, y_train, classes = [0, 1])
      y_pred = mlp.predict(X_train)
      y_pred_val = mlp.predict(X_val)
      y_pred_test = mlp.predict(X_test)

      train_loss.append(log_loss(y_train, y_pred))
      val_loss.append(log_loss(y_val, y_pred_val))
      test_loss.append(log_loss(y_test, y_pred_test))

  tr_acc = mlp.score(X_train, y_train)
  vl_acc = mlp.score(X_val, y_val)
  ts_acc = mlp.score(X_test, y_test)

  tr_loss = train_loss[-1]
  vl_loss = val_loss[-1]
  ts_loss = test_loss[-1]

  return tr_acc, vl_acc, ts_acc, tr_loss, vl_loss, ts_loss

# %% [markdown]
# # **<font color="#34ebdb">3.0 MONK 1</font>**

# %% [markdown]
# ## **<font color="#CEFF5E">KERAS</font>**

# %%
model_search = KerasClassifier(build_fn, n_hidden_units=0, learning_rate=0, regularizer=L2, lambd=0, batch_size=0, momentum=0, epochs=50, random_state=42, verbose=0)

param_grid = {'n_hidden_units': [2, 4, 6],
              'learning_rate': [.01, .1, 1],
              'momentum': [0, .01, .1],
              'lambd': [0, .01, .1],
              'batch_size': [2, 8, 32]
              }

search_loss = GridSearchCV(model_search,
                           param_grid,
                           cv = StratifiedKFold(5, shuffle=True, random_state=42),
                           verbose = 0,
                           n_jobs = -1,
                           scoring = 'neg_mean_squared_error').fit(X_MONK1_train, y_MONK1_train, verbose=0)

print('Best score:', search_loss.best_score_, '\nBest params', search_loss.best_params_)

# %%
scores_loss = search_loss.cv_results_['mean_test_score']
params_loss = search_loss.cv_results_['params']

grid_dict_loss = {}
for score, param in zip(scores_loss, params_loss):
    grid_dict_loss[str(param)] = score

optimal_model_loss = [(key, value) for key, value in grid_dict_loss.items() if value > -.05]
for opt_model in optimal_model_loss:
  print(opt_model)

# %%
model_search = KerasClassifier(build_fn, n_hidden_units=0, learning_rate=0, regularizer=L2, lambd=0, batch_size=0, momentum=0, epochs=50, random_state=42, verbose=0)

param_grid = {'n_hidden_units': [4, 5, 6, 7],
              'learning_rate': [.1, .5, 1],
              'momentum': [0, .01, .1],
              'lambd': [0],
              'batch_size': [2, 8]
              }

search_loss = GridSearchCV(model_search,
                           param_grid,
                           cv = StratifiedKFold(5, shuffle=True, random_state=42),
                           verbose = 0,
                           n_jobs = -1,
                           scoring = 'neg_mean_squared_error').fit(X_MONK1_train, y_MONK1_train, verbose=0)

print('Best score:', search_loss.best_score_, '\nBest params', search_loss.best_params_)

# %%
scores_loss = search_loss.cv_results_['mean_test_score']
params_loss = search_loss.cv_results_['params']

grid_dict_loss = {}
for score, param in zip(scores_loss, params_loss):
    grid_dict_loss[str(param)] = score

optimal_model_loss = [(key, value) for key, value in grid_dict_loss.items() if value > -.05]
for opt_model in optimal_model_loss:
  print(opt_model)

# %%
keras.backend.clear_session()
tf.random.set_seed(42)

model = Sequential()
model.add(Dense(4, input_dim = X_MONK1_train.shape[1], activation="relu", kernel_regularizer=L2(0)))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=keras.optimizers.SGD(learning_rate=.5, momentum=.1), loss='mse', metrics='accuracy')
model.summary()

history = model.fit(X1_train, y1_train, epochs=50, batch_size=2, validation_data=(X1_val, y1_val), verbose=0)

loss, acc = model.evaluate(X1_train, y1_train, verbose = 0)
print('Training accuracy:  ', round(acc,4))
print('Training loss:      ', round(loss,4))

loss_val, acc_val = model.evaluate(X1_val, y1_val, verbose = 0)
print('Validation accuracy:', round(acc_val,4))
print('Validation loss:    ', round(loss_val,4))

plot_score_loss(train_scores = history.history['accuracy'],
                val_scores = history.history['val_accuracy'],
                train_loss = history.history['loss'],
                val_loss = history.history['val_loss'],
                save = 0,
                name = '')

# %%


# %% [markdown]
# ## **<font color="#CEFF5E">KERAS: COMPARING PARAMETERS</font>**

# %%
accs_dict, losses_dict = begin_comparison("optimizer",
                                          iter = 50,
                                          set_to_use = 'val',
                                          monk = 1,
                                          epochs = 50,
                                          input_unit = 5,
                                          default_batch_size = 2,
                                          default_initializer = 'glorot_normal',
                                          default_act_input = 'relu',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = SGD(),
                                          default_kernel_regularizer=L2(0),
                                          learning_rate = .01,
                                          plot = 0)

# %%
accs_dict, losses_dict = begin_comparison("initializer",
                                          iter = 50,
                                          set_to_use = 'val',
                                          monk = 1,
                                          epochs = 50,
                                          input_unit = 5,
                                          default_batch_size = 2,
                                          default_initializer = 'glorot_normal',
                                          default_act_input = 'relu',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = Adam(learning_rate = .01),
                                          default_kernel_regularizer=L2(0),
                                          learning_rate = .01,
                                          plot = 0)

# %%
accs_dict, losses_dict = begin_comparison("activation_input",
                                          iter = 50,
                                          set_to_use = 'val',
                                          monk = 1,
                                          epochs = 50,
                                          input_unit = 5,
                                          default_batch_size = 2,
                                          default_initializer = 'he_normal',
                                          default_act_input = 'relu',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = Adam(learning_rate = .01),
                                          default_kernel_regularizer=L2(0),
                                          learning_rate = .01,
                                          plot = 0)

# %%
accs_dict, losses_dict = begin_comparison("batch_size",
                                          iter = 50,
                                          set_to_use = 'val',
                                          monk = 1,
                                          epochs = 50,
                                          input_unit = 5,
                                          default_batch_size = 2,
                                          default_initializer = 'he_normal',
                                          default_act_input = 'tanh',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = Adam(learning_rate = .01),
                                          default_kernel_regularizer=L2(0),
                                          learning_rate = .01,
                                          plot = 0)

# %%
for i in [.005, .01, .05, .1]:
    print('lr =', i)
    accs_dict, losses_dict = begin_comparison("testing",
                                              iter = 30,
                                              set_to_use = 'val',
                                              monk = 1,
                                              epochs = 50,
                                              input_unit = 5,
                                              default_batch_size = 1,
                                              default_initializer = 'he_normal',
                                              default_act_input = 'tanh',
                                              default_act_output = 'sigmoid',
                                              default_optimizer = Adam(learning_rate = i),
                                              default_kernel_regularizer=L2(0),
                                              learning_rate = .01,
                                              plot = 0)

# %%
for i in [.008, .009, .02, .03]:
    print('lr =', i)
    accs_dict, losses_dict = begin_comparison("testing",
                                              iter = 30,
                                              set_to_use = 'val',
                                              monk = 1,
                                              epochs = 50,
                                              input_unit = 5,
                                              default_batch_size = 1,
                                              default_initializer = 'he_normal',
                                              default_act_input = 'tanh',
                                              default_act_output = 'sigmoid',
                                              default_optimizer = Adam(learning_rate = i),
                                              default_kernel_regularizer=L2(0),
                                              learning_rate = .01,
                                              plot = 0)

# %%
keras.backend.clear_session()
tf.random.set_seed(45)

model = Sequential()
model.add(Dense(5, input_dim = X_MONK1_train.shape[1], kernel_initializer = 'he_normal', activation="tanh", kernel_regularizer=L2(0)))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=.02), loss='mse', metrics='accuracy')
model.summary()

history = model.fit(X1_train, y1_train, epochs=50, batch_size=1, validation_data=(X1_val, y1_val), verbose=0)

loss, acc = model.evaluate(X1_train, y1_train, verbose = 0)
print('Training accuracy:  ', round(acc,4))
print('Training loss:      ', round(loss,4))

loss_val, acc_val = model.evaluate(X1_val, y1_val, verbose = 0)
print('Validation accuracy:', round(acc_val,4))
print('Validation loss:    ', round(loss_val,4))

plot_score_loss(train_scores = history.history['accuracy'],
                val_scores = history.history['val_accuracy'],
                train_loss = history.history['loss'],
                val_loss = history.history['val_loss'],
                save = 1,
                name = 'MONK1_final')

# %%
# iterating for getting avg TR accuracy and loss

train_accs = []
train_losses = []

count = 0
list_seeds = np.random.default_rng().choice(50, size = 50, replace = False)
for i in range(50):
    print(f"Processing {i + 1}/50")
    seed = list_seeds[i]
    keras.backend.clear_session()
    tf.random.set_seed(seed)

    model = Sequential()
    model.add(Dense(5, input_dim = X_MONK1_train.shape[1], kernel_initializer = 'he_normal', activation="tanh", kernel_regularizer=L2(0)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.02), loss='mse', metrics='accuracy')

    history = model.fit(X1_train, y1_train, epochs=50, batch_size=1, validation_data=(X1_val, y1_val), verbose=0)
    loss, acc = model.evaluate(X1_train, y1_train, verbose = 0)
    train_accs.append(acc)
    train_losses.append(loss)

print("Avg TR Accuracy:", np.mean(train_accs))
print("Std TR Accuracy:", np.std(train_accs))
print("Avg TR Loss:", np.mean(train_losses))
print("Std TR Loss:", np.std(train_losses))

# %%
# Fitting final model to test set and evaluating it over n iterations to get mean score
iterations = 50
accuracy = []
auc = []

list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

for i in list_seeds:
    model = Sequential()
    model.add(Dense(5, input_dim = X_MONK1_train.shape[1], kernel_initializer = 'he_normal', activation="tanh", kernel_regularizer=L2(0)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.02), loss='mse', metrics='accuracy')
    model.fit(X_MONK1_train, y_MONK1_train)

    y_pred = model.predict(X_MONK1_test)

    acc = accuracy_score(y_MONK1_test, y_pred)
    accuracy.append(acc)
    auc_score = roc_auc_score(y_MONK1_test, y_pred)
    auc.append(auc_score)

print('Avg Accuracy:  ', round(np.mean(accuracy),4))
print('std dev:       ', round(np.std(accuracy),4))
print('AUC:           ', round(np.mean(auc),4))
print('std dev:       ', round(np.std(auc),4))

# %%
accs_dict, losses_dict = begin_comparison("testing",
                                          iter = 50,
                                          set_to_use = 'test',
                                          monk = 1,
                                          epochs = 50,
                                          input_unit = 5,
                                          default_batch_size = 1,
                                          default_initializer = 'he_normal',
                                          default_act_input = 'tanh',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = Adam(learning_rate=.02),
                                          default_kernel_regularizer = L2(0),
                                          learning_rate = 1,
                                          plot = 1)

# %%
accs_dict, losses_dict = begin_comparison("testing",
                                          iter = 50,
                                          set_to_use = 'test',
                                          monk = 1,
                                          epochs = 50,
                                          input_unit = 5,
                                          default_batch_size = 1,
                                          default_initializer = 'he_normal',
                                          default_act_input = 'tanh',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = Adam(learning_rate=.01),
                                          default_kernel_regularizer = L2(0),
                                          learning_rate = 1,
                                          plot = 1)

# %% [markdown]
# ## **<font color="#CEFF5E">SCIKITLEARN'S MLP</font>**

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
        "learning_rate": "invscaling",
        "momentum": 0,
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
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
    "inv-scaling learning w/o momentum",
    "inv-scaling with momentum",
    "inv-scaling with Nesterov's momentum",
    "adaptive learning w/o momentum",
    "adaptive with momentum",
    "adaptive with Nesterov's momentum",
    "adam",
]

# %%
# def plot_on_dataset(X, y, ax, name):
models = []
for label, param in zip(labels, params):
    print("training", label)
    mlp = MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(6,), **param)

    # some parameter combinations will not converge so they are ignored here
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
      mlp.fit(X1_train, y1_train)

    models.append(mlp)
    print("Score: %f" % mlp.score(X1_val, y1_val))
    print("Loss: %f" % mlp.loss_,'\n')

# %%
plt.figure(figsize=(12,4))
for i, label in zip(range(len(models)), labels):
  plt.plot(models[i].loss_curve_, label = label)
plt.title("Comparing different learning methods for MLP")
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
plt.show()

# %%
# Gridsearch to find optimal values for neural network
model = MLPClassifier(solver = 'sgd',
                      learning_rate= 'adaptive',
                      nesterovs_momentum = True,
                      learning_rate_init = 0.2,
                      max_iter = 50,
                      random_state = 42)

param_grid = {'hidden_layer_sizes': [(2,), (4,), (6,)],
              'momentum': [0, .001, .01, .1],
              'learning_rate_init': [.001, .01, 1],
              'batch_size': [1, 2, 8, 32],
              }

with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
      search = GridSearchCV(model,
                            param_grid,
                            cv = StratifiedKFold(5, shuffle = True, random_state = 42),
                            verbose = 1).fit(X_MONK1_train, y_MONK1_train)

print('Best score:', search.best_score_)
print('Best params', search.best_estimator_)

# %%
# Visualizing training curves for model found in gridsearch
mlp = MLPClassifier(solver = 'sgd',
                    learning_rate = 'adaptive',
                    nesterovs_momentum = True,
                    max_iter = 50,
                    batch_size=8,
                    hidden_layer_sizes=(6,),
                    learning_rate_init=1,
                    momentum=0.1,
                    random_state=42)

train_scores, train_loss, val_scores, val_loss = mlp_fit(mlp, max_iter = 50, monk=1)

plot_score_loss(train_scores = train_scores,
                val_scores = val_scores,
                train_loss = train_loss,
                val_loss = val_loss,
                save = 1,
                name = 'MONK1_sklearn')

# %%
iterations = 50
list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

train_acc = []
val_acc = []
test_acc = []

train_loss = []
val_loss = []
test_loss = []

for i in list_seeds:
    tr_acc, vl_acc, ts_acc, tr_loss, vl_loss, ts_loss = mlp_final_test(mlp=MLPClassifier(solver = 'sgd',
                                                                                         learning_rate = 'adaptive',
                                                                                         nesterovs_momentum = True,
                                                                                         max_iter = 50,
                                                                                         batch_size=8,
                                                                                         hidden_layer_sizes=(6,),
                                                                                         learning_rate_init=1,
                                                                                         momentum=0.1,
                                                                                         random_state=i), epochs=50, monk=1)
    train_acc.append(tr_acc)
    val_acc.append(vl_acc)
    test_acc.append(ts_acc)

    train_loss.append(tr_loss)
    val_loss.append(vl_loss)
    test_loss.append(ts_loss)

print('Avg train acc: ', round(np.mean(train_acc),4))
print('Std dev:       ', round(np.std(train_acc),4))
print('Avg val acc:   ', round(np.mean(val_acc),4))
print('Std dev:       ', round(np.std(val_acc),4))
print('Avg test acc:  ', round(np.mean(test_acc),4))
print('Std dev:       ', round(np.std(test_acc),4))

print('Avg train loss:', round(np.mean(train_loss),4))
print('Std dev:       ', round(np.std(train_loss),4))
print('Avg val loss:  ', round(np.mean(val_loss),4))
print('Std dev:       ', round(np.std(val_loss),4))
print('Avg test loss: ', round(np.mean(test_loss),4))
print('Std dev:       ', round(np.std(test_loss),4))

# %% [markdown]
# ## **<font color="#CEFF5E">NESTED CROSS VALIDATION</font>**

# %%
# Models, parameters and hyperparameters to test with nested cross validation for MONK 1
knn_params = ['n_neighbors', 'weights', 'metric']
knn_param_grid = [range(2,78,2), ["uniform", "distance"], ["euclidean", "cityblock", "chebyshev"]]

dt_params = ['criterion', 'max_depth', 'min_samples_split']
dt_param_grid = [['gini', 'entropy'], [1, 2, 4, 6, 8, 10, None], [2, 3, 4, 5, 6, 7, 8]]

lr_params = ['solver', 'C', 'max_iter']
lr_param_grid = [['saga', 'lbfgs'], [.0001, .001, .1, 1, 10, 100, 1000], [500, 1000]]

rf_params = ['criterion', 'max_depth', 'n_estimators']
rf_param_grid = [['gini', 'entropy'], [1, 3, 5, 7, 10, None], [50, 100, 150, 200]]

bnb_params = ['alpha', 'binarize', 'fit_prior']
bnb_param_grid = [[.0001, .001, .01, .1, 1], [0], [True, False]]

gb_params = ['criterion', 'learning_rate', 'n_estimators']
gb_param_grid = [['friedman_mse', 'squared_error'], [.001, .01, .1, 1], [50, 100, 150, 200]]

svc_params = ['C', 'kernel', 'gamma']
svc_param_grid = [[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5],
                  ['poly', 'rbf', 'sigmoid'],
                  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]]

nested_dict = {
    'K-Nearest Neighbors': [KNeighborsClassifier, knn_params, knn_param_grid],
    'Decision Tree': [DecisionTreeClassifier, dt_params, dt_param_grid],
    'Logistic Regression': [LogisticRegression, lr_params, lr_param_grid],
    'Random Forests': [RandomForestClassifier, rf_params, rf_param_grid],
    'Bernoulli Naive Bayes': [BernoulliNB, bnb_params, bnb_param_grid],
    'Gradient Boosting': [GradientBoostingClassifier, gb_params, gb_param_grid],
    'Support Vector Machine': [SVC, svc_params, svc_param_grid]}

# %%
monk = 1 # MONK set to use
print('MONK', monk, 'Nested Cross Validation results:\n')

# Ignoring convergence warning for logistic regression when max_iter is not enough to converge
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    for name, [model_to_use, model_params, param_grid] in nested_dict.items():
        avg_acc, std_dev = nested_cross_validation(monk = monk,
                                                   folds = 5,
                                                   model_to_use = model_to_use,
                                                   model_params = model_params,
                                                   param_grid = param_grid)
        print('------', name, '------')
        print('Average accuracy:  ', avg_acc)
        print('Standard deviation:', std_dev, '\n')

# %% [markdown]
# ### **<font color="#CEFF5E">SVM</font>**

# %% [markdown]
# GridSearch 1st run

# %%
model_svc = SVC()

svc_param_grid = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ['poly', 'rbf', 'sigmoid'], #default = rbf
    'class_weight': ['balanced', None],
    'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], #default=1.0
    'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], #default: auto
    }

svc_param_grid_linear = {
    'kernel': ['linear'],
    'class_weight': ['balanced', None],
    'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], #default=1.0
    }

models_params = {
    'SVM': [model_svc, svc_param_grid],
    'SVMlinear': [model_svc, svc_param_grid_linear]
    }

best_params = {}
validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Grid search starting')
    search = GridSearchCV(model,
                          param,
                          cv = KFold(10, shuffle=True, random_state=42),
                          verbose=2,
                          n_jobs = -1).fit(X_MONK1_train, y_MONK1_train)
    best_params[name] = search.best_estimator_
    validation_scores[name] = search.best_score_
    print('Best score: ', validation_scores[name])
    print('Best parameters: ', best_params[name], '\n')

# %% [markdown]
# GridSearch 2nd run

# %%
model_svc = SVC()

svc_param_grid_linear = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ['linear'],
    'class_weight': ['balanced', None],
    'C': [.5, .6, .7, .8, .9, 1, 2, 3, 4, 5], #default=1.0
    }

models_params = {'SVMlinear': [model_svc, svc_param_grid_linear]}

best_params = {}
validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Grid search starting')
    search = GridSearchCV(model,
                          param,
                          cv = KFold(10, shuffle=True, random_state=42),
                          verbose=2,
                          n_jobs = -1).fit(X_MONK1_train, y_MONK1_train)
    best_params[name] = search.best_estimator_
    validation_scores[name] = search.best_score_
    print('Best score: ', validation_scores[name])
    print('Best parameters: ', best_params[name], '\n')

# %% [markdown]
# GridSearch 3rd run

# %%
model_svc = SVC()

svc_param_grid_linear = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ['linear'],
    'class_weight': ['balanced', None],
    'C': [.65, .66, .67, .68, .69, .70, .71, .72, .73, .74, .75], #default=1.0
    }

models_params = {'SVMlinear': [model_svc, svc_param_grid_linear]}

best_params = {}
validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Grid search starting')
    search = GridSearchCV(model,
                          param,
                          cv = KFold(10, shuffle=True, random_state=42),
                          verbose=2,
                          n_jobs = -1).fit(X_MONK1_train, y_MONK1_train)
    best_params[name] = search.best_estimator_
    validation_scores[name] = search.best_score_
    print('Best score: ', validation_scores[name])
    print('Best parameters: ', best_params[name], '\n')

# %%
# Testing best parameters (svc linear) on validation set
svc = SVC(C=0.66, class_weight='balanced', kernel='linear')
svc.fit(X1_train, y1_train)

y_pred = svc.predict(X1_val)
print(classification_report(y1_val, y_pred, digits = 4))

auc_score = round(roc_auc_score(y1_val, y_pred),4)
print('AUC:', auc_score)

# %%
# Testing best parameters on validation set
svc = SVC(C=10.0, class_weight='balanced', gamma=0.1)
svc.fit(X1_train, y1_train)

y_pred = svc.predict(X1_val)
print(classification_report(y1_val, y_pred, digits = 4))

auc_score = round(roc_auc_score(y1_val, y_pred),4)
print('AUC:', auc_score)

# %%
# Testing best parameters on validation set
svc = SVC(C=10.0, class_weight='balanced', gamma=0.1)
svc.fit(X_MONK1_train, y_MONK1_train)

y_pred = svc.predict(X1_test)
print(classification_report(y1_test, y_pred, digits = 4))

auc_score = round(roc_auc_score(y1_test, y_pred),4)
print('AUC:', auc_score)

# %% [markdown]
# ### **<font color="#CEFF5E">GRADIENT BOOSTING</font>**

# %%
param_grid = {'criterion':['friedman_mse', 'squared_error'],
              'learning_rate': [0.001, 0.01, 0.1, 1.],
              'n_estimators':[50, 100, 200]}

search = GridSearchCV(GradientBoostingClassifier(),
                      param_grid=param_grid,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      verbose=1,
                      n_jobs=-1,
                      scoring='accuracy').fit(X_MONK1_train, y_MONK1_train)
print(search.best_score_, search.best_params_)

# %%
# Testing on validation set
gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=.1, criterion='friedman_mse')
gbc.fit(X1_train, y1_train)

y_pred = gbc.predict(X1_val)

print(classification_report(y1_val, y_pred, digits = 4))

auc_score = round(roc_auc_score(y1_val, y_pred),4)
print('AUC:', auc_score)

# %%
# Testing on test set
gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=.1, criterion='friedman_mse')
gbc.fit(X_MONK1_train, y_MONK1_train)

y_pred = gbc.predict(X1_test)

print(classification_report(y1_test, y_pred, digits = 4))

auc_score = round(roc_auc_score(y_MONK1_test, y_pred),4)
print('AUC:', auc_score)

# %%
# Fitting final model to test set and evaluating it over n iterations to get mean score
iterations = 50
accuracy = []
auc = []

list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

for i in list_seeds:
    gbc = GradientBoostingClassifier(n_estimators = 200, learning_rate = .1, criterion = 'friedman_mse', random_state = i)
    gbc.fit(X_MONK1_train, y_MONK1_train)

    y_pred = gbc.predict(X_MONK1_test)

    acc = accuracy_score(y_MONK1_test, y_pred)
    accuracy.append(acc)
    auc_score = roc_auc_score(y_MONK1_test, y_pred)
    auc.append(auc_score)

print('Avg Accuracy:  ', round(np.mean(accuracy),4))
print('std dev:       ', round(np.std(accuracy),4))
print('AUC:           ', round(np.mean(auc),4))
print('std dev:       ', round(np.std(auc),4))

# %% [markdown]
# ### **<font color="#CEFF5E">RANDOM FOREST</font>**

# %%
param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [1, 3, 5, 7, 10, None],
              'n_estimators': [50, 100, 150, 200]}

search = GridSearchCV(RandomForestClassifier(),
                      param_grid,
                      cv = StratifiedKFold(5, shuffle=True, random_state=42),
                      verbose = 1,
                      n_jobs = -1,
                      scoring = 'accuracy').fit(X_MONK1_train, y_MONK1_train)

print(search.best_score_, search.best_params_)

# %%
param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [6, 8, 10, 12, 14, 16, None],
              'n_estimators': [20, 30, 40, 50]}

search = GridSearchCV(RandomForestClassifier(),
                      param_grid,
                      cv = StratifiedKFold(5, shuffle=True, random_state=42),
                      verbose = 1,
                      n_jobs = -1,
                      scoring = 'accuracy').fit(X_MONK1_train, y_MONK1_train)

print(search.best_score_, search.best_params_)

# %%
# Testing on validation set
rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 30, max_depth = 6)
rf.fit(X1_train, y1_train)

y_pred = rf.predict(X1_val)

print(classification_report(y1_val, y_pred, digits = 4))

auc_score = round(roc_auc_score(y1_val, y_pred),4)
print('AUC:', auc_score)

# %%
# Testing on test set
rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 30, max_depth = 6)
rf.fit(X_MONK1_train, y_MONK1_train)

y_pred = rf.predict(X1_test)

print(classification_report(y1_test, y_pred, digits = 4))

auc_score = round(roc_auc_score(y1_test, y_pred),4)
print('AUC:', auc_score)

# %%
# Fitting final model to test set and evaluating it over n iterations to get mean score
iterations = 50
accuracy = []
auc = []

list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

for i in list_seeds:
    rf = RandomForestClassifier(criterion = 'gini', n_estimators = 30, max_depth = 6, random_state = i)
    rf.fit(X_MONK1_train, y_MONK1_train)

    y_pred = rf.predict(X_MONK1_test)

    acc = accuracy_score(y_MONK1_test, y_pred)
    accuracy.append(acc)
    auc_score = roc_auc_score(y_MONK1_test, y_pred)
    auc.append(auc_score)

print('Avg Accuracy:  ', round(np.mean(accuracy),4))
print('std dev:       ', round(np.std(accuracy),4))
print('AUC:           ', round(np.mean(auc),4))
print('std dev:       ', round(np.std(auc),4))

# %% [markdown]
# # **<font color="#34ebdb">4.0 MONK 2</font>**

# %% [markdown]
# ## **<font color="#CEFF5E">KERAS</font>**

# %%
# First Grid Search

model_search = KerasClassifier(build_fn,
                               n_hidden_units=0,
                               learning_rate=0,
                               regularizer=L2,
                               lambd=0,
                               batch_size=0,
                               momentum=0,
                               epochs=50,
                               random_state=42,
                               verbose=1)

param_grid = {'n_hidden_units': [2, 3, 4],
              'learning_rate': [.01, .1, 1],
              'momentum': [0, .01, .1],
              'lambd': [0, .01, .1],
              'batch_size': [2, 8, 32]
              }

search_loss = GridSearchCV(model_search,
                           param_grid,
                           cv = StratifiedKFold(5, shuffle=True, random_state=42),
                           verbose = 1,
                           n_jobs = -1,
                           scoring = 'neg_mean_squared_error').fit(X_MONK2_train, y_MONK2_train, verbose=0)

print('Best score:', search_loss.best_score_, '\nBest params', search_loss.best_params_)

# %%
# Checking all the optimal models

scores_loss = search_loss.cv_results_['mean_test_score']
params_loss = search_loss.cv_results_['params']

grid_dict_loss = {}
for score, param in zip(scores_loss, params_loss):
    grid_dict_loss[str(param)] = score

optimal_model_loss = [(key, value) for key, value in grid_dict_loss.items() if value == 0]
for opt_model in optimal_model_loss:
  print(opt_model)

# %%
keras.backend.clear_session()
tf.random.set_seed(42)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=15,
                   restore_best_weights = True)

model = Sequential()
model.add(Dense(3, input_dim = X_MONK2_train.shape[1], activation="relu", kernel_initializer="random_normal"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="random_normal"))
model.compile(optimizer=keras.optimizers.SGD(learning_rate=1, momentum=0.1), loss='mse', metrics='accuracy')
model.summary()

history = model.fit(X2_train, y2_train, epochs=50, batch_size=2, validation_data=(X2_val, y2_val), verbose=0)#, callbacks=[es])

plot_score_loss(history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss'])

# %%
# finer-grained search for lr and momentum

model_search = KerasClassifier(build_fn, n_hidden_units=3, learning_rate=0, regularizer=L2, lambd=0, batch_size=2, momentum=0, epochs=50, random_state=42, verbose=1)

param_grid = {'learning_rate': [.2, .4, .6, .8, 1.],
              'momentum': [0, .1, .3, .5, .7, .9],
              }

search = GridSearchCV(model_search,
                      param_grid,
                      cv = StratifiedKFold(5, shuffle=True, random_state=42),
                      verbose = 1,
                      n_jobs = -1,
                      scoring = 'neg_mean_squared_error')

search.fit(X_MONK2_train, y_MONK2_train, verbose=0)

print('Best score:', search.best_score_, '\nBest params', search.best_params_)

# %%
scores_loss = search.cv_results_['mean_test_score']
params_loss = search.cv_results_['params']

grid_dict_loss = {}
for score, param in zip(scores_loss, params_loss):
    grid_dict_loss[str(param)] = score

optimal_model_loss = [(key, value) for key, value in grid_dict_loss.items() if value == 0]
for opt_model in optimal_model_loss:
  print(opt_model)

# %%
keras.backend.clear_session()
tf.random.set_seed(42)

model = Sequential()
model.add(Dense(3, input_dim = X_MONK2_train.shape[1], activation="relu", kernel_initializer="random_normal"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="random_normal"))
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.6, momentum=0.5), loss='mse', metrics='accuracy')
model.summary()

history = model.fit(X2_train, y2_train, epochs=50, batch_size=2, validation_data=(X2_val, y2_val), verbose=0)
plot_score_loss(history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss'])

# %% [markdown]
# ## **<font color="#CEFF5E">KERAS: COMPARING PARAMETERS</font>**

# %%
accs_dict, losses_dict = begin_comparison("initializer",
                                          iter = 50,
                                          set_to_use = 'val',
                                          monk = 2,
                                          epochs = 50,
                                          input_unit = 3,
                                          default_batch_size = 2,
                                          default_kernel_regularizer = L2(0),
                                          default_initializer = 'random_normal',
                                          default_act_input = 'relu',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = SGD(learning_rate=0.6, momentum=0.5),
                                          learning_rate = 1,
                                          plot = 0)

max(accs_dict.items(), key=lambda x: x[1])

# %%
accs_dict, losses_dict = begin_comparison("activation_input",
                                          iter = 50,
                                          set_to_use = 'val',
                                          monk = 2,
                                          epochs = 50,
                                          input_unit = 3,
                                          default_batch_size = 2,
                                          default_kernel_regularizer = L2(0),
                                          default_initializer = 'he_normal',
                                          default_act_input = 'relu',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = SGD(learning_rate=0.6, momentum=0.5),
                                          learning_rate = 1,
                                          plot = 0)

max(accs_dict.items(), key=lambda x: x[1])

# %%
accs_dict, losses_dict = begin_comparison("batch_size",
                                          iter = 50,
                                          set_to_use = 'val',
                                          monk = 2,
                                          epochs = 50,
                                          input_unit = 3,
                                          default_batch_size = 32,
                                          default_kernel_regularizer = L2(0),
                                          default_initializer = 'he_normal',
                                          default_act_input = 'sigmoid',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = SGD(learning_rate=0.6, momentum=0.5),
                                          learning_rate = 1,
                                          plot = 0)

max(accs_dict.items(), key=lambda x: x[1])

# %%
accs_dict, losses_dict = begin_comparison("optimizer",
                                          iter = 50,
                                          set_to_use = 'val',
                                          monk = 2,
                                          epochs = 50,
                                          input_unit = 3,
                                          default_batch_size = 2,
                                          default_kernel_regularizer = L2(0),
                                          default_initializer = 'he_normal',
                                          default_act_input = 'sigmoid',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = SGD(learning_rate=0.6, momentum=0.5),
                                          learning_rate = 1,
                                          plot = 0)

max(accs_dict.items(), key=lambda x: x[1])

# %%
keras.backend.clear_session()
tf.random.set_seed(12)

model = Sequential()
model.add(Dense(3, input_dim = X_MONK2_train.shape[1], activation="sigmoid", kernel_initializer="he_normal"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="he_normal"))
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.8, momentum=0.5), loss='mse', metrics='accuracy') # learning rate increased for sigmoid
model.summary()

history = model.fit(X2_train, y2_train, epochs=50, batch_size=2, validation_data=(X2_val, y2_val), verbose=0)

plot_score_loss(history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss'], save=1, name = 'MONK2_final')

# %%
# iterating for get avg TR accuracy and loss

train_accs = []
train_losses = []

count = 0
list_seeds = np.random.default_rng().choice(50, size = 50, replace = False)
for i in range(50):
    print(f"Processing {i + 1}/50")
    seed = list_seeds[i]
    keras.backend.clear_session()
    tf.random.set_seed(seed)

    model = Sequential()
    model.add(Dense(3, input_dim = X_MONK2_train.shape[1], activation="sigmoid", kernel_initializer="he_normal"))
    model.add(Dense(1, activation="sigmoid", kernel_initializer="he_normal"))
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.8, momentum=0.5), loss='mse', metrics='accuracy')

    history = model.fit(X2_train, y2_train, epochs=50, batch_size=2, validation_data=(X2_val, y2_val), verbose=0)
    loss, acc = model.evaluate(X2_train, y2_train, verbose = 0)
    train_accs.append(acc)
    train_losses.append(loss)

print("Avg TR Accuracy:", np.mean(train_accs))
print("Std TR Accuracy:", np.std(train_accs))
print("Avg TR Loss:", np.mean(train_losses))
print("Std TR Loss:", np.std(train_losses))

# %%
accs_dict, losses_dict = begin_comparison("testing",
                                          iter = 50,
                                          set_to_use = 'test',
                                          monk = 2,
                                          epochs = 50,
                                          input_unit = 3,
                                          default_batch_size = 2,
                                          default_initializer = 'he_normal',
                                          default_act_input = 'sigmoid',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = SGD(learning_rate=0.7, momentum=0.5),
                                          learning_rate = 1,
                                          plot = 0)

max(accs_dict.items(), key=lambda x: x[1])

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
        "learning_rate": "invscaling",
        "momentum": 0,
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
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
    "inv-scaling learning w/o momentum",
    "inv-scaling with momentum",
    "inv-scaling with Nesterov's momentum",
    "adaptive learning w/o momentum",
    "adaptive with momentum",
    "adaptive with Nesterov's momentum",
    "adam",
]

# %%
# def plot_on_dataset(X, y, ax, name):
models = []
for label, param in zip(labels, params):
    print("training", label)
    mlp = MLPClassifier(random_state=42, max_iter=500, **param)

    # some parameter combinations will not converge so they are ignored here
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
      mlp.fit(X2_train, y2_train)

    models.append(mlp)
    print("Score: %f" % mlp.score(X2_val, y2_val))
    print("Loss: %f" % mlp.loss_,'\n')

# %%
plt.figure(figsize=(12,4))
for i, label in zip(range(len(models)), labels):
  plt.plot(models[i].loss_curve_, label = label)
plt.title("Comparing different learning methods for MLP")
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
plt.show()

# %%
model = MLPClassifier(solver = 'adam',
                      max_iter = 50,
                      # validation_fraction = 0.3,
                      # early_stopping = True,
                      # n_iter_no_change = 10,
                      random_state = 42)

param_grid = {'hidden_layer_sizes': [(1,), (2,), (4,), (6,), (8,), (2,2), (4,4)],
              'momentum': [.001, .01, .1],
              'learning_rate_init': [.001, .01, 1],
              'batch_size': [1, 2, 4, 8],
              }

with warnings.catch_warnings():
  warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
  search = GridSearchCV(model,
                        param_grid,
                        cv = StratifiedKFold(5, shuffle=True, random_state=42),
                        verbose = 1).fit(X_MONK2_train, y_MONK2_train)

print('Best score:', search.best_score_)
print('Best params', search.best_estimator_)

# %%
model = MLPClassifier(solver = 'adam',
                      max_iter = 50,
                      validation_fraction = 0.3,
                      early_stopping = True,
                      n_iter_no_change = 10,
                      random_state = 42,
                      batch_size=1,
                      hidden_layer_sizes=(2,),
                      learning_rate_init=0.01,
                      momentum=0.001).fit(X_MONK2_train, y_MONK2_train)

print('Number of epochs:', len(model.loss_curve_))

# %%
max_iter = 22
mlp = MLPClassifier(solver='adam',
                    random_state=42,
                    batch_size=1,
                    hidden_layer_sizes=(2,),
                    learning_rate_init=0.01,
                    momentum=0.001)

train_scores, train_loss, val_scores, val_loss = mlp_fit(mlp, max_iter, monk=2)

plot_score_loss(train_scores, val_scores, train_loss, val_loss)

# %%
# Fitting final model to test set
mlp = MLPClassifier(solver = 'adam',
                    max_iter = 23,
                    random_state = 42,
                    batch_size=2,
                    hidden_layer_sizes=(2,),
                    learning_rate_init=0.01,
                    momentum=0.001)

mlp.fit(X_MONK2_train, y_MONK2_train)
y_pred = mlp.predict(X_MONK2_test)

print(classification_report(y_MONK2_test, y_pred, digits = 4))

# %% [markdown]
# ### **<font color="#CEFF5E">ASSESSING OTHER MODELS</font>**

# %%
# Models, parameters and hyperparameters to test with nested cross validation for MONK 1
knn_params = ['n_neighbors', 'weights', 'metric']
knn_param_grid = [np.arange(2,54), ["uniform", "distance"], ["euclidean", "cityblock", "chebyshev"]]

dt_params = ['criterion', 'max_depth', 'min_samples_split']
dt_param_grid = [['gini', 'entropy'], range(2,108,2), range(2, 108,2)]

lr_params = ['solver', 'C', 'max_iter']
lr_param_grid = [['saga', 'lbfgs'], [.001, .1, 1, 10, 100], [500, 1000]]

rf_params = ['criterion', 'max_depth', 'n_estimators']
rf_param_grid = [['gini', 'entropy'], range(2,78,20), [100, 200, 300]]

bnb_params = ['alpha', 'binarize', 'fit_prior']
bnb_param_grid = [[.001, .01, .1, 1], [0], [True, False]]

gb_params = ['criterion', 'learning_rate', 'n_estimators']
gb_param_grid = [['friedman_mse', 'squared_error'], [.001, .01, .1, 1], [100, 200, 300]]

svc_params = ['C', 'kernel', 'gamma']
svc_param_grid = [[.1, 1, 100, 1000, 10000], ['poly', 'rbf', 'sigmoid'], [.0001, .001, .01, .1, 1, 10, 100, 1000, 10000]]

nested_dict = {
    #'K-Nearest Neighbors': [KNeighborsClassifier, knn_params, knn_param_grid],
    #'Decision Tree': [DecisionTreeClassifier, dt_params, dt_param_grid],
    #'Logistic Regression': [LogisticRegression, lr_params, lr_param_grid],
    #'Random Forests': [RandomForestClassifier, rf_params, rf_param_grid],
    #'Bernoulli Naive Bayes': [BernoulliNB, bnb_params, bnb_param_grid],
    #'Gradient Boosting': [GradientBoostingClassifier, gb_params, gb_param_grid],
    'Support Vector Machine': [SVC, svc_params, svc_param_grid]}

# %%
monk = 2 # MONK set to use
print('MONK', monk, 'Nested Cross Validation results:\n')

# Ignoring convergence warning for logistic regression when max_iter is not enough to converge
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    for name, [model_to_use, model_params, param_grid] in nested_dict.items():
        avg_acc, std_dev = nested_cross_validation(monk = monk,
                                                   folds = 5,
                                                   model_to_use = model_to_use,
                                                   model_params = model_params,
                                                   param_grid = param_grid)
        print('------', name, '------')
        print('Average accuracy:  ', avg_acc)
        print('Standard deviation:', std_dev, '\n')

# %%
param_grid = {'criterion':['friedman_mse', 'squared_error'],
              'learning_rate': [0.001, 0.01, 0.1, 1.],
              'n_estimators':[100, 1000, 10000]}

search = GridSearchCV(GradientBoostingClassifier(),
                      param_grid=param_grid,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      verbose=1,
                      n_jobs=-1,
                      scoring='accuracy').fit(X_MONK2_train, y_MONK2_train)
print(search.best_score_, search.best_params_)

# %%
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, criterion='friedman_mse')
gbc.fit(X_MONK2_train, y_MONK2_train)
predictions = gbc.predict(X2_test)
print(classification_report(y2_test, predictions))

# %% [markdown]
# ### **<font color="#CEFF5E">SVM</font>**

# %% [markdown]
# GridSearch 1st run

# %%
model_svc = SVC()

svc_param_grid = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ['poly', 'rbf', 'sigmoid'], #default = rbf
    'class_weight': ['balanced', None],
    'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], #default=1.0
    'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], #default: auto
    }

svc_param_grid_linear = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ['linear'],
    'class_weight': ['balanced', None],
    'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], #default=1.0
    }

models_params = {
    'SVM': [model_svc, svc_param_grid],
    'SVMlinear': [model_svc, svc_param_grid_linear]
    }

best_params = {}
validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Random search starting')
    search = GridSearchCV(model,
                          param,
                          cv = KFold(10, shuffle=True, random_state=42),
                          verbose=2,
                          n_jobs = -1).fit(X_MONK2_train, y_MONK2_train)
    best_params[name] = search.best_estimator_
    validation_scores[name] = search.best_score_
    print('Best score: ', validation_scores[name])
    print('Best parameters: ', best_params[name], '\n')

# %% [markdown]
# GridSearch 2nd run

# %%
model_svc = SVC()

svc_param_grid = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ["rbf", "poly", "sigmoid"], #default = rbf
    'class_weight': ['balanced', None],
    'C': [5000, 6000, 7000, 8000, 10000, 11000, 12000, 13000, 14000, 15000], #default=1.0
    'gamma': [.005, .006, .007, .008, .009, .01, .02, .03, .04, .05], #default: auto
    }

svc_param_grid_linear = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
    'kernel': ['linear'],
    'class_weight': ['balanced', None],
    'C': [.000005, .000006, .000007, .000008, .000009, 1e-5, .00002, .00003, .00004, .00005], #default=1.0
    }

models_params = {
    'SVM': [model_svc, svc_param_grid],
    'SVMlinear': [model_svc, svc_param_grid_linear]
    }

best_params = {}
validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Random search starting')
    search = GridSearchCV(model,
                          param,
                          cv = KFold(10, shuffle=True, random_state=42),
                          verbose=2,
                          n_jobs = -1).fit(X_MONK2_train, y_MONK2_train)
    best_params[name] = search.best_estimator_
    validation_scores[name] = search.best_score_
    print('Best score: ', validation_scores[name])
    print('Best parameters: ', best_params[name], '\n')

# %% [markdown]
# GridSearch 3rd run

# %%
model_svc = SVC()

svc_param_grid = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'class_weight': ['balanced', None],
    'C': [9600, 9700, 9800, 9900, 10000, 10100, 10200, 10300, 10400], #default=1.0
    'gamma': [.0006, .0007, .0008, .0009, .008, .0081, .0082, .0083, .0084], #default: auto
    # 'random_state': [0]
    }

models_params = {
    'SVM': [model_svc, svc_param_grid],
    'SVMlinear': [model_svc, svc_param_grid_linear]
    }

best_params = {}
validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Random search starting')
    search = GridSearchCV(model,
                          param,
                          cv = KFold(10, shuffle=True, random_state=42),
                          verbose=2,
                          n_jobs = -1).fit(X_MONK2_train, y_MONK2_train)
    best_params[name] = search.best_estimator_
    validation_scores[name] = search.best_score_
    print('Best score: ', validation_scores[name])
    print('Best parameters: ', best_params[name], '\n')

# %%
# Testing best parameters on test set
svc = SVC(C=9600, gamma=0.008)
svc.fit(X_MONK2_train, y_MONK2_train)

y_pred = svc.predict(X_MONK2_test)

print(accuracy_score(y_MONK2_test, y_pred))
print(classification_report(y_MONK2_test, y_pred, digits = 4))

# %% [markdown]
# ## **<font color="#CEFF5E">SCIKITLEARN'S MLP</font>**

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
        "learning_rate": "invscaling",
        "momentum": 0,
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
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
    "inv-scaling learning w/o momentum",
    "inv-scaling with momentum",
    "inv-scaling with Nesterov's momentum",
    "adaptive learning w/o momentum",
    "adaptive with momentum",
    "adaptive with Nesterov's momentum",
    "adam",
]

# %%
# def plot_on_dataset(X, y, ax, name):
models = []
for label, param in zip(labels, params):
    print("training", label)
    mlp = MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(6,), **param)

    # some parameter combinations will not converge so they are ignored here
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
      mlp.fit(X2_train, y2_train)

    models.append(mlp)
    print("Score: %f" % mlp.score(X2_val, y2_val))
    print("Loss: %f" % mlp.loss_,'\n')

# %%
plt.figure(figsize=(12,4))
for i, label in zip(range(len(models)), labels):
  plt.plot(models[i].loss_curve_, label = label)
plt.title("Comparing different learning methods for MLP")
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
plt.show()

# %%
# Gridsearch to find optimal values for neural network
model = MLPClassifier(solver = 'adam',
                      max_iter = 50)

param_grid = {'hidden_layer_sizes': [(2,), (4,), (6,)],
              'momentum': [0, .001, .01, .1],
              'learning_rate_init': [.001, .01, 1],
              'batch_size': [1, 2, 8, 32],
              }

with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
      search = GridSearchCV(model,
                            param_grid,
                            cv = StratifiedKFold(5, shuffle = True, random_state = 42),
                            verbose = 1).fit(X_MONK2_train, y_MONK2_train)

print('Best score:', search.best_score_)
print('Best params', search.best_estimator_)

# %%
# Visualizing training curves for model found in gridsearch
mlp = MLPClassifier(solver = 'adam',
                    max_iter = 50,
                    batch_size=1,
                    hidden_layer_sizes=(6,),
                    learning_rate_init=0.01,
                    momentum=0.1,
                    random_state=42)

train_scores, train_loss, val_scores, val_loss = mlp_fit(mlp, max_iter = 50, monk=2)

plot_score_loss(train_scores = train_scores,
                val_scores = val_scores,
                train_loss = train_loss,
                val_loss = val_loss,
                save = 1,
                name = 'MONK2_sklearn')

# %%
iterations = 50
list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

train_acc = []
val_acc = []
test_acc = []

train_loss = []
val_loss = []
test_loss = []

for i in list_seeds:
    tr_acc, vl_acc, ts_acc, tr_loss, vl_loss, ts_loss = mlp_final_test(mlp=MLPClassifier(solver = 'adam',
                                                                                         max_iter = 50,
                                                                                         batch_size=1,
                                                                                         hidden_layer_sizes=(6,),
                                                                                         learning_rate_init=0.01,
                                                                                         momentum=0.1,
                                                                                         random_state=i), epochs=50, monk=2)
    train_acc.append(tr_acc)
    val_acc.append(vl_acc)
    test_acc.append(ts_acc)

    train_loss.append(tr_loss)
    val_loss.append(vl_loss)
    test_loss.append(ts_loss)

print('Avg train acc: ', round(np.mean(train_acc),4))
print('Std dev:       ', round(np.std(train_acc),4))
print('Avg val acc:   ', round(np.mean(val_acc),4))
print('Std dev:       ', round(np.std(val_acc),4))
print('Avg test acc:  ', round(np.mean(test_acc),4))
print('Std dev:       ', round(np.std(test_acc),4))

print('Avg train loss:', round(np.mean(train_loss),4))
print('Std dev:       ', round(np.std(train_loss),4))
print('Avg val loss:  ', round(np.mean(val_loss),4))
print('Std dev:       ', round(np.std(val_loss),4))
print('Avg test loss: ', round(np.mean(test_loss),4))
print('Std dev:       ', round(np.std(test_loss),4))

# %% [markdown]
# ## **<font color="#CEFF5E">NESTED CROSS VALIDATION</font>**

# %%
# Models, parameters and hyperparameters to test with nested cross validation for MONK 1
knn_params = ['n_neighbors', 'weights', 'metric']
knn_param_grid = [range(2,108,2), ["uniform", "distance"], ["euclidean", "cityblock", "chebyshev"]]

dt_params = ['criterion', 'max_depth', 'min_samples_split']
dt_param_grid = [['gini', 'entropy'], [1, 2, 4, 6, 8, 10, None], [2, 3, 4, 5, 6, 7, 8]]

lr_params = ['solver', 'C', 'max_iter']
lr_param_grid = [['saga', 'lbfgs'], [.0001, .001, .1, 1, 10, 100, 1000], [500, 1000]]

rf_params = ['criterion', 'max_depth', 'n_estimators']
rf_param_grid = [['gini', 'entropy'], [1, 3, 5, 7, 10, None], [50, 100, 150, 200]]

bnb_params = ['alpha', 'binarize', 'fit_prior']
bnb_param_grid = [[.0001, .001, .01, .1, 1], [0], [True, False]]

gb_params = ['criterion', 'learning_rate', 'n_estimators']
gb_param_grid = [['friedman_mse', 'squared_error'], [.001, .01, .1, 1], [50, 100, 150, 200]]

svc_params = ['C', 'kernel', 'gamma']
svc_param_grid = [[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5],
                  ['poly', 'rbf', 'sigmoid'],
                  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]]

nested_dict = {
    'K-Nearest Neighbors': [KNeighborsClassifier, knn_params, knn_param_grid],
    'Decision Tree': [DecisionTreeClassifier, dt_params, dt_param_grid],
    'Logistic Regression': [LogisticRegression, lr_params, lr_param_grid],
    'Random Forests': [RandomForestClassifier, rf_params, rf_param_grid],
    'Bernoulli Naive Bayes': [BernoulliNB, bnb_params, bnb_param_grid],
    'Gradient Boosting': [GradientBoostingClassifier, gb_params, gb_param_grid],
    'Support Vector Machine': [SVC, svc_params, svc_param_grid]}

# %%
monk = 2 # MONK set to use
print('MONK', monk, 'Nested Cross Validation results:\n')

# Ignoring convergence warning for logistic regression when max_iter is not enough to converge
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    for name, [model_to_use, model_params, param_grid] in nested_dict.items():
        avg_acc, std_dev = nested_cross_validation(monk = monk,
                                                   folds = 5,
                                                   model_to_use = model_to_use,
                                                   model_params = model_params,
                                                   param_grid = param_grid)
        print('------', name, '------')
        print('Average accuracy:  ', avg_acc)
        print('Standard deviation:', std_dev, '\n')

# %% [markdown]
# ### **<font color="#CEFF5E">SVM</font>**

# %% [markdown]
# GridSearch 1st run

# %%
model_svc = SVC()

svc_param_grid = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ['poly', 'rbf', 'sigmoid'], #default = rbf
    'class_weight': ['balanced', None],
    'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], #default=1.0
    'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], #default: auto
    }

svc_param_grid_linear = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ['linear'],
    'class_weight': ['balanced', None],
    'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], #default=1.0
    }

models_params = {
    'SVM': [model_svc, svc_param_grid],
    'SVMlinear': [model_svc, svc_param_grid_linear]
    }

best_params = {}
validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Random search starting')
    search = GridSearchCV(model,
                          param,
                          cv = KFold(10, shuffle=True, random_state=42),
                          verbose=2,
                          n_jobs = -1).fit(X_MONK2_train, y_MONK2_train)
    best_params[name] = search.best_estimator_
    validation_scores[name] = search.best_score_
    print('Best score: ', validation_scores[name])
    print('Best parameters: ', best_params[name], '\n')

# %% [markdown]
# GridSearch 2nd run

# %%
model_svc = SVC()

svc_param_grid = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ["rbf", "poly", "sigmoid"], #default = rbf
    'class_weight': ['balanced', None],
    'C': [5000, 6000, 7000, 8000, 10000, 11000, 12000, 13000, 14000, 15000], #default=1.0
    'gamma': [.005, .006, .007, .008, .009, .01, .02, .03, .04, .05], #default: auto
    }

svc_param_grid_linear = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
    'kernel': ['linear'],
    'class_weight': ['balanced', None],
    'C': [.000005, .000006, .000007, .000008, .000009, 1e-5, .00002, .00003, .00004, .00005], #default=1.0
    }

models_params = {
    'SVM': [model_svc, svc_param_grid],
    'SVMlinear': [model_svc, svc_param_grid_linear]
    }

best_params = {}
validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Random search starting')
    search = GridSearchCV(model,
                          param,
                          cv = KFold(10, shuffle=True, random_state=42),
                          verbose=2,
                          n_jobs = -1).fit(X_MONK2_train, y_MONK2_train)
    best_params[name] = search.best_estimator_
    validation_scores[name] = search.best_score_
    print('Best score: ', validation_scores[name])
    print('Best parameters: ', best_params[name], '\n')

# %% [markdown]
# GridSearch 3rd run

# %%
model_svc = SVC()

svc_param_grid = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'class_weight': ['balanced', None],
    'C': [9600, 9700, 9800, 9900, 10000, 10100, 10200, 10300, 10400], #default=1.0
    'gamma': [.0006, .0007, .0008, .0009, .008, .0081, .0082, .0083, .0084], #default: auto
    # 'random_state': [0]
    }

models_params = {
    'SVM': [model_svc, svc_param_grid],
    'SVMlinear': [model_svc, svc_param_grid_linear]
    }

best_params = {}
validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Random search starting')
    search = GridSearchCV(model,
                          param,
                          cv = KFold(10, shuffle=True, random_state=42),
                          verbose=2,
                          n_jobs = -1).fit(X_MONK2_train, y_MONK2_train)
    best_params[name] = search.best_estimator_
    validation_scores[name] = search.best_score_
    print('Best score: ', validation_scores[name])
    print('Best parameters: ', best_params[name], '\n')

# %%
# Testing best parameters on validation set
svc = SVC(C=9600, gamma=0.008)
svc.fit(X2_train, y2_train)

y_pred = svc.predict(X2_val)
print(classification_report(y2_val, y_pred, digits = 4))

auc_score = round(roc_auc_score(y2_val, y_pred),4)
print('AUC:', auc_score)

# %%
# Testing best parameters on test set
svc = SVC(C=9600, gamma=0.008)
svc.fit(X_MONK2_train, y_MONK2_train)

y_pred = svc.predict(X2_test)
print(classification_report(y2_test, y_pred, digits = 4))

auc_score = round(roc_auc_score(y2_test, y_pred),4)
print('AUC:', auc_score)

# %% [markdown]
# ### **<font color="#CEFF5E">GRADIENT BOOSTING</font>**

# %%
param_grid = {'criterion':['friedman_mse', 'squared_error'],
              'learning_rate': [0.001, 0.01, 0.1, 1.],
              'n_estimators':[50, 100, 200]}

search = GridSearchCV(GradientBoostingClassifier(),
                      param_grid=param_grid,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      verbose=1,
                      n_jobs=-1,
                      scoring='accuracy').fit(X_MONK2_train, y_MONK2_train)
print(search.best_score_, search.best_params_)

# %%
param_grid = {'criterion':['friedman_mse', 'squared_error'],
              'learning_rate': [.5, 7, 1, 1.2, 1.5],
              'n_estimators':[50, 100, 200]}

search = GridSearchCV(GradientBoostingClassifier(),
                      param_grid=param_grid,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      verbose=1,
                      n_jobs=-1,
                      scoring='accuracy').fit(X_MONK2_train, y_MONK2_train)
print(search.best_score_, search.best_params_)

# %%
param_grid = {'criterion':['friedman_mse', 'squared_error'],
              'learning_rate': [.3, .4, .5, .6],
              'n_estimators':[150, 175, 200, 225, 250]}

search = GridSearchCV(GradientBoostingClassifier(),
                      param_grid=param_grid,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      verbose=1,
                      n_jobs=-1,
                      scoring='accuracy').fit(X_MONK2_train, y_MONK2_train)
print(search.best_score_, search.best_params_)

# %%
# Testing on validation set
gbc = GradientBoostingClassifier(n_estimators=250, learning_rate=.5, criterion='friedman_mse')
gbc.fit(X2_train, y2_train)

y_pred = gbc.predict(X2_val)

print(classification_report(y2_val, y_pred, digits = 4))

auc_score = round(roc_auc_score(y2_val, y_pred),4)
print('AUC:', auc_score)

# %%
# Testing on test set
gbc = GradientBoostingClassifier(n_estimators=250, learning_rate=.5, criterion='friedman_mse')
gbc.fit(X_MONK2_train, y_MONK2_train)

y_pred = gbc.predict(X2_test)

print(classification_report(y2_test, y_pred, digits = 4))

auc_score = round(roc_auc_score(y_MONK2_test, y_pred),4)
print('AUC:', auc_score)

# %% [markdown]
# ### **<font color="#CEFF5E">RANDOM FORESTS</font>**

# %%
param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [1, 3, 5, 7, 10, None],
              'n_estimators': [50, 100, 150, 200]}

search = GridSearchCV(RandomForestClassifier(),
                      param_grid,
                      cv = StratifiedKFold(5, shuffle=True, random_state=42),
                      verbose = 1,
                      n_jobs = -1,
                      scoring = 'accuracy').fit(X_MONK2_train, y_MONK2_train)

print(search.best_score_, search.best_params_)

# %%
param_grid = {'criterion': ['gini'],
              'max_depth': [1, 3, 5, 7, 10, None],
              'n_estimators': [80, 90, 100, 110, 120, 130]}

search = GridSearchCV(RandomForestClassifier(),
                      param_grid,
                      cv = StratifiedKFold(5, shuffle=True, random_state=42),
                      verbose = 1,
                      n_jobs = -1,
                      scoring = 'accuracy').fit(X_MONK2_train, y_MONK2_train)

print(search.best_score_, search.best_params_)

# %%
# Testing on validation set
rf = RandomForestClassifier(n_estimators=110)
rf.fit(X2_train, y2_train)

y_pred = rf.predict(X2_val)

print(classification_report(y2_val, y_pred, digits = 4))

auc_score = round(roc_auc_score(y2_val, y_pred),4)
print('AUC:', auc_score)

# %%
# Testing on test set
rf = RandomForestClassifier(n_estimators=110)
rf.fit(X_MONK2_train, y_MONK2_train)

y_pred = rf.predict(X2_test)

print(classification_report(y2_test, y_pred, digits = 4))

auc_score = round(roc_auc_score(y2_test, y_pred),4)
print('AUC:', auc_score)

# %%
# Fitting final model to test set and evaluating it over n iterations to get mean score
iterations = 50
accuracy = []
auc = []

list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

for i in list_seeds:
    rf = RandomForestClassifier(n_estimators=110, random_state = i)
    rf.fit(X_MONK2_train, y_MONK2_train)

    y_pred = rf.predict(X_MONK2_test)

    acc = accuracy_score(y_MONK2_test, y_pred)
    accuracy.append(acc)
    auc_score = roc_auc_score(y_MONK2_test, y_pred)
    auc.append(auc_score)

print('Avg Accuracy:  ', round(np.mean(accuracy),4))
print('std dev:       ', round(np.std(accuracy),4))
print('AUC:           ', round(np.mean(auc),4))
print('std dev:       ', round(np.std(auc),4))

# %% [markdown]
# # **<font color="#34ebdb">5.0 MONK 3</font>**

# %% [markdown]
# ## **<font color="#CEFF5E">KERAS</font>**

# %%
model_search = KerasClassifier(build_fn, n_hidden_units=0, learning_rate=0, regularizer=L2, lambd=0, batch_size=0, momentum=0, epochs=50, random_state=42, verbose=0)

param_grid = {'n_hidden_units': [2, 4, 6],
              'learning_rate': [.01, .1, 1],
              'momentum': [0, .01, .1],
              'lambd': [0, .01, .1],
              'batch_size': [2, 8, 32]
              }

search_loss = GridSearchCV(model_search,
                           param_grid,
                           cv = StratifiedKFold(5, shuffle=True, random_state=42),
                           verbose = 0,
                           n_jobs = -1,
                           scoring = 'neg_mean_squared_error').fit(X_MONK3_train, y_MONK3_train, verbose=0)

print('Best score:', search_loss.best_score_, '\nBest params', search_loss.best_params_)

# %%
scores_loss = search_loss.cv_results_['mean_test_score']
params_loss = search_loss.cv_results_['params']

grid_dict_loss = {}
for score, param in zip(scores_loss, params_loss):
    grid_dict_loss[str(param)] = score

optimal_model_loss = [(key, value) for key, value in grid_dict_loss.items() if value > -.07]
for opt_model in optimal_model_loss:
  print(opt_model)

# %%
keras.backend.clear_session()
tf.random.set_seed(42)

model = Sequential()
model.add(Dense(4, input_dim = X_MONK3_train.shape[1], kernel_regularizer = L2(0), activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=keras.optimizers.SGD(learning_rate=.1, momentum=0), loss='mse', metrics='accuracy')
model.summary()

history = model.fit(X3_train, y3_train, epochs=50, batch_size=2, validation_data=(X3_val, y3_val), verbose=0)

loss, acc = model.evaluate(X3_train, y3_train, verbose = 0)
print('Training accuracy:  ', round(acc,4))
print('Training loss:      ', round(loss,4))

loss_val, acc_val = model.evaluate(X3_val, y3_val, verbose = 0)
print('Validation accuracy:', round(acc_val,4))
print('Validation loss:    ', round(loss_val,4))

plot_score_loss(train_scores = history.history['accuracy'],
                val_scores = history.history['val_accuracy'],
                train_loss = history.history['loss'],
                val_loss = history.history['val_loss'],
                save = 1,
                name = 'MONK3_noreg')

# %%
keras.backend.clear_session()
tf.random.set_seed(42)

model = Sequential()
model.add(Dense(4, input_dim = X_MONK3_train.shape[1], kernel_regularizer = L2(.01), activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=keras.optimizers.SGD(learning_rate=.1, momentum=0), loss='mse', metrics='accuracy')
model.summary()

history = model.fit(X3_train, y3_train, epochs=50, batch_size=2, validation_data=(X3_val, y3_val), verbose=0)

loss, acc = model.evaluate(X3_train, y3_train, verbose = 0)
print('Training accuracy:  ', round(acc,4))
print('Training loss:      ', round(loss,4))

loss_val, acc_val = model.evaluate(X3_val, y3_val, verbose = 0)
print('Validation accuracy:', round(acc_val,4))
print('Validation loss:    ', round(loss_val,4))

plot_score_loss(train_scores = history.history['accuracy'],
                val_scores = history.history['val_accuracy'],
                train_loss = history.history['loss'],
                val_loss = history.history['val_loss'],
                save = 1,
                name = 'MONK3_reg')

# %%
keras.backend.clear_session()
tf.random.set_seed(42)

model = Sequential()
model.add(Dense(4, input_dim = X_MONK3_train.shape[1], kernel_regularizer = L2(.01), activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=keras.optimizers.SGD(learning_rate=.1, momentum=0), loss='mse', metrics='accuracy')
model.summary()

history = model.fit(X3_train, y3_train, epochs=50, batch_size=8, validation_data=(X3_val, y3_val), verbose=0)

loss, acc = model.evaluate(X3_train, y3_train, verbose = 0)
print('Training accuracy:  ', round(acc,4))
print('Training loss:      ', round(loss,4))

loss_val, acc_val = model.evaluate(X3_val, y3_val, verbose = 0)
print('Validation accuracy:', round(acc_val,4))
print('Validation loss:    ', round(loss_val,4))

plot_score_loss(train_scores = history.history['accuracy'],
                val_scores = history.history['val_accuracy'],
                train_loss = history.history['loss'],
                val_loss = history.history['val_loss'],
                save = 0,
                name = '')

# %%
keras.backend.clear_session()
tf.random.set_seed(42)

model = Sequential()
model.add(Dense(4, input_dim = X_MONK3_train.shape[1], kernel_regularizer = L2(.01), activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=keras.optimizers.SGD(learning_rate=.1, momentum=0), loss='mse', metrics='accuracy')

model.fit(X3_train, y3_train, epochs=30, batch_size=2, verbose=0)

predictions = model.predict(X3_val)
y_pred = get_pred(predictions)

print(classification_report(y3_val, y_pred, digits = 4))

# %% [markdown]
# ## **<font color="#CEFF5E">KERAS: COMPARING PARAMETERS</font>**

# %%
accs_dict, losses_dict = begin_comparison("initializer",
                                          iter = 50,
                                          set_to_use = 'val',
                                          monk = 3,
                                          epochs = 30,
                                          input_unit = 4,
                                          default_batch_size = 2,
                                          default_initializer = 'glorot_normal',
                                          default_kernel_regularizer = L2(.01),
                                          default_act_input = 'relu',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = SGD(learning_rate=.1, momentum=0),
                                          learning_rate = 1,
                                          plot = 0)

max(accs_dict.items(), key=lambda x: x[1])

# %%
accs_dict, losses_dict = begin_comparison("optimizer",
                                          iter = 50,
                                          set_to_use = 'val',
                                          monk = 3,
                                          epochs = 30,
                                          input_unit = 4,
                                          default_batch_size = 2,
                                          default_initializer = 'random_normal',
                                          default_kernel_regularizer = L2(.01),
                                          default_act_input = 'relu',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = SGD(learning_rate=.1, momentum=0),
                                          learning_rate = .01,
                                          plot = 0)

max(accs_dict.items(), key=lambda x: x[1])

# %%
accs_dict, losses_dict = begin_comparison("activation_input",
                                          iter = 50,
                                          set_to_use = 'val',
                                          monk = 3,
                                          epochs = 30,
                                          input_unit = 4,
                                          default_batch_size = 2,
                                          default_initializer = 'random_normal',
                                          default_kernel_regularizer = L2(.01),
                                          default_act_input = 'relu',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = RMSprop(learning_rate=.01),
                                          learning_rate = .01,
                                          plot = 0)

max(accs_dict.items(), key=lambda x: x[1])

# %%
accs_dict, losses_dict = begin_comparison("batch_size",
                                          iter = 50,
                                          set_to_use = 'val',
                                          monk = 3,
                                          epochs = 30,
                                          input_unit = 4,
                                          default_batch_size = 2,
                                          default_initializer = 'random_normal',
                                          default_kernel_regularizer = L2(.01),
                                          default_act_input = 'relu',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = RMSprop(learning_rate=.01),
                                          learning_rate = .01,
                                          plot = 0)

max(accs_dict.items(), key=lambda x: x[1])

# %%
keras.backend.clear_session()
tf.random.set_seed(285)

model = Sequential()
model.add(Dense(4, input_dim = X_MONK3_train.shape[1], kernel_regularizer = L2(.01), activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=.01), loss='mse', metrics='accuracy')
model.summary()

history = model.fit(X3_train, y3_train, epochs=50, batch_size=8, validation_data=(X3_val, y3_val), verbose=0)

loss, acc = model.evaluate(X3_train, y3_train, verbose = 0)
print('Training accuracy:  ', round(acc,4))
print('Training loss:      ', round(loss,4))

loss_val, acc_val = model.evaluate(X3_val, y3_val, verbose = 0)
print('Validation accuracy:', round(acc_val,4))
print('Validation loss:    ', round(loss_val,4))

plot_score_loss(train_scores = history.history['accuracy'],
                val_scores = history.history['val_accuracy'],
                train_loss = history.history['loss'],
                val_loss = history.history['val_loss'],
                save = 1,
                name = 'MONK3_final')

# %%
# iterating for get avg TR accuracy and loss

train_accs = []
train_losses = []

count = 0
list_seeds = np.random.default_rng().choice(50, size = 50, replace = False)
for i in range(50):
    print(f"Processing {i + 1}/50")
    seed = list_seeds[i]
    keras.backend.clear_session()
    tf.random.set_seed(seed)

    model = Sequential()
    model.add(Dense(4, input_dim = X_MONK3_train.shape[1], kernel_regularizer = L2(.01), activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=.01), loss='mse', metrics='accuracy')

    history = model.fit(X3_train, y3_train, epochs=50, batch_size=8, validation_data=(X3_val, y3_val), verbose=0)
    loss, acc = model.evaluate(X3_train, y3_train, verbose = 0)
    train_accs.append(acc)
    train_losses.append(loss)

print("Avg TR Accuracy:", np.mean(train_accs))
print("Std TR Accuracy:", np.std(train_accs))
print("Avg TR Loss:", np.mean(train_losses))
print("Std TR Loss:", np.std(train_losses))

# %%
accs_dict, losses_dict = begin_comparison("testing",
                                          iter = 50,
                                          set_to_use = 'test',
                                          monk = 3,
                                          epochs = 30,
                                          input_unit = 4,
                                          default_batch_size = 8,
                                          default_initializer = 'random_normal',
                                          default_kernel_regularizer = L2(.01),
                                          default_act_input = 'relu',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = RMSprop(learning_rate=.01),
                                          learning_rate = .01,
                                          plot = 1)

# %%
keras.backend.clear_session()
tf.random.set_seed(285)

model = Sequential()
model.add(Dense(4, input_dim = X_MONK3_train.shape[1], kernel_regularizer = L2(0), activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=.01), loss='mse', metrics='accuracy')
model.summary()

history = model.fit(X3_train, y3_train, epochs=50, batch_size=8, validation_data=(X3_val, y3_val), verbose=0)

loss, acc = model.evaluate(X3_train, y3_train, verbose = 0)
print('Training accuracy:  ', round(acc,4))
print('Training loss:      ', round(loss,4))

loss_val, acc_val = model.evaluate(X3_val, y3_val, verbose = 0)
print('Validation accuracy:', round(acc_val,4))
print('Validation loss:    ', round(loss_val,4))

plot_score_loss(train_scores = history.history['accuracy'],
                val_scores = history.history['val_accuracy'],
                train_loss = history.history['loss'],
                val_loss = history.history['val_loss'],
                save=0, name='MONK3_nonr')

# %%
# iterating for get avg TR accuracy and loss

train_accs = []
train_losses = []

count = 0
list_seeds = np.random.default_rng().choice(50, size = 50, replace = False)
for i in range(50):
    print(f"Processing {i + 1}/50")
    seed = list_seeds[i]
    keras.backend.clear_session()
    tf.random.set_seed(seed)

    model = Sequential()
    model.add(Dense(4, input_dim = X_MONK3_train.shape[1], kernel_regularizer = L2(0), activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=.01), loss='mse', metrics='accuracy')

    history = model.fit(X3_train, y3_train, epochs=50, batch_size=8, validation_data=(X3_val, y3_val), verbose=0)
    loss, acc = model.evaluate(X3_train, y3_train, verbose = 0)
    train_accs.append(acc)
    train_losses.append(loss)

print("Avg TR Accuracy:", np.mean(train_accs))
print("Std TR Accuracy:", np.std(train_accs))
print("Avg TR Loss:", np.mean(train_losses))
print("Std TR Loss:", np.std(train_losses))

# %%
accs_dict, losses_dict = begin_comparison("testing",
                                          iter = 50,
                                          set_to_use = 'test',
                                          monk = 3,
                                          epochs = 30,
                                          input_unit = 4,
                                          default_batch_size = 8,
                                          default_initializer = 'random_normal',
                                          default_kernel_regularizer = L2(0),
                                          default_act_input = 'relu',
                                          default_act_output = 'sigmoid',
                                          default_optimizer = RMSprop(learning_rate=.01),
                                          learning_rate = 1,
                                          plot = 1)

# %% [markdown]
# ## **<font color="#CEFF5E">SCIKITLEARN'S MLP</font>**

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
        "learning_rate": "invscaling",
        "momentum": 0,
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "nesterovs_momentum": False,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
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
    "inv-scaling learning w/o momentum",
    "inv-scaling with momentum",
    "inv-scaling with Nesterov's momentum",
    "adaptive learning w/o momentum",
    "adaptive with momentum",
    "adaptive with Nesterov's momentum",
    "adam",
]

# %%
# def plot_on_dataset(X, y, ax, name):
models = []
for label, param in zip(labels, params):
    print("training", label)
    mlp = MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(6,), **param)

    # some parameter combinations will not converge so they are ignored here
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
      mlp.fit(X3_train, y3_train)

    models.append(mlp)
    print("Score: %f" % mlp.score(X3_val, y3_val))
    print("Loss: %f" % mlp.loss_,'\n')

# %%
plt.figure(figsize=(12,4))
for i, label in zip(range(len(models)), labels):
    plt.plot(models[i].loss_curve_, label = label)
plt.title("Comparing different learning methods for MLP")
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
plt.show()

# %%
# Gridsearch to find optimal values for neural network
model = MLPClassifier(solver = 'adam',
                      max_iter = 50,
                      random_state = 42)

param_grid = {'hidden_layer_sizes': [(2,), (4,), (6,)],
              'momentum': [0, .001, .01, .1],
              'learning_rate_init': [.001, .01, 1],
              'batch_size': [1, 2, 8, 32],
              }

with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
      search = GridSearchCV(model,
                            param_grid,
                            cv = StratifiedKFold(5, shuffle = True, random_state = 42),
                            verbose = 1).fit(X_MONK3_train, y_MONK3_train)

print('Best score:', search.best_score_)
print('Best params', search.best_estimator_)

# %%
# Visualizing training curves for model found in gridsearch
mlp = MLPClassifier(solver = 'adam',
                    random_state = 42,
                    max_iter = 50,
                    batch_size=8,
                    hidden_layer_sizes=(2,),
                    learning_rate_init=0.01,
                    momentum=0)

train_scores, train_loss, val_scores, val_loss = mlp_fit(mlp, max_iter = 50, monk=3)

plot_score_loss(train_scores = train_scores,
                val_scores = val_scores,
                train_loss = train_loss,
                val_loss = val_loss,
                save = 0,
                name = '')

# %%
iterations = 50
list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

train_acc = []
val_acc = []
test_acc = []

train_loss = []
val_loss = []
test_loss = []

for i in list_seeds:
    tr_acc, vl_acc, ts_acc, tr_loss, vl_loss, ts_loss = mlp_final_test(mlp=MLPClassifier(solver = 'adam',
                                                                                         random_state = i,
                                                                                         max_iter = 50,
                                                                                         batch_size=8,
                                                                                         hidden_layer_sizes=(2,),
                                                                                         learning_rate_init=0.01,
                                                                                         momentum=0), epochs=50, monk=3)
    train_acc.append(tr_acc)
    val_acc.append(vl_acc)
    test_acc.append(ts_acc)

    train_loss.append(tr_loss)
    val_loss.append(vl_loss)
    test_loss.append(ts_loss)

print('Avg train acc: ', round(np.mean(train_acc),4))
print('Std dev:       ', round(np.std(train_acc),4))
print('Avg val acc:   ', round(np.mean(val_acc),4))
print('Std dev:       ', round(np.std(val_acc),4))
print('Avg test acc:  ', round(np.mean(test_acc),4))
print('Std dev:       ', round(np.std(test_acc),4))

print('Avg train loss:', round(np.mean(train_loss),4))
print('Std dev:       ', round(np.std(train_loss),4))
print('Avg val loss:  ', round(np.mean(val_loss),4))
print('Std dev:       ', round(np.std(val_loss),4))
print('Avg test loss: ', round(np.mean(test_loss),4))
print('Std dev:       ', round(np.std(test_loss),4))

# %% [markdown]
# ## **<font color="#CEFF5E">NESTED CROSS VALIDATION</font>**

# %%
# Models, parameters and hyperparameters to test with nested cross validation for MONK 1
knn_params = ['n_neighbors', 'weights', 'metric']
knn_param_grid = [range(2,78,2), ["uniform", "distance"], ["euclidean", "cityblock", "chebyshev"]]

dt_params = ['criterion', 'max_depth', 'min_samples_split']
dt_param_grid = [['gini', 'entropy'], [1, 2, 4, 6, 8, 10, None], [2, 3, 4, 5, 6, 7, 8]]

lr_params = ['solver', 'C', 'max_iter']
lr_param_grid = [['saga', 'lbfgs'], [.0001, .001, .1, 1, 10, 100, 1000], [500, 1000]]

rf_params = ['criterion', 'max_depth', 'n_estimators']
rf_param_grid = [['gini', 'entropy'], [1, 3, 5, 7, 10, None], [50, 100, 150, 200]]

bnb_params = ['alpha', 'binarize', 'fit_prior']
bnb_param_grid = [[.0001, .001, .01, .1, 1], [0], [True, False]]

gb_params = ['criterion', 'learning_rate', 'n_estimators']
gb_param_grid = [['friedman_mse', 'squared_error'], [.001, .01, .1, 1], [50, 100, 150, 200]]

svc_params = ['C', 'kernel', 'gamma']
svc_param_grid = [[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5],
                  ['poly', 'rbf', 'sigmoid'],
                  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]]

nested_dict = {
    'K-Nearest Neighbors': [KNeighborsClassifier, knn_params, knn_param_grid],
    'Decision Tree': [DecisionTreeClassifier, dt_params, dt_param_grid],
    'Logistic Regression': [LogisticRegression, lr_params, lr_param_grid],
    'Random Forests': [RandomForestClassifier, rf_params, rf_param_grid],
    'Bernoulli Naive Bayes': [BernoulliNB, bnb_params, bnb_param_grid],
    'Gradient Boosting': [GradientBoostingClassifier, gb_params, gb_param_grid],
    'Support Vector Machine': [SVC, svc_params, svc_param_grid]}

# %%
monk = 3 # MONK set to use
print('MONK', monk, 'Nested Cross Validation results:\n')

# Ignoring convergence warning for logistic regression when max_iter is not enough to converge
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    for name, [model_to_use, model_params, param_grid] in nested_dict.items():
        avg_acc, std_dev = nested_cross_validation(monk = monk,
                                                   folds = 5,
                                                   model_to_use = model_to_use,
                                                   model_params = model_params,
                                                   param_grid = param_grid)
        print('------', name, '------')
        print('Average accuracy:  ', avg_acc)
        print('Standard deviation:', std_dev, '\n')

# %% [markdown]
# ### **<font color="#CEFF5E">SVM</font>**

# %% [markdown]
# GridSearch 1st run

# %%
model_svc = SVC()

svc_param_grid = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ['poly', 'rbf', 'sigmoid'], #default = rbf
    'class_weight': ['balanced', None],
    'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], #default=1.0
    'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], #default: auto
    }

svc_param_grid_linear = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ['linear'],
    'class_weight': ['balanced', None],
    'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], #default=1.0
    }

models_params = {
    'SVM': [model_svc, svc_param_grid],
    'SVMlinear': [model_svc, svc_param_grid_linear]
    }

best_params = {}
validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Random search starting')
    search = GridSearchCV(model,
                          param,
                          cv = KFold(10, shuffle=True, random_state=42),
                          verbose=2,
                          n_jobs = -1).fit(X_MONK3_train, y_MONK3_train)
    best_params[name] = search.best_estimator_
    validation_scores[name] = search.best_score_
    print('Best score: ', validation_scores[name])
    print('Best parameters: ', best_params[name], '\n')

# %% [markdown]
# GridSearch 2nd run

# %%
model_svc = SVC()

svc_param_grid = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ['poly', 'rbf', 'sigmoid'], #default = rbf
    'class_weight': ['balanced', None],
    'C': [.5, .6, .7, .8, .9, 1, 2, 3, 4, 5 ], #default=1.0
    'gamma': [.01, .02, .03, .04, .05, .07, .08, .09, .1], #default: auto
    }

svc_param_grid_linear = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ['linear'],
    'class_weight': ['balanced', None],
    'C': [.01, .02, .03, .04, .05, .07, .08, .09, .1], #default=1.0
    }

models_params = {
    'SVM': [model_svc, svc_param_grid],
    'SVMlinear': [model_svc, svc_param_grid_linear]
    }

best_params = {}
validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Random search starting')
    search = GridSearchCV(model,
                          param,
                          cv = KFold(10, shuffle=True, random_state=42),
                          verbose=2,
                          n_jobs = -1).fit(X_MONK3_train, y_MONK3_train)
    best_params[name] = search.best_estimator_
    validation_scores[name] = search.best_score_
    print('Best score: ', validation_scores[name])
    print('Best parameters: ', best_params[name], '\n')

# %% [markdown]
# GridSearch 3rd run

# %%
model_svc = SVC()

svc_param_grid = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ['poly', 'rbf', 'sigmoid'], #default = rbf
    'class_weight': ['balanced', None],
    'C': [.05, .06, .07, .08, .09, .1,  .2, .3, .4, .5], #default=1.0
    'gamma': [.1], #default: auto
    }

svc_param_grid_linear = {
    #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    'kernel': ['linear'],
    'class_weight': ['balanced', None],
    'C': [.065, .066, .067, .068, .069, .070, .071, .072, .073, .074, .075], #default=1.0
    }

models_params = {
    'SVM': [model_svc, svc_param_grid],
    'SVMlinear': [model_svc, svc_param_grid_linear]
    }

best_params = {}
validation_scores = {}

for name, [model, param] in models_params.items():
    print(f'{name} Random search starting')
    search = GridSearchCV(model,
                          param,
                          cv = KFold(10, shuffle=True, random_state=42),
                          verbose=2,
                          n_jobs = -1).fit(X_MONK3_train, y_MONK3_train)
    best_params[name] = search.best_estimator_
    validation_scores[name] = search.best_score_
    print('Best score: ', validation_scores[name])
    print('Best parameters: ', best_params[name], '\n')

# %%
# Testing best parameters on validation set
svc = SVC(C=0.5, class_weight='balanced', gamma=0.1).fit(X3_train, y3_train)

y_pred = svc.predict(X3_val)
print(classification_report(y3_val, y_pred, digits = 4))

auc_score = round(roc_auc_score(y3_val, y_pred),4)
print('AUC:', auc_score)

# %%
# Testing best parameters (linear) on validation set
svc = SVC(C=0.065, class_weight='balanced', kernel='linear').fit(X3_train, y3_train)

y_pred = svc.predict(X3_val)
print(classification_report(y3_val, y_pred, digits = 4))

auc_score = round(roc_auc_score(y3_val, y_pred),4)
print('AUC:', auc_score)

# %%
# Testing best parameters on test set
svc = SVC(C=0.065, class_weight='balanced', kernel='linear').fit(X_MONK3_train, y_MONK3_train)

y_pred = svc.predict(X_MONK3_test)
print(classification_report(y_MONK3_test, y_pred, digits = 4))

auc_score = round(roc_auc_score(y_MONK3_test, y_pred),4)
print('AUC:', auc_score)

# %% [markdown]
# ### **<font color="#CEFF5E">GRADIENT BOOSTING</font>**

# %%
param_grid = {'criterion':['friedman_mse', 'squared_error'],
              'learning_rate': [0.001, 0.01, 0.1, 1.],
              'n_estimators':[50, 100, 200]}

search = GridSearchCV(GradientBoostingClassifier(),
                      param_grid=param_grid,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      verbose=1,
                      n_jobs=-1,
                      scoring='accuracy').fit(X_MONK3_train, y_MONK3_train)
print(search.best_score_, search.best_params_)

# %%
# Testing on validation set
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=.01, criterion='friedman_mse')
gbc.fit(X3_train, y3_train)

y_pred = gbc.predict(X3_val)

print(classification_report(y3_val, y_pred, digits = 4))

auc_score = round(roc_auc_score(y3_val, y_pred),4)
print('AUC:', auc_score)

# %%
# Testing on test set
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=.01, criterion='friedman_mse')
gbc.fit(X_MONK3_train, y_MONK3_train)

y_pred = gbc.predict(X3_test)

print(classification_report(y3_test, y_pred, digits = 4))

auc_score = round(roc_auc_score(y_MONK3_test, y_pred),4)
print('AUC:', auc_score)

# %%
# Fitting final model to test set and evaluating it over n iterations to get mean score
iterations = 50
accuracy = []
auc = []

list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

for i in list_seeds:
    gbc = GradientBoostingClassifier(n_estimators = 100, learning_rate = .01, criterion = 'friedman_mse', random_state = i)
    gbc.fit(X_MONK3_train, y_MONK3_train)

    y_pred = gbc.predict(X_MONK3_test)

    acc = accuracy_score(y_MONK3_test, y_pred)
    accuracy.append(acc)
    auc_score = roc_auc_score(y_MONK3_test, y_pred)
    auc.append(auc_score)

print('Avg Accuracy:  ', round(np.mean(accuracy),4))
print('std dev:       ', round(np.std(accuracy),4))
print('AUC:           ', round(np.mean(auc),4))
print('std dev:       ', round(np.std(auc),4))

# %% [markdown]
# ### **<font color="#CEFF5E">RANDOM FORESTS</font>**

# %%
param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [1, 3, 5, 7, 10, None],
              'n_estimators': [50, 100, 150, 200]}

search = GridSearchCV(RandomForestClassifier(),
                      param_grid,
                      cv = StratifiedKFold(5, shuffle=True, random_state=42),
                      verbose = 1,
                      n_jobs = -1,
                      scoring = 'accuracy').fit(X_MONK3_train, y_MONK3_train)

print(search.best_score_, search.best_params_)

# %%
param_grid = {'criterion': ['gini'],
              'max_depth': [None],
              'n_estimators': [200, 250, 300]}

search = GridSearchCV(RandomForestClassifier(),
                      param_grid,
                      cv = StratifiedKFold(5, shuffle=True, random_state=42),
                      verbose = 1,
                      n_jobs = -1,
                      scoring = 'accuracy').fit(X_MONK3_train, y_MONK3_train)

print(search.best_score_, search.best_params_)

# %%
# Testing on validation set
rf = RandomForestClassifier(n_estimators=250)
rf.fit(X3_train, y3_train)

y_pred = rf.predict(X3_val)

print(classification_report(y3_val, y_pred, digits = 4))

auc_score = round(roc_auc_score(y3_val, y_pred),4)
print('AUC:', auc_score)

# %%
# Testing on test set
rf = RandomForestClassifier(n_estimators=250)
rf.fit(X_MONK3_train, y_MONK3_train)

y_pred = rf.predict(X3_test)

print(classification_report(y3_test, y_pred, digits = 4))

auc_score = round(roc_auc_score(y3_test, y_pred),4)
print('AUC:', auc_score)

# %%
# Fitting final model to test set and evaluating it over n iterations to get mean score
iterations = 50
accuracy = []
auc = []

list_seeds = np.random.default_rng().choice(iterations, size = iterations, replace = False)

for i in list_seeds:
    rf = RandomForestClassifier(n_estimators=250, random_state = i)
    rf.fit(X_MONK3_train, y_MONK3_train)

    y_pred = rf.predict(X_MONK3_test)

    acc = accuracy_score(y_MONK3_test, y_pred)
    accuracy.append(acc)
    auc_score = roc_auc_score(y_MONK3_test, y_pred)
    auc.append(auc_score)

print('Avg Accuracy:  ', round(np.mean(accuracy),4))
print('std dev:       ', round(np.std(accuracy),4))
print('AUC:           ', round(np.mean(auc),4))
print('std dev:       ', round(np.std(auc),4))


