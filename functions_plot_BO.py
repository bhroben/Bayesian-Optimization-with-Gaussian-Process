import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import time
from tqdm import tqdm
from scipy import integrate
from scipy.stats import multivariate_normal,norm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy import integrate
from scipy.stats import multivariate_normal,norm
from sklearn.datasets import load_iris
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import fetch_openml
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

################################## Utilities ##################################

def mu(k, Sigma, y):
    """
    Mean of the marginal distribution of f(x)

    Parameters
    ----------
    k : array_like
        kernel vector w.r.t. the previous observations
    Sigma : array_like
        inverse of covariance matrix of the previous observations
    y : array_like 
        previous observations

    Returns
    -------
    float
        mean of the marginal distribution of f(x)
    """
    return k.T @ Sigma @ y


def generate_combinations(dictionary):
    """
    Generate all combinations of values in a dictionary
    """
    keys = list(dictionary.keys())
    value_lists = [dictionary[key] for key in keys]
    combinations = list(itertools.product(*value_lists))
    return combinations


def sigma(Kx, k, Sigma):
    """
    Variance of the marginal distribution of f(x)

    Parameters
    ----------
    Kx : array_like
        kernel vector w.r.t. the new observation K(x, x)
    k : array_like
        kernel vector w.r.t. the previous observations
    Sigma : array_like
        inverse of covariance matrix of the previous observations

    Returns
    -------
    float
        variance of the marginal distribution of f(x)
    """
    return Kx - k.T @ Sigma @ k


def kernel_M52(x1, x2, l=None, s=1):
    """
    Matern 5/2 kernel

    Parameters
    ----------
    x1 : array_like
        First input
    x2 : array_like
        Second input
    l : array_like 
        Length scales
    s : float
        Signal variance
    """
    if l is None:
        l = np.ones(len(x1))
    # check if l and x have the same dimension
    if len(x1) != len(l):
        raise ValueError('Dimension of x and l must be the same')
    r2 = np.sum(((x1-x2)/l)**2)
    return s*(1 + np.sqrt(5*r2) + 5/3*r2)*np.exp(-np.sqrt(5*r2))

# Expected improvement 

def acq_func_EI(y_best, mu, sigma):
    """
    Expected improvement acquisition function

    Parameters
    ----------
    y_best : float
        Best observed function value
    mu : float
        Mean of the predictive distribution
    sigma : float
        Standard deviation of the predictive distribution
    
    Returns
    -------
    float
        Expected improvement acquisition function value
    """ 
    # initialize standard normal distribution
    stand_norm = multivariate_normal(mean=[0], cov=[[1]])
    epsilon = 1e-6  # Small epsilon value to prevent division by zero
    z = (y_best - mu) / (sigma + epsilon)
    return (z*stand_norm.cdf(z) + stand_norm.pdf(z)) * sigma

def acq_func_PI(y_best, mu, sigma):
    """
    Probability of Improvement acquisition function

    Parameters
    ----------
    y_best : float
        Best observed function value
    mu : float
        Mean of the predictive distribution
    sigma : float
        Standard deviation of the predictive distribution
    
    Returns
    -------
    float
        Probability of Improvement acquisition function value
    """ 
    gamma = y_best - mu
    z = gamma / sigma
    return norm.cdf(z)

def random_grid_search(X_train, X_val, y_train, y_val, hyperparameters, model, model_type="SVM", num_trials=10):
    losses = []
    best_losses = []
    best_loss = float('inf')
    best_params = None
    first_params = None
    execution_times = []

    for _ in range(num_trials):
        start_time = time.time()

        param_dict = {param: random.choice(values) for param, values in hyperparameters.items()}

        if model_type == "CNN":
            cnn_model = model(X_train, y_train, **param_dict)
            val_loss = evaluate_cnn(cnn_model, X_val, y_val)
        elif model_type == "SVM":
            clf = model(**param_dict)
            clf.fit(X_train, y_train)
            val_loss = 1 - clf.score(X_val, y_val)
        else:
            raise ValueError("Invalid model_type. Supported types are 'SVM' and 'CNN'.")

        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

        losses.append(val_loss)

        if first_params is None:
            first_params = param_dict

        if val_loss < best_loss:
            best_loss = val_loss
            best_params = param_dict
            best_losses.append(best_loss)

    return first_params, best_params, losses, best_losses, execution_times

def custom_random_grid_search_cifar(train_loader, val_loader, hyperparameters, model, num_random_samples=10):
    losses = []
    best_losses = []
    best_loss = float('inf')
    best_params = None
    first_params = None
    execution_times = []

    for _ in range(num_random_samples):
        start_time = time.time()

        param_dict = {param: random.choice(values) for param, values in hyperparameters.items()}

        cnn_model = model(train_loader, **param_dict)
        val_loss = evaluate_NET(cnn_model, val_loader)

        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

        losses.append(val_loss)

        if not first_params:
            first_params = param_dict

        if val_loss < best_loss:
            best_loss = val_loss
            best_params = param_dict
            best_losses.append(best_loss)

    return first_params, best_params, losses, best_losses, execution_times

def custom_grid_search(X_train, X_val, y_train, y_val, hyperparameters, model, model_type="SVM"):
    losses = []
    best_losses = []
    best_loss = float('inf')
    best_params = None
    first_params = None
    execution_times = []

    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(*hyperparameters.values()))

    for idx, params in enumerate(param_combinations):
        start_time = time.time()

        param_dict = dict(zip(hyperparameters.keys(), params))

        if model_type == "CNN":
            cnn_model = model(X_train, y_train, **param_dict)
            val_loss = evaluate_cnn(cnn_model, X_val, y_val)
        elif model_type == "SVM":
            clf = model(**param_dict)
            clf.fit(X_train, y_train)
            val_loss = 1 - clf.score(X_val, y_val)
        else:
            raise ValueError("Invalid model_type. Supported types are 'SVM' and 'CNN'.")

        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

        losses.append(val_loss)

        if idx == 0:
            first_params = param_dict

        if val_loss < best_loss:
            best_loss = val_loss
            best_params = param_dict
            best_losses.append(best_loss)

    return first_params, best_params, losses, best_losses, execution_times

def custom_grid_search_cifar(train_loader, val_loader, hyperparameters, model):
    losses = []
    best_losses = []
    best_loss = float('inf')
    best_params = None
    first_params = None
    execution_times = []

    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(*hyperparameters.values()))

    for idx, params in enumerate(param_combinations):
        start_time = time.time()

        param_dict = dict(zip(hyperparameters.keys(), params))

        cnn_model = model(train_loader, **param_dict)
        val_loss = evaluate_NET(cnn_model, val_loader)

        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

        losses.append(val_loss)

        if idx == 0:
            first_params = param_dict

        if val_loss < best_loss:
            best_loss = val_loss
            best_params = param_dict
            best_losses.append(best_loss)

    return first_params, best_params, losses, best_losses, execution_times


def plot_optimization_results(grid_search_best_losses, losses_bv, losses_bv_MCMC,
                              execution_times_gs, execution_times_bo, execution_times_MCMC):
    
    # Calculate the minimum value found during grid search
    grid_search_min = min(grid_search_best_losses)

    # Plot optimization comparison
    plt.figure(figsize=(10, 6))
    plt.plot(losses_bv, label=f'Bayesian Optimization (y_best={np.round(losses_bv[-1], 3)})')
    plt.plot(losses_bv_MCMC, label=f'Bayesian Optimization MCMC (y_best={np.round(losses_bv_MCMC[-1], 3)})')
    plt.axhline(y=grid_search_min, color='r', linestyle='--', label=f'Minimum from Grid Search (value={np.round(grid_search_min, 3)})')
    plt.xlabel('Number of iterations', fontsize=14)
    plt.ylabel('f(x)', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Optimization Comparison', fontsize=16)
    plt.show()

    # Summing up the execution times for each method
    total_time_gs = sum(execution_times_gs)
    # total_time_r = sum(execution_times_r)
    total_time_bo = sum(execution_times_bo)
    total_time_MCMC = sum(execution_times_MCMC)

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(['Custom Grid Search', 'Bayesian Optimization', 'Bayesian Optimization MCMC'], [total_time_gs, total_time_bo, total_time_MCMC], color=['blue', 'orange', 'green'])
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Total Execution Time (seconds)', fontsize=14)
    plt.title('Total Execution Time for Each Method', fontsize=16)
    plt.show()
    
def plot_results_SVM(grid_search_best_losses, losses_bv, losses_bv_MCMC, execution_times_bo, execution_times_MCMC):
    
    fig, ax = plt.subplots(figsize=(12, 4), dpi=500)
    
    # Calculate the minimum value found during grid search
    grid_search_min = min(grid_search_best_losses)

    # Plot the mean values and scatter points
    ax.plot(np.mean(losses_bv, axis=0), label=f'y_GP_best={np.round(np.mean(losses_bv[:,-1]),3)}', color='orange')
    ax.plot(np.mean(losses_bv_MCMC, axis=0), label=f'y_GP_MCMC_best={np.round(np.mean(losses_bv_MCMC[:,-1]),3)}', color='b')

    # Plot axial line at grid_search_min
    ax.axhline(y=grid_search_min, color='g', linestyle='--', label=f'Grid Search Minimum={np.round(grid_search_min, 3)}')

    # Set labels and legend
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('f(x)')
    ax.legend()

    # Create zoom inset
    axins = ax.inset_axes([0.33, 0.42, 0.6, 0.35])

    # Adjust data ranges for zoom-in
    zoom_in_x = np.arange(1, 13)
    zoom_in_y_bv = np.mean(losses_bv, axis=0)[1:13]  # Selecting data from index 1 to 12
    zoom_in_y_bv_MCMC = np.mean(losses_bv_MCMC, axis=0)[1:13]  # Selecting data from index 1 to 12

    # Plot zoomed-in data
    axins.plot(zoom_in_x, zoom_in_y_bv, color='orange')
    axins.plot(zoom_in_x, zoom_in_y_bv_MCMC, color='b')

    # Set zoom-in limits
    axins.set_xlim(1, 12)
    axins.set_ylim(-0.005, 0.005)  # Set y-axis range from -0.005 to 0.005

#     # Add scatter plot to the inset
#     for i in range(len(losses_bv)):
#         axins.scatter(np.arange(1, 13), losses_bv[i][:12], color='orange', alpha=0.1)
#         axins.scatter(np.arange(1, 13), losses_bv_MCMC[i][:12], color='b', alpha=0.1)

    # Add border around the zoom inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # Sum the elements in each list
    sum_execution_times_bo = np.sum(execution_times_bo, axis=1)
    sum_execution_times_MCMC = np.sum(execution_times_MCMC, axis=1)

    # Take the mean of the sums
    mean_sum_execution_times_bo = np.mean(sum_execution_times_bo)
    mean_sum_execution_times_MCMC = np.mean(sum_execution_times_MCMC)

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(['Custom Grid Search', 'Bayesian Optimization', 'Bayesian Optimization MCMC'], [sum(execution_times_gs), mean_sum_execution_times_bo, mean_sum_execution_times_MCMC], color=['blue', 'orange', 'green'])
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Total Execution Time (seconds)', fontsize=14)
    plt.title('Total Execution Time for Each Method', fontsize=16)
    plt.show()
    
def plot_results(grid_search_best_losses, grid_search_best_losses_rgs, losses_bv_1, losses_bv_2, losses_bv_3, execution_times_gs, execution_times_rgs, execution_times_bo_1, execution_times_bo_2, execution_times_bo_3, s_or_t):
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=500)

    # Calculate the minimum value found during grid search
    grid_search_min = min(grid_search_best_losses)
    
    n_samples = len(losses_bv_1[0])
    

    # Plot the mean values and scatter points
    ax.plot(grid_search_best_losses_rgs, label=f'y_random_Grid={min(grid_search_best_losses_rgs)}', color='Green')

    ax.plot(np.mean(losses_bv_1, axis=0), label=f'y_GP_{s_or_t}_one={np.round(np.mean(losses_bv_1[:,-1]),3)}', color='orange')
    ax.plot(np.mean(losses_bv_2, axis=0), label=f'y_GP_{s_or_t}_zero_one={np.round(np.mean(losses_bv_2[:,-1]),3)}', color='b')
    ax.plot(np.mean(losses_bv_3, axis=0), label=f'y_GP_{s_or_t}_ten={np.round(np.mean(losses_bv_3[:,-1]),3)}', color='r')
    ax.scatter(np.arange(1, n_samples), np.mean(losses_bv_1, axis=0)[1:n_samples], color='orange', alpha=0.5)
    ax.scatter(np.arange(1, n_samples), np.mean(losses_bv_2, axis=0)[1:n_samples], color='b', alpha=0.5)
    ax.scatter(np.arange(1, n_samples), np.mean(losses_bv_3, axis=0)[1:n_samples], color='r', alpha=0.5)
    
    # Calculate standard deviation of losses
    std_losses_bv_1 = np.std(losses_bv_1, axis=0)
    std_losses_bv_2 = np.std(losses_bv_2, axis=0)
    std_losses_bv_3 = np.std(losses_bv_3, axis=0)

    
    # Plot error bars for scatter points
    ax.errorbar(np.arange(1, n_samples), np.mean(losses_bv_1, axis=0)[1:n_samples], yerr=std_losses_bv_1[1:n_samples], fmt='o', color='orange', alpha=0.5)
    ax.errorbar(np.arange(1, n_samples), np.mean(losses_bv_2, axis=0)[1:n_samples], yerr=std_losses_bv_2[1:n_samples], fmt='o', color='b', alpha=0.5)
    ax.errorbar(np.arange(1, n_samples), np.mean(losses_bv_3, axis=0)[1:n_samples], yerr=std_losses_bv_3[1:n_samples], fmt='o', color='r', alpha=0.5)
    
    # Plot axial line at grid_search_min
    ax.axhline(y=grid_search_min, color='g', linestyle='--', label=f'Grid Search Minimum={np.round(grid_search_min, 3)}')

    # Set labels and legend
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('1 - accuracy')
    ax.legend()

    # Create zoom inset
    axins = ax.inset_axes([0.33, 0.42, 0.6, 0.35])

    # Adjust data ranges for zoom-in
    zoom_in_x = np.arange(1, n_samples)
    zoom_in_y_bv_1 = np.mean(losses_bv_1, axis=0)[1:n_samples]  # Selecting data from index 1 to 12
    zoom_in_y_bv_2 = np.mean(losses_bv_2, axis=0)[1:n_samples]  # Selecting data from index 1 to 12
    zoom_in_y_bv_3 = np.mean(losses_bv_3, axis=0)[1:n_samples]  # Selecting data from index 1 to 12
    
    # Plot zoomed-in data
    axins.plot(zoom_in_x, zoom_in_y_bv_1, color='orange')
    axins.plot(zoom_in_x, zoom_in_y_bv_2, color='b')
    axins.plot(zoom_in_x, zoom_in_y_bv_3, color='r')
    axins.axhline(y=grid_search_min, color='g', linestyle='--', label=f'Grid Search Minimum={np.round(grid_search_min, 3)}')
    
    # Set zoom-in limits
    axins.set_xlim(1, n_samples)
    axins.set_ylim(0.01, 0.07)  # Set y-axis range from -0.005 to 0.005

    # Add scatter plot to the inset
    axins.scatter(np.arange(1, n_samples), np.mean(losses_bv_1, axis=0)[1:n_samples], color='orange', alpha=0.5)
    axins.scatter(np.arange(1, n_samples), np.mean(losses_bv_2, axis=0)[1:n_samples], color='b', alpha=0.5)
    axins.scatter(np.arange(1, n_samples), np.mean(losses_bv_3, axis=0)[1:n_samples], color='r', alpha=0.5)
    
    # Plot error bars in zoomed-in data
    axins.errorbar(zoom_in_x, zoom_in_y_bv_1, yerr=std_losses_bv_1[1:n_samples], fmt='o', color='orange', alpha=1)
    axins.errorbar(zoom_in_x, zoom_in_y_bv_2, yerr=std_losses_bv_2[1:n_samples], fmt='o', color='b', alpha=0.4)
    axins.errorbar(zoom_in_x, zoom_in_y_bv_3, yerr=std_losses_bv_3[1:n_samples], fmt='o', color='r', alpha=0.6)
    

    # Add border around the zoom inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # Sum the elements in each list
    sum_execution_times_bo_1 = np.sum(execution_times_bo_1, axis=1)
    sum_execution_times_bo_2 = np.sum(execution_times_bo_2, axis=1)
    sum_execution_times_bo_3 = np.sum(execution_times_bo_3, axis=1)

    # Take the mean of the sums
    mean_sum_execution_times_bo_1 = np.mean(sum_execution_times_bo_1)
    mean_sum_execution_times_bo_2 = np.mean(sum_execution_times_bo_2)
    mean_sum_execution_times_bo_3 = np.mean(sum_execution_times_bo_3)

    # Calculate standard deviation of execution times
    std_execution_times_bo_1 = np.std(sum_execution_times_bo_1)
    std_execution_times_bo_2 = np.std(sum_execution_times_bo_2)
    std_execution_times_bo_3 = np.std(sum_execution_times_bo_3)

    # Plotting the bar chart with error bars
    methods = ['Grid Search', 'Random Search', f'Bayes Opt {s_or_t} = 1', f'Bayes Opt {s_or_t} = 0.1', f'Bayes Opt {s_or_t} = 10']
    means = [sum(execution_times_gs), sum(execution_times_rgs), mean_sum_execution_times_bo_1, mean_sum_execution_times_bo_2, mean_sum_execution_times_bo_3]
    std_devs = [0, 0, std_execution_times_bo_1, std_execution_times_bo_2, std_execution_times_bo_3]

    plt.figure(figsize=(10, 6))
    plt.bar(methods, means, yerr=std_devs, capsize=5, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Total Execution Time (seconds)', fontsize=14)
    plt.title('Total Execution Time for Each Method', fontsize=16)
    plt.show()



####################################### METROPOLIS #######################################

def likelihood(theta, x, y, kernel, loop_number):
    """ 
    Likelihood function for the GP hyperparameters (function of theta, given data (x, y))

    Parameters
    ----------
    theta : array_like
        Hyperparameters
    y : array_like
        Observations
    x : array_like
        Sample points (raws hyperparameters, columns observations)
    kernel : function
        Kernel function
    loop_number : int
        Number of the loop for which the likelihood is computed

    Returns
    -------
    float
        Likelihood
    """
    # compute covariance matrix
    l = loop_number
    K = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            # no negative s
            K[i, j] = kernel(x[:, i], x[:, j], l=theta[1:], s=np.abs(theta[0])+1e-6)
    like = multivariate_normal(mean=np.zeros(len(y)), cov=K)
    return like.pdf(y)


def Metropolis(n_sample, cov, x0, burn_in, thinning, target, **kwargs):
    """
    Metropolis algorithm with Gaussian proposal distribution

    Parameters
    ----------
    target : function
        Target distribution
    x0 : array_like
        Initial point
    n_sample : int
        Number of samples
    cov : array_like
        Covariance matrix of the proposal distribution
    burn_in : int
        Number of burn-in samples
    thinning : int
        Thinning factor
    **kwargs :
        Additional parameters for the target distribution

    Returns
    -------
    array_like
        Samples from the target distribution
    """
    # rows: dimensions, columns: samples (take into account thinning)
    sample = np.zeros((len(x0), n_sample))
    x_current = x0
    for i in range(burn_in):
        # propose new sample
        proposal = multivariate_normal(mean=x_current, cov=cov)
        x_prop = proposal.rvs()
        # Acceptance probability
        proposal_sym = multivariate_normal(mean=x_prop, cov=cov)
        # Hastings
        A = min(1, target(x_prop, **kwargs)*proposal_sym.pdf(x_current) / target(x_current, **kwargs)*proposal.pdf(x_prop))
        #A = min(1, target(x_prop, **kwargs)/ target(x_current, **kwargs))
        if np.random.rand() < A:
            x_current = x_prop
    
    for i in range(n_sample*thinning):
        # propose new sample
        proposal = multivariate_normal(mean=x_current, cov=cov)
        x_prop = proposal.rvs()
        proposal_sym = multivariate_normal(mean=x_prop, cov=cov)
        # Acceptance probability
        A = min(1, target(x_prop, **kwargs)*proposal_sym.pdf(x_current) / target(x_current, **kwargs)*proposal.pdf(x_prop))
        if np.random.rand() < A:
            x_current = x_prop
        if i % thinning == 0:
            sample[:, i//thinning] = x_current
    return sample

##################################### TRAIN ML MODELS ########################################

################ SVM ################

def train_svm(X_train, y_train, C, gamma):
    svm = SVC(C=C, gamma=gamma, random_state=42)
    svm.fit(X_train, y_train)
    return svm

def evaluate_svm(svm, X_val, y_val):
    return 1.0 - svm.score(X_val, y_val)

################ CNN ################

def train_cnn(X_train, y_train, N_epochs, regularization_l2, dropout_rate, learning_rate, num_layers):

    # Define the CNN architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    
    for _ in range(int(num_layers) - 1):
        model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(10, activation='softmax'))  # Output layer with 10 units for 10 classes

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=int(N_epochs), validation_split=0.2, batch_size=128, verbose=1)

    return model

def evaluate_cnn(model, X_val, y_val):
    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
    return 1.0 - accuracy

################ CNN REDUCED SPACE ################

def train_cnn_reduced(X_train, y_train, dropout_rate, learning_rate):

    # Define the CNN architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    
    for _ in range(2 - 1):
        model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(10, activation='softmax'))  # Output layer with 10 units for 10 classes

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=int(3), validation_split=0.2, batch_size=128, verbose=1)

    return model

def evaluate_cnn_reduced(model, X_val, y_val):
    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
    return 1.0 - accuracy
    
################ CIFAR ######################

def train_NET_complex(trainloader, epochs, learning_rate, momentum_m, verbose=True):
    # Define the neural network model
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum_m)
    
    for epoch in range(int(epochs)):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if verbose and i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        if verbose:
            print(f'Epoch [{epoch + 1}/{epochs}] Loss: {running_loss / len(trainloader):.3f}')

    print('Finished Training')
    return net 


def evaluate_NET_complex(net, validloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validloader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return 1.0 - accuracy

def train_NET(trainloader,
              activation_function,
              learning_rate,
              epochs = 5,
              n_neurons=[[3072,512],[512,258],[258,10]],
              criterion = nn.CrossEntropyLoss(),
              verbose=True):
    
    # Define the neural network model
    class MLP(nn.Module):
        def __init__(self, n_neurons, activation_function):
            
            # Initialize the MLP model
            super().__init__()
                
            # List to store layers
            layers = []

            # Input layer
            layers.append(nn.Linear(n_neurons[0][0], n_neurons[0][1]))
            layers.append(activation_function)
            
            # Hidden layers
            for i in range(1, len(n_neurons)):
                layers.append(nn.Linear(n_neurons[i][0], n_neurons[i][1]))
                layers.append(activation_function)

            # Create the Sequential container
            self.linear_layers = nn.Sequential(*layers)

        def forward(self, x):
            batch_size = x.shape[0]
            # Flatten x
            x = x.view(batch_size, -1)
            # Apply linear 
            out = self.linear_layers(x)

            return out

    net = MLP(n_neurons, activation_function)
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    for epoch in range(int(epochs)):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if verbose and i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        if verbose:
            print(f'Epoch [{epoch + 1}/{epochs}] Loss: {running_loss / len(trainloader):.3f}')

    print('Finished Training')
    return net 

def evaluate_NET(net, validloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validloader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return 1.0 - accuracy

##################################### BAYESIAN OPTIMIZERS #####################################

def bayes_optimization(hyp_space, n_sample, n_wu, acq_function, kernel, l, s, eval_func, train_func, start_hyper_comb, X_train, X_val, y_train, y_val):
    """
    Bayesian optimization for hyperparameter tuning

    Parameters
    ----------
    hyp_space : dict
        hyperparameter dictionary 
    n_sample : int
        number of samples to draw
    n_wu : int
        number of warmup samples
    acq_function : function
        acquisition function
    kernel : function
        kernel function
    f : function
        function to optimize
    
    Returns
    -------
    hyp : array_like
        hyperparameter values
    """
    
    loss_best_vec = np.array([])
    best_hyp = np.array([])
    loss_sampled = np.array([])
    hyp_name = list(hyp_space.keys())
    loss_best = np.inf
    acq_funcs = np.array([])
    
    # all possible combinations of hyperparameters (each row is a combination)
    hyp_space_combinations = np.array(generate_combinations(hyp_space))

    # warm up

    # random sampling from hyperparameter space
    # matrix of sampled values (rows: hyperparameters, columns: samples)
    x_sampled = np.zeros((len(hyp_name), n_sample+n_wu))
    
    execution_times = []

    for j in range(n_wu):
        start_time = time.time()
            
        # sample hyperparameters from the hyperparameter space
        np.random.shuffle(hyp_space_combinations)
        shape = hyp_space_combinations.shape
        if shape[0] == 0:
            break
        else:
            if j == 0:
                x_sampled[:, j] = start_hyper_comb
                hyp_space_combinations = np.delete(hyp_space_combinations, np.where((hyp_space_combinations == x_sampled[:, j]).all(axis=1))[0][0], axis=0)
            else:
                #sample a hyperparameter combination
                x_sampled[:, j] = hyp_space_combinations[0]
                # delete the sampled value from the hyperparameter combinations 
                hyp_space_combinations = np.delete(hyp_space_combinations, 0, axis=0)

        # evaluate loss for training with those hyperparameters
        trained_model = train_func(X_train, y_train, *x_sampled[:, j])
        loss = eval_func(trained_model, X_val, y_val)
        loss_sampled = np.append(loss_sampled, loss)
        # update loss_best
        if loss < loss_best:
            loss_best = loss
            best_hyp = x_sampled[:, j]
        # append loss_best to loss_best_vec
        loss_best_vec = np.append(loss_best_vec, loss_best)
        end_time = time.time()
        execution_times.append(end_time - start_time)

    # sampling
        
    for j in range(n_wu, n_sample+n_wu):
        start_time = time.time()
        acq_max = -np.inf

        # update kernel matrix
        if j == n_wu:
            # Initialize covariance matrix between two sampled points and
            # place values in the cov matrix using the kernel
            K = np.zeros((n_wu, n_wu))
            for i in range(n_wu):
                for k in range(i):
                    # The kernel makes an assumption on the covariance
                    # Update every new sample
                    K[i, k] = kernel(x_sampled[:, i], x_sampled[:, k], l, s)
                    K[k, i] = K[i, k]  
            
        else:
            # add new row and column to kernel matrix
            K = np.pad(K, pad_width=(0,1), mode='constant', constant_values=0)
            # update new row and column
            for i in range(j-1):
                K[i, j-1] = kernel(x_sampled[:, i], x_sampled[:, j-1], l, s)
                K[j-1, i] = K[i, j-1]
                

        # compute the acquisition function for each possible value of the hyperparameters vector, keeping only the maximum while updating the hyperparameters
        # and sampling the new hyperparameters vector
        
        K_inv = np.linalg.inv(K) 
        for xs in hyp_space_combinations:
            # compute the kernel vector for each combination of hyperparameter
            # with the newly chosen HP
            k = np.array([kernel(x_sampled[:, r], xs, l, s) for r in range(j)])
            acq = acq_function(loss_best, mu(k, K_inv, loss_sampled), sigma(kernel(xs, xs, l, s), k, K_inv))
#             acq_funcs = np.append(acq_funcs, acq)
            
            # check if the new value is the maximum
            if acq > acq_max:
                acq_max = acq
                tmp_x_sampled = xs
        
        # update x_sampled
        x_sampled[:, j] = tmp_x_sampled
        # delete the sampled value from the hyperparameter combinations
        if x_sampled[:, j] in hyp_space_combinations:
            hyp_space_combinations = np.delete(hyp_space_combinations, np.where((hyp_space_combinations == x_sampled[:, j]).all(axis=1))[0][0], axis=0)
        else:
            break
        # evaluate function
        trained_model = train_func(X_train, y_train, *x_sampled[:, j])
        loss = eval_func(trained_model, X_val, y_val)
        
        loss_sampled = np.append(loss_sampled, loss)
        # update y_best
        if loss < loss_best:
            loss_best = loss
            best_hyp = x_sampled[:, j]
        # append y_best to y_best_vec
        loss_best_vec = np.append(loss_best_vec, loss_best)
        end_time = time.time()
        execution_times.append(end_time - start_time)

    return best_hyp, loss_best_vec, x_sampled, loss_sampled, execution_times

def bayes_optimization_MCMC(hyp_space, n_sample, n_wu, acq_function, kernel, eval_func, train_func, start_hyper_comb, X_train, X_val, y_train, y_val):
    """
    Bayesian optimization for hyperparameter tuning

    Parameters
    ----------
    hyp_space : dict
        hyperparameter dictionary 
    n_sample : int
        number of samples to draw
    n_wu : int
        number of warmup samples
    acq_function : function
        acquisition function
    kernel : function
        kernel function
    eval_func : function
        function to evaluate the model
    train_func : function
        function to train the model
    X_train : array_like
        training data
    X_val : array_like
        validation data
    y_train : array_like
        training labels
    y_val : array_like
        validation labels
    
    Returns
    -------
    hyp : array_like
        hyperparameter values
    """
    
    y_best_vec = np.array([])
    best_hyp = np.array([])
    y_sampled = np.array([])
    hyp_name = list(hyp_space.keys())
    y_best = np.inf

    # all possible combinations of hyperparameters (each row is a combination)
    hyp_space_combinations = np.array(generate_combinations(hyp_space))

    # warm up

    # random sampling from hyperparameter space
    # matrix of sampled values (rows: hyperparameters, columns: samples)
    n_hyper = len(hyp_name) # number of hyperparameters

    x_sampled = np.zeros((n_hyper, n_sample+n_wu))
    
    execution_times = []
    
    for j in range(n_wu):
        start_time = time.time()
        # sample hyperparameters from the hyperparameter space
        np.random.shuffle(hyp_space_combinations)
        if j == 0:
            x_sampled[:, j] = start_hyper_comb
            hyp_space_combinations = np.delete(hyp_space_combinations, np.where((hyp_space_combinations == x_sampled[:, j]).all(axis=1))[0][0], axis=0)
        else:
            #sample a hyperparameter combination
            x_sampled[:, j] = hyp_space_combinations[0]
            # delete the sampled value from the hyperparameter combinations 
            hyp_space_combinations = np.delete(hyp_space_combinations, 0, axis=0)
            
        # evaluate loss for training with those hyperparameters
        trained_model = train_func(X_train, y_train, *x_sampled[:, j])
        y = eval_func(trained_model, X_val, y_val)
        y_sampled = np.append(y_sampled, y)
        # update y_best
        if y < y_best:
            y_best = y
            best_hyp = x_sampled[:, j]
        # append y_best to y_best_vec
        y_best_vec = np.append(y_best_vec, y_best)
        end_time = time.time()
        execution_times.append(end_time - start_time)
        
    # sampling
        
    for j in range(n_wu, n_sample+n_wu):
        start_time = time.time()
        integrated_acq_max = -np.inf

        # sampling from the likelihood the hyper-hyperparameters
        M = 100
        theta_sample = Metropolis(n_sample=M, cov=np.eye(n_hyper+1), x0=np.ones(n_hyper+1), burn_in=100,
                                thinning=10, target=likelihood, x=x_sampled, y=y_sampled, kernel=kernel_M52, loop_number=j)

        # compute the integrated acquisition function for each possible value of the hyperparameters vector, keeping only the maximum while updating the 
        # hyperparameters and sampling the new hyperparameters vector
        
        for idx, xs in enumerate(hyp_space_combinations):
            acq = np.array([])
            for theta in theta_sample.T:
                # compute k,K for each theta
                k = np.array([kernel(x_sampled[:, r], xs, s=theta[0], l=theta[1:]) for r in range(j)])
                K = np.zeros((j, j))
                for i in range(j):
                    for azz in range(j):
                        K[i, azz] = kernel(x_sampled[:, i], x_sampled[:, azz], s=theta[0], l=theta[1:])
                
                # integrate the acquisition function
                K_inv = np.linalg.inv(K)
                acq = np.append(acq, acq_function(y_best, mu(k, K_inv, y_sampled), sigma(kernel(xs, xs, s=theta[0], l=theta[1:]), k, K_inv)))
            integrated_acq = np.mean(acq)
                # check if the new value is the maximum
            if integrated_acq > integrated_acq_max:
                integrated_acq_max = integrated_acq
                tmp_x_sampled = xs
                idx_sample = idx
        # update x_sampled
        x_sampled[:, j] = tmp_x_sampled
        # delete the sampled value from the hyperparameter combinations  
        #hyp_space_combinations = np.delete(hyp_space_combinations, np.where((hyp_space_combinations == tmp_x_sampled).all(axis=1))[0][0], axis=0) 
        hyp_space_combinations = np.delete(hyp_space_combinations, idx_sample, axis=0)  
        # evaluate loss for training with those hyperparameters
        trained_model = train_func(X_train, y_train, *x_sampled[:, j])
        y = eval_func(trained_model, X_val, y_val)
        y_sampled = np.append(y_sampled, y)
        # update y_best
        if y < y_best:
            y_best = y
            best_hyp = x_sampled[:, j]
        # append y_best to y_best_vec
        y_best_vec = np.append(y_best_vec, y_best)
        end_time = time.time()
        execution_times.append(end_time - start_time)
    return best_hyp, y_best_vec, x_sampled, y_sampled, execution_times

def bayes_optimization_cifar(hyp_space, n_sample, n_wu, acq_function, kernel, l, s, eval_func, train_func, start_hyper_comb, trainloader, validloader):
    loss_best_vec = np.array([])
    best_hyp = {}
    loss_sampled = np.array([])
    hyp_name = list(hyp_space.keys())
    loss_best = np.inf
    acq_funcs = np.array([])

    hyp_space_combinations = np.array(generate_combinations(hyp_space))

    x_sampled = np.zeros((len(hyp_name), n_sample + n_wu))
    execution_times = []

    for j in range(n_wu):
        start_time = time.time()

        np.random.shuffle(hyp_space_combinations)
        shape = hyp_space_combinations.shape
        if j == 0:
            x_sampled[:, j] = start_hyper_comb
            hyp_space_combinations = np.delete(hyp_space_combinations, 0, axis=0)
        else:
            x_sampled[:, j] = hyp_space_combinations[0]
            hyp_space_combinations = np.delete(
            hyp_space_combinations,
            np.where((hyp_space_combinations == x_sampled[:, j]).all(axis=1))[0][0],
            axis=0,
            )

        start_time = time.time()
        trained_model = train_func(
            trainloader,
            *x_sampled[:, j]
        )
        loss = evaluate_NET(trained_model, validloader)

        loss_sampled = np.append(loss_sampled, loss)
        if loss < loss_best:
            loss_best = loss
            best_hyp = {
                hyp_name[i]: x_sampled[i, j] for i in range(len(hyp_name))
            }
        loss_best_vec = np.append(loss_best_vec, loss_best)
        end_time = time.time()
        execution_times.append(end_time - start_time)

    for j in range(n_wu, n_sample + n_wu):
        start_time = time.time()
        acq_max = -np.inf

        if j == n_wu:
            K = np.zeros((n_wu, n_wu))
            for i in range(n_wu):
                for k in range(i):
                    K[i, k] = kernel(x_sampled[:, i], x_sampled[:, k], l, s)
                    K[k, i] = K[i, k]

        else:
            K = np.pad(K, pad_width=(0, 1), mode="constant", constant_values=0)
            for i in range(j - 1):
                K[i, j - 1] = kernel(x_sampled[:, i], x_sampled[:, j - 1], l, s)
                K[j - 1, i] = K[i, j - 1]

        K_inv = np.linalg.inv(K)
        for xs in hyp_space_combinations:
            k = np.array(
                [kernel(x_sampled[:, r], xs, l, s) for r in range(j)]
            )
            acq = acq_function(
                loss_best, mu(k, K_inv, loss_sampled), sigma(kernel(xs, xs, l, s), k, K_inv)
            )

            if acq > acq_max:
                acq_max = acq
                tmp_x_sampled = xs

        x_sampled[:, j] = tmp_x_sampled
        if x_sampled[:, j] in hyp_space_combinations:
            hyp_space_combinations = np.delete(
                hyp_space_combinations,
                np.where((hyp_space_combinations == x_sampled[:, j]).all(axis=1))[0][0],
                axis=0,
            )
        else:
            break

        trained_model = train_func(trainloader, *x_sampled[:, j])
        loss = evaluate_NET(trained_model, validloader)
        end_time = time.time()
        execution_times.append(end_time - start_time)

        loss_sampled = np.append(loss_sampled, loss)
        if loss < loss_best:
            loss_best = loss
            best_hyp = {hyp_name[i]: x_sampled[i, j] for i in range(len(hyp_name))}
        loss_best_vec = np.append(loss_best_vec, loss_best)
        end_time = time.time()
        execution_times.append(end_time - start_time)
    
    return best_hyp, loss_best_vec, x_sampled, loss_sampled, execution_times

def bayes_optimization_MCMC_cifar(hyp_space, n_sample, n_wu, acq_function, kernel, eval_func, train_func, start_hyper_comb, train_loader, val_loader):
    """
    Bayesian optimization for hyperparameter tuning

    Parameters
    ----------
    hyp_space : dict
        hyperparameter dictionary 
    n_sample : int
        number of samples to draw
    n_wu : int
        number of warmup samples
    acq_function : function
        acquisition function
    kernel : function
        kernel function
    eval_func : function
        function to evaluate the model
    train_func : function
        function to train the model
    X_train : array_like
        training data
    X_val : array_like
        validation data
    y_train : array_like
        training labels
    y_val : array_like
        validation labels
    
    Returns
    -------
    hyp : array_like
        hyperparameter values
    """
    
    y_best_vec = np.array([])
    best_hyp = np.array([])
    y_sampled = np.array([])
    hyp_name = list(hyp_space.keys())
    y_best = np.inf

    # all possible combinations of hyperparameters (each row is a combination)
    hyp_space_combinations = np.array(generate_combinations(hyp_space))

    # warm up

    # random sampling from hyperparameter space
    # matrix of sampled values (rows: hyperparameters, columns: samples)
    n_hyper = len(hyp_name) # number of hyperparameters

    x_sampled = np.zeros((n_hyper, n_sample+n_wu))
    
    execution_times = []
    
    for j in range(n_wu):
        start_time = time.time()
        # sample hyperparameters from the hyperparameter space
        np.random.shuffle(hyp_space_combinations)
        if j == 0:
            x_sampled[:, j] = start_hyper_comb
            hyp_space_combinations = np.delete(hyp_space_combinations, np.where((hyp_space_combinations == x_sampled[:, j]).all(axis=1))[0][0], axis=0)
        else:
            #sample a hyperparameter combination
            x_sampled[:, j] = hyp_space_combinations[0]
            # delete the sampled value from the hyperparameter combinations 
            hyp_space_combinations = np.delete(hyp_space_combinations, 0, axis=0)
            
        # evaluate loss for training with those hyperparameters
        trained_model = train_func(train_loader, *x_sampled[:, j])
        y = eval_func(trained_model, val_loader)
        y_sampled = np.append(y_sampled, y)
        # update y_best
        if y < y_best:
            y_best = y
            best_hyp = x_sampled[:, j]
        # append y_best to y_best_vec
        y_best_vec = np.append(y_best_vec, y_best)
        end_time = time.time()
        execution_times.append(end_time - start_time)
        
    # sampling
        
    for j in range(n_wu, n_sample+n_wu):
        start_time = time.time()
        integrated_acq_max = -np.inf

        # sampling from the likelihood the hyper-hyperparameters
        M = 100
        theta_sample = Metropolis(n_sample=M, cov=np.eye(n_hyper+1), x0=np.ones(n_hyper+1), burn_in=100,
                                thinning=10, target=likelihood, x=x_sampled, y=y_sampled, kernel=kernel_M52, loop_number=j)

        # compute the integrated acquisition function for each possible value of the hyperparameters vector, keeping only the maximum while updating the 
        # hyperparameters and sampling the new hyperparameters vector
        
        for idx, xs in enumerate(hyp_space_combinations):
            acq = np.array([])
            for theta in theta_sample.T:
                # compute k,K for each theta
                k = np.array([kernel(x_sampled[:, r], xs, s=theta[0], l=theta[1:]) for r in range(j)])
                K = np.zeros((j, j))
                for i in range(j):
                    for azz in range(j):
                        K[i, azz] = kernel(x_sampled[:, i], x_sampled[:, azz], s=theta[0], l=theta[1:])
                
                # integrate the acquisition function
                K_inv = np.linalg.inv(K)
                acq = np.append(acq, acq_function(y_best, mu(k, K_inv, y_sampled), sigma(kernel(xs, xs, s=theta[0], l=theta[1:]), k, K_inv)))
            integrated_acq = np.mean(acq)
                # check if the new value is the maximum
            if integrated_acq > integrated_acq_max:
                integrated_acq_max = integrated_acq
                tmp_x_sampled = xs
                idx_sample = idx
        # update x_sampled
        x_sampled[:, j] = tmp_x_sampled
        # delete the sampled value from the hyperparameter combinations  
        #hyp_space_combinations = np.delete(hyp_space_combinations, np.where((hyp_space_combinations == tmp_x_sampled).all(axis=1))[0][0], axis=0) 
        hyp_space_combinations = np.delete(hyp_space_combinations, idx_sample, axis=0)  
        # evaluate loss for training with those hyperparameters
        trained_model = train_func(train_loader, *x_sampled[:, j])
        y = eval_func(trained_model, val_loader)
        y_sampled = np.append(y_sampled, y)
        # update y_best
        if y < y_best:
            y_best = y
            best_hyp = x_sampled[:, j]
        # append y_best to y_best_vec
        y_best_vec = np.append(y_best_vec, y_best)
        end_time = time.time()
        execution_times.append(end_time - start_time)
    return best_hyp, y_best_vec, x_sampled, y_sampled, execution_times

