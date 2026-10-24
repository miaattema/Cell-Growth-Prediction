# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================
from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


def calculate_average(values):
    return sum(values) / len(values)

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_4factor.csv"))

features = df.columns[1:]

targets = df.columns[0]

x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(include_bias=False),
    LinearRegression()
)

param_grid = dict(polynomialfeatures__degree=[i for i in range(1, 11)])

model_gscv_poly = GridSearchCV(model, param_grid=param_grid,
                          cv=4, scoring="neg_root_mean_squared_error", return_train_score=True)

model_gscv_poly.fit(x_train, y_train)
best_poly_model = model_gscv_poly.best_estimator_

df_poly = pd.DataFrame(model_gscv_poly.cv_results_)
df_poly = df_poly.sort_values(by=['rank_test_score'], ascending=True)

columns_view = ["rank_test_score", "param_polynomialfeatures__degree",
                "mean_test_score", "std_test_score",
                "mean_train_score", "std_train_score"]

ranking_poly = df_poly.loc[:, columns_view].copy()

ranking_poly.to_csv(
    os.path.join("tables", "table_01_polynomial_ranking.csv"),
    index=False
)

best_poly_model = model_gscv_poly.best_estimator_

powers = best_poly_model["polynomialfeatures"].powers_
coefs = best_poly_model["linearregression"].coef_
intercept = best_poly_model["linearregression"].intercept_


#print("Polynomial model:\n")
#print(f"{targets} = ")
#print(f"{intercept:8.3f}")
for i in range(powers.shape[0]):
    partial_terms = []
    for j in range(powers.shape[1]):

        feature_unit_splitter = " ["
        if feature_unit_splitter in features[j]:
            feature_name = features[j].split(feature_unit_splitter)[0]
        else:
            feature_name = features[j]

        if powers[i, j] == 0:
            pass
        elif powers[i, j] == 1:
            partial_terms.append(f"[{feature_name}]")
        else:
            partial_terms.append(f"[{feature_name}]^{powers[i, j]}")

    coef = coefs[i]
    term = " * ".join(partial_terms)

    #print(f"{coef:+8.3f} * {term}")

# ========================================
# Plot Validation Curve for Polynomials
# ========================================

FIG_WIDTH = 14.39 / 2.54
FIG_HEIGHT = 12.09 / 2.54

fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12

# Use the polynomial cross-validation results, not the original dataset
df_plot = df_poly.sort_values(
    by="param_polynomialfeatures__degree"
)



degrees = df_plot["param_polynomialfeatures__degree"].to_numpy()
rmse_train = -df_plot["mean_train_score"].to_numpy()
rmse_valid = -df_plot["mean_test_score"].to_numpy()

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=200,
    layout="constrained"
)

ax.plot(
    degrees, rmse_train,
    label="Training RMSE",
    color="tab:blue",
    linestyle="-",
    marker="o"
)

ax.plot(
    degrees, rmse_valid,
    label="Validation RMSE",
    color="tab:orange",
    linestyle="--",
    marker="o"
)

ax.set_xlabel("Polynomial Degree", fontsize=fontsize_xy_label)
ax.set_ylabel("Specific Growth Rate RMSE [d$^{-1}$]", fontsize=fontsize_xy_label)
ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
ax.legend(fontsize=fontsize_legend)
ax.set_ylim(bottom=0)
ax.set_xticks(degrees)

fig.savefig(
    os.path.join("figures", "figure_01a_validation_curve_poly.png"),
    dpi=200
)
plt.close(fig)


# ==========================================
# Plot Zoomed Validation Curve (Degrees 1–4)
# ==========================================

mask = degrees <= 4

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=200,
    layout="constrained"
)

ax.plot(
    degrees[mask],
    rmse_train[mask],
    label="Training RMSE",
    color="tab:blue",
    linestyle="-",
    marker="o"
)

ax.plot(
    degrees[mask],
    rmse_valid[mask],
    label="Validation RMSE",
    color="tab:orange",
    linestyle="--",
    marker="o"
)

ax.set_xlabel("Polynomial Degree", fontsize=fontsize_xy_label)
ax.set_ylabel("Specific Growth Rate RMSE [d$^{-1}$]", fontsize=fontsize_xy_label)
ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
ax.legend(fontsize=fontsize_legend)
ax.set_ylim(bottom=0)
ax.set_xticks(degrees[mask])

fig.savefig(
    os.path.join("figures", "figure_01b_validation_curve_poly_zoom.png"),
    dpi=200
)

plt.close(fig)


best_degree = model_gscv_poly.best_params_["polynomialfeatures__degree"]


model = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor()
)


param_grid = dict(
    kneighborsregressor__n_neighbors=[i for i in range(1, 81)],  # Try every value of n_neighbors from 1 to 80
    kneighborsregressor__weights=["uniform", "distance"],  # Try both weight algorithms
)

model_gscv_knn = GridSearchCV(model, param_grid=param_grid, cv=4, scoring="neg_root_mean_squared_error",
                          return_train_score=True)

model_gscv_knn.fit(x_train, y_train)

df_knn = pd.DataFrame(model_gscv_knn.cv_results_)

df_knn = df_knn.sort_values(by=['rank_test_score'], ascending=True)

columns_view = [
    "rank_test_score",
    "param_kneighborsregressor__n_neighbors",
    "param_kneighborsregressor__weights",
    "mean_test_score", "std_test_score",
    "mean_train_score", "std_train_score"
]

ranking_knn = df_knn.loc[:, columns_view].copy()

ranking_knn.to_csv(
    os.path.join("tables", "table_02_knn_ranking.csv"),
    index=False
)


# ===================================
#   Plot Validation Curves for knn
# ===================================

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
#LINESTYLES = ["-", ":", "--", "-."]
FIG_WIDTH = 14.39 / 2.54
FIG_HEIGHT = 12.09 / 2.54

fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")

for i, weight in enumerate(param_grid["kneighborsregressor__weights"], start=1):
    mask_weight = (df_knn.loc[:, "param_kneighborsregressor__weights"] == weight)
    df_sub = df_knn.loc[mask_weight, :]
    df_sub = df_sub.sort_values(by=['param_kneighborsregressor__n_neighbors'], ascending=True)
    n_neighbors = df_sub.loc[:, "param_kneighborsregressor__n_neighbors"].to_numpy()
    rmse_train = - df_sub.loc[:, "mean_train_score"].to_numpy()
    rmse_test = - df_sub.loc[:, "mean_test_score"].to_numpy()

    ax.plot(n_neighbors, rmse_train, label=f"Train (kNN: {weight})", color=COLORS[i], linestyle="-")
    ax.plot(n_neighbors, rmse_test, label=f"Valid (kNN: {weight})", color=COLORS[i], linestyle="--")

ax.set_xlabel("k", fontsize=fontsize_xy_label)
ax.set_ylabel("Specific Growth Rate RMSE [d^-1]", fontsize=fontsize_xy_label)
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.legend(ncols=1, fontsize=fontsize_legend)
ax.set_ylim(bottom=0)

fig.savefig(os.path.join("figures", "figure_02_validation_curve_knn.png"))

best_n_neighbors = model_gscv_knn.best_params_["kneighborsregressor__n_neighbors"]
best_weight = model_gscv_knn.best_params_["kneighborsregressor__weights"]

if -model_gscv_poly.score(x_test, y_test) < -model_gscv_knn.score(x_test, y_test):
    print(f"Best Model: Polynomial, Degree: {best_degree}\n")
else: print(f"Best Model: k-Nearest Neighbours, Degree: {best_n_neighbors}, Weight: {best_weight}\n")

plt.rcParams['mathtext.default'] = 'regular'

fig_width = 14.39 / 2.54
fig_height = 12.09 / 2.54
fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12
fontsize_text = 8
str_hm1 = r"$d^{-1}$"
dx = 0.1
c_poly = "tab:blue"
c_knn = "tab:orange"
c_test = "tab:green"

kwargs_gscv = dict(cv=4, scoring="neg_root_mean_squared_error", return_train_score=True)

param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(1, 11)])

param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(1, 11)],
                      kneighborsregressor__weights=["uniform", "distance"])

rng = np.random.default_rng(seed=1)

df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_4factor.csv"))

features = df.columns[1:]
target = df.columns[0]

x_all = df.loc[:, features].to_numpy()
y_all = df.loc[:, target].to_numpy()

lc_n_training_samples = []
lc_rmse_poly_train = []
lc_rmse_knn_train = []
lc_rmse_poly_test = []
lc_rmse_knn_test = []
lc_results = []

for n_samples in range(20, 110, 10):
    # We will select a random subset of n_samples from the dataset.
    idxs = rng.choice(x_all.shape[0], size=n_samples, replace=False)
    x = x_all[idxs]
    y = y_all[idxs]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model_poly = make_pipeline(StandardScaler(), PolynomialFeatures(include_bias=False), LinearRegression())
    model_gscv_poly = GridSearchCV(model_poly, param_grid=param_grid_poly, **kwargs_gscv)
    model_gscv_poly.fit(x_train, y_train)

    model_knn = make_pipeline(StandardScaler(), KNeighborsRegressor())
    model_gscv_knn = GridSearchCV(model_knn, param_grid=param_grid_knn, **kwargs_gscv)
    model_gscv_knn.fit(x_train, y_train)

    rmse_poly_train = root_mean_squared_error(y_train, model_gscv_poly.predict(x_train))
    rmse_knn_train = root_mean_squared_error(y_train, model_gscv_knn.predict(x_train))
    rmse_poly_test = root_mean_squared_error(y_test, model_gscv_poly.predict(x_test))
    rmse_knn_test = root_mean_squared_error(y_test, model_gscv_knn.predict(x_test))

    # Extract the best hyperparameters
    best_degree = model_gscv_poly.best_params_["polynomialfeatures__degree"]
    best_n_neighbors = model_gscv_knn.best_params_["kneighborsregressor__n_neighbors"]
    best_weight = model_gscv_knn.best_params_["kneighborsregressor__weights"]

    # Extract the number of training samples
    n_training_samples = x_train.shape[0]

    lc_results.append({
        "Samples Selected": n_samples,
        "Training Samples": x_train.shape[0],
        "Test Samples": x_test.shape[0],

        "Polynomial Degree": best_degree,
        "Poly Train RMSE": rmse_poly_train,
        "Poly Test RMSE": rmse_poly_test,

        "kNN Neighbors": best_n_neighbors,
        "kNN Weights": best_weight,
        "kNN Train RMSE": rmse_knn_train,
        "kNN Test RMSE": rmse_knn_test
    })

    message = (
        f"{n_training_samples} training samples\n"
        f"Poly degree={best_degree}\n"
        f"RMSE Train: {rmse_poly_train:.6f} {str_hm1}\n"
        f"RMSE Test:  {rmse_poly_test:.6f} {str_hm1}\n"
        f"kNN: k={best_n_neighbors}, weights={best_weight}\n"
        f"RMSE Train: {rmse_knn_train:.6f} {str_hm1}\n"
        f"RMSE Test:  {rmse_knn_test:.6f} {str_hm1}"
    )

    lc_n_training_samples.append(n_training_samples)
    lc_rmse_poly_train.append(rmse_poly_train)
    lc_rmse_knn_train.append(rmse_knn_train)
    lc_rmse_poly_test.append(rmse_poly_test)
    lc_rmse_knn_test.append(rmse_knn_test)


# Plot the learning curve
fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=200, layout="constrained")
ax.plot(lc_n_training_samples, lc_rmse_poly_train, label="RMSE Train (Poly)", color=c_poly, linestyle="-")
ax.plot(lc_n_training_samples, lc_rmse_poly_test, label="RMSE Test (Poly)", color=c_poly, linestyle="--")
ax.plot(lc_n_training_samples, lc_rmse_knn_train, label="RMSE Train (kNN)", color=c_knn, linestyle="-")
ax.plot(lc_n_training_samples, lc_rmse_knn_test, label="RMSE Test (kNN)", color=c_knn, linestyle="--")
ax.set_xlabel("Number of training samples", fontsize=fontsize_xy_label)
ax.set_ylabel(f"RMSE [{str_hm1}]", fontsize=fontsize_xy_label)
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.legend(ncols=1, fontsize=fontsize_legend)

fig.savefig(os.path.join("figures", f"figure_03_learning_curve.png"))
plt.close(fig)

learning_curve_table = pd.DataFrame(lc_results)

learning_curve_table.to_csv(
    os.path.join("tables", "table_03_learning_curve.csv"),
    index=False
)


def growth_range(targets):
    return [min(targets), max(targets)]

growth_limits = growth_range(y_all)

HIGH_RANGE = [growth_limits[1] - 0.1, growth_limits[1]]
LOW_RANGE = [growth_limits[0], growth_limits[0] + 0.1]

high_mask=(y_all>=HIGH_RANGE[0]) & (y_all<=HIGH_RANGE[1])
low_mask=(y_all>=LOW_RANGE[0]) & (y_all<=LOW_RANGE[1])

high_growth_conditions = df.loc[high_mask]
low_growth_conditions = df.loc[low_mask]

mean_high_growth_temp=calculate_average(high_growth_conditions["Temperature [C]"])
mean_low_growth_temp=calculate_average(low_growth_conditions["Temperature [C]"])
mean_high_growth_ph=calculate_average(high_growth_conditions["pH"])
mean_low_growth_ph=calculate_average(low_growth_conditions["pH"])
mean_high_growth_gluc=calculate_average(high_growth_conditions["Glucose [mM]"])
mean_low_growth_gluc=calculate_average(low_growth_conditions['Glucose [mM]'])
mean_high_growth_lac=calculate_average(high_growth_conditions["Lactate [mM]"])
mean_low_growth_lac=calculate_average(low_growth_conditions["Lactate [mM]"])


temp_high_growth=[min(high_growth_conditions['Temperature [C]']),max(high_growth_conditions['Temperature [C]'])]
temp_min_growth=[min(low_growth_conditions['Temperature [C]']),max(low_growth_conditions['Temperature [C]'])]

ph_high_growth=[min(high_growth_conditions['pH']),max(high_growth_conditions['pH'])]
ph_min_growth=[min(low_growth_conditions['pH']),max(low_growth_conditions['pH'])]

glucose_high_growth=[min(high_growth_conditions['Glucose [mM]']),max(high_growth_conditions['Glucose [mM]'])]
glucose_min_growth=[min(low_growth_conditions['Glucose [mM]']),max(low_growth_conditions['Glucose [mM]'])]

lac_high_growth=[min(high_growth_conditions['Lactate [mM]']),max(high_growth_conditions['Lactate [mM]'])]
lac_min_growth=[min(low_growth_conditions['Lactate [mM]']),max(low_growth_conditions['Lactate [mM]'])]



# print("High growth conditions:")
# print(f"Temperature:{temp_high_growth} [C],pH:{ph_high_growth},Glucose:{glucose_high_growth} [mM], Lactate:{lac_high_growth} [mM]")
#
# print("Low growth conditions:")
# print(f"Temperature: {temp_min_growth} [C], pH:{ph_min_growth}, Glucose: {glucose_min_growth} [mM], Lactate: {lac_min_growth} [mM]")
#
# print("Mean High Growth Conditions")
# print(f"Temperature:{mean_high_growth_temp:.2f} [C],pH:{mean_high_growth_ph:.2f},Glucose:{mean_high_growth_gluc:.2f} [mM], Lactate:{mean_high_growth_lac:.2f} [mM]")
#
# print("Mean Low growth conditions:")
# print(f"Temperature: {mean_low_growth_temp:.2f} [C], pH:{mean_low_growth_ph:.2f}, Glucose: {mean_low_growth_gluc:.2f} [mM], Lactate: {mean_low_growth_lac:.2f} [mM]")

growth_factors=pd.DataFrame({
    "Inputs":["Temperature [C]", "pH", "Glucose [mM]","Lactate [mM]"],
    "High Growth Rate Range": [temp_high_growth, ph_high_growth, glucose_high_growth, lac_high_growth],
    "Low Growth Rate Range": [temp_min_growth, ph_min_growth, glucose_min_growth, lac_min_growth],
"Mean High Growth Conditions": [mean_high_growth_temp, mean_high_growth_ph, mean_high_growth_gluc, mean_high_growth_lac],
"Mean Low Growth Conditions": [mean_low_growth_temp, mean_low_growth_ph, mean_low_growth_gluc, mean_low_growth_lac]})

growth_factors[[
    "Mean High Growth Conditions",
    "Mean Low Growth Conditions"
]] = growth_factors[[
    "Mean High Growth Conditions",
    "Mean Low Growth Conditions"
]].round(2)

growth_factors.to_csv(
    os.path.join("tables", "table_04_growth_conditions.csv"),
    index=False
)


#determining optimal ranges
baseline = df[features].median()

results = []

best_model = best_poly_model


for variable in features:

    #Test values across the observed range of this variable
    test_values = np.linspace(
        df[variable].min(),
        df[variable].max(),
        200
    )

    #keep other inputs constant
    test_data = pd.DataFrame(
        np.tile(baseline.values, (len(test_values), 1)),
        columns=features
    )

    test_data[variable] = test_values

    predictions = best_model.predict(test_data.to_numpy())

    best_index = np.argmax(predictions)
    worst_index = np.argmin(predictions)

    results.append({
        "Variable": variable,
        "Best Predicted Value": test_values[best_index],
        "Predicted Max Growth Rate [d^-1]": predictions[best_index],
        "Worst Predicted Value": test_values[worst_index],
        "Predicted Min Growth Rate [d^-1]": predictions[worst_index]
    })

results_df = pd.DataFrame(results)

results_df[["Best Predicted Value","Predicted Max Growth Rate [d^-1]", "Worst Predicted Value",
        "Predicted Min Growth Rate [d^-1]"]]=results_df[["Best Predicted Value","Predicted Max Growth Rate [d^-1]", "Worst Predicted Value",
        "Predicted Min Growth Rate [d^-1]"]].round(2)

results_df.to_csv(
    os.path.join("tables", "table_05_optimal_conditions.csv"),
    index=False
)

# ===============================================
# Figure 4: Predicted growth rate vs. each input
# ===============================================

fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
axes = axes.flatten()

baseline = df.loc[:, features].median()

for i, variable in enumerate(features):

    test_values = np.linspace(
        df[variable].min(),
        df[variable].max(),
        200
    )

    test_data = pd.DataFrame(
        np.tile(baseline.values, (len(test_values), 1)),
        columns=features
    )

    test_data[variable] = test_values

    predictions = best_model.predict(test_data.to_numpy())

    axes[i].plot(test_values, predictions, color="tab:blue")
    axes[i].set_xlabel(variable)
    axes[i].set_ylabel("Specific Growth Rate [d$^{-1}$]")
    axes[i].set_title(f"Effect of {variable}")
    axes[i].grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Predicted Cell Growth Rate vs. Input Variables")

fig.savefig(
    os.path.join("figures", "figure_04_growth_rate_vs_inputs.png"),
    dpi=200
)

plt.close(fig)