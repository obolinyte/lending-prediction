#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    plot_confusion_matrix,
    confusion_matrix,
)
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.metrics import (
    plot_confusion_matrix,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_squared_error,
)
from typing import List

from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
import statsmodels.stats.api as sms
import statsmodels.api as sm


from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.pipeline import Pipeline
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from imblearn.pipeline import Pipeline as ImPipeline
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
from sklearn.dummy import DummyRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cbt
import optuna
import warnings

np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

custom_palette = ["#124984", "#5fa5cd", "#a7d0e4", "#dd7059", "#d25849", "#ae172a", "#8a0b25"]
primary_light_color = '#5699bf'


# In[ ]:


RANDOM = 101


# ### Project specific functions

# In[2]:


months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


# In[3]:


states = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DC": "DC",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}


# In[4]:


def group_title(title):
    if title == "medical":
        return "medical_expenses"
    elif title == "car":
        return "car_financing"
    elif title == "credit_card":
        return "credit_card_refinancing"
    elif title == "house":
        return "home_buying"
    elif "business" in title:
        return "business"
    elif title == "moving":
        return "moving_and_relocation"
    elif title == "renewable_energy":
        return "green_loan"
    return title


# In[5]:


def convert_col_name(col_name):
    col = col_name.lower().replace(" ", "_")
    return col


# In[6]:


def count_accepted(group):
    accepted = group[group == "accepted"].count()
    return accepted


# In[7]:


def count_rejected(group):
    accepted = group[group == "rejected"].count()
    return accepted


# In[8]:


def find_state(state):
    if state in top_states:
        return "background-color: #8ac0db"


# In[9]:


def convert_term(term_string):
    if term_string.strip() == '36 months':
        return 36
    else:
        return 60


# In[ ]:


def plot_avg_by_state(ax, col, df, top_list):
    
    sns.barplot(
    x="state_long",
    y=col,
    data=df, palette=['#124984' if x in top_list else '#a9d3eb' for x in df['state_long']],
    ax=ax
    )
    ax.xaxis.set_tick_params(rotation=90)
    
        
    if col == 'avg_fico':
        set_labels(ax, "Sorted list of states by avg fico score (normalized)", "state", "avg fico score (normalized)",)
    else:
        set_labels(ax, "Sorted list of states by avg dti ratio (normalized)", "state", "avg dti ratio (normalized)",)


# ### Utility functions

# In[10]:


def set_bar_values(
    ax: plt.Axes, fontsize: str, sign: str = "", y_location=3, r=1
) -> None:

    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height()
        value = f"{round(p.get_height(),r)}{sign}"
        ax.text(
            _x,
            _y + y_location,
            value,
            verticalalignment="bottom",
            ha="center",
            fontsize=fontsize,
        )


# In[55]:


def normalize_dti(string: str) -> str:
    for word in ["dti", "Dti"]:
        if word in string:
            string = string.replace(word, "DTI")
    return string


# In[56]:


def normalize_fico(string: str) -> str:
    for word in ["fico", "Fico"]:
        if word in string:
            string = string.replace(word, "FICO")
    return string


# In[57]:


def set_labels(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:

    title = title.replace("_", " ").capitalize()
    xlabel = xlabel.replace("_", " ").capitalize()
    ylabel = ylabel.replace("_", " ").capitalize()

    title = normalize_dti(title)
    xlabel = normalize_dti(xlabel)
    ylabel = normalize_dti(ylabel)
    
    title = normalize_fico(title)
    xlabel = normalize_fico(xlabel)
    ylabel = normalize_fico(ylabel)

    ax.set_title(title, pad=20, fontsize=14, fontweight="semibold")
    ax.set_xlabel(
        xlabel,
        fontsize=12,
        labelpad=12,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=12,
    )


# In[15]:


def plot_norm_value_counts(
    column: str, ax: plt.Axes, df: pd.DataFrame, y_location=0.3, font_size=10, palette='RdBu'
) -> None:
    sns.barplot(
        x=(df[column].value_counts(normalize=True) * 100).index,
        y=(df[column].value_counts(normalize=True) * 100),
        ax=ax,
        palette=palette
    )
    set_bar_values(ax, font_size, sign="%", y_location=y_location)
    set_labels(
        ax,
        "% of applications by " + column,
        column,
        "percentage",
    )


# In[16]:


def plot_top_percentage(
    column: str,
    ax: plt.Axes,
    df: pd.DataFrame,
    top_number,
    y_location=0.3,
    font_size=10,
) -> None:

    top = (df[column].value_counts(normalize=True) * 100).nlargest(top_number)
    sns.barplot(
        x=top.index,
        y=top,
        ax=ax,
        palette = custom_palette
    )
    set_bar_values(ax, font_size, sign="%", y_location=y_location)


# In[17]:


def plot_grade_boxplot_means(ax, y, title, df, showfliers=True):
    sns.boxplot(
    x="grade",
    y=y,
    data=df,
    ax=ax,
    order=["A", "B", "C", "D", "E", "F", "G"],
    showmeans=True,
    showfliers=showfliers,
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "10",
    },
    palette=custom_palette
    )
    set_labels(
        ax,
        title,
        "grade",
        y,
    )


# In[18]:


def highlight_max(s, num_to_highlight):
    is_large = s.nlargest(num_to_highlight).values
    return ["background-color: #a7d0e4" if v in is_large else "" for v in s]


# In[19]:


def make_mi_scores(
    X: pd.DataFrame, y: pd.Series, discrete_features: pd.Series
) -> pd.Series:

    if y.dtype == "object":
        mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=RANDOM)
    else:
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=RANDOM)

    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


# In[20]:


def factorize_df(X: pd.DataFrame) -> pd.DataFrame:

    X_num = X.copy()

    cols_to_factorize = X.select_dtypes(object).columns
    X_num[cols_to_factorize] = X_num[cols_to_factorize].apply(
        lambda x: pd.factorize(x)[0]
    )

    return X_num


# In[58]:


def return_mi(imputer, X: pd.DataFrame, y: pd.Series) -> pd.Series:

    X_num = factorize_df(X)

    if imputer != None:
        imputed_bmi = imputer.fit_transform(X_num)
        X_num = pd.DataFrame(imputed_bmi, columns=X_num.columns)

    discrete_features = X_num.dtypes == int
    mi_scores = make_mi_scores(X_num, y, discrete_features)

    return round(mi_scores, 3)


# ### Statistical inference functions

# In[22]:


def evaluate_pvalue(val: int) -> str:

    if val < 0.05:
        return "Stat. significant difference"
    return "Not enough evidence"


# In[23]:


def compare_proportions_ztest(
    contingency_table: pd.DataFrame, label: str = "difference in proportions"
) -> pd.DataFrame:

    successes = np.array(contingency_table["yes"])
    samples = np.array(contingency_table["total"])

    stat, p_value = proportions_ztest(
        count=successes, nobs=samples, alternative="two-sided"
    )

    lb, ub = confint_proportions_2indep(
        count1=np.array(contingency_table.iloc[0])[0],
        nobs1=np.array(contingency_table.iloc[0])[1],
        count2=np.array(contingency_table.iloc[1])[0],
        nobs2=np.array(contingency_table.iloc[1])[1],
    )

    diff_in_proportions = pd.DataFrame(
        data=[p_value, stat, lb, ub],
        index=["p-value", "z-statistic", "CI lower", "CI upper"],
        columns=[label],
    ).T

    diff_in_proportions["significance"] = diff_in_proportions["p-value"].apply(
        evaluate_pvalue
    )

    return diff_in_proportions


# In[24]:


def compare_means_ztest(sample_1: pd.DataFrame, sample_2: pd.DataFrame) -> pd.DataFrame:

    cm = sms.CompareMeans(sms.DescrStatsW(sample_1), sms.DescrStatsW(sample_2))
    z_stat, p_val = cm.ztest_ind(usevar="unequal")
    lb, ub = cm.tconfint_diff(usevar="unequal")

    diff_in_means = pd.DataFrame(
        data=[p_val, z_stat, lb, ub],
        index=["p-value", "z-statistic", "CI lower", "CI upper"],
        columns=["difference in means"],
    ).T

    diff_in_means["significance"] = diff_in_means["p-value"].apply(evaluate_pvalue)

    return diff_in_means.round(2)


# ### ML related functions

# In[43]:


def set_labels_cm(
    ax: plt.Axes, title: str, xlabel: str, ylabel: str, fontsize=12
) -> None:

    title = title.replace("_", " ").capitalize()
    xlabel = xlabel.replace("_", " ").capitalize()
    ylabel = ylabel.replace("_", " ").capitalize()

    ax.set_title(title, pad=14, fontsize=fontsize, fontweight="semibold")
    ax.set_xlabel(xlabel, fontsize=10, labelpad=12)
    ax.set_ylabel(
        ylabel,
        fontsize=10,
    )


# In[25]:


def select_top_features(df, target):

    y = df[target]
    X = df.drop([target], axis=1)
    mi_scores = pd.DataFrame(return_mi(None, X, y))
    mi_scores = mi_scores.rename(columns={"MI Scores": "mi_scores"})

    df_num = factorize_df(df)
    corr_df = pd.DataFrame(df_num.corr().abs()[target].sort_values(ascending=False))[1:]
    corr_df = corr_df.rename(columns={target: "corr_coef"})

    top_by_corr = corr_df.head(20).index

    top_features = list(top_by_corr).copy()
    for feature in list(mi_scores.nlargest(20, columns="mi_scores").index):
        if feature not in top_by_corr:
            top_features.append(feature)

    return top_features


# In[26]:


def compare_classifiers(lst_of_classifiers, X_train, y_train):

    cv_comparison = pd.DataFrame(
        columns=[
            "Classifier",
            "Fit_time",
            "Roc_auc",
            "F1-score",
            "Precision",
            "Recall",
        ]
    )

    for model_name, model in lst_of_classifiers:
        results = cross_validate(
            model,
            X_train,
            y_train,
            cv=3,
            scoring=(
                "f1_macro",
                "roc_auc",
                "precision",
                "recall",
            ),
            error_score="raise",
        )

        cv_comparison = cv_comparison.append(
            {
                "Classifier": model_name,
                "Fit_time": results["fit_time"].mean(),
                "Roc_auc": results["test_roc_auc"].mean(),
                "F1-score": results["test_f1_macro"].mean(),
                "Precision": results["test_precision"].mean(),
                "Recall": results["test_recall"].mean(),
            },
            ignore_index=True,
        )

    return cv_comparison


# In[27]:


def plot_cm_comparison(
    lst_of_classifiers: List, X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:

    f, ax = plt.subplots(2, 3, figsize=(12, 9))
    f.suptitle("Confusion matrix (normalized by actuals)", fontsize=16, y=1)

    for i in range(len(lst_of_classifiers)):
        j = i // 3
        k = i % 3

        y_pred = cross_val_predict(lst_of_classifiers[i][1], X_train, y_train, cv=6)
        g = sns.heatmap(
            confusion_matrix(y_train, y_pred, normalize="true"),
            ax=ax[j][k],
            annot=True,
            fmt=".2%",
            cmap="RdBu",
            cbar=False,
        )
        g.set_title(lst_of_classifiers[i][0])
        g.set_xlabel("Predicted")
        g.set_ylabel("Actual")
        plt.grid(False)

    plt.tight_layout(h_pad=2)


# In[28]:


def create_cm_pr(y_test, pred, pred_proba):

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    cm = confusion_matrix(y_test, pred, normalize="true")

    sns.heatmap(
        cm,
        cmap="RdBu_r",
        annot=True,
        ax=ax[0],
        cbar=False,
        square=True,
        annot_kws={"fontsize": 11},
        fmt=".2%",
    )
    set_labels_cm(
        ax[0],
        "confusion matrix (normalized by actuals)",
        "predicted",
        "actual",
    )

    precision, recall, threshold = precision_recall_curve(
        y_test, pred_proba[:, 1], pos_label=1
    )
    ap = average_precision_score(y_test, pred_proba[:, 1])
    ax[1].plot(
        recall, precision, color="#2166ac", lw=2, label="PR curve (AP = %.2f)" % ap
    )
    ax[1].fill_between(
        recall, precision, y2=np.min(precision), color="#a9d3eb", alpha=0.4, hatch="/"
    )
    set_labels_cm(ax[1], "precision-recall curve", "Recall", "Precision")
    ax[1].legend()

    plt.show()


#     fscore = (2 * precision * recall) / (precision + recall)
#     ix = np.argmax(fscore)
#     print('Best Threshold=%f, F-Score=%.3f' % (threshold[ix], fscore[ix]))


# In[29]:


class MeanByState(BaseEstimator, TransformerMixin):
    def __init__(self, column=None):
        self.column = column
        self.dti_ratio_by_state = None
        self.feature_name = "mean_" + column + "_by_state"

    def fit(self, X, y=None):
        self.dti_ratio_by_state = X.groupby("state").agg(mean=(self.column, "mean"))
        return self

    def transform(self, X, y=None):
        #         X[self.feature_name] = X['state'].transform(lambda x: self.dti_ratio_by_state.loc[x])
        X[self.feature_name] = X["state"].transform(
            lambda x: self.dti_ratio_by_state.loc[x]
        )
        return X

    def get_feature_name(self):
        return self.feature_name


# In[30]:


def get_month(X: pd.DataFrame) -> pd.DataFrame:

    X = X.copy()
    X["month"] = pd.DatetimeIndex(X["date"]).month
    X.drop("date", axis=1, inplace=True)
    return X


def get_month_cyclic(X: pd.DataFrame) -> pd.DataFrame:

    X = X.copy()
    X["month_sin"] = np.sin(X["month"] / 12 * 2 * np.pi)
    X["month_cos"] = np.cos(X["month"] / 12 * 2 * np.pi)
    X.drop("month", axis=1, inplace=True)
    return X

def get_cr_line_year(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["cr_line_year"] = pd.DatetimeIndex(X["earliest_cr_line"]).year
    return X


def get_multi_month(X: pd.DataFrame) -> pd.DataFrame:

    X = X.copy()
    X["issue_month"] = pd.DatetimeIndex(X["issue_d"]).month
    X["cr_line_month"] = pd.DatetimeIndex(X["earliest_cr_line"]).month
    X.drop(columns=["issue_d", "earliest_cr_line"], inplace=True)
    return X


# In[31]:


def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


# In[32]:


def objective_lgbm(trial, X, y, preprocessor):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 5, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
    }

    model = lgb.LGBMClassifier(
        objective="binary",
        verbosity=-1,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    X_train = preprocessor.fit_transform(X_train, y_train)
    X_valid = preprocessor.transform(X_valid)

    model.fit(
        X_train,
        y_train,
        early_stopping_rounds=100,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    yhat = model.predict(X_valid)

    return f1_score(y_valid, yhat, average="macro")


# In[33]:


def objective_xgb(trial, X, y, preprocessor):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 5, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
    }

    model = xgb.XGBClassifier(
        objective="reg:logistic", use_label_encoder=False, **params
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    X_train = preprocessor.fit_transform(X_train, y_train)
    X_valid = preprocessor.transform(X_valid)

    model.fit(
        X_train,
        y_train,
        early_stopping_rounds=100,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    yhat = model.predict(X_valid)

    #     trial.set_user_attr(key="best_booster", value=model)

    return f1_score(y_valid, yhat, average="macro")


# In[34]:


def compare_multiclass_cls(lst_of_classifiers, X_train, y_train):

    cv_comparison = pd.DataFrame(
        columns=[
            "Classifier",
            "Fit_time",
            "Accuracy",
            "F1-score",
            "Precision",
            "Recall",
        ]
    )

    for model_name, model in lst_of_classifiers:
        results = cross_validate(
            model,
            X_train,
            y_train,
            cv=5,
            scoring=(
                "accuracy",
                "f1_macro",
                "precision_macro",
                "recall_macro",
            ),
            error_score="raise",
        )

        cv_comparison = cv_comparison.append(
            {
                "Classifier": model_name,
                "Fit_time": results["fit_time"].mean(),
                "Accuracy": results["test_accuracy"].mean(),
                "F1-score": results["test_f1_macro"].mean(),
                "Precision": results["test_precision_macro"].mean(),
                "Recall": results["test_recall_macro"].mean(),
            },
            ignore_index=True,
        )

    return cv_comparison


# In[35]:


def create_multi_cm(
    y_test, pred, labels=["A", "B", "C", "D", "E", "F", "G"]
):

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    cm = confusion_matrix(y_test, pred, normalize="true")

    sns.heatmap(
        cm,
        cmap="RdBu_r",
        annot=True,
        ax=ax[0],
        cbar=False,
        square=True,
        annot_kws={"fontsize": 10},
        fmt=".1%",
        xticklabels=labels,
        yticklabels=labels,
    )
    set_labels_cm(
        ax[0],
        "confusion matrix (normalized by actuals)",
        "predicted",
        "actual",
    )

    ax[1].axis("off")
    plt.show()


# In[36]:


def objective_multi_lgbm(trial, X, y, preprocessor):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 5, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 4, 500, step=20),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 1000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
    }

    model = lgb.LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        verbosity=-1,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    X_train = preprocessor.fit_transform(X_train, y_train)
    X_valid = preprocessor.transform(X_valid)

    model.fit(
        X_train,
        y_train,
        early_stopping_rounds=100,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    yhat = model.predict(X_valid)

    return f1_score(y_valid, yhat, average="macro")


# In[37]:


def objective_lgbm_subgrade(
    trial,
    X,
    y,
):

    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 10, 300, step=5),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200, step=10),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=1),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=1),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 10),
    }

    model = lgb.LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        verbosity=-1,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    predictor = model.fit(
        X_train,
        y_train,
        early_stopping_rounds=100,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    yhat = predictor.predict(X_valid)

    trial.set_user_attr(key="best_booster", value=predictor)

    return f1_score(y_valid, yhat, average="macro")


# In[38]:


def train_lgbm_subgrade(grade, df, preprocessor):

    per_grade = df[df["grade"] == grade]
    y_s = per_grade["sub_grade"]
    X_s = per_grade.drop(["sub_grade", "grade"], axis=1)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_s, y_s, test_size=0.20, stratify=y_s, random_state=RANDOM
    )

    X_train_tr = preprocessor.fit_transform(X_train_s)
    X_test_tr = preprocessor.transform(X_test_s)

    sampler = TPESampler(seed=RANDOM)
    study_lgbm_subgrade = optuna.create_study(direction="maximize", sampler=sampler)
    study_lgbm_subgrade.optimize(
        lambda trial: objective_lgbm_subgrade(trial, X_train_tr, y_train_s),
        n_trials=5,
        show_progress_bar=True,
        callbacks=[callback],
    )

    best_value = round(study_lgbm_subgrade.best_value, 3)
    lgbm_hp = study_lgbm_subgrade.best_params
    best_cls = study_lgbm_subgrade.user_attrs["best_booster"]

    subgrade_pred = best_cls.predict(X_test_tr)

    metric_dict = {
        "Grade": grade,
        "F1-score": f1_score(y_test_s, subgrade_pred, average="macro"),
        "Precision": precision_score(y_test_s, subgrade_pred, average="macro"),
        "Recall": recall_score(y_test_s, subgrade_pred, average="macro"),
    }

    return best_value, lgbm_hp, metric_dict, best_cls


# In[39]:


def objective_lgbm_int_rate(
    trial,
    X,
    y,
):

    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 10, 300, step=5),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 300, step=10),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=1),
        #         "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=1),
        #         "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 10),
    }

    model = lgb.LGBMRegressor(
        objective="regression",
        verbosity=-1,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    predictor = model.fit(
        X_train,
        y_train,
        early_stopping_rounds=100,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        verbose=False,
    )

    yhat = predictor.predict(X_valid)

    trial.set_user_attr(key="best_booster", value=predictor)

    return np.sqrt(mean_squared_error(y_valid, yhat))


# In[40]:


def train_lgbm_int_rate(grade, df, preprocessor):

    per_grade = df[df["grade"] == grade]
    y_rate = per_grade["int_rate"]
    X_rate = per_grade.drop(["int_rate", "grade"], axis=1)
    X_train_rate, X_test_rate, y_train_rate, y_test_rate = train_test_split(
        X_rate, y_rate, test_size=0.20, random_state=RANDOM
    )

    X_train_tr = preprocessor.fit_transform(X_train_rate)
    X_test_tr = preprocessor.transform(X_test_rate)

    sampler = TPESampler(seed=RANDOM)
    study_lgbm_int_rate = optuna.create_study(direction="minimize", sampler=sampler)
    study_lgbm_int_rate.optimize(
        lambda trial: objective_lgbm_int_rate(trial, X_train_tr, y_train_rate),
        n_trials=10,
        show_progress_bar=True,
        callbacks=[callback],
    )

    best_value = round(study_lgbm_int_rate.best_value, 3)
    lgbm_hp = study_lgbm_int_rate.best_params
    best_rgs = study_lgbm_int_rate.user_attrs["best_booster"]

    #     best_rgs = Pipeline(steps=[
    #                             ("preprocessor", pipe_transform_int),
    #                             (
    #                             "classifier",
    #                             lgb.LGBMRegressor(objective="regression", verbosity=-1, **lgbm_hp)
    #                             ),
    #                             ]
    #                             ).fit(X_train_rate, y_train_rate)

    #     joblib.dump(best_rgs, f"models/{grade}_rate_regressor.pkl")

    int_rate_pred = best_rgs.predict(X_test_tr)

    metric_dict = {
        "Grade": grade,
        "R2-score": r2_score(y_test_rate, int_rate_pred),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_test_rate,
                int_rate_pred,
            )
        ),
    }

    return best_value, lgbm_hp, metric_dict, best_rgs


# In[41]:


def evaluate_model(y_test: pd.Series, predicted_values: list) -> None:

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "--",
        lw=2,
        color="r",
        alpha=0.4,
    )
    sns.scatterplot(x=y_test, y=predicted_values, color="#2166ac", ax=ax[0], s=60)
    set_labels(ax[0], "actuals vs predicted values", "actuals", "predicted")

    residuals = y_test - predicted_values
    sns.histplot(
        residuals,
        kde=True,
        ax=ax[1],
        color="#2166ac",
        bins=20,
        edgecolor="white",
    )
    set_labels(ax[1], "distribution of residuals", "predicted", "residuals")

    plt.tight_layout()
    plt.show()


# In[47]:


def append_cls_metrics(
    metrics_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    y_pred_proba: pd.Series,
    model_name: str,
) -> pd.DataFrame:

    metrics_df = metrics_df.append(
        {
            "Model": model_name,
            "Roc-auc": roc_auc_score(y_test, y_pred_proba[:, 1], average="macro"),
            "F1-score": f1_score(y_test, y_pred, average="macro"),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
        },
        ignore_index=True,
    )

    return metrics_df


# In[59]:


def create_confusion(test: pd.Series, pred: list) -> None:

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    cm = confusion_matrix(test, pred, normalize="true")

    sns.heatmap(
        cm,
        cmap="RdBu_r",
        annot=True,
        ax=ax[0],
        cbar=False,
        square=True,
        annot_kws={"fontsize": 11},
        fmt=".2%",
    )
    set_labels_cm(
        ax[0],
        "confusion matrix (normalized by actuals)",
        "predicted",
        "actual",
    )

    ax[1].axis("off")
    plt.show()


# In[60]:


def plot_prec_recall_vs_tresh(precisions, recalls, thresholds, ax):
    
    ax.plot(thresholds, precisions[:-1], '#124984', linewidth=2, label='precision')
    ax.plot(thresholds, recalls[:-1], '#8a0b25', linewidth=2, label = 'recall')
    ax.legend(loc='lower left')
    ax.axvline(x=.5, ymin=0, ymax=1, linestyle="dashed", color='grey')
    ax.text(
        x=0.5,
        y=0.35,
        s='Current threshold',
        horizontalalignment="center",
        color="black",
        rotation=90,
        alpha=1,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=1, edgecolor="none"),
    )
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    set_labels_cm(ax, "Precision & recall for probability thresholds", "Threshold", "")


# In[ ]:


def compare_regressors(
    lst_of_regressors: List, X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:

    cv_comparison = pd.DataFrame(columns=["Regressor", "Fit_time", "R-squared", "RMSE"])

    for model_name, model in lst_of_regressors:
        results = cross_validate(
            model,
            X_train,
            y_train,
            cv=6,
            scoring=("r2", "neg_root_mean_squared_error"),
        )

        cv_comparison = cv_comparison.append(
            {
                "Regressor": model_name,
                "Fit_time": results["fit_time"].mean(),
                "R-squared": results["test_r2"].mean(),
                "RMSE": results["test_neg_root_mean_squared_error"].mean() * (-1),
            },
            ignore_index=True,
        )

    return cv_comparison

