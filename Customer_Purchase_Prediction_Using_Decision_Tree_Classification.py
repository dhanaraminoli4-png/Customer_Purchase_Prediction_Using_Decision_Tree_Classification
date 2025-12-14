import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# SET RANDOM SEED
# Ensures reproducible results

np.random.seed(42)

# 3. GENERATE SYNTHETIC DATA

samples = 1000

Age = np.random.randint(18, 65, size=samples).astype(float)
Income = np.random.randint(20_000, 150_000, size=samples).astype(float)
Time_spent_on_website = np.random.randint(1, 60, size=samples).astype(float)
Number_of_previous_purchases = np.random.randint(0, 21, size=samples).astype(float)

# INTRODUCE MISSING VALUES

nan_frac = 0.1

for arr in [Age, Income, Time_spent_on_website, Number_of_previous_purchases]:
    nan_idx = np.random.choice(samples, int(samples * nan_frac), replace=False)
    arr[nan_idx] = np.nan

# CREATE CATEGORICAL DATA

regions = [
    "Western", "Central", "Southern", "Northern", "Eastern",
    "North Western", "North Central", "Uva", "Sabaragamuwa"
]

Region = np.random.choice(regions, size=samples)
Region[np.random.choice(samples, int(samples * nan_frac), replace=False)] = np.nan

# CREATE DATAFRAME

df = pd.DataFrame({
    "Age": Age,
    "Income": Income,
    "Time_spent_on_website": Time_spent_on_website,
    "Number_of_previous_purchases": Number_of_previous_purchases,
    "Region": Region
})

# CREATE TARGET VARIABLE
# Binary classification problem

df["Purchased"] = ((df["Income"] > 50_000) & (df["Age"] > 30)).astype(int)

# DATA CLEANING

df = df.drop_duplicates()

# Impute numerical features

num_cols = ["Age", "Income", "Time_spent_on_website", "Number_of_previous_purchases"]
num_imputer = SimpleImputer(strategy="mean")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Impute categorical feature

cat_imputer = SimpleImputer(strategy="most_frequent")
df[["Region"]] = cat_imputer.fit_transform(df[["Region"]])

# TEXT PREPROCESSING (REGIONS)

def preprocess(text):
    text = text.lower()
    return word_tokenize(text)

df["tokens"] = df["Region"].apply(preprocess)
df.drop(columns="Region", inplace=True)

#  NUMERICAL SCALING

df["Income"] = np.log10(df["Income"])  # log normalization

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ENCODE CATEGORICAL DATA

mlb = MultiLabelBinarizer()
encoded = mlb.fit_transform(df["tokens"])

encoded_df = pd.DataFrame(
    encoded,
    columns=mlb.classes_,
    index=df.index
)

df = pd.concat([df, encoded_df], axis=1)
df.drop(columns="tokens", inplace=True)

# DIMENSIONALITY REDUCTION (PCA)
# PCA was used to demonstrate dimensionality reduction on encoded categorical variables

pca = PCA(n_components=1)
region_pca = pca.fit_transform(df[mlb.classes_])

df["PCA_Region"] = region_pca
df.drop(columns=mlb.classes_, inplace=True)

# SAMPLING + TRAIN / TEST SPLIT

def sample_train_test(data, sample_frac=0.6, train_frac=0.8):
    sampled = data.sample(frac=sample_frac)
    train = sampled.sample(frac=train_frac)
    test = sampled.drop(train.index)

    X_train = train.drop(columns="Purchased")
    y_train = train["Purchased"]

    X_test = test.drop(columns="Purchased")
    y_test = test["Purchased"]

    return X_train, y_train, X_test, y_test

# MODEL TRAINING & EVALUATION

rounds = 3
accuracies = []

for _ in range(rounds):
    X_train, y_train, X_test, y_test = sample_train_test(df)

    model = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    accuracies.append(acc)

# RESULTS

print("Accuracy for each round:", accuracies)
print("Average accuracy:", sum(accuracies) / len(accuracies))

# VISUALIZING THE CLASSIFICATION TREE

plt.figure(figsize=(18,10))
plot_tree(model,
          feature_names=["Age", "Income", "Time_spent_on_website", "Number_of_previous_purchases"],
          filled=True,
          rounded=True,
          fontsize=12)
plt.show()

# CONFUSION MATRIX

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Purchased", "Purchased"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Decision Tree Classifier")
plt.show()

