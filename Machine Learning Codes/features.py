import os
import re
import math
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
from tabulate import tabulate
from collections import Counter
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from mlxtend.evaluate import paired_ttest_5x2cv
from sklearn.metrics import roc_auc_score, roc_curve

# Codes adapted from https://github.com/MariaZork/my-machine-learning-tutorials/blob/master/JS_obfuscaton_detection.ipynb
warnings.filterwarnings('ignore')
sns.set_theme(font_scale = 2)

SEED = 0
benign_js_path = 'benign';
malicious_obfuscated_js_path = 'big-mal';
#Data loading
filenames, scripts, labels = [], [], []

# Set labels based on the directory
file_types_and_labels = [(benign_js_path, 0), (malicious_obfuscated_js_path, 1)]

for files_path, label in file_types_and_labels:
    files = os.listdir(files_path)
    for file in tqdm(files):
        file_path = files_path + "/" + file
        try:
            with open(file_path, "r", encoding="utf8") as myfile:
                df = myfile.read().replace("\n", "")
                df = str(df)
                filenames.append(file)
                scripts.append(df)
                labels.append(label)
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError in file: {file_path}")
            # Handle the error or log the file name if needed
            pass
        except Exception as ee:
            print(f"Error processing file: {file_path}: {ee}")
            # Handle other exceptions if needed
            pass

df = pd.DataFrame(data=filenames, columns=['js_filename'])
df['js'] = scripts
df['label'] = labels

# Removing empty scripts by checking the scripts column
df = df[df['js'] != '']
# Capture duplicated rows
duplicated_rows = df[df["js"].duplicated(keep=False)]

# Apply filtering to remove duplicated rows
df = df[~df["js"].isin(duplicated_rows["js"])]

# Print duplicated rows
#print("Duplicated rows:")
#print(duplicated_rows)

#see what has been done after data preprocessing
filtered_ben = df[df['label'] == 0]
filtered_mal = df[df['label'] == 1]
print("Benign","Malicious")
print(len(filtered_ben), len(filtered_mal))

#Featurization
print('Begin Feature Engineering')
df['js_length'] = df.js.apply(lambda x: len(x))
df['num_spaces'] = df.js.apply(lambda x: x.count(' '))
df['num_parenthesis'] = df.js.apply(lambda x: (x.count('(') + x.count(')')))
df['num_slash'] = df.js.apply(lambda x: x.count('/'))
df['num_plus'] = df.js.apply(lambda x: x.count('+'))
df['num_point'] = df.js.apply(lambda x: x.count('.'))
df['num_comma'] = df.js.apply(lambda x: x.count(','))
df['num_semicolon'] = df.js.apply(lambda x: x.count(';'))
df['num_alpha'] = df.js.apply(lambda x: len(re.findall(re.compile(r"\w"),x)))
df['num_numeric'] = df.js.apply(lambda x: len(re.findall(re.compile(r"[0-9]"),x)))

df['ratio_spaces'] = df['num_spaces'] / df['js_length']
df['ratio_alpha'] = df['num_alpha'] / df['js_length']
df['ratio_numeric'] = df['num_numeric'] / df['js_length']
df['ratio_parenthesis'] = df['num_parenthesis'] / df['js_length']
df['ratio_slash'] = df['num_slash'] / df['js_length']
df['ratio_plus'] = df['num_plus'] / df['js_length']
df['ratio_point'] = df['num_point'] / df['js_length']
df['ratio_comma'] = df['num_comma'] / df['js_length']
df['ratio_semicolon'] = df['num_semicolon'] / df['js_length']


def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

df['entropy'] = df.js.apply(lambda x: entropy(x))

# String Operation: substring(), charAt(), split(), concat(), slice(), substr()

df['num_string_oper'] = df.js.apply(lambda x: x.count('substring') +
                                            x.count('charAt') +
                                            x.count('split') +
                                            x.count('concat') +
                                            x.count('slice') +
                                            x.count('substr'))

df['r_num_string_oper'] = df['num_string_oper'] / df['js_length']

# Encoding Operation: escape(), unescape(), string(), fromCharCode()

df['num_encoding_oper'] = df.js.apply(lambda x: x.count('escape') +
                                        x.count('unescape') +
                                        x.count('string') +
                                        x.count('fromCharCode'))

df['r_num_encoding_oper'] = df['num_encoding_oper'] / df['js_length']


# URL Redirection: setTimeout(), location.reload(), location.replace(), document.URL(), document.location(), document.referrer()

df['num_url_redirection'] = df.js.apply(lambda x: x.count('setTimeout') +
                                          x.count('location.reload') +
                                          x.count('location.replace') +
                                          x.count('document.URL') +
                                          x.count('document.location') +
                                          x.count('document.referrer'))

df['r_num_url_redirection'] = df['num_url_redirection'] / df['js_length']


# Specific Behaviors: eval(), setTime(), setInterval(), ActiveXObject(), createElement(), document.write(), document.writeln(), document.replaceChildren()

df['num_specific_func'] = df.js.apply(lambda x: x.count('eval') +
                                       x.count('setTime') +
                                       x.count('setInterval') +
                                       x.count('ActiveXObject') +
                                       x.count('createElement') +
                                       x.count('document.write') +
                                       x.count('document.writeln') +
                                       x.count('document.replaceChildren') +
                                       x.count('window.execScript')) # this item not in original code

df['r_num_specific_func'] = df['num_specific_func'] / df['js_length']
print('Feature Engineering Ended')
# create csv
#csv_file_path = 'output.csv'
# Specify the escape character
#escape_char = "\\"  # added to code as I had issues without it
# Write the DataFrame to a CSV file with the escapechar parameter
#df.to_csv(csv_file_path, index=False, escapechar=escape_char)
#print ('File saved')

#Extract numeric features only
num_feat=df.iloc[:, 3:]

# Calculate correlation matrix
corr_matrix = num_feat.corr()

# Display the heatmap
plt.figure(figsize=(30, 30)) #size of the heatmap
corr_heat=sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.1, annot_kws={"size": 10})
# Customize font size of x and y axis tick labels
corr_heat.set_xticklabels(corr_heat.get_xticklabels(), rotation=45, ha="right", fontsize=8)
corr_heat.set_yticklabels(corr_heat.get_yticklabels(), fontsize=8)

corr_heat.set_title('Features Correlation Heatmap', fontdict={'fontsize':12},pad=11)
#plt.title("Feature Correlation Heatmap")
plt.show()
# The line below  begins from the fourth column to the end
print('Splitting file for training')
X_train, X_test, y_train, y_test = train_test_split(num_feat, df['label'],
                                                    stratify=df['label'],
                                                    test_size=0.2,
                                                    random_state=SEED)
print('Splitting ended')
confidence_level = 0.95  # Set the desired confidence level
n = len(y_test)  # Number of samples
# The codes below are for the training and testing of the different algorithms
# Initialize dictionary to store the models to be used
models = {}
model_scores={}
models = [
    ("NB", MultinomialNB()),  # Multinomial Naive Bayes
    ("RF", RandomForestClassifier(n_estimators=100, random_state=SEED)),
    ("GB", GradientBoostingClassifier(n_estimators=100, random_state=SEED)),
    ("SVM",SVC(C=1.0, random_state=SEED)),  # Support Vector Machines
    ("LR", LogisticRegression(C=1.0, random_state=SEED))
]
# variables for the best model
best_model = None
best_accuracy = 0
bestmodel_conf_mat = None
for name, model in models:
    print(f"Training {name} model.")

    # Create a k-fold cross-validation object
    k_fold = KFold(n_splits=5, shuffle=True, random_state=SEED)

    print(f'Running cross validation for {name} model.')
    d_cross_val_score = cross_val_score(model, X_train, y_train, cv=k_fold)
    accuracy_mean = d_cross_val_score.mean()
    accuracy_std = d_cross_val_score.std()
    accuracy_median = np.median(d_cross_val_score)
    accuracy_1st_quartile = np.percentile(d_cross_val_score, 25)
    accuracy_3rd_quartile = np.percentile(d_cross_val_score, 75)


    # Train and evaluate other models
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    
        # Check if the current model has the highest accuracy
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy
        bestmodel_conf_mat = conf_mat

    # Calculate confidence interval for accuracy
    std_error = np.sqrt(accuracy * (1 - accuracy) / n)  # Standard error
    margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * std_error  # Margin of error
    lower_bound = accuracy - margin_of_error
    upper_bound = accuracy + margin_of_error
    #store model scores
    model_scores[name] = {
        "Cross Validation Mean": accuracy_mean,
        "Cross Validation Std Dev": accuracy_std,
        "Cross Validation Median": accuracy_median,
        "Cross Validation 1st Quartile": accuracy_1st_quartile,
        "Cross Validation 3rd Quartile": accuracy_3rd_quartile,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC":  auc,
        "TPR":  tpr,
        "FPR":  fpr,
        "Confusion Matrix": conf_mat,
        "Confidence Interval (Lower, Upper)": [lower_bound, upper_bound]
    }

    #save the models
    joblib.dump(model, f'{name}.jobl')
    
    

# The next here is to tabulate the results
table_data = []
for model_name, scores in model_scores.items():
    row = [
        model_name,
        scores['Cross Validation Mean'],
        scores['Cross Validation Std Dev'],
        scores['Cross Validation Median'],
        scores['Cross Validation 1st Quartile'],
        scores['Cross Validation 3rd Quartile'],
        scores['Accuracy'],
        scores['Precision'],
        scores['Recall'],
        scores['F1'],
        scores['AUC'],
        scores['TPR'],
        scores['FPR'],
        scores['Confusion Matrix'],
        scores['Confidence Interval (Lower, Upper)']
    ]
    table_data.append(row)

headers = [
    "Model",
    "Cross Validation Mean",
    "Cross Validation Std Dev",
    "Cross Validation Median",
    "Cross Validation 1st Quartile",
    "Cross Validation 3rd Quartile",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC",
    "TPR",
    "FPR",
    "Confusion Matrix",
    "Confidence Interval (Lower, Upper)"
]

table = tabulate(table_data, headers=headers, tablefmt="grid")
print("\nFinal Evaluation Results:")
print(table)

# now let's plot from the table
# Extract model names
model_names = [row[0] for row in table_data]

# Extract corresponding metric values from the table data
accuracy_values = [row[6] for row in table_data]
precision_values = [row[7] for row in table_data]
recall_values = [row[8] for row in table_data]
f1_values = [row[9] for row in table_data]
auc_values = [row[10] for row in table_data]
tpr_values = [row[11] for row in table_data]
fpr_values = [row[12] for row in table_data]

# Create subbarplots for different metrics bar plot
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].bar(model_names, accuracy_values)
axs[0, 0].set_title('Accuracy')

axs[0, 1].bar(model_names, precision_values)
axs[0, 1].set_title('Precision')

axs[0, 2].bar(model_names, recall_values)
axs[0, 2].set_title('Recall')

axs[1, 0].bar(model_names, f1_values)
axs[1, 0].set_title('F1 Score')

axs[1, 1].bar(model_names, auc_values)
axs[1, 1].set_title('AUC')

axs[1, 2].set_visible(False)

plt.tight_layout()

# Plot ROC curves
plt.figure(figsize=(10, 5))
for model_name, fpr, tpr in zip(model_names, fpr_values, tpr_values):
    plt.plot(fpr, tpr, label=f"{model_name}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# print the best model
print(f"Best model: {best_model}")
print(f"Best accuracy: {best_accuracy}")
# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10,10))
sns.heatmap(bestmodel_conf_mat, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Save the trained models

print('Run paired test')

# Perform paired t-test for model comparison
estimator1 = models[2][1]  # Gradient Booster
estimator2 = models[1][1]  # RandomForest
t, p = paired_ttest_5x2cv(estimator1=estimator1, estimator2=estimator2, X=X_test, y=y_test)
print(f'Paired t-test p-value: {p:.4f}, {t}')


