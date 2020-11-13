import pandas as pd
import time
import numpy as np
import sys
import copy

arguments = copy.deepcopy(sys.argv)

if len(sys.argv) != 5:
    raise Exception("please enter four parameters")
if sys.argv[1].endswith(('.csv')):
    pass
else:
    raise Exception("imput file should be of type - csv")

impacts = []
for imp in sys.argv[3]:
    if imp == ',':
        pass
    elif imp == '+' or '-':
        impacts.append(imp)

    else:
        raise Exception("impact must be positive(+) or negative (-)")

# make a list of weights
weights = []
for w in arguments[2]:
    if w != ',':
        weights.append(int(w))

print(impacts)
print(weights)

# compare length of weights, impacts list
if len(weights) != len(impacts):
    raise Exception("impacts and weights must be of same length")

# read the input file:
df = pd.read_csv(sys.argv[1])
num_of_col = len(df.columns)

# check number of columns in input file
if num_of_col < 3:
    raise Exception("input file must have atleast 3 columns")

# check length of weights list, and number of numeric columns
if (len(impacts) != num_of_col - 1):
    raise Exception("impacts, weights and number of columns  must be of same length")

# implementing step 1 of TOPSIS
def sum_of_squares(df, RSS):
    for i in range(1, len(df.columns)):
        col = df.iloc[:, i]
        sum = 0
        for j in col:
            if type(j) != int:
                try:
                    val = int(j)
                except:
                    raise Exception("all values must be numeric")
            sum += j * j
        sum = np.sqrt(sum)
        RSS.append(sum)
    return df

# implementing step 2 of TOPSIS
def normalized(df, RSS):
    for i in range(1, len(df.columns)):
        col = df.iloc[:, i]
        sum = 0
        for j in col:
            j = (j / RSS[i - 1]) * weights[i - 1]
    return df

# implementing step 3 of TOPSIS:
def best(df, impacts, ideal_best):
    for i in range(1, len(df.columns)):
        if impacts[i - 1] == '-':
            col = df.iloc[:, i]
            mini = col[0]
            for j in col:
                if j < mini:
                    mini = j;
            ideal_best.append(mini)

        else:
            col = df.iloc[:, i]
            maxi = col[0]
            for j in col:
                if j > maxi:
                    maxi = j;
            ideal_best.append(maxi)

def worst(df, impacts, ideal_worst):
    for i in range(1, len(df.columns)):
        if impacts[i - 1] == '+':
            col = df.iloc[:, i]
            mini = col[0]
            for j in col:
                if j < mini:
                    mini = j;
            ideal_worst.append(mini)

        else:
            col = df.iloc[:, i]
            maxi = col[0]
            for j in col:
                if j > maxi:
                    maxi = j;
            ideal_worst.append(maxi)

# implementing STEP 4 of TOPSIS:
def euclidean_distances(df, dist_best, dist_worst, ideal_best, ideal_worst):
    for i in range(df.shape[0]):  # iterate per row
        plus = 0
        minus = 0
        for j in range(1, len(df.columns)):
            plus += ((ideal_best[j - 1] - df.iloc[i, j]) * (ideal_best[j - 1] - df.iloc[i, j]))
            minus += ((ideal_worst[j - 1] - df.iloc[i, j]) * (ideal_worst[j - 1] - df.iloc[i, j]))
        dist_best.append(np.sqrt(plus))
        dist_worst.append(np.sqrt(minus))
    return df

RSS = []
df = sum_of_squares(df, RSS)
print("step 1 done")

df = normalized(df, RSS)
print("step 2 done")

ideal_best = []
ideal_worst = []
best(df, impacts, ideal_best)
worst(df, impacts, ideal_worst)
print("step 3 done")

dist_best = []
dist_worst = []
df = euclidean_distances(df, dist_best, dist_worst, ideal_best, ideal_worst)
print("step 4 done")

# calculate topsis score:
Top_score = []

for i in range(df.shape[0]):
    Top_score.append(dist_worst[i] / (dist_worst[i] + dist_best[i]))

df['TOPSIS_Score'] = Top_score
# print(df)
print("step 5 done")

copyfile = pd.DataFrame(df)
# print(copyfile)
print("output file is ready")

copyfile.sort_values(by=['TOPSIS_Score'], ascending=False, inplace=True)

dictionary = {}

rank = 1
for i in range(df.shape[0]):
    dictionary[copyfile.iloc[i, 0]] = rank
    rank += 1

ranks = []
for i in range(df.shape[0]):
    ranks.append(dictionary[df.iloc[i, 0]])

df['Rank'] = ranks
# print(df)
df.to_csv(arguments[4], index=False, header=True)