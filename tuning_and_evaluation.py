#%%
import joblib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.utils.fixes import loguniform


model = joblib.load('./model/svm_face_recognitor')
# %% Tuning
param_grid = {
    "C": loguniform(1e3, np.inf),
    "max_iter": [10000, 11000, 15000, 20000, 100000, 1000000],
    'random_state': [0, 42, 100, 200, 300, 500, 700, 10000],
    'penalty': ('l1', 'l2'),
    'loss': ('hinge', 'squared_hinge'),
    'multi_class': ('ovr', 'crammer_singer')
}
SVM_clf = RandomizedSearchCV(
    LinearSVC(), param_grid, n_iter=10
)
SVM_clf = SVM_clf.fit(trainX, trainy)
print("Best estimator found by grid search:")
print(SVM_clf.best_estimator_)

# %%
# Decision boundary
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    # Plot decision boundary:
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]        
    x1 = np.linspace(xmin, xmax, 200)
    x2 = -w[0]/w[1]*x1 - b/w[1] # Note: At the decision boundary, w1*x1 + w2*x2 + b = 0 => x2 = -w1/w2 * x1 - b/w2
    plt.plot(x1, x2, "k-", linewidth=3, label="SVM")
    
    # Plot gutters of the margin:
    margin = 1/w[1]
    right_gutter = x2 + margin
    left_gutter = x2 - margin
    plt.plot(x1, right_gutter, "k:", linewidth=2)
    plt.plot(x1, left_gutter, "k:", linewidth=2)

    # Highlight samples at the gutters (support vectors):
    skipped=True
    if not skipped:
        hinge_labels = y*2 - 1 # hinge loss label: -1, 1. our label y: 0, 1
        scores = trainX.dot(w) + b
        support_vectors_id = (hinge_labels*scores < 1).ravel()
        svm_clf.support_vectors_ = trainX[support_vectors_id]      
        svs = svm_clf.support_vectors_
        plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
        
plot_svc_decision_boundary(SVM_clf, 0, trainX.shape[0])
z = lambda x,y: (-SVM_clf.intercept_[0]-SVM_clf.coef_[0][0]*x -SVM_clf.coef_[0][1]*y) / SVM_clf.coef_[0][2]

tmp = np.linspace(-1,1,30)
x,y = np.meshgrid(tmp,tmp)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
colors = ['b', 'r', 'g', 'c', 'm', 'y']
markers = ['.', ',', 'o', '^', '<', '>', '+', 'X', 'h', '*']
c = 0
m = 0
for i in range(id.shape[0]):
    if m == 9:
        m = 0
        c += 1
        
    ax.plot(trainX[:, 0][trainy==i], trainX[:, 1][trainy==i], 0, f'{colors[c]}{markers[m]}')

    m += 1
ax.plot_surface(x, y, z(x,y))
ax.view_init(30, 60)
plt.show()

#%%
# 3D decision boundary
def plot_3D_decision_function(w, b, x1_lim, x2_lim ):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot samples
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    markers = ['.', ',', 'o', '^', '<', '>', '+', 'X', 'h', '*']
    c = 0
    m = 0
    for i in range(id.shape[0]):
        if m == 9:
            m = 0
            c += 1
            
        ax.plot(trainX[:, 0][trainy==i], trainX[:, 1][trainy==i], 0, f'{colors[c]}{markers[m]}')

        m += 1
        
    # ax.plot(trainX[:, 0][trainy==0], trainX[:, 1][trainy==0], 0, "go")
    # ax.plot(trainX[:, 0][trainy==3], trainX[:, 1][trainy==3], 0, "b^")

    # Plot surface z=0
    x1s = np.linspace(x1_lim[0], x1_lim[1], 20)
    x2s = np.linspace(x2_lim[0], x2_lim[1], 20)
    x1, x2 = np.meshgrid(x1s, x2s)
    ax.plot_surface(x1, x2, np.zeros(x1.shape),  color="w", alpha=0.3) #, cstride=100, rstride=100)
                                                       
    # Plot decision boundary (and margins)
    m = 1 / np.linalg.norm(w)
    x2s_boundary = -x1s*(w[0]/w[1])-b/w[1]
    ax.plot(x1s, x2s_boundary, 0, "k-", linewidth=3, label=r"Decision boundary")
    x2s_margin_1 = -x1s*(w[0]/w[1])-(b-1)/w[1]
    x2s_margin_2 = -x1s*(w[0]/w[1])-(b+1)/w[1]         
    ax.plot(x1s, x2s_margin_1, 0, "k--", linewidth=1, label=r"Margins at h=1 and -1") 
    ax.plot(x1s, x2s_margin_2, 0, "k--", linewidth=1)
     
    # Plot decision function surface
    xs = np.c_[x1.ravel(), x2.ravel()]
    print(xs.shape, w.shape, x1.shape, x2.shape)
    # dec_func = (xs.dot(w) + b).reshape(x1.shape)      
    #ax.plot_wireframe(x1, x2, df, alpha=0.3, color="k")
    # ax.plot_surface(x1, x2, dec_func, alpha=0.3, color="r")
    # ax.text(4, 1, 3, "Decision function $h$", fontsize=12)       

    # ax.axis(x1_lim + x2_lim)
    # ax.set_xlabel(r"Petal length", fontsize=12, labelpad=10)
    # ax.set_ylabel(r"Petal width", fontsize=12, labelpad=10)
    # ax.set_zlabel(r"$h$", fontsize=14, labelpad=5)
    # ax.legend(loc="upper left", fontsize=12)    
    
w=SVM_clf.coef_[0]
b=SVM_clf.intercept_[0]
plot_3D_decision_function(w,b,x1_lim=[0, 5.5],x2_lim=[0, 2])
# plt.show()

#%%
# test lib
from mlxtend.plotting import plot_decision_regions, plot_learning_curves

# fig, ax = plt.subplots()
# # Decision region for feature 3 = 1.5
# value = 0.5
# # Plot training sample with feature 3 = 1.5 +/- 0.75
# width = 0.1
# filler_feature_values = {}
# filler_feature_ranges = {}
# for i in range(testX.shape[1]):
#     filler_feature_values[i] = value
#     filler_feature_ranges[i] = width
    
# plot_decision_regions(testX, testy, clf=SVM_clf,
#                     #   feature_index=(164, 128),
#                       filler_feature_values=filler_feature_values,
#                     #   filler_feature_ranges=filler_feature_ranges,
#                       legend=2, ax=ax)

# # ax.set_xlabel('Feature 1')
# # ax.set_ylabel('Feature 2')
# # ax.set_title('Feature 3 = {}'.format(value))

# # Adding axes annotations
# fig.suptitle('LinearSVC')
# plt.show()

# %%
# Learning curve
plot_learning_curves(trainX, trainy, testX, testy, SVM_clf)
plt.show()