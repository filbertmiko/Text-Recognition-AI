import os
import io
import time
import pickle
import requests
import cv2
import numpy as np
from scipy import ndimage
from skimage.transform import resize
from imageio import imread, imsave
from skimage import img_as_float, color, exposure
from skimage.feature import peak_local_max, hog
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import LinearSVC
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import seaborn as sn
import keras
import warnings
from letters_extractor import Extract_Letters
from sklearn import metrics

extract = Extract_Letters()
start_time = time.time()

test_file_path = './ocr/testing/alchemist2.png'

knnC = pickle.load(open('knn.detector', 'rb'))
clf = pickle.load(open('svm.detector', 'rb'))
mlp =pickle.load(open('mlp.detector', 'rb'))

recognised_SVM = []
recognised_KNN = []
recognised_MLP = []
y_score = []

print("Extracting File...")
#extract test image so we can get the letters
letters = extract.extractFile(test_file_path)

print("Recognised text: ")
for letter in letters:
    letter = resize(letter, (200,200))
    hog_features = hog(letter, orientations = 8, pixels_per_cell=(16,16), cells_per_block=(1,1))
    result_knn = knnC.predict(hog_features.reshape(1,-1))
    result_svm = clf.predict(hog_features.reshape(1,-1))
    result_mlp = mlp.predict(hog_features.reshape(1,-1))

    recognised_SVM.append(result_svm)
    recognised_KNN.append(result_knn)
    recognised_MLP.append(result_mlp)

#truth = 'whenwestrivetobecomebetterthanweareeverythingaroundusbecomesbetter'
truth = 'whenyouwantsomethingalltheuniverseconspiresinhelpingyoutoachieveit'
#truth = 'Thesecretofhappinessistoseethemarveloftheworldandneverforgetthedropsofoilinthespoon'
#truth = 'InaregulatorydocumentfiledwiththeSECtodayAdobeannouncedthatchieftechnologyofficerKevinLynchwouldbetakinghisleaveasofthiscomingFridayOnMarch182013KevinLynchresignedfromhispositionasexecutivevicepresidentchieftechnologyofficerofAdobeSystemsIncorporatedeffectiveMarch222013topursueotheropportunitiesthefilingreadsLynchwhocametothecompanyin2005duringitsacquisitionofMacromedialedAdobeschargeintosomeofthemorecuttingedgeareasoftechnologyincludingmultiscreencomputingcloudcomputingandsocialmediaForagesAdobehadbeenrootedintheworkflowsoftheprintdesigncommunityLynchwasresponsibleforthecompanysshiftintowebpublishingstartingwithDreamweaverHealsooversawAdobesresearchandexperiencedesignteamsandwasasAdobeputsitinchargeofshapingAdobeslongtermtechnologyvisionandfocusinginnovationacrossthecompanyduringatransformativetimeRumorsaroundthewebhavepinpointedAppleasLynchsnextdestinationanditsnotanentirelynonsensicalrumorAdobestransitiontowebtechnologieshasbeennothingifnotprofitableApplestillagiantinconsumerhardwarecoulduseahelpinghandwhenitcomestomultiscreenfluiditysocialmediaandwebbasedsoftware'
truth = truth.lower()

y_true = []
for i in range(len(truth)):
    y_true.append(truth[i])

#print("Extracted: " + str(len(recognised)))
print("# of letter in the test image: " + str(len(truth)))

SVM_accuracy = metrics.accuracy_score(y_true, recognised_SVM)*100
KNN_accuracy = metrics.accuracy_score(y_true, recognised_KNN)*100
MLP_accuracy = metrics.accuracy_score(y_true, recognised_MLP)*100

model = ['SVM', 'KNN', 'MLP']
pos = np.arange(len(model))
pred_result = [SVM_accuracy, KNN_accuracy, MLP_accuracy]
print(pred_result)

fig, ax = plt.subplots()

pred_bar = plt.bar(pos, pred_result, width=0.4, label="Prediction Result")
#mean_bar = plt.bar(pos+0.2, mean, width=0.4, label="Mean")
plt.xticks(pos, model)
plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy %")

for pred in pred_bar:
    height = pred.get_height()
    ax.text(pred.get_x() + pred.get_width()/2., 0.99*height,'%d' % int(height) + "%", ha='center', va='bottom')

#for mean in mean_bar:
#    height = mean.get_height()
#    ax.text(mean.get_x() + mean.get_width()/2., 0.99*height,'%d' % int(height) + "%", ha='center', va='bottom')

plt.legend()
plt.show()


print("SVM Classification Report")
print("Accuracy Score: {}".format(SVM_accuracy))
print("F1 Score: {}".format(metrics.f1_score(y_true, recognised_SVM, average='micro') * 100))
print("Precision Score: {}".format(metrics.precision_score(y_true, recognised_SVM, average='micro') * 100))
print("Recall Score: {}".format(metrics.recall_score(y_true, recognised_SVM, average='micro') * 100))
print('')

print(metrics.classification_report(y_true, recognised_SVM))

print("KNN Classification Report")
print("Accuracy Score: {}".format(KNN_accuracy))
print("F1 Score: {}".format(metrics.f1_score(y_true, recognised_KNN, average='micro') * 100))
print("Precision Score: {}".format(metrics.precision_score(y_true, recognised_KNN, average='micro') * 100))
print("Recall Score: {}".format(metrics.recall_score(y_true, recognised_KNN, average='micro') * 100))
print('')

print(metrics.classification_report(y_true, recognised_KNN))

print("MLP Classification Report")
print("Accuracy Score: {}".format(MLP_accuracy))
print("F1 Score: {}".format(metrics.f1_score(y_true, recognised_MLP, average='micro') * 100))
print("Precision Score: {}".format(metrics.precision_score(y_true, recognised_MLP, average='micro') * 100))
print("Recall Score: {}".format(metrics.recall_score(y_true, recognised_MLP, average='micro') * 100))
print('')

print(metrics.classification_report(y_true, recognised_SVM))
print('')
print("KNN Classification Report")
print(metrics.classification_report(y_true, recognised_KNN))
print('')
print("MLP Classification Report")
print(metrics.classification_report(y_true, recognised_MLP))
print('')

print("SVM Classification Report")
print("Confusion matrix for SVM: ")
cm_SVM = metrics.confusion_matrix(y_true, recognised_SVM)
print("Confusion matrix for KNN: ")
cm_KNN = metrics.confusion_matrix(y_true, recognised_KNN)
print("Confusion matrix for MLP: ")
cm_MLP = metrics.confusion_matrix(y_true, recognised_MLP)

#Confusion Matrix
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(cm_SVM, annot=True)
plt.title("SVM")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(cm_KNN, annot=True)
plt.title("KNN")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(cm_MLP, annot=True)
plt.title("MLP")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
