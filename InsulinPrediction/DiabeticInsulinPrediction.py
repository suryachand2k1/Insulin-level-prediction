from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression

main = tkinter.Tk()
main.title("Machine learning Models for diagnosis of the diabetic patient and predicting insulin dosage")
main.geometry("1300x1200")

global filename

global diabetic_classifier
global insulin_classifier
global diabetic_X_train, diabetic_X_test, diabetic_y_train, diabetic_y_test
global insulin_X_train, insulin_X_test, insulin_y_train, insulin_y_test
global diabetic_dataset
global insulin_dataset
global diabetic_X
global diabetic_Y
global insulin_X
global insulin_Y
global gbc_acc
global lr_acc

def upload():
    global filename
    global diabetic_X_train, diabetic_X_test, diabetic_y_train, diabetic_y_test
    global insulin_X_train, insulin_X_test, insulin_y_train, insulin_y_test
    global diabetic_dataset
    global insulin_dataset
    filename = filedialog.askdirectory(initialdir = ".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'Diabetes & insulin dataset loaded\n')
    diabetic_dataset = pd.read_csv('Dataset/diabetes.csv')
    diabetic_dataset.fillna(0, inplace = True)

    insulin_dataset = pd.read_csv('Dataset/insulin_dosage.csv',sep='\t')
    insulin_dataset.fillna(0, inplace = True)
    text.insert(END,"Diabetes Dataset\n")
    text.insert(END,str(diabetic_dataset.head())+"\n\n")
    text.insert(END,"Insulin Dataset\n")
    text.insert(END,str(insulin_dataset.head())+"\n\n")
                
    insulin_dataset.drop(['date','time'], axis = 1,inplace=True)
    code = insulin_dataset['code']
    value = insulin_dataset['value']
    insulin_dataset = diabetic_dataset

    sns.pairplot(data=diabetic_dataset, hue = 'Outcome')
    plt.show()
    
def preprocess():
    text.delete('1.0', END)
    global diabetic_X
    global diabetic_Y
    global insulin_X
    global insulin_Y
    global diabetic_X_train, diabetic_X_test, diabetic_y_train, diabetic_y_test
    global insulin_X_train, insulin_X_test, insulin_y_train, insulin_y_test
    global diabetic_dataset
    global insulin_dataset
    diabetic_dataset = diabetic_dataset.values
    cols = diabetic_dataset.shape[1]-1
    diabetic_X = diabetic_dataset[:,0:cols]
    diabetic_Y = diabetic_dataset[:,cols]
    diabetic_X = normalize(diabetic_X)
    print(diabetic_X.shape)
    print(diabetic_Y)
    
    insulin_Y = insulin_dataset['Insulin']
    insulin_Y = np.asarray(insulin_Y)
    insulin_dataset.drop(['Insulin','Outcome'], axis = 1,inplace=True)
    insulin_dataset = insulin_dataset.values
    insulin_X = insulin_dataset[:,0:insulin_dataset.shape[1]]
    insulin_X = normalize(insulin_X)
    print(insulin_X.shape)

    diabetic_X_train, diabetic_X_test, diabetic_y_train, diabetic_y_test = train_test_split(diabetic_X, diabetic_Y, test_size=0.2,random_state=0)
    insulin_X_train, insulin_X_test, insulin_y_train, insulin_y_test = train_test_split(insulin_X, insulin_Y, test_size=0.2,random_state=0)
    text.insert(END,"Total records available in both datasets : "+str(diabetic_X.shape[0])+"\n")
    text.insert(END,"Total records used to train machine learning algorithm (80%) : "+str(diabetic_X_train.shape[0])+"\n")
    text.insert(END,"Total records used to test machine learning algorithm (20%)  : "+str(diabetic_X_test.shape[0])+"\n")
    

def gradientBoosting():
    global diabetic_classifier
    global diabetic_X_train, diabetic_X_test, diabetic_y_train, diabetic_y_test
    global diabetic_X
    global diabetic_Y
    global gbc_acc
    text.delete('1.0', END)

    diabetic_classifier = XGBClassifier() #object of extreme gradient boosting
    diabetic_classifier.fit(diabetic_X, diabetic_Y)#training xgb on daatset
    predict = diabetic_classifier.predict(diabetic_X_test)
    gbc_acc = accuracy_score(predict, diabetic_y_test)
    text.insert(END,"Graident Boosting Diabetes Prediction Accuracy : "+str(gbc_acc)+"\n\n")

    
    

def linearRegression():
    global insulin_classifier
    global insulin_X_train, insulin_X_test, insulin_y_train, insulin_y_test
    global insulin_X
    global insulin_Y
    global lr_acc

    insulin_classifier = LinearRegression()#object of linear regression
    insulin_classifier.fit(insulin_X, insulin_Y)#training linear regression on insulin dataset
    predict = insulin_classifier.predict(insulin_X_test)
    lr_acc = 1 - insulin_classifier.score(insulin_X, insulin_Y)
    text.insert(END,"Logistic Regression Insulin Dosage Prediction Accuracy : "+str(lr_acc)+"\n\n")
    
def graph():
    height = [gbc_acc,lr_acc]
    bars = ("Gradient Boosting Accuracy","Linear Regression Accuracy")
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
            

def predict():
    global diabetic_classifier
    global insulin_classifier
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    diabetes_test = pd.read_csv(filename)
    insulin_test = pd.read_csv(filename)
    insulin_test.drop(['Insulin'], axis = 1,inplace=True)
    
    diabetes_test = diabetes_test.values[:, 0:diabetes_test.shape[1]]
    insulin_test = insulin_test.values[:, 0:insulin_test.shape[1]]

    print(diabetes_test.shape)
    print(insulin_test.shape)

    diabetes_test = normalize(diabetes_test)
    insulin_test = normalize(insulin_test)
    
    diabetes_prediction = diabetic_classifier.predict(diabetes_test)
    insulin_prediction = insulin_classifier.predict(insulin_test)
    print(diabetes_prediction)
    print(insulin_prediction)
    for i in range(len(diabetes_test)):
        if diabetes_prediction[i] == 0:
            text.insert(END,"X=%s, Predicted = %s" % (diabetes_test[i], 'No Diabetes Detected')+"\n\n")
        if diabetes_prediction[i] == 1:
            text.insert(END,"X=%s, Predicted = %s" % (diabetes_test[i], 'Diabetes Detected & Required Insulin Dosage : '+str(int(insulin_prediction[i])))+"\n\n")
                        
               

def close():
    main.destroy()
    
font = ('times', 16, 'bold')
title = Label(main, text='Machine learning Models for diagnosis of the diabetic patient and predicting insulin dosage')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Diabetic Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=700,y=200)
preprocessButton.config(font=font1) 

gbButton = Button(main, text="Run Gradient Boosting Algorithm", command=gradientBoosting)
gbButton.place(x=700,y=250)
gbButton.config(font=font1) 

lrButton = Button(main, text="Run Logistic Regression Algorithm", command=linearRegression)
lrButton.place(x=700,y=300)
lrButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=700,y=350)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Diabetes & Insulin Dosage", command=predict)
predictButton.place(x=700,y=400)
predictButton.config(font=font1)


closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=700,y=450)
closeButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
