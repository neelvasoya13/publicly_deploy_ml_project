import numpy as np
import pickle
import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from streamlit_option_menu import option_menu
dia_model=pickle.load(open('diabetes_trained_model_svc.sav','rb'))
diascalar_model=pickle.load(open("diabetes_scalr.sav",'rb'))
fakenews_model=pickle.load(open("fake_trained_model_svc.sav",'rb'))
fakenews_vector=pickle.load(open("fake_vectorizer.pkl",'rb'))
heart_model=pickle.load(open("heart_trained_model_lr.sav",'rb'))
loan_model=pickle.load(open("loan_trained_model_dt.sav",'rb'))
parkinson_model=pickle.load(open("parkinson_trained_model_rf1.sav",'rb'))
standard_parkinson_model=pickle.load(open("parkinson_standard_function.pkl",'rb'))
spam_model=pickle.load(open("spam_trained_model_svc.sav",'rb'))
vector_spam=pickle.load(open("spam_vectorrizer.pkl",'rb'))
titanic_model=pickle.load(open("titanic_trained_model_rf1.sav",'rb'))


def dia_pred(input_data):
    arr=np.asarray(input_data)
    arr_2d=arr.reshape(1,-1)
    arr_2d=diascalar_model.transform(arr_2d)
    prediction=dia_model.predict(arr_2d)
    if(prediction[0]):
        return "The Person is Diabetic"
    else:
        return  "The Person is not Diabetic"



port_stem=PorterStemmer()
def stemming(content):
    stemmed_content=re.sub('[^a-zA-Z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=" ".join(stemmed_content)
    return stemmed_content
def fake_pred(input_data):
    input_data = stemming(input_data)
    input_data = fakenews_vector.transform([input_data])  # Wrap input_data in a list
    prediction = fakenews_model.predict(input_data)
    if prediction[0]:
        return "This News is Fake"
    else:
        return "This News is Correct"
    

def Spam_pred(input_data):
    input_data = vector_spam.transform([input_data])  # Wrap input_data in a list
    prediction = spam_model.predict(input_data)
    if prediction[0]:
        return "This Mail is Fake"
    else:
        return "This Mail is Not Fake"
    

def tit_pred(input_data):
    if(input_data[1].lower()=="male"):
        input_data[1]=1
    else:
        input_data[1]=0
    if(input_data[6]=="S" or input_data[6]=='s'):
        input_data[6]=2
    elif(input_data[6]=='C' or input_data[6]=='c'):
        input_data[6]=0
    arr=np.asarray(input_data)
    arr_2d=arr.reshape(1,-1)
    prediction=titanic_model.predict(arr_2d)
    # print(prediction)
    if(prediction[0]):
        return "The Person is Survived"
    else:
        return  "The Person is not Survived"
    

def loan_pred(input_data):
    if(input_data[0].lower()=="male"):
        input_data[0]=1
    else:
        input_data[0]=0
    if(input_data[1].lower()=="yes"):
        input_data[1]=1
    else:
        input_data[1]=0
    if(input_data[2]=="3+"):
        input_data[2]=3
    if(input_data[4].lower()=="yes"):
        input_data[4]=1
    else:
        input_data[4]=0
    if(input_data[9]=="1.0"):
        input_data[9]=1
    else:
        input_data[9]=0
    if(input_data[10].lower()=="urban"):
        input_data[10]=2
    elif(input_data[10].lower()=="semiurban"):
        input_data[10]=1
    else:
        input_data[10]=0
    
    arr=np.asarray(input_data)
    arr_2d=arr.reshape(1,-1)
    prediction=loan_model.predict(arr_2d)
    print(prediction)
    if(prediction[0]):
        return "The Person is Eligible"
    else:
        return  "The Person is not Eligible"

def heart_pred(input_data):
    input_data[0] = float(input_data[0])  
    input_data[1] = 1 if input_data[1].lower() == "male" else 0 
    input_data[2] = int(input_data[2])  
    input_data[3] = float(input_data[3])  
    input_data[4] = float(input_data[4])  
    input_data[5] = 1 if input_data[5].lower() == "yes" else 0  
    input_data[6] = int(input_data[6])  
    input_data[7] = float(input_data[7]) 
    input_data[8] = int(input_data[8])
    input_data[9] = float(input_data[9]) 
    input_data[10] = int(input_data[10])  
    input_data[11] = int(input_data[11])  
    input_data[12] = int(input_data[12])  
    
    arr=np.asarray(input_data)
    arr_2d=arr.reshape(1,-1)
    prediction=heart_model.predict(arr_2d)
    print(prediction)
    if(prediction[0]):
        return "The Person has Heart Disease"
    else:
        return  "The Person has No Heart Disease"
    

def park_pred(input_data):
    arr=np.asarray(input_data)
    arr_2d=arr.reshape(1,-1)
    arr_2d=standard_parkinson_model.transform(arr_2d)
    prediction=parkinson_model.predict(arr_2d)
    # print(prediction)
    if(prediction[0]):
        return "The Person has Parkinsons"
    else:
        return  "The Person has not Parkinsons"
    

with st.sidebar:
    selected=option_menu('Multiple Prediction Systems',['Diabetes Prediction','Fake News Prediction'
                                                        ,'Heart Diseases Prediction','Loan Status Prediction','Parkinson Diseases Prediction'
                                                        ,'Spam Mail Detection','Titanic Survival Prediction'],
                                                        icons=['activity','newspaper','heart','archive','person','envelope','tsunami'],default_index=0)
    
if(selected=='Diabetes Prediction'):
    st.title('Diabetes Predicting System')
    preg=st.text_input('No. of Pregnancies')
    gl=st.text_input('Glucose Level')
    bp=st.text_input("Blood Pressure Level")
    stv=st.text_input("Skin Thickness Value")
    ins=st.text_input("Insulin Level")
    bmi=st.text_input("BMI Value")
    dpf=st.text_input("Diabetes Pedigree Function Value")
    age=st.text_input("Age")
    
    diagnosis=""
    if st.button('Test'):
        diagnosis=dia_pred([preg,gl,bp,stv,ins,bmi,dpf,age])

    st.success(diagnosis)


if(selected=='Fake News Prediction'):
    st.title('Fake News Predicting System')
    Title=st.text_input('Title of news')
    author=st.text_input('Author of news')
    text=st.text_input("Text of News")
    ans1=""
    if st.button('Test'):
        ans1=fake_pred(Title+author)

    st.success(ans1)


if(selected=='Heart Diseases Prediction'):
    st.title('Heart Diseases Predicting System')

    agee=st.text_input("Age")
    Gender=st.text_input("Gender")
    # male->1 female->0
    cp=st.text_input("chest pain type: 0 to 3")
    trstbp =st.text_input("resting blood pressure")
    chol=st.text_input("serum cholestoral in mg/dl")
    fbs=st.text_input("fasting blood sugar > 120 mg/dl")
    # yes->1 no->0
    restecg=st.text_input("resting electrocardiographic results (values 0,1,2)")
    thalach	 =st.text_input("maximum heart rate achieved")
    exang =st.text_input("exercise induced angina")
    oldpeak=st.text_input("ST depression induced by exercise relative to rest")
    slope=st.text_input("the slope of the peak exercise ST segment")
    ca=st.text_input("number of major vessels (0-3) colored by flourosopy")
    thal=st.text_input("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")

    ans6=""
    if st.button('Test'):
        ans6=heart_pred([agee,Gender,cp,trstbp,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])

    st.success(ans6)

if(selected=='Loan Status Prediction'):
    st.title('Loan Status Predicting System')

    Gender=st.text_input("Gender")
    # male->1 female->0
    Married=st.text_input("Married: Yes or No")
    # yes->1 no->0
    Dependents=st.text_input("Dependents: 1 or 2 or 3+")
    # replace->3+->3
    Education =st.text_input("Education: Graduate->0 And Not Graduate->1")
    Self_Employed=st.text_input("Self_Employed: Yes or No")
    # yes->1 no->0
    ApplicantIncome=st.text_input("ApplicantIncome")
    CoapplicantIncome =st.text_input("CoapplicantIncome")
    LoanAmount =st.text_input("LoanAmount")
    Loan_Amount_Term =st.text_input("Loan_Amount_Term")
    Credit_History=st.text_input("Credit_History: 1.0 or 0.0")
    # 1.0->1 0.0->0
    Property_Area=st.text_input("Property_Area: Urban or Semiurban or Rural")
    # rural->0 urban->2 semiurban->1

    ans5=""
    if st.button('Test'):
        ans5=loan_pred([Gender, Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area])

    st.success(ans5)
    
if(selected=='Parkinson Diseases Prediction'):
    st.title('Parkinson Diseases Predicting System')

    MDVP_Fo_Hz=st.text_input('MDVP:Fo(Hz)')
    MDVP_Fhi_Hz=st.text_input('MDVP:Fhi(Hz)')
    MDVP_Flo_Hz=st.text_input("MDVP:Flo(Hz)")
    MDVP_Jitter_per=st.text_input("MDVP:Jitter(%)")
    MDVP_Jitter_ABS=st.text_input("MDVP:Jitter(ABS)")
    MDVP_Rap=st.text_input("MDVP_RAP")
    MDVP_Ppq=st.text_input("MDVP:PPQ")
    Jitter_DDP=st.text_input("Jitter:DDP")
    MDVP_Shimmer=st.text_input("MDVP:Shimmer")
    MDVP_Shimmer_dB=st.text_input("MDVP:Shimmer(dB)")
    Shimmer_APQ3=st.text_input("Shimmer:APQ3")
    Shimmer_APQ5=st.text_input(" Shimmer:APQ5")
    MDVP_Apq=st.text_input("MDVP:APQ")
    Shimmer_DDA=st.text_input("Shimmer:DDA")
    Nhr =st.text_input("NHR ")
    Hnr=st.text_input("HNR")
    Rpde=st.text_input("RPDE")
    Dfa =st.text_input("DFA")
    spread1  =st.text_input("spread1 ")
    spread2  =st.text_input("spread2 ")
    d_2 =st.text_input("D2")
    Ppe =st.text_input("PPE")

    ans4=""
    if st.button('Test'):
        ans4=park_pred([MDVP_Fo_Hz,MDVP_Fhi_Hz,MDVP_Flo_Hz,MDVP_Jitter_per,MDVP_Jitter_ABS,MDVP_Rap,MDVP_Ppq,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_Apq,Shimmer_DDA,Nhr,Hnr,Rpde,Dfa,spread1,spread2,d_2,Ppe])

    st.success(ans4)


if(selected=='Spam Mail Detection'):
    st.title('Spam Mail Detection System')
    Text_mail=st.text_input('Text of Mail')
    ans2=""
    if st.button('Test'):
        ans2=Spam_pred(Text_mail)

    st.success(ans2)


if(selected=='Titanic Survival Prediction'):
    st.title('Titanic Survival Predicting System')
    pclass=st.text_input('Ticket class')
    gender=st.text_input('Sex')
    ag=st.text_input("Age in years")
    sibsp=st.text_input("No. of siblings / spouses aboard the Titanic")
    parch=st.text_input("No. of parents / children aboard the Titanic")
    fare=st.text_input("Passenger fare")
    embarked=st.text_input("Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton")
    ans3=""
    if st.button('Test'):
        ans3=tit_pred([pclass,gender,ag,sibsp,parch,fare,embarked])

    st.success(ans3)
