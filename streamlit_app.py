import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings

# 处理版本警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 加载模型和标准化器
model_path = r"D:\WEB汇总\AMI-AF WEB\glmnet_model.pkl"
scaler_path = r"D:\WEB汇总\AMI-AF WEB\scaler.pkl"

with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# 定义特征名称
feature_names = [
    'ACEI/ARB',
    'aspirin',
    'reperfusion therapy',
    'Neu',
    'Hb',
    'Scr',
    'P'
]

# 创建 Web 应用的标题
st.title('Machine learning-based models to predict one-year mortality in patients with acute myocardial infarction combined with atrial fibrillation.')

# 添加介绍部分
st.markdown("""
## Introduction
This web-based calculator was developed based on the Glmnet model with an AUC of 0.87 and a Brier score of 0.178. Users can obtain the 1-year risk of death for a given case by simply selecting the parameters and clicking on the 'Predict' button.
""")

# 创建输入表单
with st.form("prediction_form"):
    acei_arb = st.selectbox('ACEI/ARB', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='ACEI/ARB')
    aspirin = st.selectbox('Aspirin', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='aspirin')
    reperfusion_therapy = st.selectbox('Reperfusion Therapy', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='reperfusion therapy')
    Neu = st.slider('Neu (10^9/L)', min_value=0, max_value=30, value=5, step=1, key='Neu')
    Hb = st.slider('Hb (g/L)', min_value=0, max_value=300, value=150, step=1, key='Hb')
    Scr = st.slider('Scr (μmol/L)', min_value=0, max_value=1300, value=100, step=10, key='Scr')
    P = st.slider('P (mmHg)', min_value=20, max_value=200, value=110, step=1, key='P')

    predict_button = st.form_submit_button("Predict")

# 处理表单提交
if predict_button:
    data = {
        "ACEI/ARB": acei_arb,
        "aspirin": aspirin,
        "reperfusion therapy": reperfusion_therapy,
        "Neu": Neu,
        "Hb": Hb,
        "Scr": Scr,
        "P": P
    }

    # 将输入数据转换为 DataFrame，并使用正确的特征名称
    data_df = pd.DataFrame([data], columns=feature_names)

    # 使用加载的标准化器对数据进行标准化
    data_scaled = scaler.transform(data_df)

    # 进行预测
    prediction = model.predict_proba(data_scaled)[:, 1][0]  # 获取类别为 1 的概率
    st.write(f'Prediction: {prediction * 100:.2f}%')  # 将概率转换为百分比
