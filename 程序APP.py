import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('rf-0.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "admission_type": {"type": "categorical", "options": [0, 2]},
    "weight": {"type": "numerical", "min": 30.0, "max": 300, "default": 50.0},
    "renal_disease": {"type": "categorical", "options": [0, 1]},
    "sofa": {"type": "numerical", "min": 0.0, "max": 24.0, "default": 6.0},
    "sirs": {"type": "categorical", "options": [0, 4]},
    "aki_score":{"type": "categorical", "options": [0, 3]},
    "InvasiveVent": {"type": "categorical", "options": [0, 1]},
    "custom_Vasoactive Drugs": {"type": "categorical", "options": [0, 1]},
    "temperature_first": {"type": "numerical", "min": 34.0, "max": 42.0, "default": 36.5},
    "heart_rate_first": {"type": "numerical", "min": 30.0, "max": 250.0, "default": 80.0},
    "resp_rate_first": {"type": "numerical", "min": 6.0, "max": 60.0, "default": 18.0},
    "sbp_ni_first": {"type": "numerical", "min": 30.0, "max": 270.0, "default": 109.0},
    "spo2_first": {"type": "numerical", "min": 40.0, "max": 100.0, "default": 94.0},
    "pco2_first": {"type": "numerical", "min": 10.0, "max": 130.0, "default": 30.0},
    "wbc_first": {"type": "numerical", "min": 0.1, "max": 100.0, "default": 10.0},
    "rdw_first": {"type": "numerical", "min": 10.0, "max": 30.0, "default": 20.0},
    "platelet_first": {"type": "numerical", "min": 1.0, "max": 3000.0, "default": 400.0},
    "calcium_total_first": {"type": "numerical", "min": 5.0, "max": 15.0, "default": 8.0},
    "ptt_first": {"type": "numerical", "min": 10.0, "max": 150.0, "default": 80.0},
    "glucose_first": {"type": "numerical", "min": 30.0, "max": 1000.0, "default": 80.0},
    "potassium_first": {"type": "numerical", "min": 2.0, "max": 7.5, "default": 5.0},
    "hemoglobin_first": {"type": "numerical", "min": 5.0, "max": 22.0, "default": 10.0},
    "magnesium_first": {"type": "numerical", "min": 0.5, "max": 10.0, "default": 5.0},
    "phosphate_first": {"type": "numerical", "min": 1.0, "max": 20.0, "default": 7.4},
    # "gcs": {"type": "numerical", "min": 3.0, "max": 15.0, "default": 15}
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
