import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def generate_realistic_health_dataset(n_samples=50000):
    """生成更符合醫學實際的健康數據集"""
    np.random.seed(42)
    
    print("🏥 基於醫學指南生成健康數據...")
    
    # 生成基本特徵
    age = np.random.randint(18, 85, n_samples)
    gender = np.random.choice([0, 1], n_samples)  # 0=男, 1=女
    height = np.where(gender == 0, 
                     np.random.normal(175, 8, n_samples),  # 男性身高
                     np.random.normal(162, 7, n_samples))  # 女性身高
    
    # 基於年齡調整體重分布
    base_weight = np.where(gender == 0, 75, 62)  # 基準體重
    age_factor = (age - 25) * 0.3  # 年齡因子
    weight = np.random.normal(base_weight + age_factor, 12, n_samples)
    weight = np.clip(weight, 40, 150)
    
    # 計算BMI
    bmi = weight / ((height / 100) ** 2)
    
    # === 更符合生理學的心率生成 ===
    # 基礎心率：男性略低，隨年齡變化
    base_hr = np.where(gender == 0, 68, 72)  # 男女基礎心率差異
    age_hr_effect = np.where(age < 30, 0, (age - 30) * 0.15)  # 年齡效應
    fitness_effect = np.where(bmi < 25, -5, (bmi - 25) * 1.5)  # 體重效應
    
    heart_rate = base_hr + age_hr_effect + fitness_effect + np.random.normal(0, 8, n_samples)
    heart_rate = np.clip(heart_rate, 45, 150)
    
    # === 更符合醫學的血氧生成 ===
    base_spo2 = 98.5
    age_spo2_effect = np.where(age > 65, -(age - 65) * 0.08, 0)  # 老年人血氧下降
    altitude_effect = np.random.normal(-0.5, 1, n_samples)  # 環境因素
    health_effect = np.where(bmi > 30, -1.5, 0)  # 肥胖影響
    
    spo2 = base_spo2 + age_spo2_effect + altitude_effect + health_effect
    spo2 = np.clip(spo2, 85, 100)
    
    # === 體溫生成 ===
    base_temp = 36.6
    individual_variation = np.random.normal(0, 0.3, n_samples)
    daily_variation = np.random.normal(0, 0.2, n_samples)
    
    temperature = base_temp + individual_variation + daily_variation
    temperature = np.clip(temperature, 35.0, 39.0)
    
    # === 心率變異性（HRV）===
    base_hrv = 35
    age_hrv_effect = -(age - 25) * 0.4  # 隨年齡下降
    fitness_hrv_effect = np.where(bmi < 25, 10, -(bmi - 25) * 2)
    
    hrv = base_hrv + age_hrv_effect + fitness_hrv_effect + np.random.normal(0, 5, n_samples)
    hrv = np.clip(hrv, 10, 70)
    
    # 創建特徵矩陣
    X = np.column_stack([
        heart_rate,
        spo2,
        temperature,
        age,
        gender,
        bmi,
        hrv
    ])
    
    # === 基於醫學指南的風險計算 ===
    
    # 1. 心血管風險 (基於ACC/AHA指南)
    cardiovascular_risk = calculate_cardiovascular_risk(heart_rate, age, gender, bmi, spo2)
    
    # 2. 呼吸系統風險
    respiratory_risk = calculate_respiratory_risk(spo2, age, heart_rate, bmi)
    
    # 3. 代謝風險
    metabolic_risk = calculate_metabolic_risk(bmi, age, gender, heart_rate)
    
    # 4. 體溫調節風險
    thermal_risk = calculate_thermal_risk(temperature, age, bmi)
    
    # 5. 綜合風險（非簡單平均，而是加權組合）
    overall_risk = calculate_overall_risk(cardiovascular_risk, respiratory_risk, 
                                        metabolic_risk, thermal_risk, age)
    
    # 組合所有風險（確保在0-1範圍內）
    y = np.column_stack([
        cardiovascular_risk,
        respiratory_risk,
        metabolic_risk,
        thermal_risk,
        overall_risk
    ])
    
    # 最終檢查：確保所有值在0-1範圍內
    y = np.clip(y, 0, 1)
    
    print(f"✅ 生成 {n_samples} 筆醫學合理的健康數據")
    print(f"📊 風險分布統計：")
    print(f"   心血管風險: {np.mean(cardiovascular_risk):.3f} ± {np.std(cardiovascular_risk):.3f}")
    print(f"   呼吸系統風險: {np.mean(respiratory_risk):.3f} ± {np.std(respiratory_risk):.3f}")
    print(f"   代謝風險: {np.mean(metabolic_risk):.3f} ± {np.std(metabolic_risk):.3f}")
    print(f"   體溫調節風險: {np.mean(thermal_risk):.3f} ± {np.std(thermal_risk):.3f}")
    print(f"   綜合風險: {np.mean(overall_risk):.3f} ± {np.std(overall_risk):.3f}")
    
    return X, y

def calculate_cardiovascular_risk(heart_rate, age, gender, bmi, spo2):
    """基於醫學指南計算心血管風險"""
    risk = np.zeros_like(heart_rate)
    
    # 心率風險
    risk += np.where(heart_rate > 100, 0.3, 0)  # 心跳過速
    risk += np.where(heart_rate < 50, 0.25, 0)  # 心跳過緩
    risk += np.where((heart_rate > 90) & (heart_rate <= 100), 0.1, 0)  # 輕度過速
    
    # 年齡風險
    risk += np.where(age > 65, 0.2, 0)
    risk += np.where(age > 75, 0.15, 0)  # 額外風險
    
    # 性別風險（男性心血管風險較高）
    risk += np.where(gender == 0, 0.1, 0)
    
    # BMI風險
    risk += np.where(bmi > 30, 0.25, 0)  # 肥胖
    risk += np.where(bmi > 35, 0.15, 0)  # 重度肥胖額外風險
    risk += np.where((bmi > 25) & (bmi <= 30), 0.1, 0)  # 過重
    
    # 血氧影響心血管
    risk += np.where(spo2 < 95, 0.15, 0)
    
    # 添加個體差異
    risk += np.random.normal(0, 0.05, len(heart_rate))
    
    return np.clip(risk, 0, 1)

def calculate_respiratory_risk(spo2, age, heart_rate, bmi):
    """計算呼吸系統風險"""
    risk = np.zeros_like(spo2)
    
    # 血氧飽和度風險
    risk += np.where(spo2 < 90, 0.6, 0)  # 嚴重缺氧
    risk += np.where((spo2 >= 90) & (spo2 < 95), 0.3, 0)  # 輕度缺氧
    risk += np.where((spo2 >= 95) & (spo2 < 97), 0.1, 0)  # 邊緣值
    
    # 年齡影響
    risk += np.where(age > 70, 0.15, 0)
    
    # 肥胖影響呼吸
    risk += np.where(bmi > 35, 0.2, 0)
    risk += np.where((bmi > 30) & (bmi <= 35), 0.1, 0)
    
    # 心率過快可能影響氧合
    risk += np.where(heart_rate > 110, 0.1, 0)
    
    # 個體差異
    risk += np.random.normal(0, 0.03, len(spo2))
    
    return np.clip(risk, 0, 1)

def calculate_metabolic_risk(bmi, age, gender, heart_rate):
    """計算代謝風險"""
    risk = np.zeros_like(bmi)
    
    # BMI風險（主要因子）
    risk += np.where(bmi < 18.5, 0.2, 0)  # 體重過輕
    risk += np.where((bmi >= 25) & (bmi < 30), 0.2, 0)  # 過重
    risk += np.where((bmi >= 30) & (bmi < 35), 0.4, 0)  # 肥胖
    risk += np.where(bmi >= 35, 0.6, 0)  # 重度肥胖
    
    # 年齡因素
    risk += np.where(age > 45, 0.1, 0)
    risk += np.where(age > 60, 0.1, 0)  # 額外風險
    
    # 性別因素（女性更年期後風險增加）
    risk += np.where((gender == 1) & (age > 50), 0.1, 0)
    
    # 靜息心率影響代謝
    risk += np.where(heart_rate > 80, 0.1, 0)
    
    # 個體差異
    risk += np.random.normal(0, 0.04, len(bmi))
    
    return np.clip(risk, 0, 1)

def calculate_thermal_risk(temperature, age, bmi):
    """計算體溫調節風險"""
    risk = np.zeros_like(temperature)
    
    # 體溫偏差風險
    temp_deviation = np.abs(temperature - 36.6)
    risk += np.where(temp_deviation > 1.5, 0.4, 0)  # 嚴重偏差
    risk += np.where((temp_deviation > 1.0) & (temp_deviation <= 1.5), 0.2, 0)  # 中度偏差
    risk += np.where((temp_deviation > 0.5) & (temp_deviation <= 1.0), 0.1, 0)  # 輕度偏差
    
    # 發燒風險
    risk += np.where(temperature > 38.0, 0.3, 0)
    risk += np.where(temperature > 39.0, 0.5, 0)  # 高燒額外風險
    
    # 體溫過低風險
    risk += np.where(temperature < 36.0, 0.3, 0)
    
    # 年齡影響體溫調節
    risk += np.where(age > 75, 0.1, 0)
    risk += np.where(age < 25, 0.05, 0)  # 年輕人調節能力稍差
    
    # 肥胖影響體溫調節
    risk += np.where(bmi > 35, 0.1, 0)
    
    # 個體差異
    risk += np.random.normal(0, 0.02, len(temperature))
    
    return np.clip(risk, 0, 1)

def calculate_overall_risk(cardio_risk, resp_risk, metab_risk, thermal_risk, age):
    """計算綜合健康風險（非簡單平均）"""
    # 加權組合，心血管和代謝風險權重較高
    weights = np.array([0.35, 0.25, 0.3, 0.1])  # 心血管、呼吸、代謝、體溫
    
    weighted_risk = (cardio_risk * weights[0] + 
                    resp_risk * weights[1] + 
                    metab_risk * weights[2] + 
                    thermal_risk * weights[3])
    
    # 年齡調整因子
    age_multiplier = np.where(age > 65, 1.2, 1.0)
    age_multiplier = np.where(age > 75, 1.3, age_multiplier)
    
    overall_risk = weighted_risk * age_multiplier
    
    # 非線性調整：多重風險的交互作用
    high_risk_count = ((cardio_risk > 0.3).astype(int) + 
                      (resp_risk > 0.3).astype(int) + 
                      (metab_risk > 0.3).astype(int) + 
                      (thermal_risk > 0.3).astype(int))
    
    # 多重風險交互作用
    interaction_bonus = high_risk_count * 0.1
    overall_risk += interaction_bonus
    
    return np.clip(overall_risk, 0, 1)

def validate_dataset(X, y):
    """驗證數據集的合理性"""
    print("\n🔍 數據集驗證：")
    
    feature_names = ['心率', '血氧', '體溫', '年齡', '性別', 'BMI', 'HRV']
    risk_names = ['心血管風險', '呼吸風險', '代謝風險', '體溫風險', '綜合風險']
    
    # 檢查特徵範圍
    print("📊 特徵統計：")
    for i, name in enumerate(feature_names):
        print(f"   {name}: {X[:, i].min():.1f} - {X[:, i].max():.1f} (平均: {X[:, i].mean():.1f})")
    
    # 檢查風險分布
    print("\n🎯 風險分布：")
    for i, name in enumerate(risk_names):
        low_risk = np.sum(y[:, i] < 0.3) / len(y) * 100
        med_risk = np.sum((y[:, i] >= 0.3) & (y[:, i] < 0.6)) / len(y) * 100
        high_risk = np.sum(y[:, i] >= 0.6) / len(y) * 100
        print(f"   {name}: 低風險{low_risk:.1f}% | 中風險{med_risk:.1f}% | 高風險{high_risk:.1f}%")
    
    # 檢查異常值
    print("\n⚠️ 異常值檢查：")
    for i, name in enumerate(feature_names):
        outliers = np.sum((X[:, i] < np.percentile(X[:, i], 1)) | 
                         (X[:, i] > np.percentile(X[:, i], 99)))
        print(f"   {name}: {outliers} 個極端值 ({outliers/len(X)*100:.2f}%)")

# 主要訓練函數（修改版）
def train_improved_health_model():
    """使用改進數據集訓練模型"""
    print("🤖 開始訓練改進版健康AI模型...")
    
    # 1. 生成改進的數據集
    X, y = generate_realistic_health_dataset(50000)
    
    # 2. 驗證數據集
    validate_dataset(X, y)
    
    # 3. 數據預處理
    print("\n🔧 數據預處理...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. 分割數據集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 5. 建立模型（更複雜的架構）
    print("🏗️ 建立改進的神經網路模型...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(7,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5, activation='sigmoid')  # 5個風險輸出
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # 6. 訓練模型
    print("🚀 開始訓練...")
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=64,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5),
            tf.keras.callbacks.ModelCheckpoint('best_health_model.h5', save_best_only=True)
        ]
    )
    
    # 7. 評估模型
    print("📈 評估模型性能...")
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"測試損失: {test_loss:.4f}")
    print(f"測試MAE: {test_mae:.4f}")
    print(f"測試MSE: {test_mse:.4f}")
    
    # 8. 詳細預測測試
    print("\n🧪 預測測試...")
    test_predictions(model, scaler, X_test, y_test)
    
    # 9. 儲存所有文件
    save_model_files(model, scaler)
    
    return model, scaler, history

def test_predictions(model, scaler, X_test, y_test):
    """測試模型預測的合理性"""
    # 選擇幾個測試樣本
    test_indices = [0, 100, 500, 1000, 5000]
    
    for idx in test_indices:
        prediction = model.predict(X_test[idx:idx+1], verbose=0)
        actual = y_test[idx]
        
        # 反標準化以查看原始特徵
        original_features = scaler.inverse_transform(X_test[idx:idx+1])[0]
        
        print(f"\n--- 測試樣本 {idx} ---")
        print(f"心率: {original_features[0]:.0f} BPM, 血氧: {original_features[1]:.1f}%, 體溫: {original_features[2]:.1f}°C")
        print(f"年齡: {original_features[3]:.0f}, 性別: {'女' if original_features[4] > 0.5 else '男'}, BMI: {original_features[5]:.1f}")
        print(f"預測風險: 心血管{prediction[0][0]:.3f} | 呼吸{prediction[0][1]:.3f} | 代謝{prediction[0][2]:.3f} | 體溫{prediction[0][3]:.3f} | 綜合{prediction[0][4]:.3f}")
        print(f"實際風險: 心血管{actual[0]:.3f} | 呼吸{actual[1]:.3f} | 代謝{actual[2]:.3f} | 體溫{actual[3]:.3f} | 綜合{actual[4]:.3f}")

def save_model_files(model, scaler):
    """儲存所有必要的文件"""
    print("\n💾 儲存模型文件...")
    
    # 儲存完整模型
    model.save('improved_health_model.h5')
    
    # 轉換為TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('improved_health_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    # 生成C++頭文件
    generate_improved_cpp_header(tflite_model, scaler)
    
    # 儲存標準化器
    import joblib
    joblib.dump(scaler, 'improved_health_scaler.pkl')
    
    print("✅ 文件儲存完成:")
    print("   • improved_health_model.h5")
    print("   • improved_health_model.tflite")
    print("   • improved_health_model.h")
    print("   • improved_health_scaler.pkl")

def generate_improved_cpp_header(tflite_model, scaler):
    """生成改進版C++頭文件"""
    model_data = ', '.join([f'0x{b:02x}' for b in tflite_model])
    
    # 提取標準化參數
    means = ', '.join([f'{mean:.6f}f' for mean in scaler.mean_])
    stds = ', '.join([f'{std:.6f}f' for std in scaler.scale_])
    
    header_content = f"""
#ifndef IMPROVED_HEALTH_MODEL_H
#define IMPROVED_HEALTH_MODEL_H

// 改進版健康AI模型 - 基於醫學指南訓練
// 模型大小: {len(tflite_model)} bytes
// 訓練數據: 50,000筆醫學合理的健康記錄

const unsigned char improved_health_model_data[] = {{
{model_data}
}};

const int improved_health_model_data_len = {len(tflite_model)};

// 標準化參數
const float feature_means[7] = {{{means}}};
const float feature_stds[7] = {{{stds}}};

// 輸入特徵順序:
// 0: heart_rate (心率 BPM, 範圍: 45-150)
// 1: spo2 (血氧飽和度 %, 範圍: 85-100)
// 2: temperature (體溫 °C, 範圍: 35.0-39.0)
// 3: age (年齡, 範圍: 18-85)
// 4: gender (性別: 0=男性, 1=女性)
// 5: bmi (BMI值, 範圍: 15-45)
// 6: hrv (心率變異性, 範圍: 10-70)

// 輸出風險評估 (0-1範圍):
// 0: cardiovascular_risk (心血管風險)
// 1: respiratory_risk (呼吸系統風險)  
// 2: metabolic_risk (代謝風險)
// 3: thermal_risk (體溫調節風險)
// 4: overall_risk (綜合健康風險)

// 風險等級判斷:
// 0.0-0.3: 低風險
// 0.3-0.6: 中風險  
// 0.6-1.0: 高風險

#endif // IMPROVED_HEALTH_MODEL_H
"""
    
    with open('improved_health_model.h', 'w') as f:
        f.write(header_content)

if __name__ == "__main__":
    # 訓練改進版模型
    model, scaler, history = train_improved_health_model()
    
    print("\n🎉 改進版健康AI模型訓練完成！")
    print("🏥 基於醫學指南的風險評估模型已準備就緒")