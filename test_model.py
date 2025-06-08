import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime

class HealthAssessmentTool:
    def __init__(self, model_path='improved_health_model.h5', scaler_path='improved_health_scaler.pkl'):
        """初始化健康評估工具"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            print("✅ AI模型載入成功！")
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            print("請確保已運行訓練程式並生成模型文件")
            self.model = None
            self.scaler = None
    
    def get_user_input(self):
        """獲取用戶輸入的健康數據"""
        print("\n" + "="*50)
        print("🏥 健康評估與就醫建議系統")
        print("="*50)
        print("請輸入您的健康數據：")
        
        try:
            # 基本資料
            print("\n👤 基本資料：")
            age = int(input("年齡 (18-85): "))
            gender_input = input("性別 (男/女 或 M/F): ").lower()
            gender = 1 if gender_input in ['女', 'f', 'female'] else 0
            height = float(input("身高 (cm): "))
            weight = float(input("體重 (kg): "))
            
            # 計算BMI
            bmi = weight / ((height / 100) ** 2)
            
            # 生理數據
            print("\n💓 生理數據：")
            heart_rate = float(input("心率 (BPM): "))
            spo2 = float(input("血氧飽和度 (%): "))
            temperature = float(input("體溫 (°C): "))
            
            # HRV（可選，如果不知道就估算）
            hrv_input = input("心率變異性 HRV (如不知道請按Enter): ")
            if hrv_input.strip():
                hrv = float(hrv_input)
            else:
                # 基於年齡和心率估算HRV
                hrv = max(10, 50 - (age - 25) * 0.4 - abs(heart_rate - 70) * 0.2)
                print(f"   估算HRV: {hrv:.1f}")
            
            # 症狀詢問
            print("\n🤒 症狀檢查：")
            has_chest_pain = input("是否有胸痛？(y/n): ").lower().startswith('y')
            has_shortness_breath = input("是否有呼吸困難？(y/n): ").lower().startswith('y')
            has_dizziness = input("是否有頭暈？(y/n): ").lower().startswith('y')
            has_fever = input("是否發燒？(y/n): ").lower().startswith('y')
            has_fatigue = input("是否極度疲勞？(y/n): ").lower().startswith('y')
            
            return {
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'bmi': bmi,
                'heart_rate': heart_rate,
                'spo2': spo2,
                'temperature': temperature,
                'hrv': hrv,
                'symptoms': {
                    'chest_pain': has_chest_pain,
                    'shortness_breath': has_shortness_breath,
                    'dizziness': has_dizziness,
                    'fever': has_fever,
                    'fatigue': has_fatigue
                }
            }
            
        except ValueError as e:
            print(f"❌ 輸入格式錯誤: {e}")
            return None
        except KeyboardInterrupt:
            print("\n👋 程式結束")
            return None
    
    def validate_input(self, data):
        """驗證輸入數據的合理性"""
        issues = []
        
        # 檢查年齡
        if not 18 <= data['age'] <= 100:
            issues.append("年齡應在18-100歲之間")
        
        # 檢查身高體重
        if not 100 <= data['height'] <= 250:
            issues.append("身高應在100-250cm之間")
        if not 30 <= data['weight'] <= 200:
            issues.append("體重應在30-200kg之間")
        
        # 檢查生理數據
        if not 30 <= data['heart_rate'] <= 200:
            issues.append("心率應在30-200 BPM之間")
        if not 70 <= data['spo2'] <= 100:
            issues.append("血氧應在70-100%之間")
        if not 32 <= data['temperature'] <= 45:
            issues.append("體溫應在32-45°C之間")
        
        return issues
    
    def predict_health_risks(self, data):
        """使用AI模型預測健康風險"""
        if self.model is None or self.scaler is None:
            return None
        
        # 準備輸入特徵
        features = np.array([[
            data['heart_rate'],
            data['spo2'],
            data['temperature'],
            data['age'],
            data['gender'],
            data['bmi'],
            data['hrv']
        ]])
        
        # 標準化
        features_scaled = self.scaler.transform(features)
        
        # 預測
        prediction = self.model.predict(features_scaled, verbose=0)[0]
        
        return {
            'cardiovascular_risk': prediction[0],
            'respiratory_risk': prediction[1],
            'metabolic_risk': prediction[2],
            'thermal_risk': prediction[3],
            'overall_risk': prediction[4]
        }
    
    def assess_emergency_symptoms(self, data):
        """評估緊急症狀"""
        emergency_flags = []
        urgent_flags = []
        
        symptoms = data['symptoms']
        
        # 緊急情況（立即就醫）
        if data['heart_rate'] > 120 and symptoms['chest_pain']:
            emergency_flags.append("心率過快伴隨胸痛")
        
        if data['spo2'] < 90:
            emergency_flags.append("血氧飽和度危險性低")
        
        if data['temperature'] > 39.5:
            emergency_flags.append("高燒（>39.5°C）")
        
        if data['temperature'] < 35.0:
            emergency_flags.append("體溫過低（<35°C）")
        
        if symptoms['chest_pain'] and symptoms['shortness_breath']:
            emergency_flags.append("胸痛伴隨呼吸困難")
        
        if data['heart_rate'] > 150 or data['heart_rate'] < 40:
            emergency_flags.append("心率嚴重異常")
        
        # 需要盡快就醫
        if data['spo2'] < 95 and symptoms['shortness_breath']:
            urgent_flags.append("血氧偏低伴隨呼吸困難")
        
        if data['temperature'] > 38.5 and symptoms['fatigue']:
            urgent_flags.append("發燒伴隨極度疲勞")
        
        if symptoms['dizziness'] and (data['heart_rate'] > 100 or data['heart_rate'] < 60):
            urgent_flags.append("頭暈伴隨心率異常")
        
        if data['bmi'] > 35 and (symptoms['chest_pain'] or symptoms['shortness_breath']):
            urgent_flags.append("肥胖伴隨心肺症狀")
        
        return emergency_flags, urgent_flags
    
    def generate_medical_advice(self, data, risks, emergency_flags, urgent_flags):
        """生成醫療建議"""
        advice = {
            'urgency_level': 'low',
            'recommendation': '',
            'reasons': [],
            'follow_up': []
        }
        
        # 緊急情況
        if emergency_flags:
            advice['urgency_level'] = 'emergency'
            advice['recommendation'] = '🚨 立即撥打119或前往急診室'
            advice['reasons'] = emergency_flags
            advice['follow_up'] = ['立即就醫，不要延遲', '可能需要心電圖、血液檢查等']
            return advice
        
        # 需要盡快就醫
        if urgent_flags or risks['overall_risk'] > 0.6:
            advice['urgency_level'] = 'urgent'
            advice['recommendation'] = '⚠️ 建議24-48小時內就醫'
            advice['reasons'] = urgent_flags + [f"AI評估整體風險較高 ({risks['overall_risk']:.3f})"]
            advice['follow_up'] = ['預約家醫科或相關專科', '準備過去病史和用藥清單']
            return advice
        
        # 中度風險
        if (risks['cardiovascular_risk'] > 0.4 or 
            risks['respiratory_risk'] > 0.4 or 
            risks['metabolic_risk'] > 0.4 or
            risks['overall_risk'] > 0.3):
            
            advice['urgency_level'] = 'moderate'
            advice['recommendation'] = '💡 建議1-2週內安排健康檢查'
            
            reasons = []
            if risks['cardiovascular_risk'] > 0.4:
                reasons.append(f"心血管風險偏高 ({risks['cardiovascular_risk']:.3f})")
            if risks['respiratory_risk'] > 0.4:
                reasons.append(f"呼吸系統風險偏高 ({risks['respiratory_risk']:.3f})")
            if risks['metabolic_risk'] > 0.4:
                reasons.append(f"代謝風險偏高 ({risks['metabolic_risk']:.3f})")
            
            advice['reasons'] = reasons
            advice['follow_up'] = ['定期監測相關指標', '考慮生活方式調整']
            return advice
        
        # 低風險
        advice['urgency_level'] = 'low'
        advice['recommendation'] = '✅ 目前健康狀況良好'
        advice['reasons'] = ['所有風險指標都在正常範圍內']
        advice['follow_up'] = ['維持健康生活習慣', '建議每年定期健康檢查']
        
        return advice
    
    def display_detailed_report(self, data, risks, advice):
        """顯示詳細的健康報告"""
        print("\n" + "="*60)
        print("📋 詳細健康評估報告")
        print("="*60)
        print(f"評估時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 基本資料
        print(f"\n👤 基本資料:")
        print(f"   年齡: {data['age']} 歲")
        print(f"   性別: {'女性' if data['gender'] else '男性'}")
        print(f"   身高: {data['height']} cm")
        print(f"   體重: {data['weight']} kg")
        print(f"   BMI: {data['bmi']:.1f} ({'正常' if 18.5 <= data['bmi'] <= 24.9 else '異常'})")
        
        # 生理數據
        print(f"\n💓 生理數據:")
        print(f"   心率: {data['heart_rate']} BPM ({'正常' if 60 <= data['heart_rate'] <= 100 else '異常'})")
        print(f"   血氧: {data['spo2']}% ({'正常' if data['spo2'] >= 95 else '偏低' if data['spo2'] >= 90 else '危險'})")
        print(f"   體溫: {data['temperature']}°C ({'正常' if 36.1 <= data['temperature'] <= 37.2 else '異常'})")
        print(f"   心率變異性: {data['hrv']:.1f}")
        
        # AI風險評估
        print(f"\n🤖 AI風險評估:")
        risk_levels = ['低', '中', '高']
        for risk_name, risk_value in [
            ('心血管風險', risks['cardiovascular_risk']),
            ('呼吸系統風險', risks['respiratory_risk']),
            ('代謝風險', risks['metabolic_risk']),
            ('體溫調節風險', risks['thermal_risk']),
            ('綜合風險', risks['overall_risk'])
        ]:
            level = 0 if risk_value < 0.3 else 1 if risk_value < 0.6 else 2
            bar = "█" * int(risk_value * 20) + "░" * (20 - int(risk_value * 20))
            print(f"   {risk_name}: {risk_value:.3f} [{bar}] {risk_levels[level]}風險")
        
        # 醫療建議
        print(f"\n🏥 醫療建議:")
        urgency_icons = {
            'emergency': '🚨',
            'urgent': '⚠️',
            'moderate': '💡',
            'low': '✅'
        }
        print(f"   {urgency_icons[advice['urgency_level']]} {advice['recommendation']}")
        
        if advice['reasons']:
            print(f"\n📋 主要原因:")
            for reason in advice['reasons']:
                print(f"   • {reason}")
        
        if advice['follow_up']:
            print(f"\n📝 後續建議:")
            for follow in advice['follow_up']:
                print(f"   • {follow}")
        
        print("\n" + "="*60)
        print("⚠️  注意：此評估僅供參考，不可取代專業醫療診斷")
        print("="*60)
    
    def run_assessment(self):
        """運行完整的健康評估"""
        # 獲取用戶輸入
        data = self.get_user_input()
        if data is None:
            return
        
        # 驗證輸入
        issues = self.validate_input(data)
        if issues:
            print("\n❌ 輸入數據有問題:")
            for issue in issues:
                print(f"   • {issue}")
            return
        
        # AI風險預測
        risks = self.predict_health_risks(data)
        if risks is None:
            print("❌ AI模型預測失敗")
            return
        
        # 症狀評估
        emergency_flags, urgent_flags = self.assess_emergency_symptoms(data)
        
        # 生成醫療建議
        advice = self.generate_medical_advice(data, risks, emergency_flags, urgent_flags)
        
        # 顯示報告
        self.display_detailed_report(data, risks, advice)
        
        # 簡要結論
        print(f"\n🎯 結論: {advice['recommendation']}")
        
        return {
            'data': data,
            'risks': risks,
            'advice': advice
        }

def main():
    """主程式"""
    print("🏥 健康評估與就醫建議系統")
    print("基於AI深度學習模型的智能健康分析")
    
    # 初始化評估工具
    tool = HealthAssessmentTool()
    
    if tool.model is None:
        print("\n❌ 無法載入AI模型，請先運行訓練程式")
        print("執行: python train_improved_health_model.py")
        return
    
    while True:
        try:
            # 運行評估
            result = tool.run_assessment()
            
            if result is None:
                continue
            
            # 詢問是否繼續
            print("\n" + "-"*50)
            continue_input = input("是否要進行另一次評估？(y/n): ")
            if not continue_input.lower().startswith('y'):
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 感謝使用健康評估系統！")
            break
        except Exception as e:
            print(f"\n❌ 程式錯誤: {e}")
            continue
    
    print("祝您身體健康！💪")

if __name__ == "__main__":
    main()