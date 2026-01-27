# analyze_physics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_gravity_and_friction(csv_file):
    df = pd.read_csv(csv_file)
    
    # 1. 데이터 전처리 (시간 간격 dt 계산)
    df['dt'] = df['timestamp'].diff().fillna(0)
    valid_df = df[df['dt'] > 0.01] # 너무 짧은 프레임 제외 (노이즈 제거)

    print(f"📂 분석 파일: {csv_file}")
    
    # --- [분석 1] Y축: 중력 가속도 (Gravity) 측정 ---
    # 조건: 공중(is_ground=0) + 사다리 아님 + 벽 충돌 아님
    air_df = valid_df[
        (valid_df['is_ground'] == 0) & 
        (valid_df['is_wall_left'] == 0) & 
        (valid_df['is_wall_right'] == 0)
    ]
    
    # 가속도 ay 분포 확인 (ay = d(vy) / dt)
    # 노이즈를 줄이기 위해 ay가 음수(낙하 가속)인 구간의 평균을 구함
    # 메이플은 위가 -y일 수도, +y일 수도 있음. 보통 아래로 떨어지면 y 증가.
    # 여기서는 ay의 '최빈값(Mode)'이나 '중앙값(Median)'이 중력 상수임.
    
    gravity_candidates = air_df['ay']
    estimated_gravity = gravity_candidates.median()
    
    print(f"   🚀 추정된 중력 가속도 (G): {estimated_gravity:.4f} pixels/s²")
    
    # --- [분석 2] X축: 공중 저항 (Air Resistance) 측정 ---
    # 조건: 공중 + 키 입력 없음 (관성 이동 중)
    inertia_df = air_df[air_df['action'] == 'None'] # 혹은 키 입력 없는 상태
    
    if not inertia_df.empty:
        # ax가 0에 가까우면 저항 없음, 음수면 공기 저항 존재
        avg_air_drag = inertia_df['ax'].median()
        print(f"   💨 추정된 공중 저항 (Drag): {avg_air_drag:.4f} pixels/s² (0에 가까우면 저항 없음)")
    else:
        print("   💨 공중 관성 데이터 부족 (키 뗀 상태의 데이터 필요)")

    return estimated_gravity

if __name__ == "__main__":
    # 방금 수집한 csv 파일 경로를 넣어주세요
    target_csv = "data/Physics_Discrete_20260127_XXXXXX.csv" 
    
    try:
        g = analyze_gravity_and_friction(target_csv)
        
        # 그래프로 포물선 확인 (시각화)
        plt.figure(figsize=(10, 5))
        plt.title(f"Vertical Velocity (vy) over Time (Est. Gravity: {g:.2f})")
        plt.plot(df['timestamp'], df['vy'], label='vy (Vertical Speed)')
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (px/s)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("csv 파일 경로를 확인해주세요.")