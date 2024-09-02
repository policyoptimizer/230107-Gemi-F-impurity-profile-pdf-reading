# 0.22 ~ 0.35 사이에 Imp1,2,3 전부 모여 있음

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# CSV 파일이 있는 폴더
csv_folder = 'D:/#.Secure Work Folder/BIG/Project/23~24Y/230106 Gemi-F SCMQuality 관리 시스템 구축/impurity profile pdf reading/module 2/'

# 해당 폴더에서 CSV 파일 찾기
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# 최종 결과를 저장할 데이터프레임 초기화
final_result_df = pd.DataFrame()

# 각 CSV 파일별로 처리
for file in csv_files:
  # CSV 파일 읽기
  file_path = os.path.join(csv_folder, file)
  df = pd.read_csv(file_path)

  # Sample Name에서 공통 부분 추출
  df['Sample Base Name'] = df['Sample Name'].str.split('-').str[0]

  # 각 샘플별로 최대 Area 값을 가지는 행의 RT 계산
  max_rt_per_sample = df.groupby('Sample Name')['Area'].idxmax().apply(lambda x: df.loc[x, 'RT'])

  # RRT 계산 및 추가
  df['RRT'] = df.apply(lambda row: row['RT'] / max_rt_per_sample[row['Sample Name']], axis=1)
  df['RRT'] = df['RRT'].round(2)

  # 각 샘플별 %Area의 평균 계산
  avg_area_per_rrt = df.groupby(['Sample Name', 'RRT'])['% Area'].mean().reset_index()

  # 피벗 테이블 생성
  pivot_df = avg_area_per_rrt.pivot(index='RRT', columns='Sample Name', values='% Area')

  # 각 Sample Base Name 별로 평균 %Area 계산
  base_names = df['Sample Base Name'].unique()
  for base_name in base_names:
      sample_columns = [col for col in pivot_df.columns if base_name in col]
      pivot_df[base_name] = pivot_df[sample_columns].mean(axis=1)

  # 최종 결과 데이터프레임에 병합
  final_result_df = pd.concat([final_result_df, pivot_df[base_names]], axis=1)

# 중복된 Sample Base Name 칼럼들 제거
final_result_df = final_result_df.loc[:, ~final_result_df.columns.duplicated()]

# 최종 결과 저장
result_file_path = os.path.join(csv_folder, 'final_processed_results.csv')
final_result_df.to_csv(result_file_path, float_format='%.2f')

# 히트맵 생성
df = final_result_df
df.index = df.index.map(str)
annot_df = df.fillna(0).astype(float).apply(lambda x: x.map('{:.2f}'.format))

plt.figure(figsize=(20, 10))
sns.heatmap(df, annot=annot_df, fmt="", cmap='coolwarm', linewidths=.5, linecolor='gray')
plt.title('Sample Base Name 별 RRT 대비 % Area 평균', fontsize=20)
plt.xlabel('Sample Base Name', fontsize=14)
plt.ylabel('RRT', fontsize=14)

# Imp1, Imp2, Imp3 영역 강조 및 라벨링을 위해 y축 범위를 조정합니다.
imp1_start, imp1_end = 1.30, 1.35
imp2_start, imp2_end = 0.90, 0.94
imp3_start, imp3_end = 0.80, 0.83

# 히트맵에 선을 그릴 때는 인덱스의 숫자 위치를 사용합니다.
plt.axhspan(imp1_start, imp1_end, color='green', alpha=0.3)
plt.text(len(df.columns)+0.5, (imp1_start + imp1_end) / 2, 'Imp1', verticalalignment='center', color='green')

plt.axhspan(imp2_start, imp2_end, color='green', alpha=0.3)
plt.text(len(df.columns)+0.5, (imp2_start + imp2_end) / 2, 'Imp2', verticalalignment='center', color='green')

plt.axhspan(imp3_start, imp3_end, color='green', alpha=0.3)
plt.text(len(df.columns)+0.5, (imp3_start + imp3_end) / 2, 'Imp3', verticalalignment='center', color='green')

# 히트맵 및 강조된 영역 저장
heatmap_file_path = os.path.join(csv_folder, 'beautiful_heatmap.png')
plt.savefig(heatmap_file_path)

# 히트맵 보여주기
plt.show()

print(df.index)