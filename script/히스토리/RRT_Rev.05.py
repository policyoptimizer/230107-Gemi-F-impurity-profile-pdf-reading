# % Area 분포 반영되도록 함
# 다만 행에 밑으로 쭉 값이 나열됨
# VoE 가 요청한대로 수정해야 함
# 현재까지 베스트. 값은 제대로 나오는데 형태가 피벗이 아님

import os
import pandas as pd

# CSV 파일이 있는 폴더
csv_folder = 'D:/#.Secure Work Folder/BIG/Project/23~24Y/230106 Gemi-F SCMQuality 관리 시스템 구축/impurity profile pdf reading/df/'

# 해당 폴더에서 CSV 파일 찾기
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# 결과를 저장할 데이터프레임 초기화
result_df = pd.DataFrame()

for file in csv_files:
   # 각 CSV 파일 읽기
   file_path = os.path.join(csv_folder, file)
   # df = pd.read_csv(file_path, sep='\t')  # 탭으로 구분된 경우
   df = pd.read_csv(file_path)

   # Sample Name에서 공통 부분 추출 (예: DPS21003)
   df['Sample Base Name'] = df['Sample Name'].str.split('-').str[0]

   # 각 샘플별로 최대 Area 값을 가지는 행의 RT 계산
   max_rt_per_sample = df.groupby('Sample Name')['Area'].idxmax().apply(lambda x: df.loc[x, 'RT'])

   # RRT 계산 및 추가
   df['RRT'] = df.apply(lambda row: row['RT'] / max_rt_per_sample[row['Sample Name']], axis=1)

   # 샘플 기반 이름으로 그룹화하여 RRT와 %Area에 대한 평균 계산
   avg_area_per_rrt = df.groupby(['Sample Base Name', 'RRT'])['% Area'].mean().reset_index()

   # 결과 데이터프레임에 병합
   if result_df.empty:
       result_df = avg_area_per_rrt
   else:
       result_df = pd.merge(result_df, avg_area_per_rrt, on='RRT', how='outer', suffixes=('', '_' + file.split('.')[0]))

# 결과 데이터프레임 저장
result_file_path = os.path.join(csv_folder, 'processed_results.csv')
result_df.to_csv(result_file_path, index=False)
