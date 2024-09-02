# 1,2,3 회 반복 시험을 전부 평균내어서 샘플 하나에 대한 값이 하나만 나오도록 해서
# final mean 파일에 저장함.
# RRT 추가된 .csv 파일이 하나 생성됨
# DPS22003 에 DP72 가 없어서 에러가 발생하는데
# 여기에서는 그냥 넘어가도록 함

import pandas as pd
import os
import numpy as np

# CSV 파일이 있는 폴더
csv_folder = 'D:/#.Secure Work Folder/BIG/Project/23~24Y/230106 Gemi-F SCMQuality 관리 시스템 구축/impurity profile pdf reading/df/'

# 해당 폴더에서 CSV 파일 찾기
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# 결과를 저장할 DataFrame 초기화
final_result_df = pd.DataFrame(columns=['Sample Base Name', 'RRT Mean', '% Area Mean'])

# CSV 파일 처리
for file in csv_files:
   file_path = os.path.join(csv_folder, file)
   df = pd.read_csv(file_path)

   # RRT 계산 및 기본 샘플 이름 추출
   df['RRT'] = np.nan
   df['Sample Base Name'] = df['Sample Name'].str.extract(r'(DPS\d+)')
   for sample in df['Sample Name'].unique():
       sample_df = df[df['Sample Name'] == sample]
       dp72_row = sample_df[sample_df['Name'] == 'DP72']
       if not dp72_row.empty:
           dp72_rt = dp72_row['RT'].values[0]
           df.loc[df['Sample Name'] == sample, 'RRT'] = sample_df['RT'] / dp72_rt

   # RRT가 추가된 새로운 CSV 파일로 저장
   processed_file_path = os.path.join(csv_folder, f'processed_{file}')
   df.to_csv(processed_file_path, index=False)

   # 기본 샘플 이름별 RRT 평균과 % Area 평균 계산
   sample_means = df.groupby('Sample Base Name').agg({'RRT': 'mean', '% Area': 'mean'}).reset_index()
   sample_means = sample_means.rename(columns={'Sample Base Name': 'Sample Base Name', 'RRT': 'RRT Mean', '% Area': '% Area Mean'})

   # 결과에 추가
   final_result_df = pd.concat([final_result_df, sample_means], ignore_index=True)

# 각 기본 샘플 이름의 최종 RRT 평균과 % Area 평균 계산
final_means = final_result_df.groupby('Sample Base Name').mean().reset_index()

# 최종 평균값을 새로운 CSV 파일로 저장
final_means.to_csv(os.path.join(csv_folder, 'final_means.csv'), index=False)
