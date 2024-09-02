# 1,2,3 회 각각의 평균값이 도출됨
# 그나마 사용 가능한 수준

import pandas as pd
import os
import numpy as np

# CSV 파일이 있는 폴더
csv_folder = 'D:/#.Secure Work Folder/BIG/Project/23~24Y/230106 Gemi-F SCMQuality 관리 시스템 구축/impurity profile pdf reading/df/'

# 해당 폴더에서 CSV 파일 찾기
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# 결과를 저장할 DataFrame 초기화
result_df = pd.DataFrame(columns=['Sample Name', 'RRT Mean', '% Area Mean'])

# CSV 파일 처리
for file in csv_files:
   file_path = os.path.join(csv_folder, file)
   df = pd.read_csv(file_path)

   # RRT 계산
   df['RRT'] = np.nan
   for sample in df['Sample Name'].unique():
       sample_df = df[df['Sample Name'] == sample]
       dp72_rt = sample_df[sample_df['Name'] == 'DP72']['RT'].values[0]
       df.loc[df['Sample Name'] == sample, 'RRT'] = sample_df['RT'] / dp72_rt

   # RRT 평균과 % Area 평균 계산
   sample_means = df.groupby('Sample Name').agg({'RRT': 'mean', '% Area': 'mean'}).reset_index()
   sample_means = sample_means.rename(columns={'RRT': 'RRT Mean', '% Area': '% Area Mean'})

   # 결과에 추가
   result_df = pd.concat([result_df, sample_means], ignore_index=True)

   # RRT가 추가된 새로운 CSV 파일로 저장
   new_file_path = os.path.join(csv_folder, f'processed_{file}')
   df.to_csv(new_file_path, index=False)

# 각 샘플의 RRT 평균과 % Area 평균 계산
final_means = result_df.groupby('Sample Name').mean().reset_index()

# 최종 평균값을 새로운 CSV 파일로 저장
final_means.to_csv(os.path.join(csv_folder, 'final_means.csv'), index=False)
