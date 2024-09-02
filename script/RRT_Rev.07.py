# % Area 분포 피벗 테이블 적용
# Best!!

import os
import pandas as pd

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

   # Sample Name에서 공통 부분 추출 (예: DPS21003)
   df['Sample Base Name'] = df['Sample Name'].str.split('-').str[0]

   # 각 샘플별로 최대 Area 값을 가지는 행의 RT 계산
   max_rt_per_sample = df.groupby('Sample Name')['Area'].idxmax().apply(lambda x: df.loc[x, 'RT'])

   # RRT 계산 및 추가
   df['RRT'] = df.apply(lambda row: row['RT'] / max_rt_per_sample[row['Sample Name']], axis=1)
   df['RRT'] = df['RRT'].round(2)  # RRT를 소수점 둘째 자리까지 반올림

   # 각 샘플별 %Area의 평균 계산
   avg_area_per_rrt = df.groupby(['Sample Name', 'RRT'])['% Area'].mean().reset_index()

   # 샘플 이름을 칼럼으로, RRT를 인덱스로 사용하는 피벗 테이블 생성
   pivot_df = avg_area_per_rrt.pivot(index='RRT', columns='Sample Name', values='% Area')

   # 각 Sample Base Name 별로 평균 %Area 계산
   base_names = df['Sample Base Name'].unique()
   for base_name in base_names:
       # 해당 base_name을 포함하는 모든 columns 선택
       sample_columns = [col for col in pivot_df.columns if base_name in col]
       # 평균 계산하여 새로운 칼럼에 저장
       pivot_df[base_name] = pivot_df[sample_columns].mean(axis=1)

   # 최종 결과 데이터프레임에 병합
   final_result_df = pd.concat([final_result_df, pivot_df[base_names]], axis=1)

# 중복된 Sample Base Name 칼럼들을 제거하고, 최종 결과를 저장
final_result_df = final_result_df.loc[:, ~final_result_df.columns.duplicated()]
result_file_path = os.path.join(csv_folder, 'final_processed_results.csv')
final_result_df.to_csv(result_file_path, float_format='%.2f')
