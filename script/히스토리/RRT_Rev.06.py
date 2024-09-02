# % Area 분포 피벗 테이블 적용
# 점점 미궁속으로 빠짐

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
   df = pd.read_csv(file_path)

   # 각 Sample Name별로 최대 % Area 값을 가지는 RT 값 찾기
   max_area_rt = df.groupby('Sample Name')['% Area'].transform('max')
   df['RRT'] = (df['RT'] / df.loc[df['% Area'] == max_area_rt, 'RT']).round(2)

   # Sample Base Name 추출
   df['Sample Base Name'] = df['Sample Name'].str.split('-').str[0]

   # RRT와 %Area를 사용하여 평균 계산
   avg_area_df = df.groupby(['Sample Name', 'RRT'])['% Area'].mean().reset_index()

   # 샘플 이름을 상단 칼럼으로, RRT를 인덱스로 사용하여 피벗 테이블 생성
   pivot_df = avg_area_df.pivot(index='RRT', columns='Sample Name', values='% Area')

# 결과 데이터프레임에 병합
result_df = pd.concat([result_df, pivot_df], axis=1)

# 결과 데이터프레임에서 NA 값은 0으로 채우기
result_df.fillna(0, inplace=True)

# 각 Sample Base Name 별로 RRT에 대한 %Area의 평균값 계산
for base_name in result_df.columns.get_level_values(0).unique():
   # 해당 base_name을 포함하는 모든 columns 선택
   base_name_columns = result_df.columns[result_df.columns.str.contains(base_name)]
   # 평균 계산
   result_df[base_name] = result_df[base_name_columns].mean(axis=1)

# 최종 결과에서 필요한 컬럼만 선택
final_result_df = result_df[[col for col in result_df.columns if '-' not in col]]

# 결과 데이터프레임 저장 (소수점 둘째자리까지 포맷)
result_file_path = os.path.join(csv_folder, 'final_results.csv')
final_result_df.to_csv(result_file_path, float_format='%.2f')
