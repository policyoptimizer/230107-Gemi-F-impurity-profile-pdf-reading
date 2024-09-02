# 좀 더 시각적으로 이쁘게 해보자!!!

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일이 있는 폴더
csv_folder = 'D:/#.Secure Work Folder/BIG/Project/23~24Y/230106 Gemi-F SCMQuality 관리 시스템 구축/impurity profile pdf reading/df/'

# 최종 결과 파일 이름
final_result_file_name = 'final_processed_results.csv'

# 최종 결과 파일 경로
final_result_file_path = os.path.join(csv_folder, final_result_file_name)

# 히트맵 생성을 위한 데이터 준비
df = pd.read_csv(final_result_file_path, index_col='RRT')

# 데이터프레임의 인덱스를 문자열로 변환
df.index = df.index.map(str)

# annot에 사용할 데이터프레임 생성
annot_df = df.applymap(lambda x: f'{x:.2f}' if x != 0 else '')

# 히트맵 생성
plt.figure(figsize=(20, 10))
sns.heatmap(df, annot=annot_df, fmt="", cmap='coolwarm', linewidths=.5, linecolor='gray')

# 그래프의 타이틀과 축 이름 설정
plt.title('Sample Base Name 별 RRT 대비 % Area 평균', fontsize=20)
plt.xlabel('Sample Base Name', fontsize=14)
plt.ylabel('RRT', fontsize=14)

# 히트맵 저장
heatmap_file_path = os.path.join(csv_folder, 'beautiful_heatmap.png')
plt.savefig(heatmap_file_path)

# 히트맵 보여주기
plt.show()
