# Name 칼럼 제외하고 txt 추출
# extract.txt 로 파싱해서 csv 까지 출력됨
# 여기까지가 1차 모듈!!!
# Best!!!

import pdfplumber
import os
import pandas as pd
import re
import datetime

# PDF 파일이 있는 폴더
pdf_folder = 'D:/#.Secure Work Folder/BIG/Project/23~24Y/230106 Gemi-F SCMQuality 관리 시스템 구축/impurity profile pdf reading/pdf/DP72/합한거_간소화/'

# 결과를 저장할 DataFrame 초기화
columns = ['Sample Name', 'Peak Number', 'RT', 'Area', 'Height', '% Area', 'Total Area', 'Int Type']
df = pd.DataFrame(columns=columns)

# 폴더 내의 모든 PDF 파일을 찾아서 처리
for filename in os.listdir(pdf_folder):
   if filename.endswith('.pdf'):
       pdf_path = os.path.join(pdf_folder, filename)
       with pdfplumber.open(pdf_path) as pdf:
           for page in pdf.pages:
               text = page.extract_text()

               if text:
                   # 샘플 이름 찾기
                   sample_name_match = re.search(r'Sample Name: ([\S]+)', text)
                   if sample_name_match:
                       sample_name = sample_name_match.group(1)
                   else:
                       continue

                   # Peak Results 찾기
                   peak_results_start = text.find('Peak Results')
                   peak_results_text = text[peak_results_start:]

                   # Peak Results 데이터 파싱
                   peaks = re.findall(r'(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([A-Z]+)', peak_results_text)
                   for peak in peaks:
                       new_row = pd.DataFrame({'Sample Name': [sample_name], 'Peak Number': [peak[0]], 'RT': [peak[1]],
                                               'Area': [peak[2]], 'Height': [peak[3]], '% Area': [peak[4]],
                                               'Total Area': [peak[5]], 'Int Type': [peak[6]]}, columns=columns)
                       df = pd.concat([df, new_row], ignore_index=True)

# 현재 날짜를 YYYYMMDD 형식으로 가져오기
current_date = datetime.datetime.now().strftime("%Y%m%d")

# 파일명에 날짜 포함
output_path = f'D:/#.Secure Work Folder/BIG/Project/23~24Y/230106 Gemi-F SCMQuality 관리 시스템 구축/impurity profile pdf reading/excel/results_{current_date}.csv'

# DataFrame을 CSV 파일로 저장
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"CSV 파일이 '{output_path}'에 저장되었습니다.")
