# 여전히 안됨
# Name 칼럼에 값이 없어도 최종 df 에 포함되어야 하는데
# Name 칼럼이 공란이면 df 에 생략됨

import pdfplumber
import os
import pandas as pd
import re
import datetime

# PDF 파일이 있는 폴더
pdf_folder = 'D:/#.Secure Work Folder/BIG/Project/23~24Y/230106 Gemi-F SCMQuality 관리 시스템 구축/impurity profile pdf reading/pdf/DP72/합한거_간소화/'

# 결과를 저장할 DataFrame 초기화
columns = ['Sample Name', 'Peak Number', 'Name', 'RT', 'Area', 'Height', '% Area', 'Total Area', 'Int Type']
df = pd.DataFrame(columns=columns)

# Peak Results 데이터 파싱 부분 수정
for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, filename)
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()

                if text:
                    # 샘플 이름 찾기
                    sample_name_match = re.search(r'Sample Name: ([\S]+)', text)
                    if not sample_name_match:
                        continue

                sample_name = sample_name_match.group(1)

                # Peak Results 찾기
                peak_results_start = text.find('Peak Results')
                if peak_results_start != -1:
                    peak_results_text = text[peak_results_start:]

                    # Peak Results 데이터 파싱
                    peaks = re.findall(r'(\d+)\s+((?:[\w-]+|))\s+([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([A-Z]+)',
                                       peak_results_text)
                    for peak in peaks:
                        new_row = pd.DataFrame({
                            'Sample Name': [sample_name],
                            'Peak Number': [peak[0]],
                            'Name': [peak[1] if peak[1] else '0'],  # Name이 비어있는 경우 '0'으로 처리
                            'RT': [peak[2]],
                            'Area': [peak[3]],
                            'Height': [peak[4]],
                            '% Area': [peak[5]],
                            'Total Area': [peak[6]],
                            'Int Type': [peak[7]]
                        }, columns=columns)
                        df = pd.concat([df, new_row], ignore_index=True)

# 현재 날짜를 YYYYMMDD 형식으로 가져오기
current_date = datetime.datetime.now().strftime("%Y%m%d")

# 파일명에 날짜 포함
output_path = f'D:/#.Secure Work Folder/BIG/Project/23~24Y/230106 Gemi-F SCMQuality 관리 시스템 구축/impurity profile pdf reading/excel/results_{current_date}.csv'

# DataFrame을 CSV 파일로 저장
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"CSV 파일이 '{output_path}'에 저장되었습니다.")
