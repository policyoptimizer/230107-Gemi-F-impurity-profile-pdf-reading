# pdf 에서 txt 를 일단 먼저 추출하자!!!
# pdf plumber 활용했음
# 테서렉트 안 썼음
# directory 에 한글 있어도 되고, 띄어쓰기 있어도 되고
# 결과 : 그나마 인식을 많이 했음

import pdfplumber
import pandas as pd
import re
import os

# PDF 파일 열기
pdf_path = '/impurity profile pdf reading/pdf/DP72/22001~413/22413-3.xps.pdf'

# 결과를 저장할 데이터프레임 생성
results = pd.DataFrame()

# Peak Results 테이블 파싱을 위한 정규 표현식 정의
# 예: r'RT\s+(\d+\.\d+)\s+Area\s+(\d+)'
pattern = re.compile(r'RT\s+(\d+\.\d+)\s+Area\s+(\d+)')

# PDF 파일을 pdfplumber로 열기
with pdfplumber.open(pdf_path) as pdf:
   # 각 페이지를 순회하며 텍스트 추출
   for page in pdf.pages:
       text = page.extract_text()

       # 페이지에 텍스트가 있는 경우에만 처리
       if text:
           print("Extracted Text:")
           print(text)  # 추출된 텍스트 출력

           # 정규 표현식에 맞는 모든 결과 찾기
           for match in pattern.finditer(text):
               rt = match.group(1)  # 첫 번째 괄호에 해당하는 부분 (RT)
               area = match.group(2)  # 두 번째 괄호에 해당하는 부분 (Area)

               # 결과를 데이터프레임에 추가
               temp_df = pd.DataFrame({'RT': [rt], 'Area': [area]})
               results = pd.concat([results, temp_df], ignore_index=True)

# 결과를 Excel 파일로 저장
output_dir = '/impurity profile pdf reading/excel'
output_path = os.path.join(output_dir, 'output.xlsx')
results.to_excel(output_path, index=False)

print(f"완료! 결과가 '{output_path}' 파일에 저장되었습니다.")
