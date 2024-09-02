# 일단 테서렉트로 진행!!
# directory 에 한글 있어도 되고, 띄어쓰기 있어도 되고

import os
import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from PIL import Image
import io
import re

# Tesseract 경로 설정 (Tesseract가 설치된 경로를 정확히 입력해야 함)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# PDF 파일 열기
# pdf_path = 'D:/#.Secure Work Folder/BIG/Project/23~24Y/231122 chart information reading/pdf/in/ARE24008/20231117115139-0001.pdf'
pdf_path = '/impurity profile pdf reading/pdf/DP72/22001~413/22413-3.xps.pdf'

doc = fitz.open(pdf_path)

# 결과를 저장할 데이터프레임 생성
results = pd.DataFrame()

# RT와 Area를 파싱하기 위한 정규 표현식 정의
# 이 정규 표현식은 실제 텍스트 데이터에 따라 조정될 필요가 있습니다.
pattern = re.compile(r'\|\s*(\d+\.\d+)\s*\|\s*(\w+)')

# 각 페이지를 순회하며 텍스트 추출
for page_num in range(len(doc)):
   # 페이지에서 이미지 추출
   page = doc.load_page(page_num)
   pix = page.get_pixmap()
   img = Image.open(io.BytesIO(pix.tobytes()))

   # 이미지에서 텍스트 추출 (OCR)
   text = pytesseract.image_to_string(img, lang='eng')
   print("Page", page_num + 1)
   print(text)

   # 정규 표현식에 맞는 모든 결과 찾기
   for match in pattern.finditer(text):
       rt = match.group(1)  # 첫 번째 괄호에 해당하는 부분 (RT)
       area = match.group(2)  # 두 번째 괄호에 해당하는 부분 (Area)

       # 결과를 데이터프레임에 추가
       temp_df = pd.DataFrame({'RT': [rt], '%Area': [area]})
       results = pd.concat([results, temp_df], ignore_index=True)

# 결과를 Excel 파일로 저장
# output_dir = 'D:/#.Secure Work Folder/BIG/Project/23~24Y/231122 chart information reading/excel'
output_dir = '/impurity profile pdf reading/excel'

output_path = os.path.join(output_dir, 'output.xlsx')
results.to_excel(output_path, index=False)

print(f"완료! 결과가 '{output_path}' 파일에 저장되었습니다.")
