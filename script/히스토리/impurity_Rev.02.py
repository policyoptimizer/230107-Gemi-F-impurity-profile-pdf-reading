# 테서렉트 활용
# 인식율 높이려고 enhance 기능 추가
# directory 에 한글 있어도 되고, 띄어쓰기 있어도 되고
# 결과 : 글자 인식 잘 못함

import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance
import io
import re
import os

# Tesseract 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# PDF 파일 열기
pdf_path = '/impurity profile pdf reading/pdf/DP72/22001~413/22413-3.xps.pdf'
doc = fitz.open(pdf_path)

# 결과를 저장할 데이터프레임 생성
results = pd.DataFrame()

# Peak Results 테이블 파싱을 위한 정규 표현식 정의
# 이 정규 표현식은 실제 텍스트 데이터에 따라 조정될 필요가 있습니다.
# 예: r'RT\s+(\d+\.\d+)\s+Area\s+(\d+)'
pattern = re.compile(r'RT\s+(\d+\.\d+)\s+Area\s+(\d+)')

# 각 페이지를 순회하며 텍스트 추출
for page_num in range(len(doc)):
   page = doc.load_page(page_num)
   pix = page.get_pixmap()
   img = Image.open(io.BytesIO(pix.tobytes()))

   # 이미지 전처리: 명암 대비 조절, 이미지 향상 등
   enhancer = ImageEnhance.Contrast(img)
   img_enhanced = enhancer.enhance(2.0)

   # 이미지에서 텍스트 추출 (OCR)
   text = pytesseract.image_to_string(img_enhanced, lang='eng', config='--psm 6')
   print("Page", page_num + 1)
   print(text)

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