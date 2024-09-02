# pdf 에서 txt 를 일단 먼저 추출하자!!!
    # pdf plumber 활용했음
    # 테서렉트 안 썼음
    # 일단 모든 텍스트를 다 추출하자는 의미에서 로직도 적용 안함
    # directory 에 한글 있어도 되고, 띄어쓰기 있어도 되고
# 결과 : extracted_text.txt 에 텍스트만 추출됨
#       여기에서 develop 하면 됨

import pdfplumber
import os

# PDF 파일 열기
pdf_path = '/impurity profile pdf reading/pdf/DP72/22001~413/22413-3.xps.pdf'

# 추출된 텍스트를 저장할 문자열 초기화
extracted_text = ""

# PDF 파일을 pdfplumber로 열기
with pdfplumber.open(pdf_path) as pdf:
    # 각 페이지를 순회하며 텍스트 추출
    for page_number, page in enumerate(pdf.pages):
        text = page.extract_text()

        # 페이지에 텍스트가 있는 경우에만 처리
        if text:
            extracted_text += f"Page {page_number + 1}:\n{text}\n\n"

# 결과를 텍스트 파일로 저장
output_dir = '/impurity profile pdf reading/excel'
output_path = os.path.join(output_dir, 'extracted_text.txt')
with open(output_path, 'w', encoding='utf-8') as file:
    file.write(extracted_text)

print(f"완료! 추출된 텍스트가 '{output_path}' 파일에 저장되었습니다.")
