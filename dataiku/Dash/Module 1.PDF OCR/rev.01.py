import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import io
import base64
import re
import datetime
import pdfplumber
import logging
from dataiku import Folder

# pdfminer 로그 레벨을 INFO로 설정하여 디버그 메시지 억제
logging.getLogger('pdfminer').setLevel(logging.INFO)

# Dash 앱 생성
#app = dash.Dash(__name__)

# csv_folder 설정 (데이터이쿠 매니지드 폴더)
csv_folder = Folder("csv_folder_id")  # csv_folder의 ID를 설정하세요

# Dash 레이아웃
app.layout = html.Div([
    html.H1("PDF 파일 업로드 및 CSV 변환"),
   
    dcc.Upload(
        id='upload-data',
        children=html.Div(['PDF 파일을 여기로 드래그하거나 ', html.A('클릭하여 선택하세요')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True
    ),
   
    html.Div(id='output-data-upload'),
   
    html.Button("CSV 파일 다운로드", id="download-button"),
    dcc.Download(id="download-csv")
])

# PDF 처리 및 CSV 저장 함수
def process_pdf(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
   
    # 결과를 저장할 DataFrame 초기화
    columns = ['Sample Name', 'Peak Number', 'RT', 'Area', 'Height', '% Area', 'Total Area', 'Int Type']
    df = pd.DataFrame(columns=columns)
   
    with pdfplumber.open(io.BytesIO(decoded)) as pdf:
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

    # CSV 파일을 Dataiku의 매니지드 폴더에 저장
    output_csv_name = f'results_{current_date}.csv'
    output_path = csv_folder.get_path(output_csv_name)
   
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
   
    return output_csv_name

# 업로드된 PDF 파일 처리 및 CSV 다운로드 생성
@app.callback(
    [Output('output-data-upload', 'children'), Output('download-csv', 'data')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        processed_files = []
       
        # 모든 업로드된 PDF 파일 처리
        for contents, name in zip(list_of_contents, list_of_names):
            output_csv_name = process_pdf(contents)
            processed_files.append(f"{name} -> {output_csv_name}")
       
        # 결과 출력 및 CSV 다운로드 링크 제공
        return (
            html.Div([html.P(f"처리된 파일: {', '.join(processed_files)}")]),
            dcc.send_file(csv_folder.get_path(output_csv_name))  # 마지막 파일을 다운로드로 제공
        )

# Dash 앱 실행
#if __name__ == '__main__':
#    app.run_server(debug=True)

