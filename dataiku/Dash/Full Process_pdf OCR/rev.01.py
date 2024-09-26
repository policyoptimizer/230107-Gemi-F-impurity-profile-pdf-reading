# pdf 는 전혀 안됨
# 엑셀이 입력에 유리함

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import io
import base64
import pdfplumber
import re
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import dataiku
from dataiku import pandasutils as pdu
import os
import logging

# Dataiku Managed Folder
folder = dataiku.Folder("uploaded_pdfs")  # 생성한 Managed Folder의 ID

# Dash 앱 생성
#app = dash.Dash(__name__)

# Layout 설정
app.layout = html.Div([
    html.H1("PDF 업로드 및 데이터 처리 웹앱"),
   
    # PDF 파일 업로드 컴포넌트
    dcc.Upload(
        id='upload-pdfs',
        children=html.Div([
            'Drag and Drop 또는 ',
            html.A('파일 선택')
        ]),
        style={
            'width': '100%',
            'height': '100px',
            'lineHeight': '100px',
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '10px',
            'textAlign': 'center',
            'margin': '20px'
        },
        multiple=True  # 여러 파일 업로드 허용
    ),
   
    html.Div(id='extracted-data-output'),
   
    html.Hr(),
   
    # 데이터 처리 및 시각화 섹션
    html.H2("데이터 처리 및 시각화"),
   
    html.Button("데이터 처리 및 시각화 실행", id="process-button", n_clicks=0, style={'margin': '20px'}),
   
    html.Div(id='processed-data-output'),
   
    html.Div(id='image-container', style={'margin': '20px'}),
   
    # 다운로드 버튼
    html.Div([
        html.Button("CSV 다운로드", id="btn_csv_download", style={'margin': '10px'}),
        html.Button("히트맵 이미지 다운로드", id="btn_img_download", style={'margin': '10px'}),
        dcc.Download(id="download-csv"),
        dcc.Download(id="download-img")
    ])
])

# Helper 함수: PDF 내용을 파싱하여 DataFrame 반환 및 Managed Folder에 저장
def parse_pdf(contents, filename):
    try:
        logging.info(f"Parsing file: {filename}")
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        with pdfplumber.open(io.BytesIO(decoded)) as pdf:
            extracted_data = []
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
                    if peak_results_start == -1:
                        continue
                    peak_results_text = text[peak_results_start:]

                    # Peak Results 데이터 파싱
                    peaks = re.findall(r'(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([A-Z]+)', peak_results_text)
                    for peak in peaks:
                        extracted_data.append({
                            'Sample Name': sample_name,
                            'Peak Number': peak[0],
                            'RT': float(peak[1]),
                            'Area': float(peak[2]),
                            'Height': float(peak[3]),
                            '% Area': float(peak[4]),
                            'Total Area': float(peak[5]),
                            'Int Type': peak[6]
                        })
        if extracted_data:
            df = pd.DataFrame(extracted_data)
            # Managed Folder에 파일 저장
            with folder.get_writer(filename) as writer:
                writer.write(decoded)
            logging.info(f"File {filename} saved to Managed Folder.")
            return df
        else:
            logging.warning(f"No data extracted from {filename}.")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error parsing {filename}: {e}")
        return pd.DataFrame()

# Helper 함수: 히트맵 생성
def create_heatmap(df):
    df['Sample Base Name'] = df['Sample Name'].str.split('-').str[0]
    max_rt_per_sample = df.groupby('Sample Name')['RT'].idxmax().apply(lambda x: df.loc[x, 'RT'])
    df['RRT'] = df.apply(lambda row: row['RT'] / max_rt_per_sample[row['Sample Name']], axis=1)
    df['RRT'] = df['RRT'].round(2)
    avg_area_per_rrt = df.groupby(['Sample Name', 'RRT'])['% Area'].mean().reset_index()
    pivot_df = avg_area_per_rrt.pivot(index='RRT', columns='Sample Name', values='% Area')
    base_names = df['Sample Base Name'].unique()
    for base_name in base_names:
        sample_columns = [col for col in pivot_df.columns if base_name in col]
        if sample_columns:
            pivot_df[base_name] = pivot_df[sample_columns].mean(axis=1)
    final_result_df = pivot_df[base_names]
    final_result_df = final_result_df.loc[:, ~final_result_df.columns.duplicated()]

    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(final_result_df, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, linecolor='gray', ax=ax)
    plt.title('Average % Area vs RRT by Sample Base Name', fontsize=20)
    plt.xlabel('Sample Base Name', fontsize=14)
    plt.ylabel('RRT', fontsize=14)

    # 이미지 저장
    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    plt.close(fig)
    img_io.seek(0)
   
    return img_io.getvalue(), final_result_df

# PDF 업로드 및 데이터 추출 콜백
@app.callback(
    Output('extracted-data-output', 'children'),
    [Input('upload-pdfs', 'contents')],
    [State('upload-pdfs', 'filename')]
)
def extract_data(contents, filenames):
    if contents is not None and filenames is not None:
        all_data = pd.DataFrame()
        for content, filename in zip(contents, filenames):
            df = parse_pdf(content, filename)
            if not df.empty:
                all_data = pd.concat([all_data, df], ignore_index=True)
        if not all_data.empty:
            # 현재 날짜를 YYYYMMDD 형식으로 가져오기
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            # CSV 파일명을 날짜 포함하여 설정
            csv_filename = f'aggregated_results_{current_date}.csv'
            # CSV를 Managed Folder에 저장
            with folder.get_writer(csv_filename) as writer:
                writer.write(all_data.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'))
            logging.info(f"Aggregated CSV saved as {csv_filename}.")
            return html.Div([
                html.H3(f"추출된 데이터 ({len(all_data)} 행)"),
                dcc.Store(id='aggregated-data', data=all_data.to_dict('records')),
                html.Button("CSV 다운로드", id="btn_download_aggregated_csv"),
                dcc.Download(id="download-aggregated-csv")
            ])
        else:
            return html.Div("업로드된 PDF에서 데이터를 추출할 수 없습니다.")
    return None

# CSV 다운로드 콜백
@app.callback(
    Output("download-aggregated-csv", "data"),
    Input("btn_download_aggregated_csv", "n_clicks"),
    State('aggregated-data', 'data'),
    prevent_initial_call=True,
)
def download_aggregated_csv(n_clicks, data):
    if n_clicks and data:
        df = pd.DataFrame(data)
        csv_string = df.to_csv(index=False, encoding='utf-8-sig')
        return dcc.send_string(csv

```python
string, "aggregated_results.csv")
    return None

# 데이터 처리 및 시각화 콜백
@app.callback(
    [Output('processed-data-output', 'children'),
     Output('image-container', 'children'),
     Output("download-csv", "data"),
     Output("download-img", "data")],
    [Input('process-button', 'n_clicks')],
    [State('aggregated-data', 'data')],
    prevent_initial_call=True
)
def process_and_visualize(n_clicks, data):
    if n_clicks and data:
        df = pd.DataFrame(data)
        img_data, processed_df = create_heatmap(df)

        # 이미지를 base64로 인코딩하여 웹앱에 표시
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        img_element = html.Img(src='data:image/png;base64,{}'.format(img_base64), style={'width': '80%', 'height': 'auto'})

        # CSV 및 이미지 다운로드 데이터 준비
        processed_csv = processed_df.to_csv(index=True).encode('utf-8-sig')
        img_bytes = img_data

        return (
            html.Div([
                html.H3("데이터 처리 완료"),
                html.P(f"처리된 데이터는 {processed_df.shape[0]}개의 행과 {processed_df.shape[1]}개의 열을 포함합니다.")
            ]),
            img_element,
            dcc.send_bytes(processed_csv, "processed_data.csv"),
            dcc.send_bytes(img_bytes, "heatmap.png")
        )
    return None, None, None, None

# CSV 다운로드 버튼 콜백
@app.callback(
    Output("download-csv", "data"),
    Input("btn_csv_download", "n_clicks"),
    State("download-csv", "data"),
    prevent_initial_call=True,
)
def trigger_csv_download(n_clicks, data):
    if n_clicks and data:
        return data
    return None

# 이미지 다운로드 버튼 콜백
@app.callback(
    Output("download-img", "data"),
    Input("btn_img_download", "n_clicks"),
    State("download-img", "data"),
    prevent_initial_call=True,
)
def trigger_img_download(n_clicks, data):
    if n_clicks and data:
        return data
    return None

# Dash 앱 실행
if __name__ == '__main__':
    # Logging 설정
    logging.basicConfig(level=logging.INFO)
    app.run_server(debug=True, host='0.0.0.0')
