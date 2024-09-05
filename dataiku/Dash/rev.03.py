# 프로세스 1 : 다수의 pdf 일괄 업로드 > 텍스트 추출 > 엑셀 변환 후 다운로드 > 
# 프로세스 2 : 변환된 엑셀 업로드 > 히트맵 이미지와 전처리된 엑셀로 다운로드

# 위 프로세스를 한 번에 실행하도록 함

# pdfplumber 설치가 안되어서 아래 코드 실행 불가함

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import io
import base64
import pdfplumber
import re
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Dash 앱 생성
app = dash.Dash(__name__)

# Layout 설정
app.layout = html.Div([
    html.H1("PDF Upload and Processing with OCR"),
   
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
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
        multiple=True  # 다수의 파일을 업로드할 수 있도록 설정
    ),
   
    html.Div(id='output-data-upload'),
   
    html.Hr(),
   
    html.Div(id='image-container'),  # 이미지가 표시될 컨테이너
    html.Div(id='download-links'),
    dcc.Download(id="download-csv"),
    dcc.Download(id="download-img")
])

# Helper functions
def parse_pdf(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
   
    # PDF 파일에서 텍스트 추출
    with pdfplumber.open(io.BytesIO(decoded)) as pdf:
        text_list = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_list.append(text)
   
    return text_list

def extract_data_from_text(text_list):
    columns = ['Sample Name', 'Peak Number', 'RT', 'Area', 'Height', '% Area', 'Total Area', 'Int Type']
    df = pd.DataFrame(columns=columns)
   
    for text in text_list:
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
            new_row = pd.DataFrame({'Sample Name': [sample_name], 'Peak Number': [peak[0]], 'RT': [peak[1]],
                                    'Area': [peak[2]], 'Height': [peak[3]], '% Area': [peak[4]],
                                    'Total Area': [peak[5]], 'Int Type': [peak[6]]}, columns=columns)
            df = pd.concat([df, new_row], ignore_index=True)
   
    return df

def create_heatmap(df):
    df['Sample Base Name'] = df['Sample Name'].str.split('-').str[0]
    max_rt_per_sample = df.groupby('Sample Name')['Area'].idxmax().apply(lambda x: df.loc[x, 'RT'])
    df['RRT'] = df.apply(lambda row: row['RT'] / max_rt_per_sample[row['Sample Name']], axis=1)
    df['RRT'] = df['RRT'].round(2)
    avg_area_per_rrt = df.groupby(['Sample Name', 'RRT'])['% Area'].mean().reset_index()
    pivot_df = avg_area_per_rrt.pivot(index='RRT', columns='Sample Name', values='% Area')
    base_names = df['Sample Base Name'].unique()
    for base_name in base_names:
        sample_columns = [col for col in pivot_df.columns if base_name in col]
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

# Callbacks
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('download-links', 'children'),
     Output('image-container', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        return None, None, None
   
    all_dfs = []
    for content, name in zip(contents, filename):
        text_list = parse_pdf(content, name)
        df = extract_data_from_text(text_list)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    img_data, processed_df = create_heatmap(combined_df)
   
    # Save image and CSV to Managed Folder
    csv_bytes = processed_df.to_csv(index=True).encode()
    csv_download_link = dcc.send_bytes(csv_bytes, f'processed_{filename[0]}.csv')
    img_download_link = dcc.send_bytes(img_data, f'heatmap_{filename[0].replace(".pdf", ".png")}')
   
    # 이미지를 base64로 인코딩하여 웹앱에 표시
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    img_element = html.Img(src='data:image/png;base64,{}'.format(img_base64), style={'width': '80%', 'height': 'auto'})
   
    return html.Div([f'File {filename[0]} processed successfully.']), html.Div([
        html.Button("Download CSV", id="btn_csv_download"),
        html.Button("Download Image", id="btn_img_download"),
        dcc.Download(id="download-csv"),
        dcc.Download(id="download-img")
    ]), img_element


@app.callback(
    Output("download-csv", "data"),
    Input("btn_csv_download", "n_clicks"),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True,
)
def download_csv(n_clicks, contents, filename):
    if n_clicks:
        all_dfs = []
        for content, name in zip(contents, filename):
            text_list = parse_pdf(content, name)
            df = extract_data_from_text(text_list)
            all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        _, processed_df = create_heatmap(combined_df)
        csv_bytes = processed_df.to_csv(index=True).encode()
        return dcc.send_bytes(csv_bytes, "processed_file.csv")


@app.callback(
    Output("download-img", "data"),
    Input("btn_img_download", "n_clicks"),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True,
)
def download_img(n_clicks, contents, filename):
    if n_clicks:
        all_dfs = []
        for content, name in zip(contents, filename):
            text_list = parse_pdf(content, name)
            df = extract_data_from_text(text_list)
            all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        img_data, _ = create_heatmap(combined_df)
        return dcc.send_bytes(img_data, "heatmap_file.png")


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

