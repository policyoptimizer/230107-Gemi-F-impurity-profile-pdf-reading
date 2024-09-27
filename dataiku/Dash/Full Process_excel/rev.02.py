# 제한적으로 돌아감
# 피크 개수가 20개를 넘지 않으면 에러남

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import io
import base64
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from dataiku import Folder

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# Dataiku에서 제공하는 Dash 앱 인스턴스 사용 (Dataiku 환경에서는 app 인스턴스를 생성하지 않습니다)
# app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Dash 업로드 크기 제한 설정 (필요 시 조정)
app.server.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 레이아웃 설정
app.layout = html.Div([
    html.H1("엑셀 파일 업로드 및 데이터 처리"),

    # 엑셀 파일 업로드 컴포넌트
    dcc.Upload(
        id='upload-excels',
        children=html.Div([
            '파일을 드래그 앤 드롭하거나 ',
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

    html.Button("데이터 처리 시작", id="process-data-button", n_clicks=0),

    # 데이터 저장을 위한 dcc.Store
    dcc.Store(id='aggregated-data-store'),
    dcc.Store(id='processed-data-store'),
    dcc.Store(id='img-data-store'),

    # 결과 출력 영역
    html.Div(id='processing-output'),

    html.Hr(),

    # 이미지 표시 영역
    html.Div(id='image-container'),

    # 다운로드 버튼들
    html.Button("CSV 다운로드", id="btn_csv_download"),
    html.Button("이미지 다운로드", id="btn_img_download"),
    dcc.Download(id="download-csv"),
    dcc.Download(id="download-img")
])

# Helper 함수: 엑셀 파일 내용을 파싱하여 DataFrame 반환
def parse_excel(contents, filename):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # 엑셀 파일 전체를 읽어 Sample Name 추출
        excel_data = pd.read_excel(io.BytesIO(decoded), header=None)
        # G3 셀 (0-based 인덱스: 행 2, 열 6)에 있는 Sample Name 추출
        sample_name = excel_data.iloc[2, 6]
        logging.info(f"Extracted Sample Name: {sample_name} from {filename}")

        # 실제 데이터는 20번째 행부터 시작 (0-based 인덱스 19)
        raw_data = pd.read_excel(io.BytesIO(decoded), header=19, dtype=str)
        logging.info(f"Data extracted from {filename} with shape {raw_data.shape}")

        # 첫 번째 세트 (A, C, F, G, H, J, L, D, E)
        df1 = pd.DataFrame({
            'Sample Name': sample_name,
            'Peak Number': pd.to_numeric(raw_data.iloc[:, 0], errors='coerce'),  # A
            'RT': pd.to_numeric(raw_data.iloc[:, 2], errors='coerce'),  # C
            'Area': pd.to_numeric(raw_data.iloc[:, 5], errors='coerce'),  # F
            'Height': pd.to_numeric(raw_data.iloc[:, 6], errors='coerce'),  # G
            '% Area': pd.to_numeric(raw_data.iloc[:, 7], errors='coerce'),  # H
            'Total Area': pd.to_numeric(raw_data.iloc[:, 9], errors='coerce'),  # J
            'Int Type': raw_data.iloc[:, 11],  # L
            'Area_extra1': pd.to_numeric(raw_data.iloc[:, 3], errors='coerce'),  # D
            'Area_extra2': pd.to_numeric(raw_data.iloc[:, 4], errors='coerce'),  # E
        })

        # 두 번째 세트 (M, O, Q, R, S, T, U, P, Q)
        df2 = pd.DataFrame({
            'Sample Name': sample_name,
            'Peak Number': pd.to_numeric(raw_data.iloc[:, 12], errors='coerce'),  # M
            'RT': pd.to_numeric(raw_data.iloc[:, 14], errors='coerce'),  # O
            'Area': pd.to_numeric(raw_data.iloc[:, 16], errors='coerce'),  # Q
            'Height': pd.to_numeric(raw_data.iloc[:, 17], errors='coerce'),  # R
            '% Area': pd.to_numeric(raw_data.iloc[:, 18], errors='coerce'),  # S
            'Total Area': pd.to_numeric(raw_data.iloc[:, 19], errors='coerce'),  # T
            'Int Type': raw_data.iloc[:, 20],  # U
            'Area_extra3': pd.to_numeric(raw_data.iloc[:, 15], errors='coerce'),  # P
            'Area_extra4': pd.to_numeric(raw_data.iloc[:, 16], errors='coerce'),  # Q
        })

        # 두 DataFrame 합치기
        combined_df = pd.concat([df1, df2], ignore_index=True)

        # Area 데이터 정제 (Area_extra1~4를 이용)
        combined_df['Area'] = combined_df['Area'].fillna(combined_df['Area_extra1']) \
                                                .fillna(combined_df['Area_extra2']) \
                                                .fillna(combined_df['Area_extra3']) \
                                                .fillna(combined_df['Area_extra4'])

        # 필요 없는 Area_extra 컬럼 삭제
        combined_df = combined_df.drop(columns=['Area_extra1', 'Area_extra2', 'Area_extra3', 'Area_extra4'])

        # Peak Number가 NaN인 행 제거
        combined_df = combined_df.dropna(subset=['Peak Number'])
        combined_df['Peak Number'] = combined_df['Peak Number'].astype(int)

        # 피크 넘버가 1~24인 데이터만 포함
        combined_df = combined_df[combined_df['Peak Number'].between(1, 24)]

        # 필요한 컬럼만 선택
        combined_df = combined_df[['Sample Name', 'Peak Number', 'RT', 'Area', 'Height', '% Area', 'Total Area', 'Int Type']]

        logging.info(f"Parsed DataFrame from {filename} with {len(combined_df)} rows.")

        return combined_df, filename, None
    except Exception as e:
        logging.error(f"Error parsing {filename}: {e}")
        return pd.DataFrame(), filename, str(e)

# 히트맵 생성 함수
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

# 콜백: 업로드된 엑셀 파일 처리
@app.callback(
    [Output('processing-output', 'children'),
     Output('aggregated-data-store', 'data')],
    Input('process-data-button', 'n_clicks'),
    State('upload-excels', 'contents'),
    State('upload-excels', 'filename')
)
def process_uploaded_excels(n_clicks, list_of_contents, list_of_names):
    if n_clicks > 0 and list_of_contents is not None and list_of_names is not None:
        aggregated_df = pd.DataFrame()
        error_messages = []
        for contents, name in zip(list_of_contents, list_of_names):
            df, fname, parse_error = parse_excel(contents, name)
            if parse_error:
                error_messages.append(f"파일 '{fname}'에서 오류 발생: {parse_error}")
                continue
            # Append to aggregated_df
            aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)
           
        if aggregated_df.empty:
            return html.Div([
                html.H3("업로드된 파일에서 데이터를 추출할 수 없습니다."),
                html.P("\n".join(error_messages))
            ]), None
           
        # Remove duplicates if any
        aggregated_df = aggregated_df.drop_duplicates()

        # 필요한 경우 정렬
        aggregated_df = aggregated_df.sort_values(by=['Sample Name', 'Peak Number']).reset_index(drop=True)

        # DataFrame을 JSON으로 변환하여 저장
        data_json = aggregated_df.to_json(date_format='iso', orient='split')

        return html.Div([
            html.H3("데이터가 성공적으로 처리되었습니다."),
            html.P(f"총 {len(aggregated_df)}개의 피크 데이터가 추출되었습니다.")
        ]), data_json
    else:
        return None, None

# 콜백: Aggregated Data를 사용하여 이미지 생성 및 데이터 저장
@app.callback(
    [Output('image-container', 'children'),
     Output('processed-data-store', 'data'),
     Output('img-data-store', 'data')],
    Input('aggregated-data-store', 'data')
)
def update_image_container(data_json):
    if data_json is not None:
        aggregated_df = pd.read_json(data_json, orient='split')
        img_data, processed_df = create_heatmap(aggregated_df)

        # DataFrame을 JSON으로 변환하여 저장
        processed_df_json = processed_df.to_json(date_format='iso', orient='split')

        # 이미지 데이터를 base64로 인코딩하여 저장
        img_data_base64 = base64.b64encode(img_data).decode('utf-8')

        # 이미지를 웹앱에 표시
        img_element = html.Img(src='data:image/png;base64,{}'.format(img_data_base64), style={'width': '80%', 'height': 'auto'})

        return img_element, processed_df_json, img_data_base64
    else:
        return None, None, None

# 콜백: CSV 다운로드
@app.callback(
    Output("download-csv", "data"),
    Input("btn_csv_download", "n_clicks"),
    State('processed-data-store', 'data'),
    prevent_initial_call=True,
)
def download_csv(n_clicks, processed_df_json):
    if n_clicks and processed_df_json is not None:
        processed_df = pd.read_json(processed_df_json, orient='split')
        csv_bytes = processed_df.to_csv(index=True, encoding='utf-8-sig').encode('utf-8-sig')
        return dict(content=csv_bytes, filename="processed_data.csv")
    else:
        return None

# 콜백: 이미지 다운로드
@app.callback(
    Output("download-img", "data"),
    Input("btn_img_download", "n_clicks"),
    State('img-data-store', 'data'),
    prevent_initial_call=True,
)
def download_img(n_clicks, img_data_base64):
    if n_clicks and img_data_base64 is not None:
        img_data = base64.b64decode(img_data_base64)
        return dict(content=img_data, filename="heatmap.png")
    else:
        return None

# Dash 앱 실행 (Dataiku 환경에 맞게 수정 필요)
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
