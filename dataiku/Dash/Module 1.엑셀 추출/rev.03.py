# 파일 1개 되는 거 확인함
# 파일 5개 처리하는 거 확인함
# 일단 이걸로 진행함

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import io
import base64
from dataiku import Folder
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# Dataiku Managed Folder 설정 (실제 사용 중인 Folder ID로 변경 필요)
folder = Folder("uploaded_excels")  # 실제 사용 중인 Folder ID로 변경

# Dataiku에서 제공하는 Dash 앱 인스턴스 사용
#app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Dash 업로드 크기 제한 설정 (필요 시 조정)
app.server.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 레이아웃 설정
app.layout = html.Div([
    html.H1("엑셀 파일 업로드 및 데이터 전처리"),

    # 엑셀 파일 업로드 컴포넌트 (multiple=True로 여러 파일 업로드 허용)
    dcc.Upload(
        id='upload-excels',
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

    html.Button("Download Aggregated CSV", id="btn-download", n_clicks=0),
    dcc.Download(id="download-dataframe-csv"),

    html.Div(id='output-excel-upload')
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

        # Area 데이터 정제 (Area_extra1~4을 이용)
        combined_df['Area'] = combined_df['Area'].fillna(combined_df['Area_extra1']) \
                                            .fillna(combined_df['Area_extra2']) \
                                            .fillna(combined_df['Area_extra3']) \
                                            .fillna(combined_df['Area_extra4'])

        # 필요 없는 Area_extra 컬럼 삭제
        combined_df = combined_df.drop(columns=['Area_extra1', 'Area_extra2', 'Area_extra3', 'Area_extra4'])

        # Peak Number가 NaN인 행 제거
        combined_df = combined_df.dropna(subset=['Peak Number'])
        combined_df['Peak Number'] = combined_df['Peak Number'].astype(int)

        # 피크 넘버가 1~24인 데이터만 포함 (필요에 따라 조정 가능)
        combined_df = combined_df[combined_df['Peak Number'].between(1, 24)]

        # Sort by 'Sample Name' and 'Peak Number'
        combined_df = combined_df.sort_values(by=['Sample Name', 'Peak Number']).reset_index(drop=True)

        # 필요한 컬럼만 선택
        combined_df = combined_df[['Sample Name', 'Peak Number', 'RT', 'Area', 'Height', '% Area', 'Total Area', 'Int Type']]

        logging.info(f"Parsed DataFrame from {filename} with {len(combined_df)} rows.")

        return combined_df, filename, None
    except Exception as e:
        logging.error(f"Error parsing {filename}: {e}")
        return pd.DataFrame(), filename, str(e)

# 콜백: 엑셀 파일 업로드 후 결과 출력
@app.callback(
    Output('output-excel-upload', 'children'),
    Input('upload-excels', 'contents'),
    State('upload-excels', 'filename')
)
def update_output(list_of_contents, list_of_names):
    aggregated_df = pd.DataFrame()
    if list_of_contents is not None and list_of_names is not None:
        for contents, name in zip(list_of_contents, list_of_names):
            df, fname, parse_error = parse_excel(contents, name)
            if parse_error:
                return html.Div([
                    html.H3(f"엑셀 파일 '{fname}'에서 데이터를 추출할 수 없습니다."),
                    html.P(f"오류 메시지: {parse_error}")
                ])
            # Append to aggregated_df
            aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)
       
        # Remove duplicates if any
        aggregated_df = aggregated_df.drop_duplicates()

        # Sort the DataFrame if needed
        aggregated_df = aggregated_df.sort_values(by=['Sample Name', 'Peak Number']).reset_index(drop=True)

        # Store aggregated_df in a hidden div or use dcc.Store to pass data
        return html.Div([
            html.H3(f"엑셀 파일들이 성공적으로 업로드 및 처리되었습니다."),
            html.P(f"총 {len(aggregated_df)}개의 피크 데이터가 추출되었습니다.")
        ])
    return None

# 콜백: Download 버튼 클릭 시 CSV 파일 다운로드
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download", "n_clicks"),
    State('upload-excels', 'contents'),
    State('upload-excels', 'filename'),
    prevent_initial_call=True,
)
def generate_csv(n_clicks, list_of_contents, list_of_names):
    if n_clicks > 0 and list_of_contents is not None and list_of_names is not None:
        aggregated_df = pd.DataFrame()
        for contents, name in zip(list_of_contents, list_of_names):
            df, fname, parse_error = parse_excel(contents, name)
            if parse_error:
                continue  # Skip files with errors
            # Append to aggregated_df
            aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)
       
        # Remove duplicates if any
        aggregated_df = aggregated_df.drop_duplicates()

        # Sort the DataFrame if needed
        aggregated_df = aggregated_df.sort_values(by=['Sample Name', 'Peak Number']).reset_index(drop=True)

        # Save to CSV as string
        csv_string = aggregated_df.to_csv(index=False, encoding='utf-8-sig')

        return dict(content=csv_string, filename="aggregated_results.csv")

    return None

# Dash 앱 실행 (Dataiku 환경에 맞게 수정 필요)
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

