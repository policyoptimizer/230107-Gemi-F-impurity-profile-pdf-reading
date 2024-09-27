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

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# Dataiku Managed Folder 설정 (필요 시 활성화)
# from dataiku import Folder
# folder = Folder("uploaded_excels")  # 실제 사용 중인 Folder ID로 변경

# Dataiku에서 제공하는 Dash 앱 인스턴스 사용
# Dataiku 환경에서는 이미 'app' 객체가 정의되어 있을 가능성이 있습니다.
# 따라서, 별도의 Dash 인스턴스를 생성하지 않고 기존 'app' 객체를 사용합니다.
# app = dash.Dash(__name__, suppress_callback_exceptions=True)

# 레이아웃 설정
app.layout = html.Div([
    html.H1("엑셀 파일 업로드 및 데이터 전처리 및 히트맵 생성"),
   
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

    # 중간 데이터를 저장할 Store 컴포넌트
    dcc.Store(id='aggregated-data-store'),
    dcc.Store(id='heatmap-image-store'),

    html.Div(id='output-excel-upload'),

    # 다운로드 버튼 및 히트맵 이미지 표시
    html.Div([
        html.Button("Download Aggregated CSV", id="btn-download-csv", n_clicks=0, style={'margin': '10px'}),
        dcc.Download(id="download-dataframe-csv"),
       
        html.Button("Download Heatmap Image", id="btn-download-img", n_clicks=0, style={'margin': '10px'}),
        dcc.Download(id="download-heatmap-img"),
    ]),

    html.Div(id='image-container')  # 히트맵 이미지가 표시될 컨테이너
])

# Helper 함수: 엑셀 파일 내용을 파싱하여 DataFrame 반환
def parse_excel(contents, filename):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # 엑셀 파일 전체를 읽어 Sample Name 추출
        excel_data = pd.read_excel(io.BytesIO(decoded), header=None)
       
        # G3 셀 (0-based 인덱스: 행 2, 열 6)에 있는 Sample Name 추출
        if excel_data.shape[0] > 2 and excel_data.shape[1] > 6:
            sample_name = excel_data.iloc[2, 6]
            logging.info(f"Extracted Sample Name: {sample_name} from {filename}")
        else:
            raise IndexError("Sample Name 위치에 데이터가 없습니다.")
       
        # 실제 데이터는 20번째 행부터 시작 (0-based 인덱스 19)
        raw_data = pd.read_excel(io.BytesIO(decoded), header=19, dtype=str)
        logging.info(f"Data extracted from {filename} with shape {raw_data.shape}")

        # 필요한 최소 열이 있는지 확인
        required_columns = [0,2,3,4,5,6,7,9,11,12,14,15,16,17,18,19,20]
        if not all(col < raw_data.shape[1] for col in required_columns):
            raise IndexError("필요한 열이 엑셀 파일에 존재하지 않습니다.")

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

        # 피크 넘버가 1~24인 데이터만 포함
        combined_df = combined_df[combined_df['Peak Number'].between(1, 24)]

        # Sort by 'Sample Name' and 'Peak Number'
        combined_df = combined_df.sort_values(by=['Sample Name', 'Peak Number']).reset_index(drop=True)

        # 필요한 컬럼만 선택
        combined_df = combined_df[['Sample Name', 'Peak Number', 'RT', 'Area', 'Height', '% Area', 'Total Area', 'Int Type']]

        logging.info(f"Parsed DataFrame from {filename} with {len(combined_df)} rows.")

        return combined_df, filename, None
    except IndexError as ie:
        logging.error(f"IndexError in {filename}: {ie}")
        return pd.DataFrame(), filename, str(ie)
    except Exception as e:
        logging.error(f"Error parsing {filename}: {e}")
        return pd.DataFrame(), filename, str(e)

# Helper 함수: 히트맵 생성
def create_heatmap(df):
    try:
        # Sample Base Name 추출
        df['Sample Base Name'] = df['Sample Name'].str.split('-').str[0]
       
        # 각 Sample Name별 최대 Area의 RT 계산
        max_rt_per_sample = df.loc[df.groupby('Sample Name')['Area'].idxmax()].set_index('Sample Name')['RT']
       
        # RRT 계산
        df['RRT'] = df.apply(lambda row: row['RT'] / max_rt_per_sample.get(row['Sample Name'], 1), axis=1)
        df['RRT'] = df['RRT'].round(2)
       
        # Sample Base Name별 RRT에 따른 평균 % Area 계산
        avg_area_per_rrt = df.groupby(['Sample Name', 'RRT'])['% Area'].mean().reset_index()
       
        # 피벗 테이블 생성
        pivot_df = avg_area_per_rrt.pivot(index='RRT', columns='Sample Name', values='% Area')
       
        # Sample Base Name별 평균 % Area 계산
        base_names = df['Sample Base Name'].unique()
        for base_name in base_names:
            sample_columns = [col for col in pivot_df.columns if base_name in col]
            if sample_columns:
                pivot_df[base_name] = pivot_df[sample_columns].mean(axis=1)
       
        # 중복된 Sample Base Name 컬럼 제거
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
       
        # 이미지 데이터를 base64로 인코딩
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        return img_base64, final_result_df
    except Exception as e:
        logging.error(f"Error creating heatmap: {e}")
        return None, pd.DataFrame()

# 콜백: 엑셀 파일 업로드 후 결과 처리
@app.callback(
    [Output('output-excel-upload', 'children'),
     Output('aggregated-data-store', 'data'),
     Output('heatmap-image-store', 'data'),
     Output('image-container', 'children')],
    [Input('upload-excels', 'contents')],
    [State('upload-excels', 'filename')]
)
def process_upload(list_of_contents, list_of_names):
    if list_of_contents is not None and list_of_names is not None:
        aggregated_df = pd.DataFrame()
        error_messages = []
        for contents, name in zip(list_of_contents, list_of_names):
            df, fname, parse_error = parse_excel(contents, name)
            if parse_error:
                error_messages.append(f"엑셀 파일 '{fname}'에서 데이터를 추출할 수 없습니다. 오류 메시지: {parse_error}")
                continue  # 해당 파일을 건너뜁니다.
            # Append to aggregated_df
            aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)
       
        if aggregated_df.empty:
            return (
                html.Div([
                    html.H3("업로드된 모든 엑셀 파일에서 데이터를 추출할 수 없었습니다."),
                    html.P("오류 메시지:"),
                    html.Ul([html.Li(msg) for msg in error_messages])
                ]),
                None,
                None,
                None
            )
       
        # Remove duplicates if any
        aggregated_df = aggregated_df.drop_duplicates()
       
        # 히트맵 생성
        img_base64, heatmap_df = create_heatmap(aggregated_df)
        if img_base64 is None:
            return (
                html.Div([
                    html.H3("히트맵 생성 중 오류가 발생했습니다.")
                ]),
                None,
                None,
                None
            )
       
        # 히트맵 이미지를 base64로 인코딩하여 웹앱에 표시
        img_element = html.Img(src='data:image/png;base64,{}'.format(img_base64), style={'width': '80%', 'height': 'auto'})
       
        # CSV 다운로드 준비
        csv_string = aggregated_df.to_csv(index=False, encoding='utf-8-sig')
        csv_data = csv_string  # 저장소에 저장할 데이터
       
        # 히트맵 이미지 다운로드 준비
        heatmap_img = img_base64  # 저장소에 저장할 데이터 as base64 string
       
        # 업데이트된 데이터를 Store에 저장
        return (
            html.Div([
                html.H3("엑셀 파일들이 성공적으로 업로드 및 처리되었습니다."),
                html.P(f"총 {len(aggregated_df)}개의 피크 데이터가 추출되었습니다."),
                html.Ul([html.Li(msg) for msg in error_messages]) if error_messages else None
            ]),
            csv_data,
            heatmap_img,
            img_element
        )
   
    return (None, None, None, None)

# 콜백: CSV 파일 다운로드
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    State('aggregated-data-store', 'data'),
    prevent_initial_call=True,
)
def download_csv(n_clicks, aggregated_data):
    if n_clicks > 0 and aggregated_data is not None:
        return dict(content=aggregated_data, filename="aggregated_results.csv")
    return None

# 콜백: 히트맵 이미지 다운로드
@app.callback(
    Output("download-heatmap-img", "data"),
    Input("btn-download-img", "n_clicks"),
    State('heatmap-image-store', 'data'),
    prevent_initial_call=True,
)
def download_heatmap(n_clicks, heatmap_image_base64):
    if n_clicks > 0 and heatmap_image_base64 is not None:
        img_bytes = base64.b64decode(heatmap_image_base64)
        return dict(content=img_bytes, filename="heatmap_results.png")
    return None

# Dash 앱 실행 (Dataiku 환경에 맞게 수정 필요)
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

