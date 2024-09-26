# 잘 안됨. 
# 처음부터 수정해야 함.

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from dataiku import Folder
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# Dataiku Managed Folder 설정 (Folder ID를 정확히 설정하세요)
folder = Folder("uploaded_excels")  # 예시: 'uploaded_excels' Managed Folder ID

# Dataiku에서 제공하는 Dash 앱 인스턴스 사용
#app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Dash 업로드 크기 제한 설정 (필요 시 조정)
app.server.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 레이아웃 설정
app.layout = html.Div([
    html.H1("엑셀 파일 업로드 및 데이터 처리"),
   
    # 엑셀 파일 업로드 컴포넌트
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
   
    html.Div(id='output-excel-upload'),
   
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
        data_df = pd.read_excel(io.BytesIO(decoded), header=19)
        logging.info(f"Data extracted from {filename} with shape {data_df.shape}")
       
        # Sample Name 컬럼 추가
        data_df['Sample Name'] = sample_name
       
        return data_df, filename
    except Exception as e:
        logging.error(f"Error parsing {filename}: {e}")
        return pd.DataFrame(), filename

# Helper 함수: Area 데이터 오류 수정
def fix_area_data(df):
    try:
        # D열 (Index 3)와 E열 (Index 4)의 데이터를 F열 ('Area')으로 이동
        df['Area'] = df['F']  # 기본적으로 F열 값을 Area로 설정
        df.loc[df['D'].notna(), 'Area'] = df.loc[df['D'].notna(), 'D']  # D열에 값이 있으면 Area로 설정
        df.loc[df['E'].notna(), 'Area'] = df.loc[df['E'].notna(), 'E']  # E열에 값이 있으면 Area로 설정

        # P열 (Index 15)의 데이터를 Q열 (Index 16)의 Area로 이동
        df['Area_Q'] = df['Q']
        df.loc[df['P'].notna(), 'Area_Q'] = df.loc[df['P'].notna(), 'P']
       
        # Area_Q가 있는 경우 Area 컬럼에 추가
        df_final = pd.concat([df, df['Area_Q']], axis=0).reset_index(drop=True)
        df_final = df_final.drop(columns=['Area_Q'])
       
        logging.info("Area data fixed successfully.")
        return df_final
    except Exception as e:
        logging.error(f"Error fixing area data: {e}")
        return df

# Helper 함수: 히트맵 생성
def create_heatmap(df):
    try:
        df['Sample Base Name'] = df['Sample Name'].str.split('-').str[0]
        max_rt_per_sample = df.groupby('Sample Name')['RT'].transform('max')
        df['RRT'] = (df['RT'] / max_rt_per_sample).round(2)
        avg_area_per_rrt = df.groupby(['Sample Name', 'RRT'])['Area'].mean().reset_index()
        pivot_df = avg_area_per_rrt.pivot(index='RRT', columns='Sample Name', values='Area')
       
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
        plt.title('Average Area vs RRT by Sample Base Name', fontsize=20)
        plt.xlabel('Sample Base Name', fontsize=14)
        plt.ylabel('RRT', fontsize=14)

        # 이미지 저장
        img_io = BytesIO()
        plt.savefig(img_io, format='png')
        plt.close(fig)
        img_io.seek(0)
       
        logging.info("Heatmap created successfully.")
        return img_io.getvalue(), final_result_df
    except Exception as e:
        logging.error(f"Error creating heatmap: {e}")
        return None, pd.DataFrame()

# 콜백: 엑셀 파일 업로드 후 결과 출력
@app.callback(
    Output('output-excel-upload', 'children'),
    [Input('upload-excels', 'contents')],
    [State('upload-excels', 'filename')]
)
def update_output(contents, filenames):
    if contents is not None and filenames is not None:
        all_data = pd.DataFrame()
        for content, filename in zip(contents, filenames):
            df, fname = parse_excel(content, filename)
            if df.empty:
                logging.warning(f"No data extracted from {fname}.")
                continue
            df = fix_area_data(df)
            if not df.empty:
                all_data = pd.concat([all_data, df], ignore_index=True)
        if not all_data.empty:
            # 현재 날짜를 YYYYMMDD 형식으로 가져오기
            current_date = pd.Timestamp.now().strftime("%Y%m%d")
            # 파일명에 날짜 포함
            csv_filename = f'aggregated_results_{current_date}.csv'
            # CSV를 Managed Folder에 저장
            try:
                with folder.get_writer(csv_filename) as writer:
                    writer.write(all_data.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'))
                logging.info(f"Aggregated CSV saved as {csv_filename}.")
                return html.Div([
                    html.H3(f"엑셀 파일 {len(filenames)}개가 성공적으로 업로드 및 처리되었습니다."),
                    html.P(f"Aggregated CSV: {csv_filename} 저장 완료.")
                ])
            except Exception as e:
                logging.error(f"Error saving aggregated CSV: {e}")
                return html.Div("CSV 파일 저장 중 오류가 발생했습니다.")
        else:
            return html.Div("업로드된 엑셀 파일에서 데이터를 추출할 수 없습니다.")
    return None

# 콜백: 데이터 처리 및 시각화 실행
@app.callback(
    [Output('processed-data-output', 'children'),
     Output('image-container', 'children')],
    [Input('process-button', 'n_clicks')],
    [State('upload-excels', 'contents'),
     State('upload-excels', 'filename')],
    prevent_initial_call=True
)
def process_and_visualize(n_clicks, contents, filenames):
    if n_clicks and contents and filenames:
        all_data = pd.DataFrame()
        for content, filename in zip(contents, filenames):
            df, fname = parse_excel(content, filename)
            if df.empty:
                logging.warning(f"No data extracted from {fname}.")
                continue
            df = fix_area_data(df)
            if not df.empty:
                all_data = pd.concat([all_data, df], ignore_index=True)
        if not all_data.empty:
            img_data, processed_df = create_heatmap(all_data)
            if img_data is None:
                return html.Div("히트맵 생성 중 오류가 발생했습니다."), None

            # 이미지를 base64로 인코딩하여 웹앱에 표시
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            img_element = html.Img(src='data:image/png;base64,{}'.format(img_base64), style={'width': '80%', 'height': 'auto'})
           
            # 최종 CSV 파일 생성 및 저장
            final_csv_filename = f'final_results_{pd.Timestamp.now().strftime("%Y%m%d")}.csv'
            try:
                with folder.get_writer(final_csv_filename) as writer:
                    processed_df.to_csv(writer, index=True, encoding='utf-8-sig')
                logging.info(f"Final CSV saved as {final_csv_filename}.")
            except Exception as e:
                logging.error(f"Error saving final CSV: {e}")
                return html.Div("최종 CSV 파일 저장 중 오류가 발생했습니다."), img_element
           
            return (
                html.Div([
                    html.H3("데이터 처리 및 시각화가 완료되었습니다."),
                    html.P(f"처리된 데이터는 {processed_df.shape[0]}개의 행과 {processed_df.shape[1]}개의 열을 포함합니다."),
                    html.P(f"Final CSV: {final_csv_filename} 저장 완료.")
                ]),
                img_element
            )
        else:
            return html.Div("데이터 처리 중 오류가 발생했습니다."), None
    return None, None

# 콜백: CSV 다운로드
@app.callback(
    Output("download-csv", "data"),
    Input("btn_csv_download", "n_clicks"),
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    if n_clicks:
        try:
            # 현재 날짜를 YYYYMMDD 형식으로 가져오기
            current_date = pd.Timestamp.now().strftime("%Y%m%d")
            final_csv_filename = f'final_results_{current_date}.csv'
            # Managed Folder에서 파일 읽기
            with folder.get_reader(final_csv_filename) as reader:
                csv_content = reader.read()
            logging.info(f"CSV 파일 {final_csv_filename} 다운로드 요청.")
            return dcc.send_bytes(csv_content, final_csv_filename)
        except Exception as e:
            logging.error(f"Error downloading CSV: {e}")
            return None
    return None

# 콜백: 히트맵 이미지 다운로드
@app.callback(
    Output("download-img", "data"),
    Input("btn_img_download", "n_clicks"),
    [State('upload-excels', 'contents'),
     State('upload-excels', 'filename')],
    prevent_initial_call=True,
)
def download_img(n_clicks, contents, filenames):
    if n_clicks and contents and filenames:
        all_data = pd.DataFrame()
        for content, filename in zip(contents, filenames):
            df, fname = parse_excel(content, filename)
            if df.empty:
                logging.warning(f"No data extracted from {fname}.")
                continue
            df = fix_area_data(df)
            if not df.empty:
                all_data = pd.concat([all_data, df], ignore_index=True)
        if not all_data.empty:
            img_data, _ = create_heatmap(all_data)
            if img_data is None:
                return None
            logging.info("히트맵 이미지 다운로드 요청.")
            return dcc.send_bytes(img_data, "heatmap.png")
    return None

# Dash 앱 실행 (Dataiku 환경에 맞게 수정 필요)
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

