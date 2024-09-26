# 그나마 되긴 됨
# 아직 깔끔하게 되진 않지만 파인 튜닝 해나가면 됨

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
# app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Dash 업로드 크기 제한 설정 (필요 시 조정)
app.server.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 레이아웃 설정
app.layout = html.Div([
    html.H1("엑셀 파일 업로드 및 데이터 전처리"),

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
        multiple=False  # 단일 파일 업로드
    ),

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
        data_df = pd.read_excel(io.BytesIO(decoded), header=19)
        logging.info(f"Data extracted from {filename} with shape {data_df.shape}")
        logging.info(f"DataFrame columns: {list(data_df.columns)}")

        # Sample Name 컬럼 추가
        data_df['Sample Name'] = sample_name

        return data_df, filename, None
    except Exception as e:
        logging.error(f"Error parsing {filename}: {e}")
        return pd.DataFrame(), filename, str(e)

# Helper 함수: Area 데이터 오류 수정
def fix_area_data(df):
    try:
        logging.info(f"Initial DataFrame columns before fixing: {list(df.columns)}")
        # 'Area' 컬럼이 있는지 확인
        if 'Area' not in df.columns:
            logging.warning("Column 'Area' not found in DataFrame. Creating 'Area' column with NaN values.")
            df['Area'] = pd.NA

        # 'D', 'E', 'P', 'Q' 컬럼이 있는 경우 'Area' 컬럼을 채움
        if 'D' in df.columns:
            df['Area'] = df['Area'].fillna(df['D'])
            logging.info("Filled 'Area' from 'D' column.")
        if 'E' in df.columns:
            df['Area'] = df['Area'].fillna(df['E'])
            logging.info("Filled 'Area' from 'E' column.")
        if 'P' in df.columns:
            df['Area'] = df['Area'].fillna(df['P'])
            logging.info("Filled 'Area' from 'P' column.")
        if 'Q' in df.columns:
            df['Area'] = df['Area'].fillna(df['Q'])
            logging.info("Filled 'Area' from 'Q' column.")

        # 필요 없는 컬럼 삭제 (선택 사항)
        cols_to_drop = [col for col in ['D', 'E', 'P', 'Q'] if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logging.info(f"Dropped columns: {cols_to_drop}")

        logging.info("Area data fixed successfully.")
        logging.info(f"DataFrame columns after fixing: {list(df.columns)}")
        return df, None
    except Exception as e:
        logging.error(f"Error fixing area data: {e}")
        return df, str(e)

# 콜백: 엑셀 파일 업로드 후 결과 출력
@app.callback(
    Output('output-excel-upload', 'children'),
    [Input('upload-excels', 'contents')],
    [State('upload-excels', 'filename')]
)
def update_output(contents, filename):
    if contents is not None and filename is not None:
        df, fname, parse_error = parse_excel(contents, filename)
        if parse_error:
            return html.Div([
                html.H3(f"엑셀 파일 '{fname}'에서 데이터를 추출할 수 없습니다."),
                html.P(f"오류 메시지: {parse_error}")
            ])

        df, fix_error = fix_area_data(df)
        if fix_error:
            return html.Div([
                html.H3(f"엑셀 파일 '{fname}'의 데이터 전처리 중 오류가 발생했습니다."),
                html.P(f"오류 메시지: {fix_error}")
            ])

        if df.empty:
            logging.warning(f"No data extracted from {fname}.")
            return html.Div([
                html.H3(f"엑셀 파일 '{fname}'에서 데이터를 추출할 수 없습니다.")
            ])

        # 현재 날짜를 YYYYMMDD 형식으로 가져오기
        current_date = pd.Timestamp.now().strftime("%Y%m%d")
        # 파일명에 날짜 포함
        csv_filename = f'aggregated_results_{current_date}.csv'
        # CSV를 Managed Folder에 저장
        try:
            # DataFrame을 CSV 문자열로 변환 후 바이트로 인코딩
            csv_content = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            with folder.get_writer(csv_filename) as writer:
                writer.write(csv_content)
            logging.info(f"Aggregated CSV saved as {csv_filename}.")
            return html.Div([
                html.H3(f"엑셀 파일 '{fname}'가 성공적으로 업로드 및 처리되었습니다."),
                html.P(f"Aggregated CSV 파일: {csv_filename} 저장 완료.")
            ])
        except Exception as e:
            logging.error(f"Error saving aggregated CSV: {e}")
            return html.Div([
                html.H3(f"엑셀 파일 '{fname}'의 CSV 저장 중 오류가 발생했습니다."),
                html.P(f"오류 메시지: {str(e)}")
            ])
    return None

# Dash 앱 실행 (Dataiku 환경에 맞게 수정 필요)
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

