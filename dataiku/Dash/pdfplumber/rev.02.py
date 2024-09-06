# 안됨. 에러남

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
#app = dash.Dash(__name__)

# Layout 설정
app.layout = html.Div([
    html.H1("PDF Upload and Generate Heatmap with Processed CSV"),
   
    dcc.Upload(
        id='upload-pdf',
        children=html.Div(['Drag and Drop or ', html.A('Select PDF Files')]),
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

    html.Div(id='pdf-upload-status'),
   
    html.Button('Process PDF and Generate Heatmap', id='process-pdf', n_clicks=0),
    html.Div(id='process-status'),
   
    dcc.Download(id='download-csv'),
    dcc.Download(id='download-heatmap'),
   
    html.Div(id='image-container')  # 이미지가 표시될 컨테이너
])

# Helper functions
def parse_pdf(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.DataFrame(columns=['Sample Name', 'Peak Number', 'RT', 'Area', 'Height', '% Area', 'Total Area', 'Int Type'])

    with pdfplumber.open(io.BytesIO(decoded)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                sample_name_match = re.search(r'Sample Name: ([\S]+)', text)
                if sample_name_match:
                    sample_name = sample_name_match.group(1)
                else:
                    continue

                peak_results_start = text.find('Peak Results')
                if peak_results_start == -1:
                    continue

                peak_results_text = text[peak_results_start:]
                peaks = re.findall(r'(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([A-Z]+)', peak_results_text)
                for peak in peaks:
                    new_row = pd.DataFrame({
                        'Sample Name': [sample_name],
                        'Peak Number': [peak[0]],
                        'RT': [peak[1]],
                        'Area': [peak[2]],
                        'Height': [peak[3]],
                        '% Area': [peak[4]],
                        'Total Area': [peak[5]],
                        'Int Type': [peak[6]]
                    })
                    df = pd.concat([df, new_row], ignore_index=True)

    return df


def process_csv_for_heatmap(df):
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

    final_result_df = pivot_df[base_names].loc[:, ~pivot_df.columns.duplicated()]
    return final_result_df


def create_heatmap(df):
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(df, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, linecolor='gray', ax=ax)
    plt.title('Average % Area vs RRT by Sample Base Name', fontsize=20)
    plt.xlabel('Sample Base Name', fontsize=14)
    plt.ylabel('RRT', fontsize=14)

    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    plt.close(fig)
    img_io.seek(0)

    return img_io.getvalue()


# Callbacks for PDF Processing and Final Outputs
@app.callback(
    [Output('process-status', 'children'),
     Output('image-container', 'children'),
     Output('download-csv', 'data'),
     Output('download-heatmap', 'data')],
    [Input('process-pdf', 'n_clicks')],
    [State('upload-pdf', 'contents')]
)
def process_pdfs_to_csv_and_heatmap(n_clicks, contents):
    if n_clicks == 0 or contents is None:
        return '', None, None, None

    combined_df = pd.DataFrame()
    for content in contents:
        df = parse_pdf(content)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # 히트맵과 전처리된 데이터 생성
    final_df = process_csv_for_heatmap(combined_df)
    img_data = create_heatmap(final_df)

    # CSV 및 이미지 파일 다운로드 설정
    csv_output = final_df.to_csv(index=True, float_format='%.2f').encode('utf-8-sig')
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    img_element = html.Img(src=f'data:image/png;base64,{img_base64}', style={'width': '80%', 'height': 'auto'})

    return 'PDFs processed successfully!', img_element, dcc.send_bytes(csv_output, 'final_processed_results.csv'), dcc.send_bytes(img_data, 'beautiful_heatmap.png')


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

