# 데이터이쿠에서 웹앱으로 이미지 생성됨
# 이미지의 한글 깨짐
# 그런데 이미지, csv 다운로드 불가

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import io
import base64
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from dataiku import Folder

# Dash 앱 생성
#app = dash.Dash(__name__)

# Layout 설정
app.layout = html.Div([
    html.H1("CSV 파일 업로드 및 처리"),
   
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
        multiple=False
    ),
   
    html.Div(id='output-data-upload'),
   
    html.Hr(),
   
    html.Div(id='download-links')
])

# Helper functions
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df, filename

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
    plt.title('Sample Base Name 별 RRT 대비 % Area 평균', fontsize=20)
    plt.xlabel('Sample Base Name', fontsize=14)
    plt.ylabel('RRT', fontsize=14)

    # 이미지 저장
    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    plt.close(fig)
    img_io.seek(0)
   
    return img_io, final_result_df

def save_to_folder(data, filename, folder_id):
    folder = Folder(folder_id)
    with folder.get_writer(filename) as writer:
        writer.write(data)

# Callbacks
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('download-links', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        return None, None
   
    df, _ = parse_contents(contents, filename)
    img_io, processed_df = create_heatmap(df)
   
    # Save image and CSV to Managed Folder
    csv_bytes = processed_df.to_csv(index=True).encode()
    save_to_folder(csv_bytes, 'processed_'+filename, 'csv_folder')
    save_to_folder(img_io.read(), 'heatmap_'+filename.replace('.csv', '.png'), 'csv_folder')
   
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode()
   
    # HTML components for displaying results
    img_element = html.Img(src='data:image/png;base64,{}'.format(img_base64))
    csv_download_link = html.A('Download Processed CSV', href=f'/managed-folder/download/csv_folder/processed_{filename}', target='_blank')
    img_download_link = html.A('Download Heatmap Image', href=f'/managed-folder/download/csv_folder/heatmap_{filename.replace(".csv", ".png")}', target='_blank')
   
    return img_element, html.Div([csv_download_link, html.Br(), img_download_link])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

