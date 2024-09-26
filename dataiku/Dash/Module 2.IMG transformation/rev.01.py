# 그나마 됨
# 이걸로 과제 마무리 하려다가 말았음

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
    html.H1("CSV File Upload and Processing"),
   
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
   
    html.Div(id='image-container'),  # 이미지가 표시될 컨테이너
    html.Div(id='download-links'),
    dcc.Download(id="download-csv"),
    dcc.Download(id="download-img")
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
   
    df, _ = parse_contents(contents, filename)
    img_data, processed_df = create_heatmap(df)
   
    # Save image and CSV to Managed Folder
    csv_bytes = processed_df.to_csv(index=True).encode()
    csv_download_link = dcc.send_bytes(csv_bytes, f'processed_{filename}')
    img_download_link = dcc.send_bytes(img_data, f'heatmap_{filename.replace(".csv", ".png")}')
   
    # 이미지를 base64로 인코딩하여 웹앱에 표시
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    img_element = html.Img(src='data:image/png;base64,{}'.format(img_base64), style={'width': '80%', 'height': 'auto'})
   
    return html.Div([f'File {filename} processed successfully.']), html.Div([
        html.Button("Download CSV", id="btn_csv_download"),
        html.Button("Download Image", id="btn_img_download"),
        dcc.Download(id="download-csv"),
        dcc.Download(id="download-img")
    ]), img_element


@app.callback(
    Output("download-csv", "data"),
    Input("btn_csv_download", "n_clicks"),
    State('upload-data', 'contents'),
    prevent_initial_call=True,
)
def download_csv(n_clicks, contents):
    if n_clicks:
        df, filename = parse_contents(contents, "processed_file.csv")
        _, processed_df = create_heatmap(df)
        csv_bytes = processed_df.to_csv(index=True).encode()
        return dcc.send_bytes(csv_bytes, "processed_file.csv")


@app.callback(
    Output("download-img", "data"),
    Input("btn_img_download", "n_clicks"),
    State('upload-data', 'contents'),
    prevent_initial_call=True,
)
def download_img(n_clicks, contents):
    if n_clicks:
        df, filename = parse_contents(contents, "heatmap_file.png")
        img_data, _ = create_heatmap(df)
        return dcc.send_bytes(img_data, "heatmap_file.png")


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

