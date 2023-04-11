import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
import os 
import plotly.graph_objects as go
from dash.dependencies import Input, Output



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

folders_list = [i for i in os.listdir('./') if os.path.isdir(i) if i != '.git']

plots_data_expl = ['scatterplot','barplot', 'lineplot', 'histogram', 'pie chart', 'boxplot', 'violin', 'heatmap']
plots_timeseries = ['scatterplot', 'lineplot', 'histogram']

app.layout = html.Div([
	dcc.Tabs([
		dcc.Tab(label='Tab one - Data Exploration', children=[
			html.Div([
				html.Label('Input folder Selection:'),
				html.Div([dcc.Dropdown(id='folder-select',options=[{'label': i, 'value': i} for i in folders_list],value=folders_list[0])]),
				html.Label('Select single files or join all the files in the folder together:'),
				html.Div([dcc.RadioItems(id='join-option', options=[{'label': i, 'value': i} for i in ['join_data','single_file']], value='single_file')]),
				html.Label('Select the data source:'),
				html.Div([dcc.Dropdown(id='df-select')]),
				html.Label('Select the plot type:'),
				html.Div([dcc.RadioItems(id='plot-type', options=[{'label': i, 'value': i} for i in plots_data_expl], value=plots_data_expl[0])]),
				html.Label('Select the variables to visualize:'),
				html.Div([html.Div([dcc.Dropdown(id='xaxis-column')], style={'width': '48%', 'display': 'inline-block'}),
						  html.Div([dcc.Dropdown(id='yaxis-column')], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})]),
				html.Label('Select the FILTER variable to use:'),
				html.Div([dcc.Dropdown(id='filtervar-select')]),
				html.Label('Select the filtered element(s) to display:'),
				html.Div([dcc.Dropdown(id='filter-select', multi=True)]),
				dcc.Graph(id='indicator-graphic')
					])
				]),
    dcc.Tab(label='Tab two - Time Series', children=[
			html.Div([
				html.Label('Input folder Selection for timeseries:'),
				html.Div([dcc.Dropdown(id='folder-select-ts',options=[{'label': i, 'value': i} for i in folders_list], value=folders_list[0])]),
				html.Label('Select single files or join all the files in the folder together:'),
				html.Div([dcc.RadioItems(id='join-option-ts', options=[{'label': i, 'value': i} for i in ['join_data','single_file']], value='single_file')]),
				html.Label('Select the data source:'),
				html.Div([dcc.Dropdown(id='df-select-ts')]),
				html.Label('Select the plot type:'),
				html.Div([dcc.RadioItems(id='plot-type-ts', options=[{'label': i, 'value': i} for i in plots_timeseries], value=plots_timeseries[0])]),
				html.Label('Select the variable to visualize:'),
				html.Div([dcc.Dropdown(id='var-select-ts')]),
				html.Label('Select the FILTER variable to use:'),
				html.Div([dcc.Dropdown(id='filtervar-select-ts')]),
				html.Label('Select the TIME variable to use:'),
				html.Div([dcc.Dropdown(id='timevar-select-ts')]),
				html.Label('Select the filtered element(s) to display:'),
				html.Div([dcc.Dropdown(id='filter-select-ts', multi=True)]),
				dcc.Graph(id='ts-graphic')
					])
				]),
	dcc.Tab(label='Tab three - Variables Exploration', children=[
			html.Div([
				html.Label('Input folder Selection:'),
				html.Div([dcc.Dropdown(id='folder-select-exp',options=[{'label': i, 'value': i} for i in folders_list], value=folders_list[0])]),
				html.Label('Select single files or join all the files in the folder together:'),
				html.Div([dcc.RadioItems(id='join-option-exp', options=[{'label': i, 'value': i} for i in ['join_data','single_file']], value='single_file')]),
				html.Label('Select the data source:'),
				html.Div([dcc.Dropdown(id='df-select-exp')]),
				html.Label('Select the variables to visualize:'),
				html.Div([dcc.Dropdown(id='var-select-exp', multi = True)]),
				html.Label('Select the FILTER variable to use:'),
				html.Div([dcc.Dropdown(id='filtervar-select-exp')]),
				html.Label('Select the filtered element(s) to display:'),
				html.Div([dcc.Dropdown(id='filter-select-exp', multi=True)]),
				dcc.Graph(id='exp-graphic')
					])
				])
	])
])


####################################################
## TAB 1 - DATA EXPLORATION
####################################################

@app.callback(
    Output('df-select', 'options'),
    Input('folder-select', 'value'),
    Input('join-option', 'value'))
def set_df_options(folder_select, join_option):
	if join_option == 'single_file':
		df_el = [{'label': i, 'value': i} for i in os.listdir('./' + folder_select + '/')]
	elif join_option == 'join_data':
		df_el = [{'label': i, 'value': i} for i in [str('joining data from path: ' + str(folder_select))]]
	return df_el

@app.callback(
	Output('xaxis-column', 'options'),
	Input('folder-select', 'value'),
	Input('df-select', 'value'),
	Input('join-option', 'value'))
def get_column_names(folder_select, df_select, join_option):
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	if join_option == 'single_file':
		if f_ext == '.parquet':
			col_names = [{'label': i, 'value': i} for i in pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow').keys()]
		elif f_ext == '.csv':
			col_names = [{'label': i, 'value': i} for i in pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';').keys()]
	elif join_option == 'join_data':
		if f_ext == '.parquet':
			df_sample = pd.read_parquet(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], engine = 'pyarrow')
		elif f_ext == '.csv':
			df_sample = pd.read_csv(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], error_bad_lines=False, sep = ';')
		col_names = [{'label': i, 'value': i} for i in df_sample.keys()]
	return col_names

@app.callback(
	Output('yaxis-column', 'options'),
	Input('folder-select', 'value'),
	Input('df-select', 'value'),
	Input('join-option', 'value'))
def get_column_names(folder_select, df_select, join_option):
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	if join_option == 'single_file':
		if f_ext == '.parquet':
			col_names = [{'label': i, 'value': i} for i in pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow').keys()]
		elif f_ext == '.csv':
			col_names = [{'label': i, 'value': i} for i in pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';').keys()]
	elif join_option == 'join_data':
		if f_ext == '.parquet':
			df_sample = pd.read_parquet(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], engine = 'pyarrow')
		elif f_ext == '.csv':
			df_sample = pd.read_csv(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], error_bad_lines=False, sep = ';')
		col_names = [{'label': i, 'value': i} for i in df_sample.keys()]
	return col_names

@app.callback(
	Output('filtervar-select', 'options'),
	Input('folder-select', 'value'),
	Input('df-select', 'value'),
	Input('join-option', 'value'))
def get_filtervar(folder_select, df_select, join_option):				
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	if join_option == 'single_file':
		if f_ext == '.parquet':
			col_names = [{'label': i, 'value': i} for i in pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow').keys()]
		elif f_ext == '.csv':
			col_names = [{'label': i, 'value': i} for i in pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';').keys()]
	elif join_option == 'join_data':
		if f_ext == '.parquet':
			df_sample = pd.read_parquet(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], engine = 'pyarrow')
		elif f_ext == '.csv':
			df_sample = pd.read_csv(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], error_bad_lines=False, sep = ';')
		col_names = [{'label': i, 'value': i} for i in df_sample.keys()]
	return col_names
	
@app.callback(
	Output('filter-select', 'options'),
	Input('folder-select', 'value'),
	Input('df-select', 'value'),
	Input('join-option', 'value'),
	Input('filtervar-select', 'value'))
def get_filterName(folder_select, df_select, join_option, filtervar_select):
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	#DATA SELECT
	if join_option == 'single_file':
		if f_ext == '.parquet':
			df = pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow')
		elif f_ext == '.csv':
			df = pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';')
	else:
		if f_ext == '.parquet':
			df = pd.DataFrame(columns = pd.read_parquet(folder_select + '/' + filenames_list[0], engine = 'pyarrow').keys())
			for i in filenames_list:
				new_df = pd.DataFrame(pd.read_parquet(folder_select + '/' + i, engine = 'pyarrow'))
				df = pd.concat([df, new_df], ignore_index=True)
		elif f_ext == '.csv':
			df = pd.DataFrame(columns = pd.read_csv(folder_select + '/' + filenames_list[0], error_bad_lines=False, sep = ';').keys())
			for i in filenames_list:
				new_df = pd.DataFrame(pd.read_csv(folder_select + '/' + i, error_bad_lines=False, sep = ';'))
				df = pd.concat([df, new_df], ignore_index=True)
	filter_list = list(df[filtervar_select].unique())
	return [{'label': i, 'value': i} for i in filter_list]

@app.callback(
    Output('indicator-graphic', 'figure'),
	Input('folder-select', 'value'),
	Input('df-select', 'value'),
	Input('join-option', 'value'),
	Input('filter-select', 'value'),
	Input('plot-type', 'value'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
	Input('filtervar-select', 'value')
	)
def update_graph(folder_select, df_select, join_option, filter_select, plot_type, xaxis_column_name, yaxis_column_name, filtervar_select):
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	#DATA SELECT
	if join_option == 'single_file':
		if f_ext == '.parquet':
			df = pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow')
		elif f_ext == '.csv':
			df = pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';')
	else:
		#filenames_list = os.listdir('./' + folder_select + '/')
		if f_ext == '.parquet':
			df = pd.DataFrame(columns = pd.read_parquet(folder_select + '/' + filenames_list[0], engine = 'pyarrow').keys())
			for i in filenames_list:
				new_df = pd.DataFrame(pd.read_parquet(folder_select + '/' + i, engine = 'pyarrow'))
				df = pd.concat([df, new_df], ignore_index=True)
		elif f_ext == '.csv':
			df = pd.DataFrame(columns = pd.read_csv(folder_select + '/' + filenames_list[0], error_bad_lines=False, sep = ';').keys())
			for i in filenames_list:
				new_df = pd.DataFrame(pd.read_csv(folder_select + '/' + i, error_bad_lines=False, sep = ';'))
				df = pd.concat([df, new_df], ignore_index=True)
	## FILTER DATA
	#if len(filter_select) > 0:
	df = df[df[filtervar_select].isin(filter_select)]
	
	## PLOT SELECT
	if plot_type == 'scatterplot':
		fig = px.scatter(x=df[xaxis_column_name], y=df[yaxis_column_name], color = df[filtervar_select])
	elif plot_type == 'barplot':
		fig = px.bar(x=df[xaxis_column_name], y=df[yaxis_column_name], color = df[filtervar_select])
	elif plot_type == 'lineplot':
		fig = px.line(df, x=xaxis_column_name, y=yaxis_column_name)
	elif plot_type == 'histogram':
		fig = px.histogram(df, x = xaxis_column_name, y= yaxis_column_name)
	elif plot_type == 'pie chart':
		fig = px.pie(values = df[xaxis_column_name], names = df[yaxis_column_name])
	elif plot_type == 'boxplot':
		fig = px.box(df, x = xaxis_column_name, y= yaxis_column_name)
	elif plot_type == 'violin':
		fig = px.violin(df, x = xaxis_column_name, y= yaxis_column_name)
	elif plot_type == 'heatmap':
		# Creating 2-D grid of features 
		[X, Y] = np.meshgrid(df[xaxis_column_name], df[yaxis_column_name])
		Z = np.cos(X / 2) + np.sin(Y / 4) 
		fig = go.Figure(data = go.Heatmap(x = df[xaxis_column_name], y = df[yaxis_column_name], z = Z,))
	
	## PLOT UPDATE
	fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
	fig.update_xaxes(title=xaxis_column_name)
	fig.update_yaxes(title=yaxis_column_name)
	return fig

####################################################
## TAB 2 - TIME SERIES
####################################################

@app.callback(
    Output('df-select-ts', 'options'),
    Input('folder-select-ts', 'value'),
    Input('join-option-ts', 'value'))
def set_df_options(folder_select, join_option):
	if join_option == 'single_file':
		df_el = [{'label': i, 'value': i} for i in os.listdir('./' + folder_select + '/')]
	elif join_option == 'join_data':
		df_el = [{'label': i, 'value': i} for i in [str('joining data from path: ' + str(folder_select))]]
	return df_el

@app.callback(
	Output('filtervar-select-ts', 'options'),
	Input('folder-select-ts', 'value'),
	Input('df-select-ts', 'value'),
	Input('join-option-ts', 'value'))
def get_filtervar(folder_select, df_select, join_option):				
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	if join_option == 'single_file':
		if f_ext == '.parquet':
			col_names = [{'label': i, 'value': i} for i in pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow').keys()]
		elif f_ext == '.csv':
			col_names = [{'label': i, 'value': i} for i in pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';').keys()]
	elif join_option == 'join_data':
		if f_ext == '.parquet':
			df_sample = pd.read_parquet(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], engine = 'pyarrow')
		elif f_ext == '.csv':
			df_sample = pd.read_csv(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], error_bad_lines=False, sep = ';')
		col_names = [{'label': i, 'value': i} for i in df_sample.keys()]
	return col_names

@app.callback(
	Output('timevar-select-ts', 'options'),
	Input('folder-select-ts', 'value'),
	Input('df-select-ts', 'value'),
	Input('join-option-ts', 'value'))
def get_timevar(folder_select, df_select, join_option):				
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	if join_option == 'single_file':
		if f_ext == '.parquet':
			col_names = [{'label': i, 'value': i} for i in pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow').keys()]
		elif f_ext == '.csv':
			col_names = [{'label': i, 'value': i} for i in pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';').keys()]
	elif join_option == 'join_data':
		if f_ext == '.parquet':
			df_sample = pd.read_parquet(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], engine = 'pyarrow')
		elif f_ext == '.csv':
			df_sample = pd.read_csv(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], error_bad_lines=False, sep = ';')
		col_names = [{'label': i, 'value': i} for i in df_sample.keys()]
	return col_names
	
@app.callback(
	Output('filter-select-ts', 'options'),
	Input('folder-select-ts', 'value'),
	Input('df-select-ts', 'value'),
	Input('join-option-ts', 'value'),
	Input('filtervar-select-ts', 'value'))
def get_filterName(folder_select, df_select, join_option, filtervar_select):
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	#DATA SELECT
	if join_option == 'single_file':
		if f_ext == '.parquet':
			df = pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow')
		elif f_ext == '.csv':
			df = pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';')
	else:
		if f_ext == '.parquet':
			df = pd.DataFrame(columns = pd.read_parquet(folder_select + '/' + filenames_list[0], engine = 'pyarrow').keys())
			for i in filenames_list:
				new_df = pd.DataFrame(pd.read_parquet(folder_select + '/' + i, engine = 'pyarrow'))
				df = pd.concat([df, new_df], ignore_index=True)
		elif f_ext == '.csv':
			df = pd.DataFrame(columns = pd.read_csv(folder_select + '/' + filenames_list[0], error_bad_lines=False, sep = ';').keys())
			for i in filenames_list:
				new_df = pd.DataFrame(pd.read_csv(folder_select + '/' + i, error_bad_lines=False, sep = ';'))
				df = pd.concat([df, new_df], ignore_index=True)
	filter_list = list(df[filtervar_select].unique())
	return [{'label': i, 'value': i} for i in filter_list]

@app.callback(
	Output('var-select-ts', 'options'),
	Input('folder-select-ts', 'value'),
	Input('df-select-ts', 'value'),
	Input('join-option-ts', 'value'))
def get_column_names(folder_select, df_select, join_option):
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	if join_option == 'single_file':
		if f_ext == '.parquet':
			col_names = [{'label': i, 'value': i} for i in pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow').keys()]
		elif f_ext == '.csv':
			col_names = [{'label': i, 'value': i} for i in pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';').keys()]
	elif join_option == 'join_data':
		if f_ext == '.parquet':
			df_sample = pd.read_parquet(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], engine = 'pyarrow')
		elif f_ext == '.csv':
			df_sample = pd.read_csv(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], error_bad_lines=False, sep = ';')
		col_names = [{'label': i, 'value': i} for i in df_sample.keys()]
	return col_names
	
@app.callback(
    Output('ts-graphic', 'figure'),
	Input('folder-select-ts', 'value'),
	Input('df-select-ts', 'value'),
	Input('plot-type-ts', 'value'),
	Input('join-option-ts', 'value'),
	Input('filter-select-ts', 'value'),
    Input('var-select-ts', 'value'),
	Input('timevar-select-ts', 'value'),
	Input('filtervar-select-ts', 'value'))
def update_graph(folder_select, df_select, plot_type, join_option, filter_select, var_select, timevar, filtervar_select):
	
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	#DATA SELECT
	if join_option == 'single_file':
		if f_ext == '.parquet':
			df = pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow')
		elif f_ext == '.csv':
			df = pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';')
	else:
		#filenames_list = os.listdir('./' + folder_select + '/')
		if f_ext == '.parquet':
			df = pd.DataFrame(columns = pd.read_parquet(folder_select + '/' + filenames_list[0], engine = 'pyarrow').keys())
			for i in filenames_list:
				new_df = pd.DataFrame(pd.read_parquet(folder_select + '/' + i, engine = 'pyarrow'))
				df = pd.concat([df, new_df], ignore_index=True)
		elif f_ext == '.csv':
			df = pd.DataFrame(columns = pd.read_csv(folder_select + '/' + filenames_list[0], error_bad_lines=False, sep = ';').keys())
			for i in filenames_list:
				new_df = pd.DataFrame(pd.read_csv(folder_select + '/' + i, error_bad_lines=False, sep = ';'))
				df = pd.concat([df, new_df], ignore_index=True)
    ## FILTER DATA
	#if len(filter_select) > 0:
	df = df[df[filtervar_select].isin(filter_select)]
        
	df = df.iloc[pd.to_datetime(df[timevar]).values.argsort()]	
	if plot_type == 'lineplot':
		fig = px.line(df, x=timevar, y=var_select, color=filtervar_select)
	elif plot_type == 'histogram':
		fig = px.histogram(df, x=timevar, y=var_select, color=filtervar_select)
	elif plot_type == 'scatterplot':
		fig = px.scatter(df, x=timevar, y=var_select, color=filtervar_select)

	## PLOT UPDATE
	fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
	fig.update_xaxes(title=timevar)
	fig.update_yaxes(title=var_select)
	return fig
    

####################################################
## TAB 3 - VARIABLES EXPLORATION
####################################################

@app.callback(
    Output('df-select-exp', 'options'),
    Input('folder-select-exp', 'value'),
    Input('join-option-exp', 'value'))
def set_df_options(folder_select, join_option):
	if join_option == 'single_file':
		df_el = [{'label': i, 'value': i} for i in os.listdir('./' + folder_select + '/')]
	elif join_option == 'join_data':
		df_el = [{'label': i, 'value': i} for i in [str('joining data from path: ' + str(folder_select))]]
	return df_el

@app.callback(
	Output('filtervar-select-exp', 'options'),
	Input('folder-select-exp', 'value'),
	Input('df-select-exp', 'value'),
	Input('join-option-exp', 'value'))
def get_filtervar(folder_select, df_select, join_option):				
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	if join_option == 'single_file':
		if f_ext == '.parquet':
			col_names = [{'label': i, 'value': i} for i in pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow').keys()]
		elif f_ext == '.csv':
			col_names = [{'label': i, 'value': i} for i in pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';').keys()]
	elif join_option == 'join_data':
		if f_ext == '.parquet':
			df_sample = pd.read_parquet(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], engine = 'pyarrow')
		elif f_ext == '.csv':
			df_sample = pd.read_csv(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], error_bad_lines=False, sep = ';')
		col_names = [{'label': i, 'value': i} for i in df_sample.keys()]
	return col_names
	

@app.callback(
	Output('filter-select-exp', 'options'),
	Input('folder-select-exp', 'value'),
	Input('df-select-exp', 'value'),
	Input('join-option-exp', 'value'),
	Input('filtervar-select-exp', 'value'))
def get_filterName(folder_select, df_select, join_option, filtervar_select):
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	#DATA SELECT
	if join_option == 'single_file':
		if f_ext == '.parquet':
			df = pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow')
		elif f_ext == '.csv':
			df = pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';')
	else:
		if f_ext == '.parquet':
			df = pd.DataFrame(columns = pd.read_parquet(folder_select + '/' + filenames_list[0], engine = 'pyarrow').keys())
			for i in filenames_list:
				new_df = pd.DataFrame(pd.read_parquet(folder_select + '/' + i, engine = 'pyarrow'))
				df = pd.concat([df, new_df], ignore_index=True)
		elif f_ext == '.csv':
			df = pd.DataFrame(columns = pd.read_csv(folder_select + '/' + filenames_list[0], error_bad_lines=False, sep = ';').keys())
			for i in filenames_list:
				new_df = pd.DataFrame(pd.read_csv(folder_select + '/' + i, error_bad_lines=False, sep = ';'))
				df = pd.concat([df, new_df], ignore_index=True)
	filter_list = list(df[filtervar_select].unique())
	return [{'label': i, 'value': i} for i in filter_list]
	
@app.callback(
	Output('var-select-exp', 'options'),
	Input('folder-select-exp', 'value'),
	Input('df-select-exp', 'value'),
	Input('join-option-exp', 'value'))
def get_column_names(folder_select, df_select, join_option):
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	if join_option == 'single_file':
		if f_ext == '.parquet':
			col_names = [{'label': i, 'value': i} for i in pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow').keys()]
		elif f_ext == '.csv':
			col_names = [{'label': i, 'value': i} for i in pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';').keys()]
	elif join_option == 'join_data':
		if f_ext == '.parquet':
			df_sample = pd.read_parquet(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], engine = 'pyarrow')
		elif f_ext == '.csv':
			df_sample = pd.read_csv(folder_select + '/' + os.listdir('./' + folder_select + '/')[0], error_bad_lines=False, sep = ';')
		col_names = [{'label': i, 'value': i} for i in df_sample.keys()]
	return col_names
	
@app.callback(
	Output('exp-graphic', 'figure'),
	Input('folder-select-exp', 'value'),
	Input('df-select-exp', 'value'),
	Input('join-option-exp', 'value'),
	Input('filtervar-select-exp', 'value'),
    Input('filter-select-exp', 'value'),
	Input('var-select-exp', 'value'))
def update_graph(folder_select, df_select, join_option, filtervar_select, filter_select, varselect):
	
	filenames_list = os.listdir('./' + folder_select + '/')
	f_name, f_ext = os.path.splitext(filenames_list[0])
	#DATA SELECT
	if join_option == 'single_file':
		if f_ext == '.parquet':
			df = pd.read_parquet(folder_select + '/' + df_select, engine = 'pyarrow')
		elif f_ext == '.csv':
			df = pd.read_csv(folder_select + '/' + df_select, error_bad_lines=False, sep = ';')
	else:
		#filenames_list = os.listdir('./' + folder_select + '/')
		if f_ext == '.parquet':
			df = pd.DataFrame(columns = pd.read_parquet(folder_select + '/' + filenames_list[0], engine = 'pyarrow').keys())
			for i in filenames_list:
				new_df = pd.DataFrame(pd.read_parquet(folder_select + '/' + i, engine = 'pyarrow'))
				df = pd.concat([df, new_df], ignore_index=True)
		elif f_ext == '.csv':
			df = pd.DataFrame(columns = pd.read_csv(folder_select + '/' + filenames_list[0], error_bad_lines=False, sep = ';').keys())
			for i in filenames_list:
				new_df = pd.DataFrame(pd.read_csv(folder_select + '/' + i, error_bad_lines=False, sep = ';'))
				df = pd.concat([df, new_df], ignore_index=True)
	
	## FILTER DATA
	#if len(filter_select) > 0:
	df = df[df[filtervar_select].isin(filter_select)]

	fig = px.scatter_matrix(df, dimensions=varselect, color=filtervar_select)

	## PLOT UPDATE
	#fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
	#fig.update_xaxes(title=x_name)
	#fig.update_yaxes(title=y_name)
	return fig

   
if __name__ == '__main__':
    app.run_server(debug=True)
	
	
