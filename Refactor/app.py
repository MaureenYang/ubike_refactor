import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np

from dash.dependencies import Input, Output
from plotly import graph_objs as go
from plotly.graph_objs import *
import datetime as dt
import dash_table

import json
import time
import os, sys
import backend

app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])

mapbox_access_token = "pk.eyJ1IjoibW9ycGhldXMyNyIsImEiOiJja2Zrd3J0dWMwb2pxMnlwY3g0Zmtza3ZuIn0.obFXuRnZeFgcHdzxq-Co4g"

station_info_json = backend.getStationInformation()

def getStationHoverInfo(stationlist):

    station_hover_info = []
    for station in stationlist:
        sno = station['sno']
        sna = station['sna']
        tot = station['tot']
        sbi = station['sbi']
        bemp = station['sbi']
        str_station = '[{sno}] {sna}<br> bike:{sbi}<br> empty:{bemp}<br> total:{tot}'.format(sno=sno,sna=sna,sbi=sbi,bemp=bemp,tot=tot)
        station_hover_info = station_hover_info + [str_station]

    return station_hover_info

# ------------------ layout ------------------
def value_block(title, value, blk_id, wper):
    block_obj = html.Div(
        className = 'div-value-block',
        style={'float':'left', 'width': wper},
        children = [
            html.P(title, style={'text-align':'left','font-size':'18px'}),
            html.P(value, style={'text-align':'center','font-size':'30px'}, id=blk_id)
        ]
    )
    return block_obj

graph_div = html.Div(
    className = "graph-div",
    children=[
        dcc.Graph(
            id='map-graph',
        ),
        dcc.Graph(
            id='line-chart',
        ),
    ]
)
pick_dt_div = html.Div(
        className = "div-datetime-ctrl",
        children=[
        dcc.DatePickerSingle(
            id="date-picker",
            min_date_allowed=dt.datetime(2018, 3, 1),
            max_date_allowed=dt.datetime(2018, 3, 5),
            initial_visible_month=dt.datetime(2018, 3, 1),
            date=dt.datetime(2018, 3, 1).date(),
            display_format="MMM D, YYYY",
        ),

        dcc.Dropdown(
            id="bar-selector",
            options=[
                {
                    "label": str(n) + ":00",
                    "value": str(n),
                } for n in range(24)
            ],
            multi=False,
            placeholder="Hour",
        )]
)

user_control_div = html.Div (
        className = "div-user-control",
        children=[
            html.H4("Please enter the informations below:"),
            dcc.Dropdown(
                id="location-dropdown",
                options=[ {"value":data['idx'] , "label": "[{sno}] {name}".format(sno=data['sno'],name=data['sna'])} for data in station_info_json],
                placeholder="Select a location",
            ),
            pick_dt_div,
            dcc.Dropdown(
                id="bar2-selector",
                options=[
                    {
                        "label": str(n) + " hour after",
                        "value": str(n),
                    } for n in range(1,13)
                ],
                multi=False,
                placeholder="Select predict hours",
            ),
        ],
)

search_result = html.Div(
    className = 'search-result-div',
    children=[
        value_block('Ground Truth Empty Num(y)','--', 'truth_empty_value',"40%"),
        value_block('Ground Truth Bike Num(y)','--','truth_bike_value',"40%"),
        value_block('Predict Empty Num(ŷ)','--', 'pred_empty_value',"40%"),
        value_block('Predict Bike Num(ŷ)','--','pred_bike_value',"40%")
    ]
)
current_status_div = html.Div(
    className = 'current-station-div',
    children = [
        html.P("No.",id='station-no'),
        html.H3("Station Name",id='station-name'),
        html.P("Station Address",id='station-addr'),
        html.P("Status",id='station-status'),
        html.Div(
            children = [
                value_block("Bike",'--',"station-bike-value",'80px'),
                value_block("Empty",'--',"station-empty-value",'80px') ,
                value_block("Total",'--',"station-total-value","80px")
            ]
        )]
)
weather_table = dash_table.DataTable(
        id='weather-table',
    )
weather_table_div = html.Div(
    id='weather-table-div',
)

info_up_div = html.Div(
    className = "info-up-div",
    children=[
        user_control_div,
        #current_status_div,
        weather_table,
        weather_table_div

    ]
)
info_down_div = html.Div(
    className = "info-down-div",
    children=[
        current_status_div,
        search_result
    ]
)
info_div = html.Div(
    className = "info-div",
    children=[
        info_up_div,
        info_down_div
    ]
)
body_div = html.Div(
    className = "body-div",
    children=[
        info_div,
        graph_div
    ]
)
header_div = html.Div(
    className = "header-div",
    children=[
        html.H1(children='Youbike Preidction')
    ]
)
app.layout = html.Div(
    className = "main-div",
    children=[
        header_div,
        body_div,
])


# ------------------ Figure ------------------

# parameters for Figure 2
title = 'Select behavior'
labels = ['Empty Truth', 'Bike Truth', 'Empty Prediction', 'Bike Prediction']
colors = ['rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)']
mode_size = [8, 8, 8, 8]
line_size = [2, 2, 2, 2]
x_data = np.vstack((np.arange(0, 24),)*4)


# TODO:
# Get the Coordinates of the chosen months, dates and times
if False: #for multiple hour
    def getLatLonColor(selectedData, datepiked):

        listCoords = empty_df[datepiked]            # TODO:

        # No times selected, output all times for chosen month and date
        if selectedData is None or len(selectedData) == 0:
            return listCoords.sum()

        selectnum = 0
        for time in selectedData: # multi select data
            selectnum = selectnum + listCoords[listCoords.index.hour == time]

        return selectnum


# Update Map Graph based on date-picker, selected data on histogram and location dropdown
@app.callback(
    Output("station-empty-value", "children"),
    Output("station-bike-value", "children"),
    Output("station-total-value", "children"),

    Output("station-no", "children"),
    Output("station-name", "children"),
    Output("station-addr", "children"),
    Output("station-status", "children"),

    Output("truth_empty_value", "children"),
    Output("truth_bike_value", "children"),

    Output("pred_empty_value", "children"),
    Output("pred_bike_value", "children"),

    Output("map-graph", "figure"),
    Output("line-chart", "figure"),
    #Output('weather-table-div', 'children'),

    [
        Input("date-picker", "date"),
        Input("bar-selector", "value"),

        Input("bar2-selector", "value"),
        Input("location-dropdown", "value"),
    ],
)
def update_graph(datePicked,hourPicked, selectedData, selectedLocation):

    #init Coordinates to station 1,
    zoom = 12.0
    latInitial = 25.0408578889
    lonInitial = 121.567904444
    bearing = 0

    total_num = '--'
    bike_num = '--'
    empty_num = '--'
    true_bemp = '--'
    true_sbi = '--'
    pred_bemp = '--'
    pred_sbi = '--'

    station_no = "No. "
    station_name = "Station Name"
    station_addr = "Address"
    station_status = "Status: Unknown"

    #fig 2
    graph_data = pd.DataFrame(range(0,24),index=pd.date_range("2018-03-01", periods=24, freq="H"))
    y_data = []
    for i in range(0,4):
        y_data = y_data + [graph_data]

    station_info_json = backend.getStationInformation()
    hover_info_txt = getStationHoverInfo(station_info_json)

    #update coordinates and parameters for the selected station
    if selectedLocation: #if select station, update map and raw status
        zoom = 15.0
        station_data = station_info_json[selectedLocation-1]
        latInitial = float(station_data["lat"])
        lonInitial = float(station_data["lng"])
        total_num = station_data['tot']
        bike_num = station_data['sbi']
        empty_num = station_data['bemp']
        station_no = station_no + station_data['sno']
        station_name = station_data['sna']
        station_addr =  station_data['ar']

        if(int(station_data['act']) == 1):
            station_status = "Status: Active"
        else:
            station_status = "Status: Inactive"

    # fig1: map
    fig = go.Figure(
        data=[
            # Plot of important locations on the map
            Scattermapbox(
                lat=[station_info_json[i]["lat"] for i in range(len(station_info_json))],
                lon=[station_info_json[i]["lng"] for i in range(len(station_info_json))],
                mode="markers",
                hoverinfo="text",
                text= hover_info_txt,
                marker=dict(size=15, symbol='bicycle'),
            ),
        ],

        layout=Layout(
            autosize=True,
            margin=go.layout.Margin(l=0, r=0, t=0, b=0),
            showlegend=False,
            height=400,
            mapbox=dict(
                accesstoken=mapbox_access_token,
                center=dict(lat=latInitial, lon=lonInitial),  # 40.7272  # -73.991251
                style='mapbox://styles/mapbox/streets-v11',
                bearing=bearing,
                zoom=zoom,
            ),
            updatemenus=[
                dict(
                    buttons=(
                    [
                        dict(
                            args=[
                                {
                                    "mapbox.zoom": 12,
                                    "mapbox.center.lon": "121.567904444",
                                    "mapbox.center.lat": "25.0408578889",
                                    "mapbox.bearing": 0,
                                    "mapbox.style": "dark",
                                }
                            ],
                        label="Reset Zoom",
                        method="relayout",
                    )
                    ]
                    ),
                    direction="left",
                    pad={"r": 0, "t": 0, "b": 0, "l": 0},
                    showactive=False,
                    type="buttons",
                    x=0.45,
                    y=0.02,
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="#323130",
                    borderwidth=1,
                    bordercolor="#6d6d6d",
                    font=dict(color="#FFFFFF"),
                )
            ],
        ),
    )

    #----------------------------------------------------------
    # get data from the date

    #for mulitple select
    #listCoords = getLatLonColor(selectedData, datePicked)
    if selectedLocation and hourPicked:
        try:
            sno = station_info_json[selectedLocation-1]['sno']
            date_picked = dt.datetime.strptime(datePicked, "%Y-%m-%d")
            hour_sel = int(hourPicked)
            newdt = dt.datetime.combine(date_picked, dt.datetime.min.time())
            newdt = newdt.replace(hour=hour_sel)
            dt_idx = newdt.strftime("%Y-%m-%d %H:%M:%S")
            y_data = backend.getBikeDatabyDate(sno,newdt)

            if selectedData:
                pre_hour_sel = int(selectedData)
                true_bemp_idx = y_data[4]
                true_sbi_idx = y_data[5]
                true_bemp = true_bemp_idx[true_bemp_idx['predict_hour']==pre_hour_sel].y_bemp
                true_sbi = true_sbi_idx[true_sbi_idx['predict_hour']==pre_hour_sel].y_sbi
                end_time = newdt + dt.timedelta(hours=12)
                end_dt_idx = end_time.strftime("%Y-%m-%d %H:%M:%S")
                idx = pd.date_range(start=dt_idx, end=end_dt_idx,freq='H')
                idx = idx[1:]
                pred_bemp_idx = y_data[2]
                pred_bemp = pred_bemp_idx[pred_bemp_idx['predict_hour']==pre_hour_sel].y_bemp_predict
                pred_sbi_idx = y_data[3]
                pred_sbi = pred_sbi_idx[pred_sbi_idx['predict_hour']==pre_hour_sel].y_sbi_predict
                y_data[2] = y_data[2].y_bemp_predict
                y_data[3] = y_data[3].y_sbi_predict
                y_data[2].index = idx
                y_data[3].index = idx

        except Exception as e:
            print(e)
            print("!!!! pred_bemp",pred_bemp)
            print("!!!! pred_sbi",pred_sbi)

    try:
        print("one")
        # Create figure
        fig2 = go.Figure()

        fig2.update_layout(
            margin=dict(l=5, r=5, t=5, b=5),
        )

        for i in range(0, 4):
            fig2.add_trace(
                go.Scatter(x=y_data[i].index, y=y_data[i])
            )

        # Add range slider
        fig2.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1h",
                             step="hour",
                             stepmode="todate"),
                        dict(count=1,
                             label="day",
                             step="day",
                             stepmode="todate"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
    except Exception as e:
        print(e)

    try:
        print("two")
        #update weather table
        print('selectedLocation:',selectedLocation)
        dfw = backend.get_weather_by_station(1)
        cell_list = []
        for col in dfw.columns:
            cell_list = cell_list + [dfw[col]]

        data = dfw.to_dict('rows')
        columns =  [{"name": i, "id": i,} for i in (dfw.columns)]
    except Exception as e:
        print(e)

    try:
        print("three")
        wtable =  dash_table.DataTable(
            data=data, columns=columns,
            style_header={'backgroundColor': 'rgb(30, 30, 30)'},
            style_cell={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white'})
    except Exception as e:
        print(e)

    return empty_num, bike_num, total_num,station_no,station_name,station_addr,station_status,true_bemp,true_sbi,pred_bemp,pred_sbi, fig,fig2 #,wtable


if __name__ == '__main__':
    app.run_server(debug=True)
