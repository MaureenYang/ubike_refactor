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

bk = backend.backend()
bk.update_all_station()
all_station_list = bk.all_station

def getStationHoverInfo(stationlist):
    station_hover_info = []
    for station in stationlist:
        info = bk.get_station_info(station)
        status = bk.get_station_status(station)
        sno = info['sno']
        sna = info['name']
        tot = info['tot']
        sbi = status['sbi']
        bemp = status['bemp']
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
        #need a option div
        dcc.Graph(
            id='histogram',
        )
    ]
)

user_control_div = html.Div (
        className = "div-user-control",
        children=[
            html.H2("YOUBIKE APP",style={'text-align':'center'}),
            html.H2("Search",style={'text-align':'right'}),
            dcc.Dropdown(
                id="location-dropdown",
                options=[ {"value":bk.all_station[data].sno , "label": "[{sno}] {name}".format(sno=bk.all_station[data].sno,name=bk.all_station[data].name)} for data in all_station_list],
                placeholder="Select a location",
            ),
            dcc.Dropdown(
                id="bar-selector",
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
        value_block('Predict Empty Num','--', 'pred_empty_value',"40%"),
        value_block('Predict Bike Num','--','pred_bike_value',"40%")
    ]
)
current_status_div = html.Div(
    className = 'current-station-div',
    children = [
        html.P("No.",id='station-no'),
        html.H2("Station Name",id='station-name'),
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
info_div = html.Div(
    className = "info-up-div",
    children=[
        user_control_div,
        weather_table,
        weather_table_div
    ]
)
station_info_div = html.Div(
    className = "station-info-div",
    children=[
        current_status_div,
        search_result
    ]
)
body_div = html.Div(
    className = "body-div",
    children=[
        info_div,
        station_info_div,
        graph_div
    ]
)
'''
header_div = html.Div(
    className = "header-div",
    children=[
        html.H2(children='Youbike Prediction')
    ]
)
'''
app.layout = html.Div(
    className = "main-div",
    children=[
        body_div,
])


# ------------------ Figure ------------------
totalList = []
def get_selection(month, day, selection):
    xVal = []
    yVal = []
    xSelected = []
    colorVal = [
        "#F4EC15",
        "#DAF017",
        "#BBEC19",
        "#9DE81B",
        "#80E41D",
        "#66E01F",
        "#4CDC20",
        "#34D822",
        "#24D249",
        "#25D042",
        "#26CC58",
        "#28C86D",
        "#29C481",
        "#2AC093",
        "#2BBCA4",
        "#2BB5B8",
        "#2C99B4",
        "#2D7EB0",
        "#2D65AC",
        "#2E4EA4",
        "#2E38A4",
        "#3B2FA0",
        "#4E2F9C",
        "#603099",
    ]

    # Put selected times into a list of numbers xSelected
    xSelected.extend([int(x) for x in selection])

    for i in range(24):
        # If bar is selected then color it white
        if i in xSelected and len(xSelected) < 24:
            colorVal[i] = "#FFFFFF"
        xVal.append(i)
        # Get the number of rides at a particular time
        yVal.append(len(totalList[month][day][totalList[month][day].index.hour == i]))

    return [np.array(xVal), np.array(yVal), np.array(colorVal)]

@app.callback(
    Output("histogram", "figure"),
    [
        Input("bar-selector", "value"),
        Input("location-dropdown", "value"),
    ],
)
def update_histogram(selection,selectedLocation):

    # get current data and 12 hour before data
    # how?

    xVal = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    yVal = np.array([ 138,66,53,93,166,333,722,1001,780,532,477,547,469,591,736,967,1152, 1262, 1122, 1018,597,878, 529 ,317])

    colorVal = [
        "#F4EC15",
        "#DAF017",
        "#BBEC19",
        "#9DE81B",
        "#80E41D",
        "#66E01F",
        "#4CDC20",
        "#34D822",
        "#24D249",
        "#25D042",
        "#26CC58",
        "#28C86D",
        "#29C481",
        "#2AC093",
        "#2BBCA4",
        "#2BB5B8",
        "#2C99B4",
        "#2D7EB0",
        "#2D65AC",
        "#2E4EA4",
        "#2E38A4",
        "#3B2FA0",
        "#4E2F9C",
        "#603099",
    ]

    layout = go.Layout(
        bargap=0.01,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=50),
        showlegend=False,
        plot_bgcolor="#323130",
        paper_bgcolor="#323130",
        dragmode="select",
        font=dict(color="white"),
        xaxis=dict(
            range=[-0.5, 23.5],
            showgrid=False,
            nticks=25,
            fixedrange=True,
            ticksuffix=":00",
        ),
        yaxis=dict(
            range=[0, max(yVal) + max(yVal) / 4],
            showticklabels=False,
            showgrid=False,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=str(yi),
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(color="white"),
            )
            for xi, yi in zip(xVal, yVal)
        ],
    )

    return go.Figure(
        data=[
            go.Bar(x=xVal, y=yVal, marker=dict(color=colorVal), hoverinfo="x"),
            go.Scatter(
                opacity=0,
                x=xVal,
                y=yVal / 2,
                hoverinfo="none",
                mode="markers",
                marker=dict(color="rgb(66, 134, 244, 0)", symbol="square", size=40),
                visible=True,
            ),
        ],
        layout=layout,
    )


# parameters for Figure 2
title = 'Select behavior'
labels = ['Empty Truth', 'Bike Truth', 'Empty Prediction', 'Bike Prediction']
colors = ['rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)']
mode_size = [8, 8, 8, 8]
line_size = [2, 2, 2, 2]
x_data = np.vstack((np.arange(0, 24),)*4)


# Update Map Graph based on date-picker, selected data on histogram and location dropdown
@app.callback(
    Output("station-empty-value", "children"),
    Output("station-bike-value", "children"),
    Output("station-total-value", "children"),
    Output("station-no", "children"),
    Output("station-name", "children"),
    Output("station-addr", "children"),
    Output("station-status", "children"),
    Output("pred_empty_value", "children"),
    Output("pred_bike_value", "children"),
    Output("map-graph", "figure"),
    [
        Input("bar-selector", "value"),
        Input("location-dropdown", "value"),
    ],
)
def update_graph(selectedData, selectedLocation):

    #init Coordinates to station 1,
    zoom = 12.0
    latInitial = 25.0408578889
    lonInitial = 121.567904444
    bearing = 0

    total_num = '--'
    bike_num = '--'
    empty_num = '--'
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

    station_list = bk.get_all_station_list()
    hover_info_txt = getStationHoverInfo(station_list)

    #update coordinates and parameters for the selected station
    if selectedLocation: #if select station, update map and raw status
        zoom = 15.0
        station_info= bk.get_station_info(selectedLocation)
        station_status = bk.get_station_status(selectedLocation)

        latInitial = station_info['coordinate'][0]
        lonInitial = station_info['coordinate'][1]
        total_num = station_info['tot']
        bike_num = station_status['sbi']
        empty_num = station_status['bemp']
        station_no = station_no + str(station_info['sno'])
        station_name = station_info['name']
        station_addr =  station_info['addr']

        if(int(station_status['act']) == 1):
            station_status = "Status: Active"
        else:
            station_status = "Status: Inactive"

    fig = go.Figure(
        data=[
            # Plot of important locations on the map
            Scattermapbox(
                lat=[all_station_list[i].coordinate[0] for i in all_station_list],
                lon=[all_station_list[i].coordinate[1] for i in all_station_list],
                mode="markers",
                hoverinfo="text",
                text= hover_info_txt,
                marker=dict(
                size=10,
                #symbol='bicycle',
                color=[int((all_station_list[i].bemp/all_station_list[i].tot)*100) for i in all_station_list],
                colorscale='jet',
                showscale=True
                ),
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
    return empty_num, bike_num, total_num,station_no,station_name,station_addr,station_status,pred_bemp,pred_sbi,fig


if __name__ == '__main__':
    app.run_server(debug=True)
