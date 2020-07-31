import os
import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_colorscales as dcs
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from mni import create_mesh_data, default_colorscale

import plotly.graph_objects as go
import numpy as np

#from envs.DeepWellEnvSpher import DeepWellEnvSpher
from envs.DeepWellEnvSpherSmallObs import DeepWellEnvSpherSmallObs

from datetime import datetime

#from stable_baselines import DQN
from stable_baselines import PPO2

import flask



app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

#server = flask.Flask(__name__) # define flask app.server


GITHUB_LINK = os.environ.get(
    "GITHUB_LINK",
    "https://github.com/equinor/deepwell/",
)

default_colorscale_index = [ea[1] for ea in default_colorscale]

axis_template = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(255, 255, 255)",
    "zerolinecolor": "rgb(255, 255, 255)",
    "range": [0, 3000],
}

z_axis_template = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(255, 255, 255)",
    "zerolinecolor": "rgb(255, 255, 255)",
    "range": [3000, 0],
}

plot_layout = {
    "title": "",
    "margin": {"t": 40, "b": 20, "l": 20, "r": 0},
    "font": {"size": 12, "color": "white"},
    "showlegend": False,
    "plot_bgcolor": "#141414",
    "paper_bgcolor": "#141414",
    "width": 800,
    "height": 750,
    "scene": {
        "xaxis": axis_template,
        "yaxis": axis_template,
        "zaxis": z_axis_template,
        "aspectratio": {"x": 1, "y": 1.2, "z": 1},
        "camera": {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}},
        "annotations": [],
    },
}


app.layout = html.Div(
    [   dcc.Store(id='point-store', storage_type='memory'),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Img(
                                            src=app.get_asset_url("dash-logo.png")
                                        ),
                                        html.H4("Summer Internship Project"),
                                    ],
                                    className="header__title",
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            "Click and hold on the plot to rotate it."
                                        )
                                    ],
                                    className="header__info pb-0",
                                ),
                            ],
                            className="header pb-0",
                        ),
                        html.Div(
                            [   
                                dcc.Graph(id='wellplot')
                            ],
                            className="graph__container",
                        ),
                    ],
                    className="container",
                )
            ],
            className="two-thirds column app__left__section",
        ),
        html.Div(
            [
                html.Ul(id='point-list',children=[]
                #html.Ul(id='point-list',children=[dcc.Input(id='input-points', value='asdf')]
                ),
                html.Div(
                    [
                        dcc.Input(id='input-txt-box', type='text', value='1500,1500,1500'),
                        html.Button(id='add-button-state', n_clicks=0, children='Add'),
                        html.Button(id='reset-button-state', n_clicks=0, children='Reset'),
                        html.Div(id='output-state')
                    ]
                ),
                html.Div(
                    [
                        html.Button(id='load-button-state', n_clicks=0, children='Load path'),
                    ]
                ),
                html.Div(
                    [
                        html.P(
                            [
                                "See code on ",
                                html.A(
                                    children="GitHub.",
                                    target="_blank",
                                    href=GITHUB_LINK,
                                    className="red-ish",
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            className="one-third column app__right__section",
        ),
        dcc.Store(id="annotation_storage"),
        html.Div(id='intermediate-value', style={'display': 'none'}),
    ]
)


#model = PPO2.load("C:/Users/torst/Desktop/deepwell/webapp/trained_models/sphere_na_def_retrained50M_smallobs.zip")
model = PPO2.load("/usr/src/app/trained_models/sphere_na_def_retrained50M_smallobs.zip")


@app.callback(
                [Output('point-list', 'children'), Output('point-store', 'data'), Output('wellplot', 'figure')],
              [
                Input('add-button-state', 'n_clicks'), 
                Input('reset-button-state', 'n_clicks'),
                Input('load-button-state', 'n_clicks'),
              ],[
                State('input-txt-box', 'value'),
                State('point-store', 'data'),
            ])
def update_figure(n_clicks,n_clicks2,n_clicks3,point_txtinput,stored_points):

    user_click = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    stored_points = stored_points or []                       #Initialize stored_points first time it is run

    figure = go.Figure()
    figure.update_layout(plot_layout)
    
    #print(n_clicks,n_clicks2,point_txtinput)

    stored_points_html = []


    if user_click == "add-button-state":
        input_list = [ int(value) for value in point_txtinput.split(',') ]
        stored_points.append(input_list)
        
    elif user_click == "reset-button-state":
        stored_points.clear()
        figure.data = []
        print("Reset button pressed")

    elif user_click == "load-button-state":
        figure.data = []
        print("Load button pressed with points: ", stored_points)
        plot_wellpath(figure,stored_points)
    
    for point in stored_points:
        plot_ball(figure,"point",'greens',point[0],point[1],point[2],100)        #Add point to figure
        stored_points_html.append( html.Li([str(point)]) )                                #Add point to stored_points_html to show in sidebar

    return stored_points_html, stored_points, figure


    #PLotting points using markers are not as resource intensive as plotting spheres with surface like in plot_ball. Below code can be used for that, but it is not adpted nor tested
    #def plot_balls(self, figure, name, color, x_list, y_list, z_list, radius_list):
    #    color_list = [color]*len(x_list)        #Color needs a list of colors for each point
#
    #    figure.add_trace(go.Scatter3d(
    #    x = x_list,
    #    y = y_list,
    #    z = z_list,
    #    name=name,
    #    mode = 'markers',
    #    marker = dict(
    #        size = diameter,
    #        sizeref = 2.3,                #The radius gets scaled by this number. Lower = larger points. Info on sizeref: https://plotly.com/python/reference/#scatter-marker-sizeref
    #        size = radius_list,
    #        color = color_list,
    #        )
    #    ))



def plot_ball(figure, name, color, x0, y0, z0, radius):
    # Make data

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = x0 + radius * np.outer(np.cos(u), np.sin(v))
    y = y0 + radius * np.outer(np.sin(u), np.sin(v))
    z = z0 + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    figure.add_trace(
        go.Surface(x=x, y=y, z=z, colorscale=color, showscale=False, name=name),)



def plot_wellpath(figure,point_list):
    #env = gym.make('DeepWellEnvSpherlevel5-v0')
    env = DeepWellEnvSpherSmallObs()
    env.set_targets(point_list)
    env.initialize()

    obs = env.reset()
    pos_list = [env.get_pos()]      #Initialize list of path coordinates with initial position

    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        pos_list.append(info['pos'])
        if done: break
    pos_list = np.array(pos_list)

    targets = info['targets']
    hazards = info['hazards']

     
    figure.add_trace(go.Scatter3d(x=pos_list[:,0],
                        y=pos_list[:,1],
                        z=pos_list[:,2],
                mode='lines', name="Well path", line=dict(width=10.0)) )

    
    #for i in range(len(targets)):
    #    plot_ball2(figure, "Target", 'greens', targets[i])

    for i in range(len(hazards)):
        x0, y0, z0 = hazards[i]['pos']
        radius = hazards[i]['rad']
        plot_ball(figure,"point",'reds',x0,y0,z0,radius)


    print("=========== PLOTTING WELL PATH =============")
    print("Minimum total distance: ", info['min_dist'])
    print("Distance traveled: ", info['tot_dist'])    
    print("Target hits:     ", info['hits'])
    print("============================================")



if __name__ == "__main__":
    #app.run_server(debug=False)
    app.run_server(host='0.0.0.0', debug=True, port=8050)

