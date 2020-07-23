
#Dash server for showing plot in browser. Taken from bottom of https://plotly.com/python/3d-line-plots/

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go # or plotly.express as px
import numpy as np

class PlotServer():

    def start_server(self, figure):
        app = dash.Dash()
        app.layout = html.Div([
            dcc.Graph(
                figure=figure ,
            )
        ], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})

        app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter   
        

    def show_model_3d(self,env , model):
        pos_list, info = self.get_well_path_3d(env, model)
        targets = info['targets']
        hazards = info['hazards']

        figure = go.Figure(data=[go.Scatter3d(x=pos_list[:,0],
                                              y=pos_list[:,1],
                                              z=pos_list[:,2],
                        mode='lines', name="Well path", line=dict(width=10.0))])
        
        for i in range(len(targets)):
            self.plot_ball(figure, "Target", 'greens', targets[i])

        for i in range(len(hazards)):
            self.plot_ball(figure, "Hazard", 'reds', hazards[i])

        figure.update_layout(
            scene_aspectmode='cube',
            title='Deepwell plot',
            width=800, height=700,
            margin = dict(t=40, r=0, l=20, b=20),
            scene = dict(
                xaxis = dict(nticks=4, range=[env.xmin, env.xmax], title_text="East",),
                yaxis = dict(nticks=4, range=[env.ymin, env.ymax], title_text="North",),
                zaxis = dict(nticks=4, range=[env.zmax, env.zmin], title_text="TVD",),
            ),
        )

        print("Minimum total distance: ", info['min_dist'])
        print("Distance traveled: ", info['tot_dist'])    
        print("Target hits:     ", info['hits'])

        self.start_server(figure)


    #def show_model_2d(self, xcoord_list, ycoord_list, info):
    #    self.start_server(figure)


    #Test the trained model, run until done, return list of visited coords
    def get_well_path_3d(self, env, model):
        obs = env.reset()
        pos_list = [env.get_pos()]      #Initialize list of path coordinates with initial position

        while True:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)

            print("reward: ",rewards) 
            pos_list.append(info['pos'])
            if done: break
        pos_list = np.array(pos_list)

        return pos_list, info


    def plot_ball(self, figure, name, color, object):
        # Make data
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x0, y0, z0 = object['pos']
        radius = object['rad']

        x = x0 + radius * np.outer(np.cos(u), np.sin(v))
        y = y0 + radius * np.outer(np.sin(u), np.sin(v))
        z = z0 + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot the surface
        figure.add_trace(
            go.Surface(x=x, y=y, z=z, colorscale=color, showscale=False, name=name),)
