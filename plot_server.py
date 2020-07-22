
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
        


    def show_model_3d(self,env,model):
        xcoord_list, ycoord_list, zcoord_list, info = self.get_well_path_3d(env,model)

        figure = go.Figure(data=[go.Scatter3d(x=xcoord_list, y=ycoord_list, z=zcoord_list, mode='lines', name="Well path", line=dict(width=10.0))])

        x_targets = info['xtargets']
        y_targets = info['ytargets']
        z_targets = info['ztargets']
        radius_targets = info['t_radius']

        x_hazards = info['xhazards']
        y_hazards = info['yhazards']
        z_hazards = info['zhazards']
        radius_hazards = info['h_radius']

        for i in range(len(x_targets)):
            self.plot_ball(figure, "Target", 'greens', x_targets[i], y_targets[i], z_targets[i], radius_targets[i])

        for i in range(len(x_hazards)):
            self.plot_ball(figure, "Hazard", 'reds', x_hazards[i], y_hazards[i], z_hazards[i], radius_hazards[i])

        figure.update_layout(
            scene_aspectmode='cube',
            title='Deepwell plot',
            width=800, height=700,
            margin = dict(t=40, r=0, l=20, b=20),
            scene = dict(
                xaxis = dict(nticks=4, range=[env.xmin,env.xmax], title_text="East",),
                yaxis = dict(nticks=4, range=[env.ymin,env.ymax], title_text="North",),
                zaxis = dict(nticks=4, range=[env.zmax,env.zmin], title_text="TVD",),
            ),
        )

        print("Minimum total distance: ", info['min_dist'])
        print("Distance traveled: ", info['tot_dist'])    
        print("Target hits:     ", info['hits'])

        self.start_server(figure)


    #def show_model_2d(self, xcoord_list, ycoord_list, info):
    #    self.start_server(figure)


    #Test the trained model, run until done, return list of visited coords
    def get_well_path_3d(self,env,model):
        obs = env.reset()
        xcoord_list = [env.x]      #Initialize list of path coordinates with initial position
        ycoord_list = [env.y]
        zcoord_list = [env.z]
        info = {}

        while True:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)

            print("reward: ",rewards) 
            xcoord_list.append(info['x'])
            ycoord_list.append(info['y'])
            zcoord_list.append(info['z'])

            if done: break

        return xcoord_list, ycoord_list, zcoord_list, info



    def plot_ball(self, figure, name, color, x0, y0, z0, radius):
        # Make data
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = x0 + radius * np.outer(np.cos(u), np.sin(v))
        y = y0 + radius * np.outer(np.sin(u), np.sin(v))
        z = z0 + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot the surface
        figure.add_trace(
            go.Surface(x=x, y=y, z=z, colorscale=color, showscale=False, name=name),)
