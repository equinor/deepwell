
#Dash server for showing plot in browser. Taken from bottom of https://plotly.com/python/3d-line-plots/

import dash
import dash_core_components as dcc
import dash_html_components as html

class Plotter():

    def show3d(self, figure, port=8080):
        
        figure.update_layout(
            scene_aspectmode='cube',
            title='Deepwell plot',
            width=800, height=700,
            margin=dict(t=40, r=0, l=20, b=20)
        )


        app = dash.Dash()
        app.layout = html.Div([
            dcc.Graph(
                figure=figure ,
            )
        ], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})


        app.run_server(host='0.0.0.0', port=port, debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter   


    #def show2d(self, figure, port=8080):