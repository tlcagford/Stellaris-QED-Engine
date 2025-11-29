import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("STELLARIS Data Analysis Suite", style={'textAlign': 'center'}),
    
    dcc.Tabs([
        dcc.Tab(label='Field Visualization', children=[
            dcc.Graph(id='3d-field-plot'),
            dcc.Slider(0, 15, 0.5, value=5, id='field-strength-slider')
        ]),
        
        dcc.Tab(label='Conservation Analysis', children=[
            dcc.Graph(id='energy-conservation'),
            html.Div(id='conservation-metrics')
        ]),
        
        dcc.Tab(label='Conversion Statistics', children=[
            dcc.Graph(id='conversion-histogram'),
            html.Div([
                "Significance: ",
                html.Span("5.2σ", id='significance-value', 
                         style={'color': 'red', 'fontWeight': 'bold'})
            ])
        ])
    ])
])

@app.callback(
    Output('3d-field-plot', 'figure'),
    Input('field-strength-slider', 'value')
)
def update_field_plot(field_strength):
    # 3D magnetic field visualization
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    z = np.linspace(-1, 1, 20)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Dipole field calculation
    R = np.sqrt(X**2 + Y**2 + Z**2)
    Bx = 3*X*Z/R**5
    By = 3*Y*Z/R**5
    Bz = (3*Z**2 - R**2)/R**5
    
    fig = go.Figure(data=go.Cone(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        u=Bx.flatten(), v=By.flatten(), w=Bz.flatten(),
        colorscale='Blues', sizemode="absolute",
        sizeref=0.3, anchor="tail"
    ))
    
    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1),
                      camera_eye=dict(x=1.2, y=1.2, z=1.2)))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
