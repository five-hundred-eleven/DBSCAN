from flask import Flask
import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    'https://use.fontawesome.com/releases/v5.9.0/css/all.css',
]

meta_tags = [
    {
        "name": "viewport",
        "content": "width=device-width, initial-scale=1",
    }
]

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets, meta_tags=meta_tags)
app.config.suppress_callback_exceptions = True
app.title = "DBSCAN"

navbar = dbc.NavbarSimple(
    brand="DBSCAN",
    brand_href="/",
    children=[
    ],
    sticky="top",
    color="#511",
    light=False,
    dark=True,
)

footer = (
    html.Div([
        dbc.Container(
            dbc.Row(
                dbc.Col(
                    html.P(
                        [
                            html.A(html.Span("Erik Cowley", className="mr-2"), href="https://datascience.stromsy.com"),
                            html.A(html.I(className='fas fa-envelope-square mr-1'), href='mailto:ecowley@protonmail.com'),
                            html.A(html.I(className='fab fa-github-square mr-1'), href='https://github.com/ekoly/ufc-fight-prediction'),
                            html.A(html.I(className='fab fa-linkedin mr-1'), href='https://www.linkedin.com/in/erik-cowley-89090120/'),
                        ],
                        className="lead"
                    )
                )
            ),
        ),
    ], id="footer")
)

content = dbc.Container(id="page-content", className="mt-4")

app.layout = (
    html.Div([
        dcc.Location(id="url", refresh=False),
        navbar,
        html.Hr(),
        content,
        html.Hr(),
        footer,
    ])
)

from routes import index_routes

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname"),],
)
def displayPage(path_name):
    if path_name == "/":
        return index_routes.layout
