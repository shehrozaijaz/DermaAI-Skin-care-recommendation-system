import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from monitoring.metrics_logger import MetricsLogger
from monitoring.drift_detector import DriftDetector
from datetime import datetime, timedelta

# Initialize
logger = MetricsLogger()
drift_detector = DriftDetector(logger)

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1('🔬 DermaAI Monitoring Dashboard', style={'textAlign': 'center', 'color': '#d68c90'}),
        html.P('Real-time monitoring and drift detection', style={'textAlign': 'center', 'color': '#888'})
    ], style={'padding': '20px', 'backgroundColor': '#f9f7f7'}),
    
    # Refresh interval
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # Update every 30 seconds
        n_intervals=0
    ),
    
    # Statistics Cards
    html.Div(id='stats-cards', style={'padding': '20px'}),
    
    # Charts
    html.Div([
        html.Div([
            dcc.Graph(id='query-volume-chart')
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(id='confidence-distribution')
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
    ]),
    
    html.Div([
        html.Div([
            dcc.Graph(id='entity-distribution')
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(id='severity-distribution')
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
    ]),
    
    # Drift Detection Section
    html.Div([
        html.H2('📊 Data Drift Detection', style={'color': '#d68c90', 'padding': '20px'}),
        html.Div(id='drift-status', style={'padding': '20px'})
    ]),
    
    html.Div([
        dcc.Graph(id='drift-metrics-chart')
    ], style={'padding': '20px'})
    
], style={'fontFamily': 'Outfit, sans-serif', 'backgroundColor': '#f9f7f7'})

@app.callback(
    Output('stats-cards', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_stats(n):
    stats = logger.get_statistics(hours=24)
    
    if not stats or stats['total_queries'] is None:
        return html.Div('No data available yet. Start making predictions!', 
                       style={'textAlign': 'center', 'padding': '40px', 'color': '#888'})
    
    cards = html.Div([
        create_stat_card('Total Queries', int(stats['total_queries'] or 0), '📊'),
        create_stat_card('Avg Confidence', f"{stats['avg_confidence'] or 0:.2f}", '🎯'),
        create_stat_card('Avg Response Time', f"{(stats['avg_response_time'] or 0)*1000:.0f}ms", '⚡'),
        create_stat_card('Avg Entities', f"{stats['avg_entity_count'] or 0:.1f}", '🏷️'),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'})
    
    return cards

def create_stat_card(title, value, icon):
    return html.Div([
        html.Div(icon, style={'fontSize': '2rem', 'marginBottom': '10px'}),
        html.Div(title, style={'fontSize': '0.9rem', 'color': '#888'}),
        html.Div(str(value), style={'fontSize': '1.8rem', 'fontWeight': 'bold', 'color': '#4a4a4a'})
    ], style={
        'backgroundColor': 'white',
        'padding': '20px',
        'borderRadius': '12px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.05)',
        'textAlign': 'center',
        'minWidth': '200px',
        'margin': '10px'
    })

@app.callback(
    Output('query-volume-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_query_volume(n):
    predictions = logger.get_recent_predictions(limit=100)
    
    if not predictions:
        return go.Figure().add_annotation(text="No data yet", showarrow=False)
    
    # Group by hour
    from collections import defaultdict
    hourly_counts = defaultdict(int)
    
    for p in predictions:
        if p['timestamp']:
            hour = p['timestamp'][:13]  # YYYY-MM-DD HH
            hourly_counts[hour] += 1
    
    hours = sorted(hourly_counts.keys())
    counts = [hourly_counts[h] for h in hours]
    
    fig = go.Figure(data=[
        go.Scatter(x=hours, y=counts, mode='lines+markers', 
                  line=dict(color='#e8aeb2', width=3),
                  marker=dict(size=8))
    ])
    
    fig.update_layout(
        title='Query Volume Over Time',
        xaxis_title='Time',
        yaxis_title='Number of Queries',
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output('confidence-distribution', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_confidence_dist(n):
    predictions = logger.get_recent_predictions(limit=100)
    confidences = [p['confidence'] for p in predictions if p['confidence']]
    
    if not confidences:
        return go.Figure().add_annotation(text="No data yet", showarrow=False)
    
    fig = go.Figure(data=[
        go.Histogram(x=confidences, nbinsx=20, 
                    marker=dict(color='#b8e0d2'))
    ])
    
    fig.update_layout(
        title='Confidence Score Distribution',
        xaxis_title='Confidence',
        yaxis_title='Frequency',
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output('entity-distribution', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_entity_dist(n):
    entity_dist = logger.get_entity_distribution(hours=24)
    
    if not entity_dist:
        return go.Figure().add_annotation(text="No entities detected yet", showarrow=False)
    
    # Top 10 entities
    sorted_entities = sorted(entity_dist.items(), key=lambda x: x[1], reverse=True)[:10]
    entities, counts = zip(*sorted_entities) if sorted_entities else ([], [])
    
    fig = go.Figure(data=[
        go.Bar(x=list(entities), y=list(counts), 
              marker=dict(color='#e8aeb2'))
    ])
    
    fig.update_layout(
        title='Top Detected Entities',
        xaxis_title='Entity',
        yaxis_title='Count',
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output('severity-distribution', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_severity_dist(n):
    severity_dist = logger.get_severity_distribution(hours=24)
    
    if not severity_dist:
        return go.Figure().add_annotation(text="No data yet", showarrow=False)
    
    fig = go.Figure(data=[
        go.Pie(labels=list(severity_dist.keys()), 
               values=list(severity_dist.values()),
               marker=dict(colors=['#b8e0d2', '#feca57', '#ff6b6b']))
    ])
    
    fig.update_layout(
        title='Severity Level Distribution',
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output('drift-status', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_drift_status(n):
    # Set baseline if not set
    if not drift_detector.baseline_data:
        success = drift_detector.set_baseline()
        if not success:
            return html.Div('⏳ Collecting baseline data... (need at least 30 predictions)', 
                          style={'color': '#888'})
    
    drift_results = drift_detector.detect_drift()
    
    if not drift_results:
        return html.Div('⏳ Not enough data for drift detection', style={'color': '#888'})
    
    drifted = [k for k, v in drift_results.items() if v['drift_detected']]
    
    if not drifted:
        return html.Div('✅ No drift detected - system is stable', 
                       style={'color': '#4caf50', 'fontSize': '1.2rem', 'fontWeight': 'bold'})
    
    alerts = [
        html.Div([
            html.Span('⚠️ ', style={'color': '#ff6b6b'}),
            html.Span(f"Drift in {metric}: ", style={'fontWeight': 'bold'}),
            html.Span(f"baseline={drift_results[metric]['baseline_mean']:.2f}, recent={drift_results[metric]['recent_mean']:.2f}")
        ], style={'margin': '10px 0'})
        for metric in drifted
    ]
    
    return html.Div(alerts)

@app.callback(
    Output('drift-metrics-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_drift_chart(n):
    if not drift_detector.baseline_data:
        return go.Figure().add_annotation(text="Setting baseline...", showarrow=False)
    
    drift_results = drift_detector.detect_drift()
    
    if not drift_results:
        return go.Figure().add_annotation(text="Not enough data", showarrow=False)
    
    metrics = list(drift_results.keys())
    baseline_means = [drift_results[m]['baseline_mean'] for m in metrics]
    recent_means = [drift_results[m]['recent_mean'] for m in metrics]
    
    fig = go.Figure(data=[
        go.Bar(name='Baseline', x=metrics, y=baseline_means, marker=dict(color='#b8e0d2')),
        go.Bar(name='Recent', x=metrics, y=recent_means, marker=dict(color='#e8aeb2'))
    ])
    
    fig.update_layout(
        title='Baseline vs Recent Metrics',
        barmode='group',
        template='plotly_white'
    )
    
    return fig

if __name__ == '__main__':
    print("Starting DermaAI Monitoring Dashboard...")
    print("Dashboard available at: http://localhost:8050")
    app.run(debug=True, port=8050)
