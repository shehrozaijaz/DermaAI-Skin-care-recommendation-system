import sqlite3
import json
from datetime import datetime
from pathlib import Path

class MetricsLogger:
    """Logs prediction metrics to SQLite database for monitoring and drift detection."""
    
    def __init__(self, db_path='monitoring/metrics.db'):
        self.db_path = db_path
        Path('monitoring').mkdir(exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                query_length INTEGER,
                entities TEXT,
                entity_count INTEGER,
                severity TEXT,
                confidence REAL,
                response_time REAL,
                top_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                drift_score REAL,
                threshold REAL,
                alert_message TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, query, entities, severity, confidence, response_time, top_score):
        """Log a single prediction."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (query, query_length, entities, entity_count, severity, confidence, response_time, top_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            query,
            len(query.split()),
            json.dumps(entities),
            len(entities),
            severity,
            confidence,
            response_time,
            top_score
        ))
        
        conn.commit()
        conn.close()
    
    def log_drift_alert(self, metric_name, drift_score, threshold, message):
        """Log a drift detection alert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO drift_alerts (metric_name, drift_score, threshold, alert_message)
            VALUES (?, ?, ?, ?)
        ''', (metric_name, drift_score, threshold, message))
        
        conn.commit()
        conn.close()
    
    def get_recent_predictions(self, limit=100):
        """Retrieve recent predictions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_statistics(self, hours=24):
        """Get aggregated statistics for the last N hours."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f'''
            SELECT 
                COUNT(*) as total_queries,
                AVG(query_length) as avg_query_length,
                AVG(entity_count) as avg_entity_count,
                AVG(confidence) as avg_confidence,
                AVG(response_time) as avg_response_time,
                AVG(top_score) as avg_top_score
            FROM predictions
            WHERE timestamp > datetime('now', '-{hours} hours')
        ''')
        
        columns = [desc[0] for desc in cursor.description]
        result = dict(zip(columns, cursor.fetchone()))
        
        conn.close()
        return result
    
    def get_entity_distribution(self, hours=24):
        """Get distribution of entities over time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f'''
            SELECT entities FROM predictions
            WHERE timestamp > datetime('now', '-{hours} hours')
        ''')
        
        all_entities = []
        for row in cursor.fetchall():
            entities = json.loads(row[0])
            all_entities.extend(entities)
        
        conn.close()
        
        # Count occurrences
        from collections import Counter
        return dict(Counter(all_entities))
    
    def get_severity_distribution(self, hours=24):
        """Get distribution of severity levels."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f'''
            SELECT severity, COUNT(*) as count
            FROM predictions
            WHERE timestamp > datetime('now', '-{hours} hours')
            GROUP BY severity
        ''')
        
        result = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return result
