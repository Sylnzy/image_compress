from app import app, db
from sqlalchemy import text

def add_missing_columns():
    """Add missing columns to the compressed_image table"""
    with app.app_context():
        # Check if the columns already exist
        conn = db.engine.connect()
        inspector = db.inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('compressed_image')]
        
        if 'compression_method' not in columns:
            print("Adding compression_method column...")
            conn.execute(text("ALTER TABLE compressed_image ADD COLUMN compression_method VARCHAR(50) DEFAULT 'PIL'"))
        
        if 'parameter_value' not in columns:
            print("Adding parameter_value column...")
            conn.execute(text("ALTER TABLE compressed_image ADD COLUMN parameter_value VARCHAR(50)"))
        
        conn.commit()
        conn.close()
        print("Database migration completed successfully!")

if __name__ == '__main__':
    add_missing_columns()