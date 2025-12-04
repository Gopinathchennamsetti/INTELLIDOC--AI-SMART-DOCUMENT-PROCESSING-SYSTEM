import logging
from routes import app  # imports the Flask app configured in routes.py
import config  # ensures folders are created

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logging.info('Starting the Flask application.')

if __name__ == '__main__':
    app.run(debug=True)