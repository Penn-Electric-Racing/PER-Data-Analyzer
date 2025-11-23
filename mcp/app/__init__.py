from flask import Flask
from .config import Config
from .routes.main import main_bp
from .routes.chat import chat_bp

def create_app(config_class=Config):
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config.from_object(config_class)

    app.register_blueprint(main_bp)
    app.register_blueprint(chat_bp)

    return app
