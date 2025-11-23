from flask import Blueprint, render_template, request, redirect, url_for, session, flash
import os
import logging

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle login page and authentication."""
    if request.method == 'POST':
        password = request.form.get('password', '')
        app_password = os.getenv('APP_PASSWORD', '')
        
        if not app_password:
            logger.error("APP_PASSWORD not configured in .env file")
            flash('Authentication not configured. Please contact administrator.', 'error')
            return render_template('login.html')
        
        if password == app_password:
            session['authenticated'] = True
            logger.info("User successfully authenticated")
            return redirect(url_for('main.index'))
        else:
            logger.warning("Failed login attempt")
            flash('Incorrect password. Please try again.', 'error')
            return render_template('login.html')
    
    # GET request - show login form
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    """Clear authentication and redirect to login."""
    session.pop('authenticated', None)
    logger.info("User logged out")
    return redirect(url_for('auth.login'))
