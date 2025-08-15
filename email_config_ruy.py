"""
RUY.APP Email Configuration Helper
Ubden® Team - Specific configuration for ruy.app email domain
"""

import os
import logging

logger = logging.getLogger(__name__)

class RuyAppEmailConfig:
    """Configuration helper for ruy.app email domain"""
    
    @staticmethod
    def get_email_settings():
        """Get email settings for ruy.app domain"""
        
        # ruy.app email configuration
        # Based on the screenshot showing POP3, IMAP, SMTP settings
        config = {
            'imap_host': 'ruy.app',  # Try direct domain first
            'imap_port': 993,        # Standard IMAP SSL port
            'imap_ssl': True,
            
            'smtp_host': 'ruy.app',  # Try direct domain first
            'smtp_port': 465,        # Try SSL SMTP port first
            'smtp_ssl': True,
            
            # Alternative configurations to try
            'alternatives': [
                # Try with mail. prefix
                {
                    'imap_host': 'mail.ruy.app',
                    'smtp_host': 'mail.ruy.app',
                    'imap_port': 993,
                    'smtp_port': 465,
                },
                # Try with imap./smtp. prefixes
                {
                    'imap_host': 'imap.ruy.app',
                    'smtp_host': 'smtp.ruy.app',
                    'imap_port': 993,
                    'smtp_port': 587,  # Try STARTTLS port
                },
                # Try non-SSL ports
                {
                    'imap_host': 'ruy.app',
                    'smtp_host': 'ruy.app',
                    'imap_port': 143,  # Non-SSL IMAP
                    'smtp_port': 587,  # STARTTLS SMTP
                },
            ]
        }
        
        return config
    
    @staticmethod
    def setup_environment_for_ruy_app():
        """Setup environment variables for ruy.app"""
        
        # Set default values if not already set
        if not os.getenv('mail_host'):
            os.environ['mail_host'] = 'ruy.app'
            
        if not os.getenv('IMAP_PORT'):
            os.environ['IMAP_PORT'] = '993'
            
        if not os.getenv('SMTP_HOST'):
            os.environ['SMTP_HOST'] = 'ruy.app'
            
        if not os.getenv('SMTP_PORT'):
            os.environ['SMTP_PORT'] = '465'
            
        logger.info("🔧 RUY.APP email configuration applied")
        
        return {
            'mail_host': os.getenv('mail_host'),
            'imap_port': os.getenv('IMAP_PORT'),
            'smtp_host': os.getenv('SMTP_HOST'),
            'smtp_port': os.getenv('SMTP_PORT'),
        }

def detect_and_configure_ruy_app():
    """Auto-detect and configure ruy.app email settings"""
    
    mail_username = os.getenv('mail_username', '')
    
    if 'ruy.app' in mail_username.lower():
        logger.info("🎯 Detected ruy.app email domain - applying specific configuration")
        return RuyAppEmailConfig.setup_environment_for_ruy_app()
    
    return None

# Auto-configure if this module is imported
if __name__ != "__main__":
    detect_and_configure_ruy_app()
