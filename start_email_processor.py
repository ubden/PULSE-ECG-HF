#!/usr/bin/env python3
"""
PULSE-7B Email ECG Processor Starter
Ubden® Team - Email processor başlatıcı script
"""

import os
import sys
import signal
import logging
from email_ecg_processor import EmailECGProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailProcessorService:
    """Email processor service wrapper"""
    
    def __init__(self):
        self.processor = None
        self.running = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📡 Received signal {signum}, shutting down gracefully...")
        self.running = False
        sys.exit(0)
    
    def check_environment(self):
        """Check required environment variables"""
        required_vars = [
            'mail_host',
            'mail_username', 
            'mail_pw',
            'hf_key'
        ]
        
        optional_vars = [
            'deep_key',
            'PULSE_ENDPOINT_URL',
            'HF_SPACE_NAME'
        ]
        
        logger.info("🔍 Checking environment configuration...")
        
        # Check required variables
        missing_required = []
        for var in required_vars:
            if not os.getenv(var):
                missing_required.append(var)
            else:
                logger.info(f"✅ {var}: {'*' * len(os.getenv(var))}")
        
        if missing_required:
            logger.error(f"❌ Missing required environment variables: {', '.join(missing_required)}")
            logger.info("\n📋 Required Environment Variables:")
            logger.info("• mail_host: Email server (e.g., imap.gmail.com)")
            logger.info("• mail_username: Email username")
            logger.info("• mail_pw: Email password (App Password for Gmail)")
            logger.info("• hf_key: HuggingFace API token")
            logger.info("\n📋 Optional Environment Variables:")
            logger.info("• deep_key: DeepSeek API key (for Turkish commentary)")
            logger.info("• PULSE_ENDPOINT_URL: Direct endpoint URL")
            logger.info("• HF_SPACE_NAME: HuggingFace space name")
            return False
        
        # Check optional variables
        for var in optional_vars:
            value = os.getenv(var)
            if value:
                logger.info(f"✅ {var}: {'*' * min(len(value), 10)}")
            else:
                logger.warning(f"⚠️ {var}: Not set (optional)")
        
        return True
    
    def start(self, check_interval: int = 300):
        """Start the email processor service"""
        logger.info("🚀 Starting PULSE-7B Email ECG Processor Service")
        logger.info("━" * 60)
        
        # Check environment
        if not self.check_environment():
            logger.error("❌ Environment check failed")
            return False
        
        logger.info("━" * 60)
        
        try:
            # Initialize processor
            self.processor = EmailECGProcessor()
            self.running = True
            
            logger.info(f"📧 Email processor initialized successfully")
            logger.info(f"⏰ Check interval: {check_interval} seconds")
            logger.info("🔄 Starting email monitoring...")
            logger.info("━" * 60)
            
            # Start processing
            self.processor.run_email_processor(check_interval)
            
        except KeyboardInterrupt:
            logger.info("🛑 Service stopped by user")
        except Exception as e:
            logger.error(f"❌ Service error: {e}")
            return False
        finally:
            logger.info("🔚 Email processor service stopped")
            
        return True

def print_usage():
    """Print usage information"""
    print("""
🏥 PULSE-7B Email ECG Processor Service

Usage:
    python start_email_processor.py [check_interval]

Arguments:
    check_interval    Check interval in seconds (default: 300 = 5 minutes)

Examples:
    python start_email_processor.py           # Check every 5 minutes
    python start_email_processor.py 60        # Check every 1 minute
    python start_email_processor.py 900       # Check every 15 minutes

Environment Variables Required:
    mail_host         Email IMAP server (e.g., imap.gmail.com)
    mail_username     Email username
    mail_pw          Email password (App Password for Gmail)
    hf_key           HuggingFace API token

Environment Variables Optional:
    deep_key         DeepSeek API key (for Turkish commentary)
    PULSE_ENDPOINT_URL  Direct endpoint URL
    HF_SPACE_NAME    HuggingFace space name

Gmail Setup:
    1. Enable 2-Factor Authentication
    2. Generate App Password: Google Account → Security → App passwords
    3. Use App Password as mail_pw (not your regular password)

HuggingFace Setup:
    1. Create API token: HuggingFace → Settings → Access Tokens
    2. Deploy PULSE-7B endpoint with this handler
    3. Add DeepSeek API key to endpoint environment (deep_key)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Ubden® Team - AI-Powered Healthcare Solutions
    """)

def main():
    """Main function"""
    # Parse command line arguments
    check_interval = 300  # Default 5 minutes
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            print_usage()
            return
        
        try:
            check_interval = int(sys.argv[1])
            if check_interval < 30:
                logger.warning("⚠️ Check interval too low, setting to minimum 30 seconds")
                check_interval = 30
            elif check_interval > 3600:
                logger.warning("⚠️ Check interval too high, setting to maximum 1 hour")
                check_interval = 3600
        except ValueError:
            logger.error("❌ Invalid check interval. Must be a number in seconds.")
            print_usage()
            return
    
    # Start service
    service = EmailProcessorService()
    service.start(check_interval)

if __name__ == "__main__":
    main()
