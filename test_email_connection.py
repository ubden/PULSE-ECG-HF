#!/usr/bin/env python3
"""
Email Connection Test for RUY.APP
Ubden® Team - Test email connectivity before running full processor
"""

import os
import sys
import logging
from email_ecg_processor import EmailECGProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_email_connection():
    """Test email connection with current configuration"""
    
    print("🧪 Testing Email Connection for RUY.APP")
    print("=" * 50)
    
    # Check environment variables
    required_vars = ['mail_username', 'mail_pw']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\n📋 Required Environment Variables:")
        print("export mail_username='ekgsonuc@ruy.app'")
        print("export mail_pw='your_password'")
        return False
    
    # Show current configuration
    print(f"📧 Testing connection for: {os.getenv('mail_username')}")
    print(f"🏠 Mail host: {os.getenv('mail_host', 'auto-detected')}")
    print(f"🔌 IMAP port: {os.getenv('IMAP_PORT', 'auto-detected')}")
    print(f"📤 SMTP host: {os.getenv('SMTP_HOST', 'auto-detected')}")
    print(f"🔌 SMTP port: {os.getenv('SMTP_PORT', 'auto-detected')}")
    print()
    
    try:
        # Initialize processor (this will auto-configure ruy.app settings)
        print("🔧 Initializing email processor...")
        processor = EmailECGProcessor()
        
        # Test IMAP connection
        print("📥 Testing IMAP connection...")
        mail = processor.connect_to_email()
        
        # Test basic operations
        print("📬 Testing email operations...")
        unread_emails = processor.get_unread_emails(mail)
        print(f"✅ Found {len(unread_emails)} unread emails")
        
        # Close connection
        mail.close()
        mail.logout()
        print("✅ IMAP connection test successful!")
        
        # Test SMTP by sending a test email to self
        print("📤 Testing SMTP connection...")
        test_msg_content = """
Konu: 🧪 Email Connection Test

Bu bir test mesajıdır. Email sistemi çalışıyor!

Test zamanı: {timestamp}

---
PULSE-7B Email ECG Processor Test
        """.format(timestamp=__import__('datetime').datetime.now())
        
        # Create a simple test message
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        test_msg = MIMEMultipart()
        test_msg['From'] = processor.mail_username
        test_msg['To'] = processor.mail_username  # Send to self
        test_msg['Subject'] = "🧪 PULSE-7B Email Test"
        test_msg.attach(MIMEText(test_msg_content, 'plain', 'utf-8'))
        
        processor._send_email_with_retry(test_msg)
        print("✅ SMTP connection test successful!")
        
        print("\n🎉 All tests passed! Email system is working correctly.")
        print("📧 You should receive a test email shortly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Connection test failed: {e}")
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Check if email credentials are correct")
        print("2. Verify that IMAP/SMTP is enabled for your account")
        print("3. Check firewall/network connectivity")
        print("4. Try different ports (993, 143 for IMAP; 465, 587 for SMTP)")
        print("5. Contact your email provider for server settings")
        return False

def show_ruy_app_settings():
    """Show recommended settings for ruy.app"""
    print("\n📋 Recommended RUY.APP Settings:")
    print("Based on your email provider screenshot:")
    print()
    print("IMAP Settings:")
    print("  - Server: ruy.app (or mail.ruy.app)")
    print("  - Port: 993 (SSL) or 143 (STARTTLS)")
    print("  - Security: SSL/TLS")
    print()
    print("SMTP Settings:")
    print("  - Server: ruy.app (or mail.ruy.app)")  
    print("  - Port: 465 (SSL) or 587 (STARTTLS)")
    print("  - Security: SSL/TLS")
    print()
    print("POP3 Settings (if needed):")
    print("  - Server: ruy.app")
    print("  - Port: 995 (SSL)")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("Email Connection Test for RUY.APP")
        print()
        print("Usage:")
        print("  python test_email_connection.py")
        print()
        print("Environment Variables:")
        print("  mail_username - Your ruy.app email address")
        print("  mail_pw       - Your email password")
        print()
        show_ruy_app_settings()
        sys.exit(0)
    
    success = test_email_connection()
    
    if not success:
        show_ruy_app_settings()
        sys.exit(1)
    else:
        print("\n🚀 Ready to run the full email processor!")
        print("Run: python start_email_processor.py")
        sys.exit(0)
