"""
PULSE-7B Email ECG Processor
Ubden® Team - Email-based ECG Analysis System
Automatically processes ECG images from emails and sends Turkish commentary
"""

import imaplib
import smtplib
import email
import base64
import os
import time
import re
import requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.header import decode_header
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecg_email_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmailECGProcessor:
    """Email-based ECG analysis processor with retry logic and error handling"""
    
    def __init__(self):
        """Initialize with environment variables"""
        self.mail_host = os.getenv('mail_host', 'imap.gmail.com')
        self.mail_username = os.getenv('mail_username')
        self.mail_password = os.getenv('mail_pw')
        self.hf_token = os.getenv('hf_key')
        self.deepseek_key = os.getenv('deep_key')
        
        # PULSE endpoint configuration
        self.endpoint_url = self._get_endpoint_url()
        
        # Email configuration
        self.smtp_host = self.mail_host.replace('imap', 'smtp')
        self.smtp_port = 587
        self.imap_port = 993
        
        # Processing configuration
        self.max_retries = 3
        self.retry_delay = 60  # 1 minute
        self.max_wait_time = 300  # 5 minutes for cold start
        
        # Supported image formats for ECG detection
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        
        self._validate_configuration()
        logger.info("📧 Email ECG Processor initialized successfully")
    
    def _get_endpoint_url(self) -> str:
        """Get endpoint URL from environment or construct from HF space"""
        # Try direct endpoint URL first
        endpoint_url = os.getenv('PULSE_ENDPOINT_URL')
        if endpoint_url:
            return endpoint_url
        
        # Try to construct from HF space name
        hf_space = os.getenv('HF_SPACE_NAME')
        if hf_space:
            return f"https://{hf_space}.hf.space"
        
        # If running as part of HF Inference Endpoint, try to detect automatically
        hf_endpoint_name = os.getenv('ENDPOINT_NAME')
        hf_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        if hf_endpoint_name:
            return f"https://{hf_endpoint_name}.{hf_region}.aws.endpoints.huggingface.cloud"
        
        # Last resort - try localhost (for development)
        if os.getenv('DEVELOPMENT_MODE'):
            return "http://localhost:8000"
        
        raise ValueError("PULSE_ENDPOINT_URL, HF_SPACE_NAME, or ENDPOINT_NAME must be set in environment variables")
    
    def _validate_configuration(self):
        """Validate required environment variables"""
        required_vars = {
            'mail_username': self.mail_username,
            'mail_pw': self.mail_password,
            'hf_key': self.hf_token
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        logger.info("✅ Configuration validation passed")
    
    def connect_to_email(self) -> imaplib.IMAP4_SSL:
        """Connect to email server"""
        try:
            mail = imaplib.IMAP4_SSL(self.mail_host, self.imap_port)
            mail.login(self.mail_username, self.mail_password)
            mail.select('INBOX')
            logger.info(f"📧 Connected to email server: {self.mail_host}")
            return mail
        except Exception as e:
            logger.error(f"❌ Failed to connect to email server: {e}")
            raise
    
    def get_unread_emails(self, mail: imaplib.IMAP4_SSL) -> List[str]:
        """Get list of unread email IDs"""
        try:
            # Search for unread emails
            _, messages = mail.search(None, 'UNSEEN')
            message_ids = messages[0].split() if messages[0] else []
            logger.info(f"📬 Found {len(message_ids)} unread emails")
            return message_ids
        except Exception as e:
            logger.error(f"❌ Failed to get unread emails: {e}")
            return []
    
    def parse_email(self, mail: imaplib.IMAP4_SSL, message_id: str) -> Optional[Dict[str, Any]]:
        """Parse email and extract relevant information"""
        try:
            _, msg_data = mail.fetch(message_id, '(RFC822)')
            email_body = msg_data[0][1]
            email_message = email.message_from_bytes(email_body)
            
            # Extract email headers
            subject = decode_header(email_message['subject'])[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode()
            
            sender = email_message['from']
            date = email_message['date']
            
            # Extract body and attachments
            body_text = ""
            attachments = []
            
            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_maintype() == 'multipart':
                        continue
                    elif part.get_content_maintype() == 'text':
                        charset = part.get_content_charset() or 'utf-8'
                        try:
                            body_text += part.get_payload(decode=True).decode(charset)
                        except:
                            body_text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    elif part.get_content_maintype() == 'image':
                        filename = part.get_filename()
                        if filename and self._is_supported_image(filename):
                            image_data = part.get_payload(decode=True)
                            attachments.append({
                                'filename': filename,
                                'data': image_data,
                                'content_type': part.get_content_type()
                            })
            else:
                charset = email_message.get_content_charset() or 'utf-8'
                try:
                    body_text = email_message.get_payload(decode=True).decode(charset)
                except:
                    body_text = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            return {
                'message_id': message_id,
                'subject': subject,
                'sender': sender,
                'date': date,
                'body': body_text,
                'attachments': attachments
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to parse email {message_id}: {e}")
            return None
    
    def _is_supported_image(self, filename: str) -> bool:
        """Check if the image has a supported format"""
        filename_lower = filename.lower()
        is_supported = any(filename_lower.endswith(ext) for ext in self.supported_image_formats)
        
        logger.info(f"🔍 Image format check for '{filename}': {is_supported}")
        return is_supported
    
    def analyze_ecg_with_retry(self, image_data: bytes, query: str, content_type: str) -> Dict[str, Any]:
        """Analyze ECG image with retry logic for cold start handling"""
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 Analysis attempt {attempt + 1}/{self.max_retries}")
                
                # Convert image to base64
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                mime_type = content_type or 'image/jpeg'
                base64_string = f"data:{mime_type};base64,{image_base64}"
                
                # Prepare payload
                payload = {
                    "inputs": {
                        "query": query,
                        "image": base64_string
                    },
                    "parameters": {
                        "max_new_tokens": 512,
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "repetition_penalty": 1.05,
                        "enable_turkish_commentary": True,
                        "deepseek_timeout": 30
                    }
                }
                
                headers = {
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type": "application/json"
                }
                
                # Make request
                response = requests.post(
                    self.endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()[0]
                    elapsed_time = time.time() - start_time
                    logger.info(f"✅ ECG analysis completed in {elapsed_time:.1f} seconds")
                    return result
                    
                elif response.status_code == 503:
                    # Service Unavailable - Cold start
                    elapsed_time = time.time() - start_time
                    if elapsed_time < self.max_wait_time and attempt < self.max_retries - 1:
                        wait_time = min(self.retry_delay, self.max_wait_time - elapsed_time)
                        logger.info(f"⏳ Endpoint cold start detected. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception("Endpoint startup timeout")
                        
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"⏰ Request timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise Exception("Analysis timeout after all retries")
                    
            except Exception as e:
                logger.error(f"❌ Analysis error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise
        
        raise Exception("Analysis failed after all retry attempts")
    
    def send_analysis_result(self, recipient: str, original_subject: str, 
                           filename: str, analysis_result: Dict[str, Any]) -> bool:
        """Send analysis result via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.mail_username
            msg['To'] = recipient
            msg['Subject'] = f"🏥 EKG Sonucunuz: {filename}"
            
            # Extract sender name for personalization
            sender_name = recipient.split('@')[0].title()
            
            # Prepare email body
            english_analysis = analysis_result.get('generated_text', 'Analysis not available')
            turkish_commentary = analysis_result.get('comment_text', 'Türkçe yorum mevcut değil')
            
            # Format the email body
            body = f"""
Sayın {sender_name},

EKG görüntünüzün analizi tamamlanmıştır. Aşağıda detaylı sonuçları bulabilirsiniz:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔬 **İNGİLİZCE TEKNİK ANALİZ:**
{english_analysis}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{turkish_commentary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  **ÖNEMLİ UYARI:**
Bu analiz sadece bilgilendirme amaçlıdır ve kesinlikle tıbbi tanı yerine geçmez. 
Kesin tanı ve tedavi için mutlaka bir kardiyoloji uzmanına başvurunuz.

🏥 **ACİL DURUMLARDA:**
Göğüs ağrısı, nefes darlığı, çarpıntı gibi şikayetleriniz varsa 
derhal en yakın acil servise başvurunuz veya 112'yi arayınız.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 **Analiz Detayları:**
• Dosya: {filename}
• Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}
• Model: PULSE-7B + DeepSeek AI
• İşlem Durumu: {analysis_result.get('commentary_status', 'Başarılı')}

Geçmiş olsun! 🙏

---
🤖 Bu e-posta PULSE-7B EKG Analiz Sistemi tarafından otomatik olarak gönderilmiştir.
💡 Ubden® Team - AI-Powered Healthcare Solutions
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.mail_username, self.mail_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"✅ Analysis result sent to {recipient}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send analysis result to {recipient}: {e}")
            return False
    
    def send_error_notification(self, recipient: str, original_subject: str, 
                              filename: str, error_message: str) -> bool:
        """Send error notification via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.mail_username
            msg['To'] = recipient
            msg['Subject'] = f"⚠️ EKG Analizi Hatası: {filename}"
            
            sender_name = recipient.split('@')[0].title()
            
            body = f"""
Sayın {sender_name},

Maalesef EKG görüntünüzün analizi sırasında bir hata oluştu.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 **Hata Detayları:**
• Dosya: {filename}
• Hata: {error_message}
• Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 **Olası Çözümler:**
1. EKG görüntüsünün net ve okunaklı olduğundan emin olun
2. Desteklenen dosya formatlarını kullanın (JPG, PNG, GIF, BMP)
3. Dosya boyutunun 10MB'dan küçük olduğundan emin olun
4. Birkaç dakika sonra tekrar deneyin (sistem başlatılıyor olabilir)

🔄 **Tekrar Deneme:**
EKG görüntünüzü yeniden gönderebilirsiniz. Sistem otomatik olarak işleme alacaktır.

📞 **Destek:**
Problem devam ederse lütfen sistem yöneticisi ile iletişime geçin.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Özür dileriz! 🙏

---
🤖 Bu e-posta PULSE-7B EKG Analiz Sistemi tarafından otomatik olarak gönderilmiştir.
💡 Ubden® Team - AI-Powered Healthcare Solutions
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.mail_username, self.mail_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"📧 Error notification sent to {recipient}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send error notification to {recipient}: {e}")
            return False
    
    def mark_as_read(self, mail: imaplib.IMAP4_SSL, message_id: str) -> bool:
        """Mark email as read"""
        try:
            mail.store(message_id, '+FLAGS', '\\Seen')
            logger.info(f"📖 Email {message_id} marked as read")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to mark email {message_id} as read: {e}")
            return False
    
    def process_single_email(self, mail: imaplib.IMAP4_SSL, message_id: str) -> bool:
        """Process a single email"""
        logger.info(f"📧 Processing email {message_id}")
        
        try:
            # Parse email
            email_data = self.parse_email(mail, message_id)
            if not email_data:
                logger.warning(f"⚠️ Failed to parse email {message_id}")
                return False
            
            logger.info(f"📬 Email from: {email_data['sender']}, Subject: {email_data['subject']}")
            
            # Process image attachments (assuming they are ECG images)
            processed_any = False
            for attachment in email_data['attachments']:
                logger.info(f"🖼️ Processing image attachment: {attachment['filename']}")
                
                try:
                    # Analyze image as ECG
                    query = f"Analyze this ECG image from email. Subject: {email_data['subject']}. Context: {email_data['body'][:200]}..."
                    
                    analysis_result = self.analyze_ecg_with_retry(
                        attachment['data'],
                        query,
                        attachment['content_type']
                    )
                    
                    # Send result
                    success = self.send_analysis_result(
                        email_data['sender'],
                        email_data['subject'],
                        attachment['filename'],
                        analysis_result
                    )
                    
                    if success:
                        processed_any = True
                        logger.info(f"✅ Successfully processed {attachment['filename']}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to analyze {attachment['filename']}: {e}")
                    
                    # Send error notification
                    self.send_error_notification(
                        email_data['sender'],
                        email_data['subject'],
                        attachment['filename'],
                        str(e)
                    )
            
            # Mark as read if we processed any attachments
            if processed_any:
                self.mark_as_read(mail, message_id)
                return True
            elif not email_data['attachments']:
                # No image attachments found, mark as read to avoid reprocessing
                logger.info(f"📭 No image attachments found in email {message_id}")
                self.mark_as_read(mail, message_id)
                return True
            else:
                logger.warning(f"⚠️ Failed to process any attachments in email {message_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing email {message_id}: {e}")
            return False
    
    def run_email_processor(self, check_interval: int = 300) -> None:
        """Main email processing loop"""
        logger.info(f"🚀 Starting Email ECG Processor (check interval: {check_interval}s)")
        
        while True:
            try:
                logger.info("🔄 Checking for new emails...")
                
                # Connect to email
                mail = self.connect_to_email()
                
                # Get unread emails
                unread_emails = self.get_unread_emails(mail)
                
                if unread_emails:
                    logger.info(f"📬 Processing {len(unread_emails)} unread emails")
                    
                    for message_id in unread_emails:
                        self.process_single_email(mail, message_id)
                        time.sleep(2)  # Brief pause between emails
                else:
                    logger.info("📭 No unread emails found")
                
                # Close connection
                mail.close()
                mail.logout()
                
                # Wait for next check
                logger.info(f"⏰ Waiting {check_interval} seconds until next check...")
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Email processor stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                logger.info(f"⏰ Waiting {check_interval} seconds before retry...")
                time.sleep(check_interval)

def main():
    """Main function to run the email processor"""
    try:
        processor = EmailECGProcessor()
        processor.run_email_processor(check_interval=300)  # Check every 5 minutes
    except Exception as e:
        logger.error(f"❌ Failed to start email processor: {e}")
        raise

if __name__ == "__main__":
    main()
