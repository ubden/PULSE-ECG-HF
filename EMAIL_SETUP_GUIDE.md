# 📧 PULSE-7B Email ECG Processor Setup Guide

Bu rehber, e-posta tabanlı EKG analiz sisteminin kurulumu ve kullanımı için adım adım talimatları içerir.

## 🎯 Sistem Özellikleri

✅ **Otomatik E-posta İşleme** - Gelen e-postaları sürekli izler
✅ **Görüntü İşleme** - Tüm resim formatlarını (JPG, PNG, GIF, BMP, TIFF) EKG olarak analiz eder
✅ **AI Tabanlı Analiz** - PULSE-7B ile medikal analiz
✅ **Türkçe Yorum** - DeepSeek ile hasta dostu açıklamalar
✅ **Otomatik Yanıt** - Sonuçları e-posta ile gönderir
✅ **Cold Start Handling** - HuggingFace endpoint sorunlarını çözer
✅ **Hata Yönetimi** - Kapsamlı retry logic ve error handling

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler

- Python 3.8+
- Gmail hesabı (veya başka IMAP destekli e-posta)
- HuggingFace hesabı ve PULSE-7B endpoint
- DeepSeek API key (isteğe bağlı)

### 2. Kurulum

```bash
# Repository'yi klonla
git clone <your-repo>
cd pulse-ecg-system

# Python bağımlılıklarını yükle
pip install -r requirements.txt

# Environment variables'ları ayarla
cp email_config_example.txt .env
# .env dosyasını düzenle (aşağıda detayları var)
```

### 3. Environment Variables Ayarlama

```bash
# Email Configuration
export mail_host="imap.gmail.com"
export mail_username="your-email@gmail.com"
export mail_pw="your-app-password"

# HuggingFace Configuration  
export hf_key="hf_your_token_here"
export PULSE_ENDPOINT_URL="endpont_url"

# DeepSeek Configuration (Optional)
export deep_key="sk-your-deepseek-key"
```

### 4. Çalıştırma

```bash
# Doğrudan çalıştır
python start_email_processor.py

# Veya Docker ile
docker-compose -f docker-compose.email.yml up -d
```

## 📋 Detaylı Kurulum Adımları

### 1. Gmail App Password Oluşturma

1. **Google Hesabınıza giriş yapın**
2. **Güvenlik** → **2 Adımlı Doğrulama** → Etkinleştirin
3. **Uygulama şifreleri** → **Posta** seçin
4. **16 haneli şifreyi kopyalayın**
5. Bu şifreyi `mail_pw` olarak kullanın

### 2. HuggingFace Endpoint Kurulumu

```bash
# 1. HuggingFace'de endpoint oluştur
# 2. PULSE-7B handler'ını deploy et
# 3. Environment variables'ları ekle:
#    - deep_key: DeepSeek API key
# 4. Endpoint URL'ini kopyala
```

### 3. DeepSeek API Key Alma

1. **DeepSeek.com'a kaydolun**
2. **API Keys** → **Create New Key**
3. Key'i kopyalayın ve `deep_key` olarak ayarlayın

## 🔧 Konfigürasyon Seçenekleri

### Email Providers

```bash
# Gmail
mail_host=imap.gmail.com

# Outlook
mail_host=outlook.office365.com

# Yahoo
mail_host=imap.mail.yahoo.com

# Custom IMAP
mail_host=your-imap-server.com
```

### Processing Settings

```bash
# Check interval (seconds)
CHECK_INTERVAL=300  # 5 dakika

# Log level
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## 📧 Kullanım

### E-posta Gönderme Formatı

```
To: your-configured-email@gmail.com
Subject: EKG Analizi (veya herhangi bir konu)
Body: Hasta bilgileri, şikayetler vb.
Attachment: ecg_image.jpg (veya png, gif, bmp)
```

### Örnek E-posta

```
Konu: Acil EKG Değerlendirmesi

Merhaba,

65 yaşındaki erkek hastada göğüs ağrısı şikayeti mevcut. 
EKG çekildi, değerlendirmenizi rica ederim.

Teşekkürler.

[EKG görüntüsü ekte]
```

### Alacağınız Yanıt

```
Konu: 🏥 EKG Sonucunuz: ecg_image.jpg

Sayın Kullanıcı,

EKG görüntünüzün analizi tamamlanmıştır...

🔬 İNGİLİZCE TEKNİK ANALİZ:
Answer: This ECG shows sinus rhythm with ST elevation...

**Türkçe Açıklama:**
Bu EKG sonucu, kalbin normal bir ritimle attığını gösteriyor...

⚠️ ÖNEMLİ UYARI:
Bu analiz sadece bilgilendirme amaçlıdır...
```

## 🐳 Docker ile Çalıştırma

### 1. Docker Build

```bash
# Image'i oluştur
docker build -f Dockerfile.email -t pulse-email-processor .
```

### 2. Docker Compose

```bash
# Environment variables'ları ayarla
cp email_config_example.txt .env

# Servisi başlat
docker-compose -f docker-compose.email.yml up -d

# Logları izle
docker-compose -f docker-compose.email.yml logs -f

# Servisi durdur
docker-compose -f docker-compose.email.yml down
```

### 3. Docker Environment

```yaml
# docker-compose.email.yml içinde
environment:
  - mail_host=imap.gmail.com
  - mail_username=your-email@gmail.com
  - mail_pw=your-app-password
  - hf_key=your-hf-token
  - PULSE_ENDPOINT_URL=your-endpoint-url
  - deep_key=your-deepseek-key
```

## 🔍 İzleme ve Debugging

### Log Dosyaları

```bash
# Ana log dosyası
tail -f ecg_email_processor.log

# Docker logs
docker logs pulse-ecg-email-processor -f
```

### Health Check

```bash
# Sistem durumu kontrolü
python -c "from email_ecg_processor import EmailECGProcessor; p = EmailECGProcessor(); print('System OK')"

# Docker health check
docker exec pulse-ecg-email-processor python -c "import email_ecg_processor; print('healthy')"
```

### Common Issues

#### 1. Authentication Errors
```bash
# Gmail için App Password kullanın
# 2FA aktif olmalı
# IMAP enabled olmalı
```

#### 2. Connection Errors
```bash
# Firewall ayarlarını kontrol edin
# Port 993 (IMAP) ve 587 (SMTP) açık olmalı
```

#### 3. HuggingFace 503 Errors
```bash
# Endpoint cold start - 3-5 dakika bekleyin
# Retry logic otomatik çalışacak
```

## 🔐 Güvenlik

### Best Practices

1. **App Password kullanın** (Gmail için)
2. **API key'leri güvenli tutun**
3. **Environment variables kullanın**
4. **Regular password rotation**
5. **Log dosyalarını güvenli tutun**

### Production Deployment

```bash
# Docker secrets kullanın
# Encrypted storage
# Network security
# Access logging
# Regular updates
```

## 📊 Monitoring

### Metrics

- **E-posta işleme sayısı**
- **Başarılı analiz oranı**  
- **Yanıt süresi**
- **Hata oranları**
- **Endpoint durumu**

### Alerts

- **Authentication failures**
- **Endpoint downtime**
- **High error rates**
- **Processing delays**

## 🚨 Troubleshooting

### Problem: E-postalar işlenmiyor

```bash
# 1. Authentication kontrolü
# 2. IMAP ayarları
# 3. Firewall/network
# 4. Log dosyalarını kontrol et
```

### Problem: Resimler işlenmiyor

```bash
# 1. Dosya formatı kontrol et (jpg, png, gif, bmp, tiff)
# 2. Dosya boyutu (<10MB)
# 3. E-posta attachment olarak gönderilmiş mi
# 4. Resim dosyası bozuk mu
```

### Problem: Türkçe yorum gelmiyor

```bash
# 1. DeepSeek API key kontrolü
# 2. deep_key environment variable
# 3. DeepSeek API quota
# 4. Network connectivity
```

### Problem: HuggingFace endpoint hatası

```bash
# 1. Endpoint durumu kontrol et
# 2. HF token geçerliliği
# 3. Cold start bekle (3-5 dakika)
# 4. Retry logic çalışıyor mu
```

## 📞 Destek

### Log Analysis

```bash
# Error patterns
grep "ERROR" ecg_email_processor.log

# Processing stats  
grep "Successfully processed" ecg_email_processor.log | wc -l

# Performance metrics
grep "completed in" ecg_email_processor.log
```

### Contact

- **Technical Issues**: Check logs first
- **Configuration Help**: Review this guide
- **Feature Requests**: Create GitHub issue
- **Emergency**: Contact system administrator

---

## 🎉 Başarı!

Sistem kurulumu tamamlandı! Artık e-posta ile EKG görüntülerini gönderebilir ve otomatik analiz sonuçlarını alabilirsiniz.

**Test için:**
1. Sistemi başlatın
2. Kendinize bir test e-postası gönderin (EKG resmi ile)
3. 5 dakika içinde yanıt almalısınız
4. Log dosyalarını kontrol edin

**🏥 Sağlıklı günler! 💙**

---
💡 **Ubden® Team - AI-Powered Healthcare Solutions**
