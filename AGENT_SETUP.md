# AIC Data Collection — AWS Kurulum ve Çalıştırma Kılavuzu

Bu doküman AWS g4dn.xlarge üzerinde sıfırdan kurulum ve veri toplamayı
tam otomatik olarak yürütmek için yazılmıştır.
Gemini CLI veya Claude Code ile bu dosyayı açıp adım adım uygulayabilirsin.

## AWS Instance Gereksinimleri

```
Instance type : g4dn.xlarge  (spot request önerilir)
AMI           : ami-0c7217cdde317cfec  (Ubuntu 22.04 Deep Learning, us-east-1)
               veya: "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)" arayın
Storage       : 80 GB gp3 (minimum)
Security group: SSH (22) açık
GPU           : NVIDIA T4 (16 GB VRAM) — Gazebo rendering + eğitim için şart
```

---

## ADIM 0 — Mac'te (Bağlanmadan Önce): Kodu AWS'e Gönder

Fork veya push'a gerek yok. AWS'de orijinal repo'yu clone'layıp,
kendi eklediğimiz dosyaları scp ile direkt göndereceğiz.

### 0a. AWS instance IP'sini öğren ve SSH erişimini doğrula

```bash
# AWS Console → EC2 → Instances → Public IPv4 address kopyala
AWS_IP="<buraya-aws-ip-yaz>"   # örnek: 54.123.45.67

# Key dosyasının izinlerini ayarla (bir kez)
chmod 400 ~/Desktop/aic-key.pem

# Bağlantıyı test et
ssh -i ~/Desktop/aic-key.pem ubuntu@$AWS_IP "echo 'SSH bağlantısı OK ✓'"
```

### 0b. AWS'de workspace hazırla (SSH ile gir, bir komut çalıştır)

```bash
ssh -i ~/Desktop/aic-key.pem ubuntu@$AWS_IP \
  "mkdir -p ~/ws_aic/src && \
   cd ~/ws_aic/src && \
   git clone https://github.com/intrinsic-dev/aic && \
   echo 'Repo clone edildi ✓'"
```

### 0c. Kendi kodlarımızı scp ile gönder

```bash
# Değişkeni hâlâ terminalde tutuyorsan:
AWS_IP="<buraya-aws-ip-yaz>"

# Eklediğimiz paket + config + kurulum rehberi
scp -i ~/Desktop/aic-key.pem -r \
  /Users/y.aykut/Desktop/aic/aic_data_collector \
  ubuntu@$AWS_IP:~/ws_aic/src/aic/

scp -i ~/Desktop/aic-key.pem \
  /Users/y.aykut/Desktop/aic/pixi.toml \
  /Users/y.aykut/Desktop/aic/AGENT_SETUP.md \
  ubuntu@$AWS_IP:~/ws_aic/src/aic/

echo "Transfer tamamlandı ✓"
```

### 0d. Transfer'i doğrula

```bash
ssh -i ~/Desktop/aic-key.pem ubuntu@$AWS_IP \
  "ls ~/ws_aic/src/aic/aic_data_collector/ && echo 'Dosyalar geldi ✓'"
```

---

## ADIM 1 — Temel Kurulum (AWS'de, ilk bağlantıda bir kez)

```bash
# 1a. Sistem güncellemesi
sudo apt-get update && sudo apt-get upgrade -y

# 1b. Docker kurulumu (eğer yoksa)
which docker || (
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER
  newgrp docker
)

# Docker çalışıyor mu?
docker info | head -5

# 1c. Distrobox kurulumu
which distrobox || sudo apt-get install -y distrobox

# 1d. Pixi kurulumu
which pixi || (
  curl -fsSL https://pixi.sh/install.sh | sh
  source ~/.bashrc
)

# 1e. NVIDIA Docker toolkit (Deep Learning AMI'de genellikle mevcut)
nvidia-smi  # GPU görünüyor mu?
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

---

## ADIM 2 — AIC Toolkit Kurulumu

```bash
# 2a. Workspace oluştur ve repo klonla
mkdir -p ~/ws_aic/src
cd ~/ws_aic/src
git clone https://github.com/intrinsic-dev/aic
cd aic

# 2b. Pixi bağımlılıklarını kur (birkaç dakika sürer)
pixi install

# 2c. Eval container image'ı çek
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest

# 2d. Distrobox container'ı oluştur (GPU ile)
export DBX_CONTAINER_MANAGER=docker
distrobox create --root --nvidia \
  --image ghcr.io/intrinsic-dev/aic/aic_eval:latest \
  --name aic_eval

# 2e. Container'ı bir kez başlat ve çıkma (kurulum tamamlanır)
distrobox enter --root aic_eval -- echo "Container hazır ✓"
```

---

## ADIM 3 — Test: Eval Environment + WaveArm

Bu adımda her şeyin çalıştığını doğruluyoruz.

```bash
# Terminal 1: Eval container başlat (ground_truth=true ile — DataCollector için şart!)
export DBX_CONTAINER_MANAGER=docker
distrobox enter --root aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true

# Terminal 2 (yeni SSH bağlantısı): WaveArm'ı çalıştır
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.WaveArm
```

Beklenen: Gazebo açılır, robot dalgalanır, 3 trial tamamlanır, score görünür.

---

## ADIM 4 — DataCollectorPolicy Kurulumu

```bash
cd ~/ws_aic/src/aic

# Yeni paketi pixi'ye kayıt ettir
pixi reinstall ros-kilted-aic-data-collector

# Kurulum doğrula
pixi run python3 -c "from aic_data_collector.ros.DataCollectorPolicy import DataCollectorPolicy; print('Import OK ✓')"
```

---

## ADIM 5 — Veri Toplama (Otomatik, ~2-3 Saat)

> **Önemli:** aic_engine README'ye göre her 3 trial sonunda "Completed" state'e geçip
> duruyor. Bu yüzden her batch için eval container'ı da yeniden başlatmamız gerekiyor.
> tmux ile iki pencereyi paralel yönetiyoruz.

### 5a. tmux kur ve oturum başlat

```bash
# tmux kurula (genellikle Deep Learning AMI'de mevcut)
which tmux || sudo apt-get install -y tmux

# Kalıcı bir tmux oturumu başlat (SSH kesilse bile devam eder)
tmux new-session -s aic
# Ctrl+B, D ile detach edebilirsin. Geri dönmek için: tmux attach -t aic
```

### 5b. Tam Otomatik Batch Döngüsü (67 batch = ~200 episode)

tmux içinde şu scripti çalıştır. Her batch için:
1. eval container'ı başlatır (3 trial yapıp durur)
2. DataCollectorPolicy'yi bağlar (3 episode kaydeder)
3. Eval container'ı temizler
4. Tekrar eder

```bash
cd ~/ws_aic/src/aic
export DBX_CONTAINER_MANAGER=docker

DATASET_DIR="$HOME/aic_dataset"   # $HOME = EBS, /tmp değil — spot stop'ta korunur
S3_BUCKET="s3://aic-yusuf/aic_dataset"
N_BATCHES=67        # 67 batch × 3 episode = ~200 episode
BATCH_TIMEOUT=360   # saniye (CheatCode ~30s/episode → 3 ep = ~90s + overhead)

mkdir -p "$DATASET_DIR/logs"

echo "=== VERİ TOPLAMA BAŞLIYOR: $N_BATCHES batch ==="
echo "Başlangıç zamanı: $(date)"

for i in $(seq 1 $N_BATCHES); do
  START_TIME=$(date +%s)
  EPISODES_BEFORE=$(ls "$DATASET_DIR"/episode_*.hdf5 2>/dev/null | wc -l)

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "BATCH $i / $N_BATCHES  |  Mevcut episode: $EPISODES_BEFORE"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # 1. Eval container'ı arka planda başlat
  distrobox enter --root aic_eval -- /entrypoint.sh \
    ground_truth:=true \
    start_aic_engine:=true \
    > "$DATASET_DIR/logs/eval_batch_$(printf '%04d' $i).log" 2>&1 &
  EVAL_PID=$!
  echo "  [1/3] Eval container başlatıldı (PID=$EVAL_PID), hazır olması bekleniyor..."

  # 2. Eval container'ın hazır olmasını bekle (Gazebo + aic_engine)
  sleep 45

  # 3. DataCollectorPolicy'yi çalıştır (3 episode kaydeder ve çıkar)
  echo "  [2/3] DataCollectorPolicy başlatılıyor..."
  timeout $BATCH_TIMEOUT pixi run ros2 run aic_model aic_model \
    --ros-args -p use_sim_time:=true \
    -p policy:=aic_data_collector.ros.DataCollectorPolicy
  DC_EXIT=$?

  # 4. Eval container'ı durdur ve temizle
  echo "  [3/3] Eval container durduruluyor..."
  kill $EVAL_PID 2>/dev/null
  distrobox stop --root aic_eval 2>/dev/null || true
  sleep 5

  # 5. Sonuç raporu
  EPISODES_AFTER=$(ls "$DATASET_DIR"/episode_*.hdf5 2>/dev/null | wc -l)
  NEW_EPS=$((EPISODES_AFTER - EPISODES_BEFORE))
  END_TIME=$(date +%s)
  ELAPSED=$((END_TIME - START_TIME))

  echo ""
  echo "  ✓ Batch $i tamamlandı:"
  echo "    Yeni episode: $NEW_EPS  |  Toplam: $EPISODES_AFTER  |  Süre: ${ELAPSED}s"
  du -sh "$DATASET_DIR" | awk '{print "    Disk: " $1}'

  # DataCollector timeout olduysa (3 episode gelmedi) — uyar ama devam et
  if [ $DC_EXIT -eq 124 ]; then
    echo "  ⚠ UYARI: Batch $i timeout oldu! Log: $DATASET_DIR/logs/eval_batch_$(printf '%04d' $i).log"
  fi

  # 6. S3'e sync — spot terminate olsa bile veri kaybolmaz
  echo "  S3 sync başlıyor..."
  aws s3 sync "$DATASET_DIR/" "$S3_BUCKET/" \
    --exclude "logs/*" \
    && echo "  S3 sync tamamlandı ✓ ($S3_BUCKET)" \
    || echo "  ⚠ S3 SYNC HATASI! IAM rolünü kontrol et: aws s3 ls s3://aic-yusuf/"

  # Sonraki batch için kısa bekleme (Zenoh bağlantı temizliği)
  sleep 10
done

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     VERİ TOPLAMA TAMAMLANDI ✓            ║"
echo "╚══════════════════════════════════════════╝"
echo "Bitiş zamanı: $(date)"
echo ""
pixi run python3 aic_data_collector/scripts/inspect_dataset.py "$DATASET_DIR"
```

### 5c. Anlık İzleme (Ayrı bir SSH bağlantısından)

```bash
# Yeni terminal/SSH ile bağlanarak ilerlemeyi izle:
watch -n 10 'echo "=== Dataset ===" && \
  ls ~/aic_dataset/episode_*.hdf5 2>/dev/null | wc -l && \
  du -sh ~/aic_dataset/ && \
  echo "" && \
  echo "=== Son log satırları ===" && \
  ls -t ~/aic_dataset/logs/run_*.log 2>/dev/null | head -1 | xargs tail -5 2>/dev/null'
```

---

## ADIM 6 — Dataset'i Sakla / Spot Terminate Sonrası Resume

Batch döngüsü her episode'dan sonra S3'e sync yaptığı için veri zaten güvende.
Bu adım final yükleme veya yeni instance'ta resume için kullanılır.

```bash
# Final S3 yüklemesi (batch script bitmişse)
aws s3 sync ~/aic_dataset/ s3://aic-yusuf/aic_dataset/ --exclude "logs/*"

# Dataset'i Mac'e indir (Mac terminalinde çalıştır)
scp -i ~/Desktop/aic-key.pem \
  ubuntu@<aws-ip>:~/aic_dataset/summary.jsonl ~/Desktop/
# Tüm dataset için (büyük olabilir):
# aws s3 sync s3://aic-yusuf/aic_dataset/ ~/Desktop/aic_dataset/
```

### Spot Terminate Sonrası Yeni Instance'ta Resume

```bash
# 1. Yeni instance'ta kurulum tamamlandıktan sonra (Adım 1-4 tekrar):

# 2. S3'ten veriyi geri al
aws s3 sync s3://aic-yusuf/aic_dataset/ ~/aic_dataset/
echo "Kaç episode var: $(ls ~/aic_dataset/episode_*.hdf5 2>/dev/null | wc -l)"

# 3. Batch script'i olduğu gibi başlat — policy kaldığı yerden devam eder.
# DataCollectorPolicy __init__'te mevcut .hdf5 sayısını sayar ve
# episode_count'u otomatik ayarlar. Üzerine yazmaz.
```

---

## ADIM 7 — Dataset İstatistikleri Kontrol

```bash
cd ~/ws_aic/src/aic
pixi run python3 aic_data_collector/scripts/inspect_dataset.py ~/aic_dataset
```

Beklenen çıktı:
```
============================================================
Dataset: /home/ubuntu/aic_dataset
Toplam episode: 200
============================================================

Başarı oranı:     195/200 = 97.5%
SFP insertion:    134
SC insertion:     66

Adım istatistikleri:
  Ortalama:       530.2 adım
  ...

Disk kullanımı:  1.6 GB
```

---

## Sorun Giderme

### Eval container başlamıyor
```bash
# Container'ı sil ve yeniden oluştur
distrobox rm --root aic_eval --force
distrobox create --root --nvidia \
  --image ghcr.io/intrinsic-dev/aic/aic_eval:latest \
  --name aic_eval
```

### "ground_truth TF frame bulunamadı" hatası
```bash
# ground_truth:=true ile başlatıldığından emin ol:
distrobox enter --root aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=true
# NOT: ground_truth:=false ile başlatırsan CheatCode çalışmaz → DataCollector veri kaydedemez
```

### Pixi reinstall gerekiyor
```bash
cd ~/ws_aic/src/aic
pixi reinstall ros-kilted-aic-data-collector
```

### Dataset çok küçük (az adım)
CheatCode'un approach + descent süresini kontrol et. Normal değerler:
- Approach: ~100 adım (5 saniye)  
- Descent: ~400-450 adım (20-22 saniye)
- Toplam: ~500-550 adım/episode

---

## Sonraki Adım: Eğitim

Veri toplama bittikten sonra bu aynı instance'ta eğitimi başlatacağız.
Eğitim için AGENT_TRAIN.md dosyasını kullan (henüz yazılmadı, veri toplandıktan sonra devreye girecek).
