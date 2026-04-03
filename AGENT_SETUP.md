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

Önce local repo'yu commit et ve push'la. Sonra SSH ile AWS'e bağlan.

```bash
# Mac'te çalıştır:
cd /Users/y.aykut/Desktop/aic
git add aic_data_collector/ pixi.toml
git commit -m "feat: DataCollectorPolicy ve veri toplama paketi eklendi"
git push
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

İki terminal aç (veya tmux kullan):

### Terminal 1 — Eval Container (ground_truth=true ile)

```bash
export DBX_CONTAINER_MANAGER=docker
distrobox enter --root aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true
```

### Terminal 2 — DataCollector Policy

```bash
cd ~/ws_aic/src/aic

# DataCollectorPolicy'yi çalıştır
# aic_engine otomatik olarak 3 trial yapıp kapanacak,
# sonra yeniden başlatman gerekiyor (veya bir loop kullan).

# TEK ÇALIŞTIRMA (3 episode = 3 trial):
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=aic_data_collector.ros.DataCollectorPolicy

# Dataset'e bak:
ls -lh /tmp/aic_dataset/
pixi run python3 aic_data_collector/scripts/inspect_dataset.py /tmp/aic_dataset
```

### Terminal 2 — LOOP ile ~67 çalıştırma = ~200 episode

```bash
cd ~/ws_aic/src/aic

for i in $(seq 1 67); do
  echo ""
  echo "======================================="
  echo "Çalıştırma $i / 67 (episode $((($i-1)*3+1))-$(($i*3)))"
  echo "======================================="

  # Eval container'ın hazır olmasını bekle (30 sn)
  sleep 30

  # DataCollector'ı çalıştır — aic_engine 3 trial yapıp bitince kapanır
  timeout 300 pixi run ros2 run aic_model aic_model \
    --ros-args -p use_sim_time:=true \
    -p policy:=aic_data_collector.ros.DataCollectorPolicy

  echo "Çalıştırma $i tamamlandı."
  echo "Şu anki dataset:"
  ls /tmp/aic_dataset/ | wc -l
  du -sh /tmp/aic_dataset/

  # Eval container yeniden başlatıldıktan sonra hazır olması için bekle
  sleep 10
done

echo ""
echo "=== VERİ TOPLAMA TAMAMLANDI ==="
pixi run python3 aic_data_collector/scripts/inspect_dataset.py /tmp/aic_dataset
```

> **NOT:** Her "çalıştırma"da eval container zaten çalışıyor olmalı (Terminal 1).
> aic_engine her run'da 3 trial yapar ve otomatik reset eder.
> Terminal 1'deki eval container'ı 67 çalıştırma boyunca kapalı tutma.

---

## ADIM 6 — Dataset'i Sakla

Veri toplama bittikten sonra dataset'i S3'e yükle veya tar.gz ile sakla:

```bash
# Seçenek A: tar.gz
cd /tmp
tar -czf aic_dataset_$(date +%Y%m%d).tar.gz aic_dataset/
ls -lh aic_dataset_*.tar.gz

# Seçenek B: AWS S3 (bucket önceden oluşturulmuş olmalı)
# aws s3 cp /tmp/aic_dataset_$(date +%Y%m%d).tar.gz s3://your-bucket/

# Seçenek C: Dataset'i Mac'e indir (SCP)
# Yerel terminalde:
# scp -i key.pem ubuntu@<aws-ip>:/tmp/aic_dataset_*.tar.gz ~/Desktop/
```

---

## ADIM 7 — Dataset İstatistikleri Kontrol

```bash
cd ~/ws_aic/src/aic
pixi run python3 aic_data_collector/scripts/inspect_dataset.py /tmp/aic_dataset
```

Beklenen çıktı:
```
============================================================
Dataset: /tmp/aic_dataset
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
