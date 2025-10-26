# Fix GPU NVIDIA GeForce 4060 MAX-Q

## Problema

Kernel-ul Linux 6.14 este prea nou pentru driverul nvidia-driver-580-open instalat.

## Soluție 1: Reinstalare driver NVIDIA

```bash
# Actualizează sistemul
sudo apt update && sudo apt upgrade -y

# Reinstalează driverul NVIDIA pentru kernel-ul curent
sudo apt install --reinstall nvidia-driver-580-open

# Sau instalează ultimul driver disponibil
sudo ubuntu-drivers install

# Rebuild module pentru kernel
sudo dkms autoinstall

# Reboot
sudo reboot
```

După reboot, verifică:

```bash
nvidia-smi
```

## Soluție 2: Downgrade kernel (dacă Soluția 1 nu merge)

```bash
# Listează kernel-urile instalate
dpkg --list | grep linux-image

# Boot cu un kernel mai vechi (6.8 sau 6.11) din GRUB
# și apoi:
sudo apt-mark hold linux-image-generic linux-headers-generic
```

## Verificare după fix

```bash
cd /home/gabriel/code/working/cameraSocket
./venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Optimizări făcute oricum (fără GPU)

✅ **Camera buffer**: Set la 1 pentru latență mică
✅ **Rezoluție optimizată**: 640x480 pentru procesare rapidă  
✅ **JPEG quality**: 80% pentru encoding mai rapid
✅ **Frame size**: imgsz=640 pentru YOLOv8
✅ **Sleep redus**: 0.01s între frame-uri
✅ **GPU ready**: Codul detectează automat CUDA când devine disponibil

Aceste optimizări reduc lag-ul și fără GPU!
