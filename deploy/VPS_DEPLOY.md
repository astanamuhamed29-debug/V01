# SELF-OS VPS deploy (Ubuntu + systemd + nginx)

## 1) Подготовка сервера

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git nginx
```

## 2) Код и окружение

```bash
sudo mkdir -p /opt/self-os
sudo chown -R $USER:$USER /opt/self-os
git clone https://github.com/astanamuhamed29-debug/V01.git /opt/self-os
cd /opt/self-os
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp deploy/.env.vps.example .env
```

Заполните `.env` реальными секретами.

## 3) systemd

```bash
sudo cp deploy/systemd/self-os-bot.service /etc/systemd/system/self-os-bot.service
sudo systemctl daemon-reload
sudo systemctl enable self-os-bot
sudo systemctl start self-os-bot
sudo systemctl status self-os-bot
```

Логи:

```bash
journalctl -u self-os-bot -f
```

## 4) nginx

```bash
sudo cp deploy/nginx/self-os.conf /etc/nginx/sites-available/self-os.conf
sudo ln -sf /etc/nginx/sites-available/self-os.conf /etc/nginx/sites-enabled/self-os.conf
sudo nginx -t
sudo systemctl reload nginx
```

Проверка:

```bash
curl -i http://YOUR_SERVER_IP/healthz
```

## 5) Обновление релиза

```bash
cd /opt/self-os
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart self-os-bot
```
