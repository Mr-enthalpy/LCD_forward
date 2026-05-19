# Server usage guide (sanitized)

> Operational notes for remote GPU servers used in the thesis workflow.
> This document contains no passwords, IPs, or personal paths.
> Placeholders `<...>` indicate values to fill per user.

## 1. SSH access

```
ssh -p <port> <user>@<server_ip>
```

Some servers use a non-root user; root requires `su` with the root password after login.

## 2. File transfer

Transfer files/directories via `scp`:

```bash
scp -r -P <port> <local_path> <user>@<server_ip>:<remote_path>
```

## 3. SSH known_hosts conflict after reinstall

If SSH fails with a host-key mismatch after the server is reinstalled:

```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
```

Fix: open `~/.ssh/known_hosts`, delete the line for the conflicting IP, then reconnect.

## 4. Create a personal directory on the server

Common data mount points:
- `/home/data1/`
- `/home/data2/`

```bash
mkdir /home/data2/<your_name>
```

## 5. Docker basics

### View containers and images

```bash
sudo docker ps -a
sudo docker image ls
```

### Create and enter a container

```bash
sudo docker run -d \
  -p <free_port>:22 \
  -it \
  -v /home/data2/<your_dir>:/home \
  --gpus '"device=0,1,2,3"' \
  --shm-size 8G \
  <image_name> \
  /bin/bash
```

| Flag | Meaning |
|------|---------|
| `-p <host_port>:22` | Map host port to container SSH port 22 |
| `-v <host_dir>:<container_dir>` | Mount host directory into container |
| `--gpus '"device=0,1,2,3"'` | Assign specific GPUs |
| `--shm-size 8G` | Shared memory size (important for PyTorch dataloaders) |

Rename container:
```bash
sudo docker rename <old_name> <new_name>
```

Attach to running container:
```bash
sudo docker attach <container_name>
```

> Avoid mounting directories that belong to other users into your container.

## 6. Container SSH configuration

### Install SSH server in container

```bash
apt-get install -y vim openssh-server
```

### Configure sshd

```bash
vim /etc/ssh/sshd_config
```

Common settings to enable:
```
PermitRootLogin yes
PasswordAuthentication yes
```

Then start SSH:
```bash
service ssh start
# or
/usr/sbin/sshd
```

## 7. DNS fix inside container

If the container cannot resolve hostnames, check current DNS:

```bash
cat /etc/resolv.conf
```

Set to a reliable public DNS (temporary; lost on container restart):

```bash
echo "nameserver 8.8.8.8" > /etc/resolv.conf
echo "nameserver 8.8.4.4" >> /etc/resolv.conf
```

## 8. apt mirror configuration (speed up downloads)

### Check OS release

```bash
cat /etc/issue
```

Example output: `Debian GNU/Linux 11 (bullseye)`

### Replace apt sources with Aliyun mirror

Backup the existing sources:

```bash
mv /etc/apt/sources.list /etc/apt/sources.list.bak
```

Write the mirror sources (example for Debian 11 bullseye):

```bash
cat >> /etc/apt/sources.list << EOF
deb https://mirrors.aliyun.com/debian/ bullseye main non-free contrib
deb-src https://mirrors.aliyun.com/debian/ bullseye main non-free contrib
deb https://mirrors.aliyun.com/debian-security/ bullseye-security main
deb-src https://mirrors.aliyun.com/debian-security/ bullseye-security main
deb https://mirrors.aliyun.com/debian/ bullseye-updates main non-free contrib
deb-src https://mirrors.aliyun.com/debian/ bullseye-updates main non-free contrib
deb https://mirrors.aliyun.com/debian/ bullseye-backports main non-free contrib
deb-src https://mirrors.aliyun.com/debian/ bullseye-backports main non-free contrib
EOF
```

Update:

```bash
apt-get update
```

For other distributions, find mirror addresses at:
- Aliyun: https://developer.aliyun.com/mirror/
- Tsinghua: https://mirrors.tuna.tsinghua.edu.cn/

## 9. Reference

- Docker basics: https://yeasy.gitbook.io/docker_practice/
- Debian Aliyun mirror: https://developer.aliyun.com/mirror/debian
- apt mirror setup: https://blog.csdn.net/gu19930914/article/details/119000362
