# Docker

## Common

| Command | Description | Example |
| --- | --- | --- |
| `docker pull` | Download image from a registry | `docker pull hello-world` | 
| `docker ps` | List running containers | `docker ps` <> `docker ps --all` |
| `docker images` | List all images | - |
| `docker run` | Create and run a new container from an image | `docker run hello-world` <> `docker run --name my_container hello-world` <> `docker run -it --entrypoint bash hello-world` | 
| `docker exec` | Execute a command in a running container | `docker exec -it my_container_name bash` | 
| `docker build` | Build an image from Dockerfile | `docker build -t my_container_name -f /path/to/dockerfile .`  | 
| `docker export` | Export running container filesystem to .tar | `docker run -d --name my_container_name my_image_name && docker export my_container_name -o my_container_filesystem.tar`  | 
| `docker save` | Save image to a tar archive (streamed to STDOUT by default) | `docker save my_image_name > my_image_name.tar`  | 
| `docker inspect` | Returns low-level information about container | `docker inspect my_image_name`  | 
| `docker image prune` | Removes all unused images | - | 
| `docker cp` | Copy files/folders between a container and the local filesystem | `docker cp my_running_container:/path/to/file.txt file.txt` (and vice versa) | 
| `sudo systemctl restart docker` | Restart docker daemon (start / stop /restart) | -  | 
| `sudo usermod -aG docker $USER` | Add user to docker group to manage docker as non-root user | -  | 


## Installing docker
[docs](https://docs.docker.com/engine/install/ubuntu/)

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update 
```
```bash
# Install the docker packages
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Verify installation
sudo docker run hello-world
```


## Installing Nvidia Container Toolkit
[docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
# Configure repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update

# Install 
sudo apt-get install -y nvidia-container-toolkit

# Configure the container runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart daemon
sudo systemctl restart docker

# Verify installation
sudo docker run --gpus all -it ubuntu:22.04 nvidia-smi
```


# Examples

## Unpack image
...see exapmle-unpack...

## Use docker as environment to run scripts
...see example-iris...

## Use docker as a service: jupyterhub
[docs](https://jupyterhub.readthedocs.io/en/latest/tutorial/quickstart-docker.html)
...see example-jupyterhub...
