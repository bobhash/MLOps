В данном примере мы разберем внутренности контейнера.

```bash
# Используем готовый image в качестве базового слоя
docker run -d --name container_to_export python:3.12-slim && \
    docker export container_to_export -o container_to_export.tar && \
    docker stop container_to_export && \
    docker rm container_to_export

# Импортируем тарник в качестве базового слоя
docker import container_to_export.tar my_python_container:latest

# Билдим
docker build . -t container_to_unpack

# Сохраняем
docker save container_to_unpack > container_to_unpack.tar

# Распаковываем
mkdir container_to_unpack && \
    tar -xvf container_to_unpack.tar -C container_to_unpack

# Удаляем
docker rmi my_python_container container_to_unpack && \
    rm -rf container_to_unpack && \
    rm -f container_to_export.tar container_to_unpack.tar
```
