FROM ubuntu:24.04

RUN apt update && apt install -y python3 python3-pip

# default values are for macOS users
ARG UID=501
ARG GID=20
ARG USERNAME=zivlazarov

RUN useradd $USERNAME -m -u $UID -g $GID && usermod -aG sudo $USERNAME

USER $USERNAME

RUN python3 -m pip install numpy pandas scikit-learn matplotlib seaborn \
            torch torchvision torchtext --break-system-packages
