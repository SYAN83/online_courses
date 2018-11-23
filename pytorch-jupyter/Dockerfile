
FROM pytorch/pytorch:latest

# install jupyter
RUN conda install -y jupyter

# Copy configs to Docker image
COPY configs/ /conf/

# Create ubuntu user
RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu

# Jupyter notebook Port
EXPOSE 8888

ENTRYPOINT /conf/bootstrap.sh; bash
