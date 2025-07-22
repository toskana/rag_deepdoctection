FROM deepdoctection/deepdoctection:latest

WORKDIR /repo

# requirements.txt in den Container kopieren
COPY requirements.txt .

RUN apt-get update && \
    apt install -y qpdf

RUN python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8888
CMD ["bash"]
