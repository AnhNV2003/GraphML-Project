# GPU-enabled image: PyTorch 2.8 + CUDA 12.8 (matches torch==2.8.0+cu128 in requirements.txt).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# No apt packages needed: assets/external_repos/LightGCN-PyTorch is baked into
# the image below (COPY . .), so official.py's git-clone fallback never
# triggers in the default flow; torch-geometric's GATConv works with the
# pure-Python/pip-installed build (no compiled extensions required).

# Install Python deps first (better layer caching across code-only changes).
# Base image already ships the matching torch build, so skip reinstalling it.
COPY requirements.txt .
RUN grep -v '^torch==' requirements.txt > requirements.no-torch.txt \
    && pip install --no-cache-dir -r requirements.no-torch.txt

COPY . .

# assets/data/ is volume-mounted at run time (see docker-compose.yml) since the
# dataset (several GB) shouldn't be baked into the image; assets/external_repos/
# (the 6 vendored reference repos, ~250MB) IS baked in via `COPY . .` above.

CMD ["bash"]
