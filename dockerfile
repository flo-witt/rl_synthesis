# Použijeme oficiální Python 3.10 obraz
FROM python:3.10-slim

# Nastavíme pracovní adresář
WORKDIR /app

# Nastavíme proměnné prostředí
ENV PAYNT_ROOT=/app \
    PREREQUISITES=${PAYNT_ROOT}/prerequisites \
    COMPILE_JOBS=8 \
    DEBIAN_FRONTEND=noninteractive

# Instalace systémových závislostí
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    libboost-all-dev \
    libcln-dev \
    libgmp-dev \
    libginac-dev \
    automake \
    libglpk-dev \
    libhwloc-dev \
    libz3-dev \
    libxerces-c-dev \
    libeigen3-dev \
    maven \
    uuid-dev \
    graphviz && \
    rm -rf /var/lib/apt/lists/*

# Vytvoření adresáře pro závislosti
RUN mkdir -p ${PAYNT_ROOT}/prerequisites

# Nastavení a aktivace Python virtuálního prostředí
RUN python3 -m venv ${PAYNT_ROOT}/prerequisites/venv
ENV PATH="${PAYNT_ROOT}/prerequisites/venv/bin:$PATH"

# Upgrade pip a instalace základních balíčků
RUN pip install --upgrade pip && \
    pip install wheel

# Instalace Paynt a dalších Python závislostí
RUN pip install paynt click z3-solver psutil graphviz && \
    pip install jax==0.5.3 && \
    pip install tensorflow==2.15 ml-dtypes==0.2.0 && \
    pip install tf-agents==0.19.0 tqdm dill matplotlib pandas seaborn networkx && \
    pip install aalpy scikit-learn

# Zkopírujeme zdrojové soubory do kontejneru
COPY . .

# Instalace VecStorm a rl_src v "editable" módu
RUN cd /app/VecStorm && \
    pip install -e . && \
    cd /app/rl_src && \
    pip install -e .

# Příkaz pro spuštění (uprav podle potřeby)

RUN pip install tf-agents==0.19.0
CMD ["bash"]
