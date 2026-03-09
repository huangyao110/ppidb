# Base image with Miniconda3
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential for gcc, git, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install mmseqs2 via bioconda
RUN conda install -c conda-forge -c bioconda mmseqs2 python-igraph pandas polars tqdm ipykernel -y

# Copy project files
COPY . .

# Install ppidb in editable mode
RUN pip install -e .

# Default command: launch a bash shell
CMD ["/bin/bash"]
