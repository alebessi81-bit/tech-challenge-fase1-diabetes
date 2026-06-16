# Imagem base Python slim (leve e estável)
FROM python:3.11-slim

# Metadados
LABEL maintainer="Tech Challenge Fase 1"
LABEL description="Diagnóstico de Diabetes com Machine Learning"

# Diretório de trabalho dentro do container
WORKDIR /app

# Copiar requirements primeiro (aproveita cache do Docker)
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar o restante do projeto
COPY . .

# Criar pastas necessárias (caso não existam)
RUN mkdir -p data/processed data/models notebooks

# Comando padrão: rodar o pipeline completo
CMD ["python", "main.py"]
