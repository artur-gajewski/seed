FROM node:24-bookworm-slim AS python-base
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
ENV SEED_PYTHON=/opt/seed-python/bin/python \
    SEED_DEVICE=cpu \
    PYTHONDONTWRITEBYTECODE=1
RUN python3 -m venv /opt/seed-python \
    && /opt/seed-python/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/seed-python/bin/pip install --no-cache-dir 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu \
    && /opt/seed-python/bin/pip install --no-cache-dir 'numpy>=2,<3'
COPY python/pyproject.toml ./python/pyproject.toml
COPY python/seed_torch ./python/seed_torch
RUN /opt/seed-python/bin/pip install --no-cache-dir --no-deps ./python

FROM python-base AS build
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
COPY public ./public
COPY shared ./shared
COPY server ./server
COPY tests ./tests
COPY scripts ./scripts
COPY python/tests ./python/tests
COPY data/inputs ./data/inputs
COPY data/references ./data/references
RUN npm test && /opt/seed-python/bin/python -m unittest discover -s python/tests -v && npm run build

FROM python-base AS runtime
ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=3001
COPY --from=build --chown=node:node /app/dist ./dist
COPY --from=build --chown=node:node /app/server ./server
COPY --from=build --chown=node:node /app/shared ./shared
COPY --from=build --chown=node:node /app/package.json ./package.json
COPY --from=build --chown=node:node /app/scripts ./scripts
COPY --from=build --chown=node:node /app/src/lib/formatDuration.ts ./src/lib/formatDuration.ts
COPY --from=build --chown=node:node /app/data/inputs ./data/inputs
COPY --from=build --chown=node:node /app/data/references ./data/references
RUN mkdir -p /app/data/models && chown -R node:node /app/data
USER node
EXPOSE 3001
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:' + process.env.PORT + '/api/status').then(async r => process.exit(r.ok && !(await r.json()).error ? 0 : 1)).catch(() => process.exit(1))"
CMD ["node", "server/index.ts"]
