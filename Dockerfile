# Stage 1: Node.js builder
FROM node:18-slim as node-builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev

# ============ FINAL RUNTIME STAGE ============
FROM python:3.11-slim

# Install Node.js (from NodeSource) + other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl bash ca-certificates wget gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Node modules
COPY --from=node-builder /app/node_modules ./node_modules
COPY --from=node-builder /app/package*.json ./

# Copy app code
COPY . .

# Create required directories for temp files
RUN mkdir -p yolo/uploads yolo/outputs yolo/temp

EXPOSE 5000
CMD ["node", "src/server.js"]





