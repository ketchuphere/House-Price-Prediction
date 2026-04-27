# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Node builder: install deps and build the React app
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS builder

WORKDIR /app

# Copy package files first to exploit Docker layer caching.
# npm install only re-runs when package*.json changes.
COPY package.json package-lock.json* ./

RUN npm ci --silent

# Copy source and build
COPY . .
RUN npm run build
# Output: /app/dist/


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — nginx: serve the static build
# ─────────────────────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy our nginx config (injected from project root via docker-compose)
COPY ../nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
